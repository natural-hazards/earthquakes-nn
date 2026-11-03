"""Train one ternary (E/M/T) model and store metrics in results/<name>.json.

Selection mode (default): train on the train split, per-epoch F1 printed during
training is on the VALIDATION split; val and test metrics are stored.

Final mode (--final): train on train + validation (the adapter's full 70 % training
split) for a fixed number of epochs and evaluate once on the test split. Results go
to results_final/, the model weights to models/.

Usage: python exp.py --model lstm --duration 20 --weighting none --log 0 --seed 0
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score, balanced_accuracy_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from quake.data.dataset import WaveformDataset
from quake.data.augmentation import Compose, RandomApply, TimeShift, AddNoise, AmplitudeScale, TimeStretch
from quake.data.transforms import FFTTransform, STFTTransform, ToTensor
from quake.models.lstm import LSTMModel
from quake.models.lstm_mhsa import LSTMAttentionModel
from quake.models.lstm_svm import LSTMAttentionSVMModel
from quake.models.lstm_fcn import LSTMFCNModel
from quake.models import EfficientNet, EfficientNetBackbone, HybridViT, Backbone
from quake.procs.train import train_model, train_svm_model

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['lstm', 'lstm_attn', 'lstm_svm', 'lstm_fcn', 'effnet', 'hvit'], required=True)
p.add_argument('--fcn', choices=['karim', 'small'], default='karim',
               help='lstm_fcn preset: karim = LSTM 128 + FCN 128/256/128; small = LSTM 16 + FCN 16/16/16 (bachelor thesis)')
p.add_argument('--task', choices=['EMT', 'E_M', 'T_M', 'E_T'], default='EMT',
               help='ternary, or a binary task; the first class of the name is the positive class (label 0)')
p.add_argument('--duration', type=int, default=20)
p.add_argument('--weighting', choices=['none', 'sqrt', 'inverse'], default='none')
p.add_argument('--log', type=int, default=0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--gamma', type=float, default=0.85)
p.add_argument('--layers', type=int, default=3, help='LSTM layers (lstm_attn, lstm_svm)')
p.add_argument('--hidden', type=int, default=128, help='LSTM hidden size (lstm_attn, lstm_svm)')
p.add_argument('--dropout', type=float, default=0.3, help='dropout (lstm_attn, lstm_svm)')
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--clip', type=float, default=0, help='max gradient norm, 0 = no clipping')
p.add_argument('--final', action='store_true', help='train on train+val, evaluate on test')
p.add_argument('--tag', default='', help='suffix for the results/models directories (e.g. timing)')
p.add_argument('--undersample', type=int, default=0, help='keep at most N training events of class M (index 1)')
p.add_argument('--limit', type=int, default=0, help='debug: use only N training events')
a = p.parse_args()

name = f"{a.model}{'' if a.task == 'EMT' else '_' + a.task}_L{a.duration}_w{a.weighting}_log{a.log}_g{a.gamma}_e{a.epochs}"
if a.model in ('lstm_attn', 'lstm_svm'):
    name += f'_l{a.layers}_h{a.hidden}_d{a.dropout}_lr{a.lr:g}_c{a.clip:g}'
if a.model == 'lstm_fcn':
    name += f'_{a.fcn}_lr{a.lr:g}'
if a.undersample:
    name += f'_u{a.undersample}'
name += f'_s{a.seed}'
print('RUN', name, 'FINAL' if a.final else 'SELECTION', flush=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
device = 'cuda'

d = np.load(f'cache/L{a.duration}s.npz')
all_classes = [str(c) for c in d['classes']]  # ['E', 'M', 'T']
classes = all_classes if a.task == 'EMT' else a.task.split('_')
keep_idx = [all_classes.index(c) for c in classes]
lut = np.full(len(all_classes), -1)
lut[keep_idx] = np.arange(len(classes))


def select(split):
    """Events of the task's classes, relabelled 0..K-1 in task order."""
    x, y = d[f'x_{split}'], d[f'y_{split}']
    m = np.isin(y, keep_idx)
    return x[m], lut[y[m]]


x_tr, y_tr = select('train')
x_va, y_va = select('val')
x_te, y_te = select('test')
if a.final:
    x_tr = np.concatenate([x_tr, x_va])
    y_tr = np.concatenate([y_tr, y_va])
if a.limit:
    x_tr, y_tr = x_tr[:a.limit], y_tr[:a.limit]
if a.undersample:
    rng = np.random.default_rng(42)
    idx_m = np.flatnonzero(y_tr == 1)
    drop = rng.choice(idx_m, size=max(0, len(idx_m) - a.undersample), replace=False)
    keep = np.setdiff1d(np.arange(len(y_tr)), drop)
    x_tr, y_tr = x_tr[keep], y_tr[keep]

if a.model in ('lstm', 'lstm_attn', 'lstm_svm', 'lstm_fcn'):
    seq_len = x_tr.shape[1]
    fft_size = 1 << (seq_len - 1).bit_length()
    aug = Compose([
        RandomApply(TimeShift(max_shift=0.1), p=0.5), RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
        RandomApply(AmplitudeScale(0.9, 1.1), p=0.3), RandomApply(TimeStretch(0.95, 1.05), p=0.3)])
    tf_train = nn.Sequential(aug, FFTTransform(fft_size, log=bool(a.log)), ToTensor())
    tf_test = nn.Sequential(FFTTransform(fft_size, log=bool(a.log)), ToTensor())
    batch_size = 32
else:
    aug = Compose([
        RandomApply(TimeShift(max_shift=0.1), p=0.5), RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
        RandomApply(AmplitudeScale(0.9, 1.1), p=0.3)])
    stft = STFTTransform(nperseg=64, hop=32, log=bool(a.log))
    tf_train = nn.Sequential(aug, stft, ToTensor(), T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)))
    tf_test = nn.Sequential(stft, ToTensor())
    batch_size = 64

ds_train = WaveformDataset(x_tr, y_tr, tf_train)
ds_val = WaveformDataset(x_va, y_va, tf_test)
ds_test = WaveformDataset(x_te, y_te, tf_test)
shape = tuple(ds_test[0][0].shape)
print(f'task {a.task} classes {classes} input {shape}, train {np.bincount(y_tr)} val {np.bincount(ds_val.labels)} test {np.bincount(ds_test.labels)}', flush=True)

K = len(classes)
if a.model == 'lstm':
    model = LSTMModel(channels=3, classes=K, hidden=128, layers=3, dropout=0.3)
elif a.model == 'lstm_attn':
    model = LSTMAttentionModel(channels=3, classes=K, hidden=a.hidden, layers=a.layers, heads=4, dropout=a.dropout)
elif a.model == 'lstm_svm':
    model = LSTMAttentionSVMModel(channels=3, classes=K, hidden=a.hidden, layers=a.layers, heads=4,
                                  dropout=a.dropout, svm_alpha=1e-6, svm_eta0=1e-3)
elif a.model == 'lstm_fcn':
    preset = dict(hidden=128, conv_channels=(128, 256, 128)) if a.fcn == 'karim' else dict(hidden=16, conv_channels=(16, 16, 16))
    model = LSTMFCNModel(channels=3, classes=K, layers=1, kernels=(8, 5, 3), dropout=0.8, **preset)
elif a.model == 'effnet':
    model = EfficientNet(backbone=EfficientNetBackbone.B0, in_channels=3, num_classes=K, dropout=0.2)
else:
    model = HybridViT(backbone=Backbone.TINY, depth=4, in_channels=3, num_classes=K,
                      img_size=shape[1:], dropout=0.2)
model = model.to(device)

# In final mode the number of epochs is fixed in advance (chosen from the selection runs);
# the per-epoch printout on the test split is informational only.
train_fn = train_svm_model if a.model == 'lstm_svm' else train_model
t = time.time()
train_fn(ds_train, ds_test if a.final else ds_val, model, batch_size=batch_size, batch_size_test=batch_size,
         epochs=a.epochs, device=device, compile_model=False,
         weighting=None if a.weighting == 'none' else a.weighting, lr_gamma=a.gamma,
         lr=a.lr, grad_clip=a.clip or None)
elapsed = time.time() - t


def predict(ds):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).float()
            # binary SVM returns a single decision value (> 0 -> class 1); CE models return logits
            score = torch.stack([-out, out], 1) if out.dim() == 1 else torch.softmax(out, 1)
            ps.append(score.cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


res = dict(name=name, args=vars(a), time=elapsed)
# inference cost on the test split (incl. on-the-fly FFT/STFT and data loading)
t0 = time.time(); predict(ds_test); res['infer_time'] = time.time() - t0; res['n_test'] = len(ds_test)
print(f"inference: {len(ds_test)} events in {res['infer_time']:.1f} s = {len(ds_test)/res['infer_time']:.0f} events/s")
splits = (('test', ds_test),) if a.final else (('val', ds_val), ('test', ds_test))
for split, ds in splits:
    y, score = predict(ds)
    pr = score.argmax(1)
    res[split] = dict(macro_f1=f1_score(y, pr, average='macro'), f1=f1_score(y, pr, average=None).tolist(),
                      cm=confusion_matrix(y, pr).tolist(), balanced_acc=balanced_accuracy_score(y, pr))
    if K == 2:
        res[split]['roc_auc'] = roc_auc_score(y == 0, score[:, 0])  # positive class = first class of the task
        print(f"{split} ROC-AUC {res[split]['roc_auc']:.4f}")
    print(f'== {split}\n{confusion_matrix(y, pr)}\n{classification_report(y, pr, target_names=classes, digits=3)}')
print(f'time {elapsed:.0f}s')

if not a.limit:
    out_dir = ('results_final' if a.final else 'results') + (f'_{a.tag}' if a.tag else '')
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f'{out_dir}/{name}.json', 'w'), indent=1)
    if a.final:
        mdir = 'models' + (f'_{a.tag}' if a.tag else '')
        os.makedirs(mdir, exist_ok=True)
        torch.save(model.state_dict(), f'{mdir}/{name}.pt')
print('DONE', name, flush=True)
