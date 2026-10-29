"""Train one model on the combined 100 Hz dataset and store metrics as JSON.

  --data  all | orig | okc   source(s) used for training and validation
  --task  EMT | E_M | T_M | E_T   ternary or one binary task (classes kept, others dropped)
  The test split is always evaluated as a whole and per source.

Selection mode (default): per-epoch F1 printed during training is on the validation split.
Final mode (--final): train on train + validation, evaluate once on the test split,
results go to results_final/ and the weights to models/.

Usage: python exp.py --model effnet --duration 8 --data all --task EMT --seed 0
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from quake.data.dataset import WaveformDataset
from quake.data.augmentation import Compose, RandomApply, TimeShift, AddNoise, AmplitudeScale, TimeStretch
from quake.data.transforms import FFTTransform, STFTTransform, ToTensor
from quake.models.lstm import LSTMModel
from quake.models.lstm_mhsa import LSTMAttentionModel
from quake.models import EfficientNet, EfficientNetBackbone, HybridViT, Backbone
from quake.procs.train import train_model

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['lstm', 'lstm_attn', 'effnet', 'hvit'], required=True)
p.add_argument('--duration', type=int, default=8, help='record length in seconds (cache/D<d>s.npz)')
p.add_argument('--data', choices=['all', 'orig', 'okc'], default='all')
p.add_argument('--task', choices=['EMT', 'E_M', 'T_M', 'E_T'], default='EMT')
p.add_argument('--log', type=int, default=None, help='log spectrum; default: 1 for spectrogram, 0 for FFT')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--lr', type=float, default=None, help='default 1e-3 (1e-4 for lstm_attn)')
p.add_argument('--gamma', type=float, default=None, help='default 0.85 (0.9 for lstm_attn)')
p.add_argument('--final', action='store_true')
p.add_argument('--tag', default='', help='dataset variant: uses cache_<tag>/, results_<tag>/ ...')
p.add_argument('--init', default='', help='state dict to start from (fine-tuning); included in the run name')
p.add_argument('--limit', type=int, default=0, help='debug: use only N training events')
a = p.parse_args()

spectro = a.model in ('effnet', 'hvit')
log = int(spectro) if a.log is None else a.log
lr = a.lr or (1e-4 if a.model == 'lstm_attn' else 1e-3)
gamma = a.gamma or (0.9 if a.model == 'lstm_attn' else 0.85)
name = f'{a.model}_{a.task}_{a.data}_D{a.duration}_log{log}_lr{lr:g}_g{gamma}_e{a.epochs}'
if a.init:
    name += '_ft'  # fine-tuned from a model pretrained on the other source
name += f'_s{a.seed}'
sfx = f'_{a.tag}' if a.tag else ''
print('RUN', name, 'FINAL' if a.final else 'SELECTION', flush=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
device = 'cuda'
SOURCES = {0: 'orig', 1: 'okc'}

d = np.load(f'cache{sfx}/D{a.duration}s.npz')
all_classes = [str(c) for c in d['classes']]  # ['E', 'M', 'T']
keep = all_classes if a.task == 'EMT' else a.task.split('_')
keep_idx = [all_classes.index(c) for c in keep]
classes = keep
K = len(classes)


def select(split, sources):
    """Return x, y (relabelled 0..K-1), source for the given split and source filter."""
    x, y, s = d[f'x_{split}'], d[f'y_{split}'], d[f's_{split}']
    m = np.isin(y, keep_idx) & np.isin(s, sources)
    lut = np.full(len(all_classes), -1)
    lut[keep_idx] = np.arange(K)  # first class of the task name becomes label 0
    return x[m], lut[y[m]], s[m]


train_sources = {'all': [0, 1], 'orig': [0], 'okc': [1]}[a.data]
x_tr, y_tr, s_tr = select('train', train_sources)
x_va, y_va, s_va = select('val', train_sources)
x_te, y_te, s_te = select('test', [0, 1])
if a.final:
    x_tr, y_tr, s_tr = np.concatenate([x_tr, x_va]), np.concatenate([y_tr, y_va]), np.concatenate([s_tr, s_va])
if a.limit:
    x_tr, y_tr = x_tr[:a.limit], y_tr[:a.limit]

if spectro:
    aug = Compose([RandomApply(TimeShift(max_shift=0.1), p=0.5), RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
                   RandomApply(AmplitudeScale(0.9, 1.1), p=0.3)])
    stft = STFTTransform(nperseg=64, hop=32, log=bool(log))
    tf_train = nn.Sequential(aug, stft, ToTensor(), T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)))
    tf_test = nn.Sequential(stft, ToTensor())
    batch_size = 64
else:
    seq_len = x_tr.shape[1]
    fft_size = 1 << (seq_len - 1).bit_length()
    aug = Compose([RandomApply(TimeShift(max_shift=0.1), p=0.5), RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
                   RandomApply(AmplitudeScale(0.9, 1.1), p=0.3), RandomApply(TimeStretch(0.95, 1.05), p=0.3)])
    tf_train = nn.Sequential(aug, FFTTransform(fft_size, log=bool(log)), ToTensor())
    tf_test = nn.Sequential(FFTTransform(fft_size, log=bool(log)), ToTensor())
    batch_size = 32

ds_train = WaveformDataset(x_tr, y_tr, tf_train)
ds_val = WaveformDataset(x_va, y_va, tf_test)
ds_test = WaveformDataset(x_te, y_te, tf_test)
shape = tuple(ds_test[0][0].shape)
print(f'classes {classes} input {shape}', flush=True)
print(f'train {np.bincount(y_tr, minlength=K)} val {np.bincount(y_va, minlength=K)} test {np.bincount(y_te, minlength=K)}'
      f' (test orig {np.bincount(y_te[s_te == 0], minlength=K)}, okc {np.bincount(y_te[s_te == 1], minlength=K)})', flush=True)

if a.model == 'lstm':
    model = LSTMModel(channels=3, classes=K, hidden=128, layers=3, dropout=0.3)
elif a.model == 'lstm_attn':
    model = LSTMAttentionModel(channels=3, classes=K, hidden=128, layers=3, heads=4, dropout=0.3)
elif a.model == 'effnet':
    model = EfficientNet(backbone=EfficientNetBackbone.B0, in_channels=3, num_classes=K, dropout=0.2)
else:
    model = HybridViT(backbone=Backbone.TINY, depth=4, in_channels=3, num_classes=K, img_size=shape[1:], dropout=0.2)
if a.init:
    model.load_state_dict(torch.load(a.init, map_location='cpu'))
    print('initialised from', a.init, flush=True)
model = model.to(device)

t = time.time()
train_model(ds_train, ds_test if a.final else ds_val, model, batch_size=batch_size, batch_size_test=batch_size,
            epochs=a.epochs, device=device, compile_model=False, weighting=None, lr=lr, lr_gamma=gamma)
elapsed = time.time() - t


def predict(ds):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)
    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)).float(), 1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def metrics(y, prob):
    pred = prob.argmax(1)
    m = dict(n=int(len(y)), support=np.bincount(y, minlength=K).tolist(),
             macro_f1=f1_score(y, pred, average='macro', labels=range(K), zero_division=0),
             f1=f1_score(y, pred, average=None, labels=range(K), zero_division=0).tolist(),
             cm=confusion_matrix(y, pred, labels=range(K)).tolist(),
             balanced_acc=balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else float('nan'))
    if K == 2 and len(np.unique(y)) > 1:
        m['roc_auc'] = roc_auc_score(y == 0, prob[:, 0])  # class 0 (first in task name) is the positive class
    return m


def evaluate(ds, src):
    y, prob = predict(ds)
    out = {'all': metrics(y, prob)}
    for s, nm in SOURCES.items():
        if (src == s).any():
            out[nm] = metrics(y[src == s], prob[src == s])
    return out


res = dict(name=name, args=vars(a), log=log, lr=lr, gamma=gamma, classes=classes, time=elapsed)
if not a.final:
    res['val'] = evaluate(ds_val, s_va)
res['test'] = evaluate(ds_test, s_te)
for split in ('val', 'test'):
    if split in res:
        for src, m in res[split].items():
            extra = f" auc {m['roc_auc']:.3f}" if 'roc_auc' in m else ''
            print(f"{split:4s} {src:4s} n={m['n']:5d} macroF1 {m['macro_f1']:.3f} f1 {np.round(m['f1'], 3)}{extra} cm {m['cm']}")
print(f'time {elapsed:.0f}s')

if not a.limit:
    out_dir = ('results_final' if a.final else 'results') + sfx
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f'{out_dir}/{name}.json', 'w'), indent=1)
    if a.final:
        os.makedirs('models' + sfx, exist_ok=True)
        torch.save(model.state_dict(), f'models{sfx}/{name}.pt')
print('DONE', name, flush=True)
