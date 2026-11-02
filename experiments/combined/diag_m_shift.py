"""Diagnostics of the "M-class shift": which OKC events does the SPF-only model call tectonic?

Reconstructs the identity of every OKC window from okc/okc_events_s20_meta.csv (one row
per stored window, same order as the pickle), classifies all OKC events with the final
SPF-only EfficientNet models (D = 20 s, three seeds; these models never saw OKC data) and
with the pooled model (test events only), and summarises the SPF-only predictions of the
mining-induced events by year, by bulletin source description and by S-P time.

Outputs (in diag_m_shift/):
  predictions.csv   one row per OKC event with metadata, split, per-model class probabilities
  summary.txt       the tables printed below
"""
import os, sys, json, glob, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from quake.data.loader import read_pickle
from quake.data.transforms import STFTTransform, ToTensor
from quake.models import EfficientNet, EfficientNetBackbone

OUT = 'diag_m_shift'; os.makedirs(OUT, exist_ok=True)
D, FS = 20, 100
CLASSES = ['E', 'M', 'T']
device = 'cuda'
log = open(f'{OUT}/summary.txt', 'w')
def say(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)

# ------------------------------------------------------------------ data + metadata
t0 = time.time()
events, labels = read_pickle(f'{REPO}/resources/okc_events_s20.pkl')
labels = np.asarray(labels)
meta = pd.read_csv(f'{REPO}/okc/okc_events_s20_meta.csv')
assert len(meta) == len(events), (len(meta), len(events))
assert (meta['label'].values == labels).all(), 'label order mismatch between meta and pickle'
assert (meta['n_samples'].values == np.array([len(e) for e in events])).all(), 'length mismatch'
say(f'loaded {len(events)} OKC events, metadata aligned ({time.time()-t0:.0f} s)')

meta['p_time'] = pd.to_datetime(meta['p_time']); meta['s_time'] = pd.to_datetime(meta['s_time'])
meta['year'] = meta['p_time'].dt.year
meta['sp'] = (meta['s_time'] - meta['p_time']).dt.total_seconds()
meta['max_counts'] = [float(np.abs(e.to_numpy()).max()) for e in events]

# ------------------------------------------------------------------ reproduce the s20 split (prepare.py)
n_spf = 39839  # all_events.pkl precedes the OKC events in the combined index
_, lab_spf = read_pickle(f'{REPO}/resources/all_events.pkl') if False else (None, None)
# the split only depends on the stratification key (source*3 + class) and random_state;
# the SPF part of the key is reconstructed from the known class counts in file order
spf_labels = np.asarray(read_pickle(f'{REPO}/resources/all_events.pkl')[1])
y_all = np.searchsorted(CLASSES, np.concatenate([spf_labels, labels]))
source = np.concatenate([np.zeros(len(spf_labels), int), np.ones(len(labels), int)])
idx = np.arange(len(y_all)); strat = source * 3 + y_all
idx_trval, idx_test = train_test_split(idx, test_size=0.3, stratify=strat, random_state=42)
idx_tr, idx_val = train_test_split(idx_trval, test_size=0.2, stratify=strat[idx_trval], random_state=42)
split = np.full(len(y_all), 'train', dtype=object); split[idx_val] = 'val'; split[idx_test] = 'test'
meta['split'] = split[len(spf_labels):]
say('OKC split counts:', meta['split'].value_counts().to_dict())
say(f'split reproduced ({time.time()-t0:.0f} s)')

# ------------------------------------------------------------------ preprocessing as in prepare.py / exp.py
n = D * FS
x = np.zeros((len(events), n, 3), np.float32)
for i, e in enumerate(events):
    a = stats.zscore(e[['Z', 'N', 'E']].to_numpy(np.float32)[:n], axis=0)
    x[i, :len(a)] = np.nan_to_num(a)
del events
stft = nn.Sequential(STFTTransform(nperseg=64, hop=32, log=True), ToTensor())
specs = torch.stack([stft(xi) for xi in x])  # [N, 3, 33, T]
say(f'spectrograms {tuple(specs.shape)} ({time.time()-t0:.0f} s)')


def predict(state_dict_path):
    model = EfficientNet(backbone=EfficientNetBackbone.B0, in_channels=3, num_classes=3, dropout=0.2)
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu')); model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(specs), 512):
            out.append(torch.softmax(model(specs[i:i+512].to(device)).float(), 1).cpu())
    return torch.cat(out).numpy()


models = {
    'spf': sorted(glob.glob('models_s20/effnet_EMT_orig_D20_log1_lr0.001_g0.85_e30_s*.pt')),
    'pooled': sorted(glob.glob('models_s20/effnet_EMT_all_D20_log1_lr0.001_g0.85_e30_s*.pt')),
    'okc': sorted(glob.glob('models_s20/effnet_EMT_okc_D16_log1_lr0.001_g0.85_e30_s*.pt')),
}
for name, paths in models.items():
    if name == 'okc':
        continue  # different record length (16 s); not needed for this question
    assert len(paths) == 3, (name, paths)
    probs = np.mean([predict(p) for p in paths], axis=0)
    for k, c in enumerate(CLASSES):
        meta[f'p_{c}_{name}'] = probs[:, k]
    meta[f'pred_{name}'] = np.array(CLASSES)[probs.argmax(1)]
    say(f'{name}: predictions done ({time.time()-t0:.0f} s)')

meta.to_csv(f'{OUT}/predictions.csv', index=False)

# ------------------------------------------------------------------ summaries
pd.set_option('display.width', 200, 'display.max_rows', 200)
m = meta[meta['label'] == 'M'].copy()
m['spf_T'] = m['pred_spf'] == 'T'
say('\n=== SPF-only model on ALL OKC events: confusion (rows true, cols predicted)')
say(pd.crosstab(meta['label'], meta['pred_spf']))
say('\n=== SPF-only model on OKC TEST events only (should match the paper table)')
say(pd.crosstab(meta.loc[meta.split == 'test', 'label'], meta.loc[meta.split == 'test', 'pred_spf']))
say('\n=== pooled model on OKC TEST events')
say(pd.crosstab(meta.loc[meta.split == 'test', 'label'], meta.loc[meta.split == 'test', 'pred_pooled']))

say('\n=== M events: share predicted T by the SPF-only model, by year')
g = m.groupby('year').agg(n=('spf_T', 'size'), frac_T=('spf_T', 'mean'), frac_M=('pred_spf', lambda s: (s == 'M').mean()),
                          median_sp=('sp', 'median'))
say(g.round(3))

say('\n=== M events: by bulletin source description (raw_label), top 40 by count')
g = m.groupby('raw_label').agg(n=('spf_T', 'size'), frac_T=('spf_T', 'mean'), median_sp=('sp', 'median'),
                               first=('year', 'min'), last=('year', 'max')).sort_values('n', ascending=False)
say(g.head(40).round(3))

say('\n=== M events: by S-P time (distance proxy)')
bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, 60]
m['sp_bin'] = pd.cut(m['sp'], bins)
g = m.groupby('sp_bin', observed=True).agg(n=('spf_T', 'size'), frac_T=('spf_T', 'mean'))
say(g.round(3))
say('\n--- same for T and E events (how the SPF-only model sees them vs S-P)')
for cl in ('T', 'E'):
    d = meta[meta['label'] == cl].copy(); d['sp_bin'] = pd.cut(d['sp'], bins)
    say(f'class {cl}:'); say(pd.crosstab(d['sp_bin'], d['pred_spf']))

say('\n=== M events: by amplitude (max raw counts, quartiles)')
m['amp_q'] = pd.qcut(m['max_counts'], 4, labels=['q1 (weak)', 'q2', 'q3', 'q4 (strong)'])
say(m.groupby('amp_q', observed=True).agg(n=('spf_T', 'size'), frac_T=('spf_T', 'mean')).round(3))

say('\n=== M events: mean class probabilities of the SPF-only model for events called T vs called M')
say(m.groupby('pred_spf')[['p_E_spf', 'p_M_spf', 'p_T_spf']].mean().round(3))

say('\n=== reference: S-P distribution of OKC T events (by raw_label, top 15)')
t = meta[meta['label'] == 'T']
say(t.groupby('raw_label').agg(n=('sp', 'size'), median_sp=('sp', 'median')).sort_values('n', ascending=False).head(15).round(2))
say(f'\ndone in {time.time()-t0:.0f} s; predictions in {OUT}/predictions.csv')
