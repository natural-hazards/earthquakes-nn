"""Build cached train/val/test arrays for the combined dataset at 100 Hz.

Sources:
  0 = all_events.pkl (125 Hz, resampled to 100 Hz with anti-aliasing)
  1 = okc_events.pkl (100 Hz)

Every event is cropped to D seconds from its beginning, z-scored per channel
(before padding) and zero-padded at the end. Splits: 30 % test, then 20 % of
the rest as validation, both stratified jointly by (source, label).

Usage: python prepare.py 8 16 [--okc okc_events_s20.pkl] [--tag s20]
       durations in seconds; --tag selects the cache directory cache_<tag>/ (default: cache/)
"""
import os, sys
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from quake.data.loader import read_pickle
from quake.data.resample import resample_events

FS = 100
RES = f'{REPO}/resources'
CHANNELS = ['Z', 'N', 'E']

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('durations', type=int, nargs='+')
ap.add_argument('--okc', default='okc_events.pkl', help='OKC pickle in resources/')
ap.add_argument('--tag', default='', help='cache directory suffix')
args = ap.parse_args()
CACHE = f'cache_{args.tag}' if args.tag else 'cache'
durations = args.durations
todo = [d for d in durations if not os.path.exists(f'{CACHE}/D{d}s.npz')]
if not todo:
    print('nothing to do'); sys.exit()

ev0, lab0 = read_pickle(f'{RES}/all_events.pkl')
ev0 = resample_events(ev0, 125, FS)
ev1, lab1 = read_pickle(f'{RES}/{args.okc}')
events = list(ev0) + list(ev1)
labels = np.concatenate([np.asarray(lab0), np.asarray(lab1)])
source = np.concatenate([np.zeros(len(ev0), int), np.ones(len(ev1), int)])
del ev0, ev1
classes = np.unique(labels)
y = np.searchsorted(classes, labels)
lengths = np.array([len(e) for e in events])
print('events', len(events), 'classes', classes, 'per source x class:')
for s in (0, 1):
    print('  source', s, np.bincount(y[source == s], minlength=len(classes)),
          'length pct 5/50/95:', np.percentile(lengths[source == s], [5, 50, 95]))

# one fixed split shared by all durations
idx = np.arange(len(events))
strat = source * len(classes) + y
idx_trval, idx_test = train_test_split(idx, test_size=0.3, stratify=strat, random_state=42)
idx_tr, idx_val = train_test_split(idx_trval, test_size=0.2, stratify=strat[idx_trval], random_state=42)
os.makedirs(CACHE, exist_ok=True)

for D in todo:
    n = D * FS
    x = np.zeros((len(events), n, 3), np.float32)
    for i, e in enumerate(events):
        a = e[CHANNELS].to_numpy(np.float32)[:n]
        a = stats.zscore(a, axis=0)
        x[i, :len(a)] = np.nan_to_num(a)  # constant (dead) channel -> zeros
    np.savez(f'{CACHE}/D{D}s.npz',
             x_train=x[idx_tr], y_train=y[idx_tr], s_train=source[idx_tr],
             x_val=x[idx_val], y_val=y[idx_val], s_val=source[idx_val],
             x_test=x[idx_test], y_test=y[idx_test], s_test=source[idx_test],
             classes=classes, fs=FS)
    print(f'{CACHE}/D{D}s.npz: train {np.bincount(y[idx_tr])} val {np.bincount(y[idx_val])} test {np.bincount(y[idx_test])}'
          f' | padded fraction {np.mean(lengths < n):.2f}', flush=True)
    del x
