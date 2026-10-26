"""Preprocess all_events.pkl once per record length and cache train/val/test arrays.

The test split is identical to WaveformDataAdapter(test_ratio=0.3, random_state=42);
validation is a stratified 20 % of the adapter's training split.

Usage: python prepare.py 10 20 30   (record length in units of 100 samples = 0.8 s at 125 Hz;
       the cache files are named L10s/L20s/L30s for historical reasons and hold 1000/2000/3000 samples)
"""
import os, sys
import numpy as np
from sklearn.model_selection import train_test_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from quake.data.adapter import WaveformDataAdapter, TransformOP
from pipelines.utils import load_events

SAMPLING_RATE = 100  # samples per 'duration' unit, NOT the true sampling rate (125 Hz)
events, labels = load_events([f'{REPO}/resources/all_events.pkl'])

for duration in map(int, sys.argv[1:]):
    out = f'cache/L{duration}s.npz'
    if os.path.exists(out):
        print(f'{out} exists, skipping')
        continue
    adapter = WaveformDataAdapter(events, labels, ('Z', 'N', 'E'),
        transforms=TransformOP.DROP_NAN | TransformOP.PADDING | TransformOP.ZSCORE,
        test_ratio=0.3, seq_len=duration * SAMPLING_RATE, random_state=42)
    ds_train, ds_test = adapter.get_datasets()
    x_tr = np.stack([ds_train[i][0].numpy() for i in range(len(ds_train))])
    x_te = np.stack([ds_test[i][0].numpy() for i in range(len(ds_test))])
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, ds_train.labels, test_size=0.2, stratify=ds_train.labels, random_state=42)
    os.makedirs('cache', exist_ok=True)
    np.savez(out, x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va,
             x_test=x_te, y_test=ds_test.labels, classes=np.unique(labels))
    print(f'{out}: train {np.bincount(y_tr)} val {np.bincount(y_va)} test {np.bincount(ds_test.labels)}')
    del adapter, ds_train, ds_test, x_tr, x_va, x_te
