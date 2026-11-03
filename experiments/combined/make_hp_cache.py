"""Supplementary experiment: band-limit both data sets before classification.

Creates cache_s20hp<cut>/D20s.npz from cache_s20/D20s.npz by applying a zero-phase
Butterworth high-pass (4th order, cut-off <cut> Hz) to every record and re-normalizing it
(z-score per channel). Split, labels and sources are unchanged, so the runs are directly
comparable with the unfiltered s20 study via `exp.py --tag s20hp<cut>`.

The high-pass is linear and the per-event z-score is affine, so filtering the already
normalized cache and re-normalizing equals filtering the raw records before normalization.
Zero-padded tails stay (nearly) zero; the small filter transient at the padding boundary
is the only difference to filtering before padding.

Usage: python make_hp_cache.py 0.5 1.0
"""
import os, sys
import numpy as np
from scipy import signal

FS = 100
src = np.load('cache_s20/D20s.npz')
for cut in map(float, sys.argv[1:]):
    tag = f's20hp{cut:g}'.replace('.', '')
    out_dir = f'cache_{tag}'; os.makedirs(out_dir, exist_ok=True)
    out = f'{out_dir}/D20s.npz'
    if os.path.exists(out):
        print(out, 'exists'); continue
    sos = signal.butter(4, cut, btype='highpass', fs=FS, output='sos')
    arrays = {k: src[k] for k in src.files}
    for split in ('train', 'val', 'test'):
        x = arrays[f'x_{split}']
        y = signal.sosfiltfilt(sos, x, axis=1).astype(np.float32)
        # re-normalize over the unpadded part of each record (padding = exact zeros in the source)
        valid = (np.abs(x) > 0).any(axis=2)                      # [N, T]
        n = valid.sum(axis=1, keepdims=True)                     # [N, 1]
        m = (y * valid[..., None]).sum(axis=1, keepdims=True) / n[..., None]
        s = np.sqrt(((y - m) ** 2 * valid[..., None]).sum(axis=1, keepdims=True) / n[..., None])
        y = ((y - m) / np.maximum(s, 1e-6)) * valid[..., None]
        arrays[f'x_{split}'] = np.nan_to_num(y).astype(np.float32)
        print(f'{tag} {split}: {x.shape} filtered', flush=True)
    np.savez(out, **arrays)
    print('written', out, f'(tag {tag})', flush=True)
