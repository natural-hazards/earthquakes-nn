"""Does the M-class shift come from the instrument response?

The SPF geophones (WDS-202, f0 = 2 Hz) suppress energy below ~2 Hz, whereas the OKC
broadband sensor (CMG-3ESP) passes it. If the archive model calls OKC events tectonic
because their z-scored records are dominated by low-frequency energy, a high-pass filter
applied to the OKC records before classification should reduce the M -> T confusion.

Classifies all OKC events with the SPF-only EfficientNet (D = 20 s, 3 seeds) after a
zero-phase Butterworth high-pass with several cut-off frequencies and prints the
confusion matrices. Output: diag_m_shift/highpass.txt
"""
import os, sys, glob, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy import stats, signal

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
from quake.data.loader import read_pickle
from quake.data.transforms import STFTTransform, ToTensor
from quake.models import EfficientNet, EfficientNetBackbone

OUT = 'diag_m_shift'; os.makedirs(OUT, exist_ok=True)
log = open(f'{OUT}/highpass.txt', 'w')
def say(*a): print(*a, flush=True); print(*a, file=log, flush=True)

D, FS = 20, 100; n = D * FS; CLASSES = ['E', 'M', 'T']; device = 'cuda'
t0 = time.time()
events, labels = read_pickle(f'{REPO}/resources/okc_events_s20.pkl'); labels = np.asarray(labels)
raw = [e[['Z', 'N', 'E']].to_numpy(np.float64) for e in events]; del events
say(f'loaded {len(raw)} OKC events ({time.time()-t0:.0f} s)')

stft = nn.Sequential(STFTTransform(nperseg=64, hop=32, log=True), ToTensor())
models = []
for p in sorted(glob.glob('models_s20/effnet_EMT_orig_D20_log1_lr0.001_g0.85_e30_s*.pt')):
    m = EfficientNet(backbone=EfficientNetBackbone.B0, in_channels=3, num_classes=3, dropout=0.2)
    m.load_state_dict(torch.load(p, map_location='cpu')); models.append(m.to(device).eval())
assert len(models) == 3


def classify(cutoff):
    if cutoff:
        sos = signal.butter(4, cutoff, btype='highpass', fs=FS, output='sos')
    x = np.zeros((len(raw), n, 3), np.float32)
    for i, a in enumerate(raw):
        b = signal.sosfiltfilt(sos, a, axis=0) if cutoff else a
        b = stats.zscore(b[:n], axis=0); x[i, :len(b)] = np.nan_to_num(b)
    specs = torch.stack([stft(xi) for xi in x])
    probs = []
    with torch.no_grad():
        for m in models:
            probs.append(torch.cat([torch.softmax(m(specs[i:i+512].to(device)).float(), 1).cpu()
                                    for i in range(0, len(specs), 512)]).numpy())
    pred = np.array(CLASSES)[np.mean(probs, 0).argmax(1)]
    return pred


for cutoff in (0, 0.5, 1.0, 2.0, 3.0, 5.0):
    pred = classify(cutoff)
    ct = pd.crosstab(pd.Series(labels, name='true'), pd.Series(pred, name='pred'))
    rec = {c: ct.loc[c, c] / ct.loc[c].sum() for c in CLASSES if c in ct.index}
    say(f'\n=== high-pass {cutoff} Hz  (recall E {rec["E"]:.3f}  M {rec["M"]:.3f}  T {rec["T"]:.3f};  M->T {ct.loc["M","T"] if "T" in ct.columns else 0})')
    say(ct.to_string())
say(f'\ndone in {time.time()-t0:.0f} s')
