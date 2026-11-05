"""LSTM-FCN on the raw waveform (no FFT), as in B. Rieznikov's thesis.

Phase 1: selection runs (3 seeds): karim preset at D = 8 s and 24 s, small preset at 8 s, lr 1e-3.
Phase 2: if the best raw configuration beats the best FFT configuration of run_fcn.py on the
         validation split, it is retrained on train+val (results_final/) and timed (results_final_timing/).
Start detached: setsid nohup python run_fcn_raw.py > fcn_raw.log 2>&1 < /dev/null &
"""
import glob, json, os, subprocess, time
import numpy as np

PY = os.environ.get('PY', '/home/skotnica/python_envs/seismo/bin/python')
SEEDS = [0, 1, 2]
GRID = [('karim', 10), ('karim', 30), ('small', 10)]  # (preset, duration units of 100 samples)


def name(preset, D, seed):
    return f'lstm_fcn_L{D}_wnone_log0_g0.85_e30_{preset}_lr0.001_raw_s{seed}'


def cmd(preset, D, seed, extra=()):
    return [PY, 'exp.py', '--model', 'lstm_fcn', '--fcn', preset, '--input', 'raw', '--duration', str(D),
            '--log', '0', '--lr', '1e-3', '--gamma', '0.85', '--seed', str(seed), *extra]


def run_jobs(jobs, out_dir, n_parallel):
    os.makedirs('logs', exist_ok=True); env = dict(os.environ, MPLBACKEND='Agg'); running = []
    for nm, c in jobs:
        if os.path.exists(f'{out_dir}/{nm}.json'):
            continue
        while len(running) >= n_parallel:
            running = [(n, p) for n, p in running if p.poll() is None]; time.sleep(10)
        print(time.strftime('%H:%M:%S'), 'start', nm, out_dir, flush=True)
        suffix = '' if out_dir == 'results' else '_' + out_dir
        running.append((nm, subprocess.Popen(c, stdout=open(f'logs/{nm}{suffix}.log', 'w'), stderr=subprocess.STDOUT, env=env)))
    for n, p in running:
        p.wait()


run_jobs([(name(p, D, s), cmd(p, D, s)) for s in SEEDS for p, D in GRID], 'results', 3)
print(time.strftime('%H:%M:%S'), 'PHASE 1 DONE', flush=True)


def val_of(prefix):
    rs = [json.load(open(f)) for f in glob.glob(f'results/{prefix}_s*.json')]
    return (np.mean([r['val']['macro_f1'] for r in rs]) if rs else -1), len(rs)


best_raw = max(((val_of(name(p, D, 0)[:-3]), p, D) for p, D in GRID), key=lambda t: t[0][0])
best_fft = max((val_of(f'lstm_fcn_L10_wnone_log0_g{g}_e30_{p}_lr{lr:g}') for p in ('karim', 'small') for lr, g in ((1e-3, 0.85), (1e-4, 0.9))), key=lambda t: t[0])
for p, D in GRID:
    print(f'  raw {p:6s} D={D}: val macro-F1 {val_of(name(p, D, 0)[:-3])[0]:.3f}', flush=True)
print(f'best raw {best_raw[0][0]:.3f} ({best_raw[1]}, D={best_raw[2]})  vs  best FFT {best_fft[0]:.3f}', flush=True)
if best_raw[0][0] > best_fft[0]:
    _, p, D = best_raw
    print(f'FINAL lstm_fcn (raw): preset {p}, D={D}', flush=True)
    run_jobs([(name(p, D, s), cmd(p, D, s, ('--final',))) for s in SEEDS], 'results_final', 3)
    run_jobs([(name(p, D, s), cmd(p, D, s, ('--final', '--tag', 'timing'))) for s in SEEDS], 'results_final_timing', 1)
else:
    print('FFT input remains the better LSTM-FCN configuration; no final runs', flush=True)
print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
