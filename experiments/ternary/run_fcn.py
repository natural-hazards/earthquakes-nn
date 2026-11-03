"""LSTM-FCN study on the 125 Hz data (replacing LSTM-SVM in the paper, code of both kept).

Phase 1: selection runs (train/val, 3 seeds) for preset {karim, small} x lr {1e-3 (g 0.85), 1e-4 (g 0.9)}.
Phase 2: the preset/lr with the best mean validation macro-F1 is retrained on train+val
         (3 seeds, results_final/, models/).
Phase 3: the same configuration one run at a time with --tag timing (results_final_timing/).
Finished runs are skipped.  Start detached: setsid nohup python run_fcn.py > fcn.log 2>&1 < /dev/null &
"""
import glob, json, os, subprocess, time
import numpy as np

PY = os.environ.get('PY', '/home/skotnica/python_envs/seismo/bin/python')
SEEDS = [0, 1, 2]
D, LOG = 10, 0  # 8 s at 125 Hz, linear spectrum, as for the other recurrent models
GRID = [(preset, lr, g) for preset in ('karim', 'small') for lr, g in ((1e-3, 0.85), (1e-4, 0.9))]


def name(preset, lr, g, seed):
    return f'lstm_fcn_L{D}_wnone_log{LOG}_g{g}_e30_{preset}_lr{lr:g}_s{seed}'


def cmd(preset, lr, g, seed, extra=()):
    return [PY, 'exp.py', '--model', 'lstm_fcn', '--fcn', preset, '--duration', str(D), '--log', str(LOG),
            '--lr', str(lr), '--gamma', str(g), '--seed', str(seed), *extra]


def run_jobs(jobs, out_dir, n_parallel):
    os.makedirs('logs', exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg'); running = []
    for nm, c in jobs:
        if os.path.exists(f'{out_dir}/{nm}.json'):
            continue
        while len(running) >= n_parallel:
            running = [(n, p) for n, p in running if p.poll() is None]; time.sleep(10)
        print(time.strftime('%H:%M:%S'), 'start', nm, out_dir, flush=True)
        running.append((nm, subprocess.Popen(c, stdout=open(f'logs/{nm}{"_" + out_dir if out_dir != "results" else ""}.log', 'w'),
                                             stderr=subprocess.STDOUT, env=env)))
    for n, p in running:
        p.wait()


run_jobs([(name(p, lr, g, s), cmd(p, lr, g, s)) for s in SEEDS for p, lr, g in GRID], 'results', 3)
print(time.strftime('%H:%M:%S'), 'PHASE 1 DONE', flush=True)

best = None
for p, lr, g in GRID:
    rs = [json.load(open(f)) for f in glob.glob(f'results/{name(p, lr, g, 0)[:-3]}_s*.json')]
    val = np.mean([r['val']['macro_f1'] for r in rs]) if rs else -1
    print(f'  {p:6s} lr {lr:g}: val macro-F1 {val:.3f} (n={len(rs)})', flush=True)
    if best is None or val > best[0]:
        best = (val, p, lr, g)
_, p, lr, g = best
print(f'FINAL lstm_fcn: preset {p}, lr {lr:g}, gamma {g}', flush=True)
run_jobs([(name(p, lr, g, s), cmd(p, lr, g, s, ('--final',))) for s in SEEDS], 'results_final', 3)
print(time.strftime('%H:%M:%S'), 'PHASE 2 DONE', flush=True)
run_jobs([(name(p, lr, g, s), cmd(p, lr, g, s, ('--final', '--tag', 'timing'))) for s in SEEDS], 'results_final_timing', 1)
print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
