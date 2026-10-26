"""Phase 1: selection runs (train/val, 3 seeds) for the configs below.
Phase 2: for every model family pick the config with the best mean VAL macro F1
(over all results/*.json, old and new) and retrain it on train+val with 3 seeds,
evaluating once on the test split (results_final/, models/).

Finished runs are skipped, so the queue can be restarted at any time.
Start detached:  ./run2.sh
"""
import glob, json, os, subprocess, time, collections
import numpy as np

N_PARALLEL = 3
PY = os.environ.get('PY', '/home/skotnica/python_envs/seismo/bin/python')  # interpreter with the project's dependencies
SEEDS = [0, 1, 2]
FAMILIES = ['effnet', 'hvit', 'lstm', 'lstm_attn', 'lstm_svm']

# dicts of exp.py arguments (seed added later)
SELECTION = []
for dur in (10, 30):
    for m in ('effnet', 'hvit'):
        SELECTION.append(dict(model=m, duration=dur, log=1))
for m in ('lstm_attn', 'lstm_svm'):
    for dur in (10, 20):
        SELECTION.append(dict(model=m, duration=dur, lr=3e-4, gamma=0.85))
        SELECTION.append(dict(model=m, duration=dur, lr=1e-4, gamma=0.9))

ARG_KEYS = ('model', 'duration', 'weighting', 'log', 'epochs', 'gamma', 'layers', 'hidden', 'dropout', 'lr', 'clip')


def run_name(args):
    """Replicates the naming in exp.py."""
    a = {**dict(weighting='none', log=0, epochs=30, gamma=0.85, layers=3, hidden=128, dropout=0.3, lr=1e-3, clip=0), **args}
    name = f"{a['model']}_L{a['duration']}_w{a['weighting']}_log{a['log']}_g{a['gamma']}_e{a['epochs']}"
    if a['model'] in ('lstm_attn', 'lstm_svm'):
        name += f"_l{a['layers']}_h{a['hidden']}_d{a['dropout']}_lr{a['lr']:g}_c{a['clip']:g}"
    return name + f"_s{a['seed']}"


def cmd_for(args, final=False):
    cmd = [PY, 'exp.py']
    for k, v in args.items():
        cmd += [f'--{k}', str(v)]
    return cmd + (['--final'] if final else [])


def run_jobs(jobs, out_dir):
    """jobs: list of (name, cmd). Runs N_PARALLEL at a time, skips finished ones."""
    os.makedirs('logs', exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg')
    running = []
    for name, cmd in jobs:
        if os.path.exists(f'{out_dir}/{name}.json'):
            continue
        while len(running) >= N_PARALLEL:
            running = [(n, p) for n, p in running if p.poll() is None]
            time.sleep(10)
        print(time.strftime('%H:%M:%S'), 'start', name, flush=True)
        log = open(f'logs/{name}.log', 'w')
        running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
    for n, p in running:
        p.wait()


# ---- Phase 1 ----
jobs = []
for seed in SEEDS:
    for cfg in SELECTION:
        args = {**cfg, 'seed': seed}
        jobs.append((run_name(args), cmd_for(args)))
run_jobs(jobs, 'results')
print(time.strftime('%H:%M:%S'), 'PHASE 1 DONE', flush=True)

# ---- Phase 2 ----
groups = collections.defaultdict(list)
for fn in glob.glob('results/*.json'):
    r = json.load(open(fn))
    groups[r['name'].rsplit('_s', 1)[0]].append(r)

best = {}
for cfg, rs in groups.items():
    fam = rs[0]['args']['model']
    val = np.mean([r['val']['macro_f1'] for r in rs])
    if fam not in best or val > best[fam][0]:
        best[fam] = (val, cfg, rs[0]['args'])

jobs = []
for fam in FAMILIES:
    if fam not in best:
        print('no results for', fam)
        continue
    val, cfg, args = best[fam]
    print(f'FINAL {fam}: {cfg} (val macro F1 {val:.3f})', flush=True)
    args = {k: args[k] for k in ARG_KEYS if k in args}
    for seed in SEEDS:
        a = {**args, 'seed': seed}
        jobs.append((run_name(a), cmd_for(a, final=True)))
run_jobs(jobs, 'results_final')
print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
