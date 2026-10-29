"""Study with the longer OKC records (okc_events_s20.pkl, tag 's20').

Phase 1: selection runs (train/val, 3 seeds).
Phase 2: per (task, data) family the config with the best mean VALIDATION macro-F1 is
         retrained on train+val (3 seeds) and evaluated once on the test split.
Phase 3: fine-tuning: the final model trained on the old data only is used to initialise
         training on the OKC data only (lower learning rate, shorter schedule), giving a third
         strategy next to "OKC only" and "pooled".

Finished runs are skipped, so the queue can be restarted at any time.  Start: ./run_s20.sh
"""
import glob, json, os, subprocess, time, collections
import numpy as np

TAG = 's20'
N_PARALLEL = 3
PY = os.environ.get('PY', '/home/skotnica/python_envs/seismo/bin/python')  # interpreter with the project's dependencies
SEEDS = [0, 1, 2]
FT = dict(lr=1e-4, gamma=0.85, epochs=15)  # fine-tuning schedule

SELECTION = []
# ternary: training source x record length (EfficientNet)
for data in ('all', 'okc', 'orig'):
    for D in (8, 16, 20):
        SELECTION.append(dict(model='effnet', duration=D, data=data, task='EMT'))
# other architectures at the longest window, pooled data
for m in ('hvit', 'lstm'):
    SELECTION.append(dict(model=m, duration=20, data='all', task='EMT'))
# binary tasks at the longest window, pooled data
for task in ('E_M', 'T_M'):
    SELECTION.append(dict(model='effnet', duration=20, data='all', task=task))


def run_name(args):
    m = args['model']
    log = args.get('log', 1 if m in ('effnet', 'hvit') else 0)
    lr = args.get('lr', 1e-4 if m == 'lstm_attn' else 1e-3)
    gamma = args.get('gamma', 0.9 if m == 'lstm_attn' else 0.85)
    name = f"{m}_{args['task']}_{args['data']}_D{args['duration']}_log{log}_lr{lr:g}_g{gamma}_e{args.get('epochs', 30)}"
    if args.get('init'):
        name += '_ft'
    return name + f"_s{args['seed']}"


def cmd_for(args, final=False):
    cmd = [PY, 'exp.py', '--tag', TAG]
    for k, v in args.items():
        cmd += [f'--{k}', str(v)]
    return cmd + (['--final'] if final else [])


def run_jobs(jobs, out_dir):
    os.makedirs(f'logs_{TAG}', exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg')
    running = []
    for name, cmd in jobs:
        if os.path.exists(f'{out_dir}/{name}.json'):
            continue
        while len(running) >= N_PARALLEL:
            running = [(n, p) for n, p in running if p.poll() is None]
            time.sleep(10)
        print(time.strftime('%H:%M:%S'), 'start', name, flush=True)
        log = open(f'logs_{TAG}/{name}.log', 'w')
        running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
    for n, p in running:
        p.wait()


# ---- Phase 1 ----
jobs = [(run_name({**cfg, 'seed': s}), cmd_for({**cfg, 'seed': s})) for s in SEEDS for cfg in SELECTION]
run_jobs(jobs, f'results_{TAG}')
print(time.strftime('%H:%M:%S'), 'PHASE 1 DONE', flush=True)

# ---- Phase 2 ----
groups = collections.defaultdict(list)
for fn in glob.glob(f'results_{TAG}/*.json'):
    r = json.load(open(fn))
    groups[r['name'].rsplit('_s', 1)[0]].append(r)

best = {}
for cfg, rs in groups.items():
    a = rs[0]['args']
    fam = (a['task'], a['data'])
    val = np.mean([r['val']['all']['macro_f1'] for r in rs])
    if fam not in best or val > best[fam][0]:
        best[fam] = (val, cfg, a)

jobs = []
for fam in sorted(best):
    val, cfg, a = best[fam]
    print(f'FINAL {fam}: {cfg} (val macro-F1 {val:.3f})', flush=True)
    args = {k: a[k] for k in ('model', 'duration', 'data', 'task', 'epochs')}
    for s in SEEDS:
        jobs.append((run_name({**args, 'seed': s}), cmd_for({**args, 'seed': s}, final=True)))
run_jobs(jobs, f'results_final_{TAG}')
print(time.strftime('%H:%M:%S'), 'PHASE 2 DONE', flush=True)

# ---- Phase 3: fine-tune the old-data model on OKC, at every record length that has an orig model
jobs = []
for fn in sorted(glob.glob(f'models_{TAG}/effnet_EMT_orig_D*_s0.pt')):
    r = json.load(open(fn.replace(f'models_{TAG}', f'results_final_{TAG}').replace('.pt', '.json')))
    D = r['args']['duration']
    for s in SEEDS:
        init = fn.replace('_s0.pt', f'_s{s}.pt')
        args = dict(model='effnet', duration=D, data='okc', task='EMT', seed=s, init=init, **FT)
        jobs.append((run_name(args), cmd_for(args, final=True)))
print(f'fine-tuning jobs: {len(jobs)}', flush=True)
run_jobs(jobs, f'results_final_{TAG}')
print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
