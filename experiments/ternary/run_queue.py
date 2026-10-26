"""Run all experiment configs, N_PARALLEL at a time. Finished runs (results/*.json) are skipped,
so the queue can be restarted at any time. Logs go to logs/<name>.log.

Start detached:  ./run.sh
"""
import os, subprocess, time

N_PARALLEL = 3
PY = os.environ.get('PY', '/home/skotnica/python_envs/seismo/bin/python')  # interpreter with the project's dependencies

# (model, duration, weighting, log, gamma, epochs)
CONFIGS = [
    ('lstm', 20, 'none', 0, 0.85, 30),
    ('lstm', 20, 'none', 1, 0.85, 30),
    ('lstm', 10, 'none', 0, 0.85, 30),
    ('lstm', 30, 'none', 0, 0.85, 30),
    ('lstm', 20, 'sqrt', 0, 0.85, 30),
    ('lstm', 20, 'none', 1, 0.90, 40),
    ('lstm_attn', 20, 'none', 0, 0.85, 30),
    ('lstm_attn', 20, 'none', 1, 0.85, 30),
    ('effnet', 20, 'none', 0, 0.85, 30),
    ('effnet', 20, 'none', 1, 0.85, 30),
    ('hvit', 20, 'none', 0, 0.85, 30),
    ('hvit', 20, 'none', 1, 0.85, 30),
]
SEEDS = [0, 1, 2]

jobs = []
for seed in SEEDS:
    for m, dur, w, log, g, ep in CONFIGS:
        name = f'{m}_L{dur}_w{w}_log{log}_g{g}_e{ep}_s{seed}'
        cmd = [PY, 'exp.py', '--model', m, '--duration', str(dur), '--weighting', w,
               '--log', str(log), '--gamma', str(g), '--epochs', str(ep), '--seed', str(seed)]
        jobs.append((name, cmd))

os.makedirs('logs', exist_ok=True)
env = dict(os.environ, MPLBACKEND='Agg')
running = []
for name, cmd in jobs:
    if os.path.exists(f'results/{name}.json'):
        continue
    while len(running) >= N_PARALLEL:
        running = [(n, p) for n, p in running if p.poll() is None]
        time.sleep(10)
    print(time.strftime('%H:%M:%S'), 'start', name, flush=True)
    log = open(f'logs/{name}.log', 'w')
    running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))

for n, p in running:
    p.wait()
print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
