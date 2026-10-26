"""Binary tasks (E vs M, T vs M, E vs T) on the original 125 Hz data with the configurations
selected for the ternary problem: EfficientNet-B0 (24 s, log STFT) and LSTM (8 s, FFT).
Selection runs (train/val) and final runs (train+val -> test), 3 seeds each.  Start: ./run_binary.sh
"""
import os, subprocess, time

N_PARALLEL = 2
PY = os.environ.get('PY', '/home/skotnica/python_envs/seismo/bin/python')  # interpreter with the project's dependencies
SEEDS = [0, 1, 2]
MODELS = [dict(model='effnet', duration=30, log=1), dict(model='lstm', duration=10, log=0)]
TASKS = ['E_M', 'T_M', 'E_T']


def run_name(m, task, seed):
    return f"{m['model']}_{task}_L{m['duration']}_wnone_log{m['log']}_g0.85_e30_s{seed}"


def run_jobs(jobs, out_dir):
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
        running.append((name, subprocess.Popen(cmd, stdout=open(f'logs/{name}.log', 'w'),
                                               stderr=subprocess.STDOUT, env=env)))
    for n, p in running:
        p.wait()


for final in (False, True):
    jobs = []
    for seed in SEEDS:
        for m in MODELS:
            for task in TASKS:
                cmd = [PY, 'exp.py', '--model', m['model'], '--duration', str(m['duration']), '--log', str(m['log']),
                       '--task', task, '--seed', str(seed)] + (['--final'] if final else [])
                jobs.append((run_name(m, task, seed), cmd))
    run_jobs(jobs, 'results_final' if final else 'results')
    print(time.strftime('%H:%M:%S'), 'FINAL DONE' if final else 'SELECTION DONE', flush=True)
print(time.strftime('%H:%M:%S'), 'ALL DONE', flush=True)
