#!/bin/bash
# Re-train the five final models one at a time on an otherwise idle machine to measure
# training and inference time (results_final_timing/, models_timing/). Start detached:
#   setsid nohup ./run_timing.sh > timing.log 2>&1 < /dev/null &
cd "$(dirname "$0")"; export MPLBACKEND=Agg; PY=${PY:-/home/skotnica/python_envs/seismo/bin/python}
mkdir -p logs_timing
run() { name=$1; shift; echo "$(date +%T) start $name"; $PY exp.py --final --tag timing "$@" > logs_timing/$name.log 2>&1; echo "$(date +%T) done  $name ($(grep -E '^time ' logs_timing/$name.log))"; }
for s in 0 1 2; do
  run effnet_s$s    --model effnet    --duration 30 --log 1 --seed $s
  run hvit_s$s      --model hvit      --duration 30 --log 1 --seed $s
  run lstm_s$s      --model lstm      --duration 10 --log 0 --seed $s
  run lstm_attn_s$s --model lstm_attn --duration 10 --log 0 --seed $s --lr 1e-4 --gamma 0.9
  run lstm_svm_s$s  --model lstm_svm  --duration 10 --log 0 --seed $s --lr 3e-4 --gamma 0.85
done
echo "$(date +%T) TIMING DONE"
