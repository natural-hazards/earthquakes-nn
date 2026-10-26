#!/bin/bash
# Class-imbalance ablation for LSTM 20 s within the validation protocol (3 seeds each).
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PY=${PY:-/home/skotnica/python_envs/seismo/bin/python}
(
for s in 0 1 2; do
  $PY exp.py --model lstm --duration 20 --weighting inverse --seed $s > logs/lstm_L20_winverse_log0_g0.85_e30_s$s.log 2>&1 &
  $PY exp.py --model lstm --duration 20 --undersample 2000 --seed $s > logs/lstm_L20_wnone_log0_g0.85_e30_u2000_s$s.log 2>&1 &
  wait
done
echo IMBALANCE DONE
) > imbalance.log 2>&1
