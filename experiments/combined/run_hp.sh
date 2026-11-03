#!/bin/bash
# Supplementary band-limiting experiment (no change to the main pipeline): EfficientNet, D = 20 s,
# training data {orig, okc, all}, high-pass 0.5 and 1 Hz, selection + final runs, 3 seeds.
# Start detached:  setsid nohup ./run_hp.sh > hp.log 2>&1 < /dev/null &
cd "$(dirname "$0")"; export MPLBACKEND=Agg; PY=${PY:-/home/skotnica/python_envs/seismo/bin/python}
$PY make_hp_cache.py 0.5 1.0
N=3
run() { tag=$1; data=$2; seed=$3; final=$4
  name="effnet_EMT_${data}_D20_${tag}_s${seed}${final:+_final}"
  mkdir -p logs_$tag
  echo "$(date +%T) start $name"
  $PY exp.py --tag $tag --model effnet --duration 20 --data $data --task EMT --seed $seed $final > logs_$tag/$name.log 2>&1 &
}
for final in "" "--final"; do
  for tag in s20hp05 s20hp1; do
    for data in orig okc all; do
      for seed in 0 1 2; do
        while [ "$(jobs -rp | wc -l)" -ge $N ]; do sleep 10; done
        run $tag $data $seed "$final"
      done
    done
  done
  wait
done
echo "$(date +%T) HP DONE"
