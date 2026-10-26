# Experiments

Scripts and results behind the tables in `docs/results.tex`. Each directory is
self-contained; run scripts from inside it. `PY` selects the interpreter
(default: `/home/skotnica/python_envs/seismo/bin/python`), e.g.
`PY=python3 ./run.sh`.

| Directory | Data | Purpose |
|---|---|---|
| `lstm/` | `resources/all_events.pkl`, 125 Hz | First pass: LSTM with inverse class weights vs. undersampling vs. no weighting (single seed, test split only). Superseded by `ternary/`. |
| `ternary/` | `resources/all_events.pkl`, 125 Hz | Main results: E/M/T classification with five architectures, model selection on a validation split (3 seeds), final retraining on train+val, ablations (record length, log spectrum, class imbalance, attention optimisation) and the binary tasks E vs M, T vs M, E vs T. |
| `combined/` | `all_events.pkl` resampled to 100 Hz + `okc_events.pkl` | Generalisation to the current (OKC) stations: combined training, per-source evaluation, cross-source transfer, OKC-only baselines. |

## Workflow (`ternary/`, `combined/`)

1. `prepare.py` – crop/pad, z-score and split the events once per record length into `cache/*.npz`
   (`ternary/`: the `--duration` argument counts 100-sample units, i.e. 10/20/30 = 8/16/24 s at 125 Hz;
   `combined/`: seconds at 100 Hz).
2. `exp.py` – train and evaluate one configuration; writes `results/<name>.json`
   (selection: validation + test metrics) or, with `--final`, `results_final/<name>.json` and `models/<name>.pt`.
3. `run_queue*.py` / `run_*.sh` – the experiment queues; finished runs are skipped, so a queue can be restarted.
4. `summarize*.py` – console tables; `ternary/gen_results_tex.py` writes `docs/results_tables.tex`.

`cache/`, `logs/` and `models/` are not versioned (several GB); they are rebuilt by steps 1–3.
