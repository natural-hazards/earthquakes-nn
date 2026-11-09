# Experiments

Scripts and results behind the tables of the manuscript *Machine learning
approaches to seismic event classification in the Ostrava region* (submitted to
the International Journal of Rock Mechanics and Mining Sciences). The data sets
are on Zenodo: https://doi.org/10.5281/zenodo.24010919. Each directory is
self-contained; run scripts from inside it. `PY` selects the interpreter
(default: `/home/skotnica/python_envs/seismo/bin/python`), e.g. `PY=python3 ./run_timing.sh`.

| Directory | Data | Purpose |
|---|---|---|
| `ternary/` | SPF archive (`resources/all_events.pkl`, 125 Hz) | Main results: E/M/T classification with five architectures, model selection on a validation split (3 seeds), final retraining on train+val, ablations (record length, log spectrum, class imbalance, attention optimisation, LSTM-FCN input) and the binary tasks E vs M, T vs M, E vs T. |
| `combined/` | SPF resampled to 100 Hz + OKC (`resources/okc_events_s20.pkl`) | Transfer to the present-day station OKC: OKC-only, SPF-only, pooled and fine-tuned models with per-source evaluation, record length selection, high-pass (band-limiting) experiment, and the diagnostics of the M-class shift. |

## Workflow

1. `prepare.py` – crop/pad, z-score and split the events once per record length into `cache*/*.npz`
   (`ternary/`: the `--duration` argument counts 100-sample units, i.e. 10/20/30 = 8/16/24 s at 125 Hz;
   `combined/`: seconds at 100 Hz; `--okc`/`--tag` select the OKC pickle and the cache name).
2. `exp.py` – train and evaluate one configuration; writes `results[_tag]/<name>.json`
   (selection: validation + test metrics) or, with `--final`, `results_final[_tag]/<name>.json` and `models[_tag]/<name>.pt`.
3. The experiment queues (finished runs are skipped, so a queue can be restarted):
   - `ternary/run_queue.py`, `run_queue2.py` – architectures, record lengths, log compression, attention settings;
     `run_fcn.py`, `run_fcn_raw.py` – LSTM-FCN on the spectrum and on the raw waveform;
     `run_binary.py` – binary tasks; `run_imbalance.sh` – class weighting and undersampling;
     `run_timing.sh` – final models retrained one at a time on an idle GPU for the cost table.
   - `combined/run_queue_s20.py` – selection, final and fine-tuning runs of the transfer study;
     `make_hp_cache.py` + `run_hp.sh` – high-pass filtered caches and runs (tags `s20hp05`, `s20hp1`);
     `diag_m_shift.py`, `diag_highpass.py` – diagnostics of the SPF-only model on the OKC events.
4. `ternary/gen_results_tex.py` and `combined/gen_transfer_tex.py` – LaTeX tables from the result JSON files.

`cache*/`, `logs*/` and `models*/` are not versioned (several GB); they are rebuilt by steps 1–3.
The `results*/` directories with the JSON metrics of every run are versioned.
