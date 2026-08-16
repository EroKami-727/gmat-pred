# OrbitGuard — AI Context

> **SUPERSEDED — generation G2 (2026-06-21 to 06-23).**
> Describes the local 80K reproduction and the multi-planet Transformer
> at 87.67% accuracy / F1 0.838. The current models are the per-planet G3
> rebuild (F1 0.9981), trained differently and evaluated on a different
> split. Retained for provenance and for the analyses that still stand
> (feature-shift error analysis, target-upweighting experiment).
> See [`README.md`](README.md) for the generation map.

Pipeline orientation. Much of it still holds — the generator, the feature
pipeline and the cadence facts are unchanged — but the model and result sections
describe G2. For the current entry point read [`README.md`](README.md), then
[`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md).

## What it does

Predicts spacecraft trajectory outcomes early from physics-invariant telemetry.
The repository now includes a calibrated synthetic pipeline from the Moon
through Neptune, sequential neural models, and non-neural baselines.

Current paper framing must be evidence-led: XGBoost on trajectory features
outperforms the Transformer on the current random mission split. Do not claim
that the Transformer is the strongest classifier.

## Stack

- **ML:** PyTorch `TrajectoryTransformer`, `TrajectoryLSTM`, and XGBoost baselines.
- **Backend:** FastAPI at `src/api/main.py` — `uvicorn src.api.main:app --port 8000` from project root
- **Frontend:** React 18 + Vite (no TypeScript), Recharts, Tailwind v4 (unreliable — use inline styles)
- **Local multi-planet data:** generated on this machine at
  `/media/Data/Coding/gmat-pred/data/` (NTFS drive, not the repo — too
  large for the Linux partition). Mercury, Venus, Mars, Jupiter, Saturn,
  Uranus generated via the original `build_database.py`; Neptune generated
  via the Numba JIT fast path (`experiments/numba_jit/`, see
  `docs/NUMBA_JIT_PROPAGATOR.md`). Individual planet source folders are
  kept at `/media/Data/Coding/gmat-pred/data/<planet>/` (regenerating
  Uranus/Saturn took 5-12 hours each with the original pipeline — do not
  delete without explicit confirmation). Final merged dataset (80K
  missions, all 8 sources) at
  `/media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet` —
  this is the canonical path everything in this session's reports points
  at. An earlier intermediate merge without Neptune (`merged_all/`,
  70K missions) was deleted as redundant once Neptune was folded in.
  The `data/merged_through_neptune_15min/` path referenced elsewhere in
  this doc was the teammate's machine, not this one.
- **Important:** generated data, reports, logs, and checkpoints are ignored by Git.

## Model — TrajectoryTransformer (src/ml/model.py)

- d_model=128, nhead=8, num_layers=4, dim_feedforward=512, Pre-LayerNorm
- Learnable CLS token prepended to sequence, CLS output → classification head
- Input: 13 physics-invariant features → Output: logit → sigmoid → **P(success)**
- **CRITICAL: model outputs P(success). Always invert: `p_fail = 1 - prob`**

## 13 input features

```
rel_x, rel_y, rel_z    synodic frame position
spec_energy            specific orbital energy
fpa_deg                flight path angle
norm_target_dist       distance to target / SOI
radial_vel             dr/dt toward target
vel_mag                total speed
earth_rmag             distance from source body
ecc                    eccentricity
mu_ratio               target μ / Sun μ  (cross-planet context)
soi_ratio              target SOI / transfer dist
dist_ratio             transfer dist / 1 AU
```

## Current final dataset

- 80,000 missions: 25,590 success / 54,410 failure.
- Final success rate: 0.320.
- Moon telemetry cadence: 900 seconds.
- Interplanetary telemetry cadence: 54,000 seconds = 15 hours.
- Folder names containing `15min` are historical names; do not interpret them as cadence.
- Final model configuration: early exit 40%, downsample factor 10, seed 42.
- For interplanetary missions, downsample factor 10 means every tenth 15-hour
  record, not ten minutes.
- `create_dataloaders()` splits by mission ID and fits `RobustScaler` on train only.
- Baselines now use the exact same mission ordering and seeded split.

## Current verified results

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority class | 67.35% | 0.0000 | 0.5000 |
| Energy threshold | 35.81% | 0.4964 | 0.5233 |
| Transformer, 40% trajectory | 79.73% | 0.7447 | 0.9363 |
| XGBoost, full summaries | 99.34% | 0.9899 | 0.9998 |
| XGBoost, first/last only | 99.33% | 0.9898 | 0.9998 |
| XGBoost, first row only | 98.42% | 0.9760 | 0.9987 |
| XGBoost, first row without context features | 98.42% | 0.9760 | 0.9987 |

Transformer training was interrupted after epoch 27 of 30 by laptop shutdown.
The best validation-loss checkpoint from epoch 23 survived and was evaluated
on the deterministic untouched test split.

The initial-only XGBoost result shows that current synthetic mission outcomes
are highly separable from initial conditions. This is not explicit target
column leakage, but it makes random mission splits optimistic. The next
paper-grade experiment should use grouped planet-held-out and/or
parameter-corridor-held-out evaluation.

Removing `mu_ratio`, `soi_ratio`, and `dist_ratio` from the initial-only
baseline does not reduce accuracy. Target-regime context is therefore not the
main cause of the near-perfect result; the initial dynamical state and
calibrated targeting corridors are.

## File map

```
src/
  ml/
    model.py         TrajectoryLSTM, TrajectoryTransformer
                       → Transformer accepts use_cls_token, use_pos_encoding flags
    dataset.py       TrajectoryDataset, create_dataloaders
    train.py         training loop — accepts --seed for reproducibility
    ablation.py      early-exit fraction sweep [0.1,0.2,0.3,0.4,0.6,1.0]
    evaluate.py      inference + metrics report on any parquet
    baselines.py     MajorityClass, EnergyThreshold, XGBoost baselines
                       → full-summary, endpoint-only, and initial-only diagnostics
    multi_seed.py    5-seed robustness experiment → reports/multi_seed/
    arch_ablation.py component ablation (CLS, pos enc, context feats, LSTM)
    results_table.py reads all report JSONs → Markdown + LaTeX tables
  api/
    main.py          FastAPI endpoints (see below)
  data_collection/
    build_database.py   GMAT sim runner
    merge_datasets.py   merge parquet runs
    mars_targeter.py    diagnostic targeting with production dynamics
    exact_targeter.py   checkpointed exact calibration grids
    adaptive_targeter.py adaptive outer-planet corridor refinement

experiments/
  numba_jit/           Numba-JIT-compiled fast path for outer-planet
                       generation (26-37x speedup at production cadence).
                       Validated against real Jupiter data — see
                       docs/NUMBA_JIT_PROPAGATOR.md. Used to generate the
                       Neptune dataset. Does not modify gmat_runner.py.

frontend/src/
  App.jsx                tab router: OVERVIEW/SIMULATOR/TRAINING/ABLATION/DATASET
  panels/
    Overview.jsx         polls /api/system every 5s, GPU bar, model stats, live log
    Training.jsx         real training via SSE, command paste box auto-fills form
    Ablation.jsx         reads ablation_results.json, clickable rows show epoch curves
    Simulator.jsx        realtime per-mission P(fail) stream with cancel logic
    Dataset.jsx          static documentation panel

reports/
  ablation/
    ablation_results.json       early-exit sweep, consumed by Ablation panel
  baselines/
    baseline_results.json       XGBoost + heuristic baselines per exit fraction
  multi_seed/
    seed_{N}_metrics.json       per-seed training history
    summary.json                mean ± std across all seeds
  arch_ablation/
    arch_ablation_results.json  component contribution deltas
  tables/
    main_comparison.{md,tex}    Table 1: Transformer vs baselines
    arch_ablation.{md,tex}      Table 2: Architecture ablation
    multi_seed.{md,tex}         Table 3: Robustness statistics
    summary.json                all experiment data in one place

models/
  transformer_production/
    best_model_transformer_binary.pt
    scaler_transformer_binary.pkl
    metrics_transformer_binary.json
```

## API endpoints (src/api/main.py)

```
GET  /api/health                    GPU/device info
GET  /api/system                    realtime snapshot: GPU mem, active jobs, ablation progress
GET  /api/status                    best model path, AUC, param count
GET  /api/ablation                  serve reports/ablation/ablation_results.json
GET  /api/metrics                   serve models/.../metrics_transformer_binary.json
POST /api/train/start               spawn training subprocess, returns job_id
GET  /api/train/stream/{job_id}     SSE stdout stream with structured epoch events
GET  /api/train/stop/{job_id}       terminate training job
GET  /api/simulator/missions        sample N mission IDs + true labels
GET  /api/simulator/stream          SSE realtime P(fail) per timestep for one mission
```

## Gotchas

- **Model outputs P(success)** — invert to get P(fail) for display and cancel logic
- **Simulator cancel gate** — only cancel after `min_elapsed_pct` to stay in trained distribution
- **Tailwind v4** — unreliable for dynamic classes, use inline styles
- **Gradient text** — must use CSS class, not inline React style (WebKit bug)
- **ReduceLROnPlateau** — `verbose` kwarg removed in PyTorch 2.x
- **PyArrow** — `.to_numpy()` is read-only in numpy 2.x, always `.copy()`
- **Scaler** — fitted on train split only, saved alongside checkpoint, required at inference
- **Run server from project root** — `src.api.main` import path requires it
- **No explicit leakage features** — models use only the 13 features above;
  label, failure_type, min_target_rmag, mission_id, source_body, and target_body
  are not input features.
- **Context-feature caveat** — mu_ratio, soi_ratio, and dist_ratio identify
  mission regimes indirectly. Report this and test grouped generalization.
- **Initial-condition separability** — first-row XGBoost reaches 98.42%
  accuracy; random mission splits do not establish transfer to unseen planets
  or unseen targeting corridors.
