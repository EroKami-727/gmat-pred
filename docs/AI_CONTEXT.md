# OrbitGuard — AI Context

Read this first before touching any code.

## What it does

Predicts failing spacecraft trajectories early using a Transformer on physics-invariant telemetry. Goal: cancel doomed Monte Carlo simulations before they complete (early-exit). The model streams in telemetry timestep-by-timestep and outputs P(fail) in realtime.

## Stack

- **ML:** PyTorch `TrajectoryTransformer` (primary). `TrajectoryLSTM` exists but deprecated.
- **Backend:** FastAPI at `src/api/main.py` — `uvicorn src.api.main:app --port 8000` from project root
- **Frontend:** React 18 + Vite (no TypeScript), Recharts, Tailwind v4 (unreliable — use inline styles)
- **Data:** 10K missions, 85.3M rows, 13.44 GB Parquet at `data/merged/missions.parquet`
- **Venv:** `/home/haise/Coding/venvs/gmat-pred/`

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

## Dataset

- label=1 → success (35.3%), label=0 → failure (64.7%)
- Downsampled 15x: 60s → 15min steps, ~576 steps/mission full trajectory
- `early_exit_frac` slices first N% of each trajectory at training time
- `create_dataloaders()` splits by mission_id, RobustScaler fitted on train only
- **Bug fixed:** `.to_numpy().copy()` required — PyArrow returns read-only arrays in numpy 2.x

## Ablation results (transformer, 30 epochs)

| Exit % | AUC   | F1    | Acc    |
|--------|-------|-------|--------|
| 10%    | 0.889 | 0.778 | 83.93% |
| 20%    | 0.964 | 0.854 | 90.67% |
| 30%    | 0.982 | 0.893 | 92.80% |
| 40%    | 0.997 | 0.948 | 96.53% |
| 60%    | 0.998 | 0.971 | 98.00% |
| 100%   | 1.000 | 0.989 | 99.27% |

Production model trained at `--early-exit 0.4`. In simulator, `min_elapsed_pct=0.4` gates cancellation.

## File map

```
src/
  ml/
    model.py      TrajectoryLSTM, TrajectoryTransformer
    dataset.py    TrajectoryDataset, create_dataloaders
    train.py      training loop, default: --model transformer
    ablation.py   early-exit fraction sweep [0.1,0.2,0.3,0.4,0.6,1.0]
    evaluate.py   inference + metrics report on any parquet
  api/
    main.py       FastAPI endpoints (see below)
  data_collection/
    build_database.py   GMAT sim runner
    merge_datasets.py   merge parquet runs

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
    ablation_results.json    output of ablation sweep, consumed by Ablation panel

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
