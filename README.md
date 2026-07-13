# OrbitGuard — Progress Report

> **What this project does in one sentence:** OrbitGuard watches a spacecraft's early flight path and predicts whether the mission will succeed or fail — so we can cancel doomed simulations before they waste hours of compute.

---

## The Problem

NASA GMAT runs Monte Carlo simulations to test spacecraft trajectories. Most of these simulations fail. The problem is that GMAT doesn't know a mission is doomed until it finishes running the full trajectory — which can take a long time for outer planets.

OrbitGuard fixes this: a machine learning model watches just the **first 40% of the trajectory** and makes a Go/No-Go call. This saves up to **80% of compute** per simulation batch.

---

## What We Built

### Earlier

- Built the GMAT simulation pipeline and feature engineering (13 physics-invariant features)
- Generated 10,000 Moon missions — the first dataset
- Trained original Transformer and LSTM models on Moon-only data
- Established initial results: AUC 0.936, F1 0.745

### This session

- Generated 7 more planet datasets (Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune) — **70,000 more missions**
- Neptune's 37-year-per-mission Hohmann transfer was too slow normally — built a **Numba JIT-compiled physics engine** that is 26–37× faster, cutting generation from ~20 hours to ~51 minutes
- Merged all 8 datasets → **80,000 missions total**, independently verified on local hardware
- Re-trained the Transformer on all 8 planets (50 epochs), achieving AUC 0.984
- Ran **8 paper-grade evaluations** (see below)

---

## Key Numbers at a Glance

| Metric | Value | Notes |
|--------|-------|-------|
| Total missions | **80,000** | 8 planets × 10K each |
| Dataset size | ~71 GB | merged_all_v2, Parquet |
| Overall success rate | 32.0% | Matches teammate's earlier independent run |
| Best F1 (XGBoost-summary) | **0.992 ± 0.001** | 5-seed confidence interval |
| Best AUC (XGBoost-summary) | **1.000 ± 0.000** | 5-seed CI |
| Transformer AUC (calibrated) | **0.984** | Multi-planet, 50 epochs |
| Transformer F1 (tuned threshold) | **0.921** | Up from 0.838 at default 0.5 |
| Calibration ECE (isotonic) | **0.0045** | 11.5× better than uncalibrated |
| JIT speedup | **26–37×** | Neptune generation |

---

## Model Comparison

> Tested at 40% trajectory seen (early exit). Random 70/15/15 split, seed 42.

```
Model                 F1      AUC     Notes
──────────────────────────────────────────────────────
Majority baseline    0.000   0.500   Always predicts failure
Energy threshold     0.536   0.535   Physics heuristic
XGBoost-initial      0.974   0.998   Uses first timestep only
XGBoost-summary      0.992   1.000   Uses full early-exit segment
Transformer (raw)    0.838   0.984   Trained online on telemetry
Transformer (tuned)  0.921   0.984   Threshold optimised to 0.557
```

XGBoost is still the strongest baseline because the task is highly separable from initial conditions. The Transformer is valuable as the **online classifier** — it processes telemetry one step at a time, which is more realistic for real-time abort decisions.

---

## Generalisation — Leave-One-Target-Out (LOTO)

> Train on 7 planets, test on 1 held-out planet. Repeated across 5 seeds — results were identical (std = 0.000), meaning the result is not noise.

| Target  | AUC   | F1@0.5 | Status              |
|---------|-------|--------|---------------------|
| Saturn  | 1.000 | 1.000  | ✅ Perfect           |
| Neptune | 1.000 | 1.000  | ✅ Perfect           |
| Jupiter | 0.998 | 0.997  | ✅ Near-perfect      |
| Uranus  | 0.992 | 0.000  | ⚠️ Threshold wrong   |
| Venus   | 0.856 | 0.000  | ⚠️ Threshold wrong   |
| Mars    | 0.509 | 0.000  | ❌ Regime shift      |
| Mercury | 0.496 | 0.000  | ❌ Regime shift      |
| Moon    | 0.296 | 0.000  | ❌ Hardest target    |

**Two different types of failure:**
- **Uranus & Venus** — the model ranks missions correctly (high AUC) but the default 0.5 threshold is wrong for these targets. Fixable without retraining via isotonic calibration or threshold tuning.
- **Mars, Mercury, Moon** — the model genuinely doesn't transfer. These planets exist in a physically different regime: their `dist_ratio` and `earth_rmag` features shift by 1–43 standard deviations from the training distribution. No threshold trick fixes this.

---

## Calibration Improvements

After the model outputs a probability, we can improve it without retraining:

```
Configuration          F1      ECE     Notes
──────────────────────────────────────────────────────
Default threshold=0.5  0.838   0.052   Out of the box
Optimised threshold    0.921   0.052   0.557 found by grid search
Isotonic calibration   0.919   0.0045  11.5× ECE reduction
```

ECE (Expected Calibration Error) measures how trustworthy the probability estimates are. A model that says "70% chance of success" should be right 70% of the time — isotonic calibration makes this much more accurate.

---

## Domain Generalisation Experiment

We tried oversampling the 4 weakest targets (Moon, Mars, Mercury, Venus) at 2× during training to see if it helps. Result: **mixed**.

| Target  | F1 (unbalanced) | F1 (balanced) | Change  |
|---------|-----------------|---------------|---------|
| Moon    | 0.577           | **0.826**     | +0.249 ✅ |
| Mars    | 0.899           | **0.923**     | +0.024 ✅ |
| Jupiter | 0.965           | 0.976         | +0.011 ✅ |
| Mercury | 0.531           | 0.500         | -0.031 ⚠️ |
| Uranus  | 0.981           | 0.977         | -0.004  |
| Saturn  | 0.997           | 0.997         | —       |
| Neptune | 0.998           | 0.999         | —       |
| Venus   | 0.902           | **0.000**     | -0.902 ❌ |

Moon improved a lot. Venus completely collapsed. There is no simple fix — future work would tune per-target upweight factors individually rather than applying one fixed 2× to all four.

---

## Neptune: Numba JIT Speed

Neptune missions simulate a 37-year Hohmann transfer — each mission requires ~1.3 million RK4 substeps. The standard Python pipeline would take 15–24 hours to generate 10,000 missions.

We wrote a Numba JIT-compiled version of the RK4 hot loop:

```
Method              Time (10K missions)
────────────────────────────────────────
Normal Python       ~20 hours (estimated)
Numba JIT           ~51 minutes (actual)
Speedup             26–37×
```

Validated by comparing 50 JIT missions against real Jupiter data — 50/50 exact outcome match, max position error 0.036 km.

---

## What the 8 Paper Upgrades Were

We completed 8 evaluations that weren't in the original codebase:

1. **Multi-seed confidence intervals** — XGBoost reported with ± std across 5 seeds
2. **Calibration metrics** — Brier score, ECE, PR-AUC added to all experiments
3. **Formal 3-way ablation** — XGBoost-initial vs XGBoost-summary vs Transformer on identical splits
4. **Target-family parameter holdout** — train on inner corridor, test on unseen corridor
5. **Domain generalisation baseline** — balanced vs unbalanced sampling experiment
6. **Leave-one-target-out (LOTO)** — per-planet generalisation audit across 5 seeds
7. **Error analysis tables** — feature-shift analysis for all weak held-out cases
8. **Calibration plots** — reliability diagrams, isotonic fitting, threshold sweep

---

## Quick Start

```bash
# Activate environment
source /home/haise/Coding/venvs/gmat-pred/bin/activate

# Start API backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (separate terminal)
cd frontend && npm run dev
```

---

## Train a Model

```bash
# Multi-planet Transformer — 40% early exit, 50 epochs
python3 -m src.ml.train \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --model transformer \
  --epochs 50 \
  --early-exit 0.4 \
  --downsample-factor 10 \
  --seed 42 \
  --output-dir models/transformer_multiplanet

# Continue from checkpoint
python3 -m src.ml.train \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --model transformer \
  --epochs 70 \
  --early-exit 0.4 \
  --downsample-factor 10 \
  --seed 42 \
  --output-dir models/transformer_multiplanet \
  --resume-from models/transformer_multiplanet/checkpoint_best.pt \
  --epoch-offset 50
```

---

## Run Paper Experiments

```bash
DATA=/media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet
PARAMS=/media/Data/Coding/gmat-pred/data/merged_all_v2/mission_params.parquet

# XGBoost baselines + LOTO + parameter holdout
python3 -m src.ml.grouped_baselines --data $DATA --output reports/baselines/leave_one_target_out_exit40_ds10_calibrated.json
python3 -m src.ml.parameter_holdout_baselines --data $DATA --params $PARAMS

# Formal 3-way ablation with 5-seed CIs
python3 -m src.ml.formal_ablation --data $DATA \
  --transformer-metrics models/transformer_multiplanet/metrics_transformer_binary.json \
  --output reports/baselines/formal_ablation.json

# Multi-seed wrappers
python3 -m src.ml.multi_seed_grouped --data $DATA --seeds 0 1 2 3 4
python3 -m src.ml.multi_seed_parameter_holdout --data $DATA --params $PARAMS --seeds 0 1 2 3 4

# Error analysis for weak targets
python3 -m src.ml.error_analysis \
  --data $DATA --params $PARAMS \
  --grouped reports/baselines/leave_one_target_out_exit40_ds10_calibrated.json \
  --parameter reports/baselines/parameter_holdout_exit40_ds10_calibrated.json \
  --out docs/ERROR_ANALYSIS.md
```

---

## Docs

- [`docs/PAPER_READY_SUMMARY.md`](docs/PAPER_READY_SUMMARY.md) — all findings consolidated for paper writing
- [`docs/RESEARCH_LEDGER.md`](docs/RESEARCH_LEDGER.md) — full history of decisions and verified numbers
- [`docs/NUMBA_JIT_PROPAGATOR.md`](docs/NUMBA_JIT_PROPAGATOR.md) — JIT implementation and validation
- [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) — feature-shift tables for weak held-out targets
- [`docs/AI_CONTEXT.md`](docs/AI_CONTEXT.md) — full technical context for AI assistants
- [`src/ml/README_EXPERIMENTS.md`](src/ml/README_EXPERIMENTS.md) — usage docs for all experiment scripts
