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

## OrbitGuard Live Simulator

The dashboard includes a real-time mission simulator that runs the ML model live on trajectory data. Key features:

- **Mission queue**: load N random missions from the dataset (with planet filter), or create synthetic ones
- **Mission creator**: define target, launch energy (C3), and phase angle; the physics engine propagates a two-body RK4 trajectory and feeds it to the ML in real time
- **Buffered playback**: SSE streams all ML steps at full speed; the client plays them back at 4×/2×/1×/½×/¼× or frame-by-frame (STEP mode)
- **Orbital map**: synodic-frame trajectory visualisation; hover after completion to scrub through telemetry at any point
- **Abort details**: when the model fires, shows the calibrated threshold, P(fail) at abort, and elapsed % alongside a CORRECT/FALSE-POSITIVE verdict

### Per-planet model architecture

The production system trains **one model per target body**. This replaced an
earlier regime-split design (one model for inner planets, one for outer) that
silently failed — see the root-cause note below.

Each model is the same Transformer (d_model=128, nhead=8, 4 Pre-LN layers, CLS
token) with two heads sharing one trunk:

- **outcome head** — will this mission fail?
- **failure-mode head** — *how* will it fail (surface impact / orbit too high / missed target)?

`PlanetRouter` (`src/ml/planet_router.py`) selects the model, its per-timestep
normalisation statistics, and its calibrated threshold at inference time.

#### Root cause: why the regime-split models failed

The regime models fitted **one scaler across 3–4 planets**. That scaler's IQR
spans the *cross-planet* range (`spec_energy` IQR = 23.6, covering Mercury
through Mars), so *within-planet* mission-to-mission variation collapsed to
~1e-5 of the input range — below what gradient descent can learn to amplify,
and Pre-LN LayerNorm removes what survives.

The result was a model that emitted a near-constant probability per planet:
every Venus mission scored P(fail) = 0.020910 regardless of outcome, so no
threshold could ever separate them.

This went unnoticed because **XGBoost baselines were unaffected** — trees split
on absolute feature values and need no amplification, so they reported
AUC ≈ 1.0 on the very same data that the transformer could not learn.

Two changes fix it:

1. **Per-planet fit** — the scaler sees one planet, so within-planet spread
   occupies the full dynamic range.
2. **Per-timestep z-scoring** — each feature is standardised against its
   distribution *at that timestep index* across missions, making
   mission-to-mission deviation O(1). Mars val AUC 0.939 → 0.998 from this alone.

Models are also trained on **random-length prefixes**, so predictions are
in-distribution at any point of the stream rather than only at the 40% horizon.

### Results (held-out test split, 40% of trajectory observed)

1,200 held-out missions per planet — never seen in training or validation.

| Target  | AUC | Recall | Precision | F1 | Failure-mode acc |
|---------|-----|--------|-----------|-----|------------------|
| Moon    | 0.9994 | 0.9925 | 0.9851 | 0.9888 | 1.0000 |
| Mercury | 0.9996 | 0.9989 | 0.9923 | 0.9956 | 0.9812 |
| Venus   | 1.0000 | 0.9975 | 0.9987 | 0.9981 | 0.9861 |
| Mars    | 0.9998 | 0.9978 | 0.9933 | 0.9956 | 0.9989 |
| Jupiter | 1.0000 | 1.0000 | 0.9987 | 0.9994 | 0.9449 |
| Saturn  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9769 |
| Uranus  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 |
| Neptune | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9293 |
| **Overall** | — | **0.9983** | **0.9959** | **0.9971** | **0.9773** |

11 false negatives in 9,600 held-out missions. All eight targets carry a fused
tree assist.

**Moon** was missing entirely until 2026-08-02: it has 10,000 missions in the
dataset (85M rows, 6-day transfers at 60 s cadence) but was never extracted or
trained, so the simulator listed Moon missions it could not score — the router
returned `available: false`, `p_fail = 0.0`, and no abort could ever fire. Its
cadence differs from the interplanetary targets by 900x, which
`planet_config.CADENCE_HOURS` now handles. Previous regime models on the same
missions: Venus constant P(fail)=0.020910 (AUC ≈ 0.5), Mars 0.61, Jupiter 0.87.

### Tree assist for rare failure modes

The Transformer misses rare failure modes even when the signal is fully present
in its own input. Uranus `surface_impact` is 119 of 6,611 failures and sequence
recall on it was **0.000** — while XGBoost on the *identical per-timestep
z-normalised window* separates it from success at **AUC 1.000**. Oversampling the
mode up to 45x (`--mode-alpha 1.0`) did not help, so this is an optimisation
limit of the sequence model, not missing information.

`src/ml/train_assist.py` fits a per-planet gradient-boosted classifier on that
same window; `PlanetRouter` fuses it with the Transformer (`max`) once the prefix
is long enough, and the threshold is recalibrated on the fused score. Uranus
`surface_impact` recall went 0.000 → 1.000 and overall F1 0.9960 → 0.9981.

The Transformer remains primary: it streams at any prefix length and predicts the
failure mode. The tree only contributes at the fixed decision window.

```bash
/home/haise/Coding/venvs/gmat-pred/bin/python3 -m src.ml.train_assist --all
/home/haise/Coding/venvs/gmat-pred/bin/python3 -m src.ml.recalibrate
```

### Does the trajectory actually help? (`src/ml/prune_economics.py`)

The honest answer for this dataset is **no**. Screening from the 6 launch-burn
offsets *before running anything* matches the telemetry model and costs nothing:

| Screen | Compute saved | Good missions destroyed |
|--------|---------------|-------------------------|
| **T0** — launch parameters, before propagating | **64.6%** | **0.8%** |
| T40 — telemetry Transformer at 40% | 38.9% | 0.2% |
| Cascade — T0 where confident, else T40 | 65.5% | 1.3% |

Compute is charged in propagation-days at 99% failure recall. T0 reaches
AUC 0.9975–1.0000 per planet and predicts the failure *mode* at 0.96–0.99,
matching the sequence model on both tasks.

The cascade does not pay for itself: it buys 0.9 pp of savings for 0.5 pp more
false prunes. **A 6-feature tabular classifier is sufficient for pruning here.**

This is expected once stated plainly: the simulator is deterministic, so the
outcome is a fixed function of the injection parameters and the trajectory is
just their integral — it carries no information the parameters do not already
have. A sequential model earns its place only where that stops being true
(mid-flight stochasticity, unmodelled dynamics, sensor noise, or missions whose
launch parameters are not recorded).

> Linear models are **not** sufficient: logistic regression on the same six
> features scores AUC 0.49 — chance — on every planet. The map is strongly
> nonlinear, so this is a case for ML, just not for a sequence model.

For comparison, the previous regime models on the same missions: Venus emitted a
constant P(fail)=0.020910 (AUC ≈ 0.5), Mars AUC 0.61, Jupiter AUC 0.87.

Accuracy is essentially flat from 10% to 40% of the trajectory — the outcome is
largely determined by injection conditions — so aborts could be taken earlier
than the current 40% operating point if desired.

Reproduce with:

```bash
/home/haise/Coding/venvs/gmat-pred/bin/python3 test_ml.py --limit 1200
```

### Known limitations

- **The extracts are in FILE order, not mission_id order.** `missions.parquet`
  is not sorted by `mission_id`, and neither is `summary.parquet`. Joining any
  two of {extract, params, summary} by row position silently pairs unrelated
  missions. Each `.npz` now carries a `mission_ids` array
  (`src/data_collection/recover_mission_ids.py`) — **always join on it.** Two
  analyses in this repo were wrong because of this before it was caught, so the
  join is now asserted in `prune_economics.py`.
- **Mission list sampling.** `/api/simulator/missions` previously collected the
  first `n*20` missions in file order. Since each planet's file opens with the
  targeter's nominal seeds, that pool held zero failures for Jupiter/Saturn/
  Neptune and, for Uranus, only the grazing failures the model misses. It now
  reads row groups in random order with a per-group cap.

### Mission creator

`src/api/mission_builder.py` builds user-defined missions with the **same
propagator and feature code that produced the dataset**
(`gmat_runner.run_synthetic`), so created missions are in-distribution and
scorable. Verified identical to real telemetry at step 0 (`rel_x` −4.14341e+07,
`spec_energy` 9.0969, `earth_rmag` 6564.43 for a nominal Venus transfer).

Missions are parameterised as the dataset is — a circular parking orbit plus an
impulsive TOI burn in the VNB frame — with offsets relative to the Hohmann
nominal. `/api/simulator/planet_info` returns both the nominal and the 1-sigma
dispersions for slider scaling.

> The earlier `src/api/trajectory_gen.py` propagated heliocentric two-body motion
> and computed Sun-referenced elements, so its features did not match the
> training schema despite its docstring claiming they did (Venus `spec_energy`
> −514.9 vs the correct Earth-centric +9.10). Every mission it produced scored
> |z| ~ 1e13 and was unscorable. It is still used for the orbital-map preview
> only, not for inference.

Transfers are extremely sensitive: dispersions are small (Venus `dv_V`
sigma = 0.003 km/s) and a 2-sigma burn error already flips a mission to failure,
consistent with the dataset's 32% success rate. Beyond ~5 sigma the input is
flagged out-of-distribution — advisory only, since P(fail) remains correct
there; it is not a veto.

### Calibrated thresholds

Thresholds are chosen on the **validation** split as the midpoint of the plateau
of thresholds maximising **failure-class F1** (`pos_label=0`), then reported
against the untouched test split. Taking the plateau midpoint rather than the
first maximum keeps the operating point away from a cliff edge.

To recalibrate without retraining (models stay frozen; only the operating point moves):

```bash
/home/haise/Coding/venvs/gmat-pred/bin/python3 -m src.ml.recalibrate
```

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

## Train Per-Planet Models

The whole rebuild runs unattended in one detached process (survives editor
restarts; progress in `.runlogs/pipeline.log`):

```bash
setsid nohup ./run_pipeline.sh > /dev/null 2>&1 &
tail -f .runlogs/pipeline.log
```

Or run the stages by hand:

```bash
VENV=/home/haise/Coding/venvs/gmat-pred/bin/python3
DATA=/media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet

# Step 1 — Extract compact per-planet arrays (one streaming pass, ~70 MB each).
# Each planet is downsampled to ~100 steps/mission so Mercury (202 raw rows) and
# Neptune (21,471) end up with comparable sequence lengths.
$VENV -m src.data_collection.extract_per_planet \
  --data $DATA --out-dir data/per_planet

# Step 2 — Train one dual-head model per planet (~50s each on an RTX 4060)
$VENV -m src.ml.per_planet_train --all --epochs 60

# Step 3 — Recalibrate thresholds (optional; step 2 already calibrates)
$VENV -m src.ml.recalibrate
```

> **Memory note:** Neptune is 214M rows. Extraction streams one batch at a time
> with Arrow readahead disabled, but still peaks near 17 GB because Arrow's pool
> does not fully return freed blocks. It is a one-time cost — the `.npz` outputs
> are cached, and step 1 skips planets already extracted. Close memory-hungry
> apps before a full re-extract on a 24 GB machine.

---

## Train a Multi-planet Baseline Model (paper experiments only)

```bash
python3 -m src.ml.train \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --model transformer \
  --epochs 50 \
  --early-exit 0.4 \
  --downsample-factor 10 \
  --seed 42 \
  --output-dir models/transformer_multiplanet
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
