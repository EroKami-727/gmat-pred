# OrbitGuard Research Ledger

This file is the running source of truth for paper writing. Update it whenever
data, code, model results, or scientific interpretation changes.

## Current Goal

Build a defensible OrbitGuard research paper on early spacecraft trajectory
outcome prediction across calibrated Moon-to-Neptune synthetic transfers.

The paper must be honest about what is proven and what is not proven. Strong
results on random mission splits are not enough if initial-condition corridors
make the task easy.

## Current Dataset State

- Final merged dataset: `data/merged_through_neptune_15min`
- Missions: 80,000
- Success count: 25,590
- Failure count: 54,410
- Success rate: 0.320
- Moon telemetry cadence: 900 seconds
- Interplanetary telemetry cadence: 54,000 seconds = 15 hours
- Historical folder names containing `15min` must not be interpreted as true
  interplanetary cadence.
- Final EDA: `reports/eda_merged_through_neptune_15min/eda_report.html`

**Note:** the numbers above came from a teammate's machine and were never
reproduced locally until the local-reproduction pass below. The dataset path
`data/merged_through_neptune_15min` does not exist on this machine — treat
it as historical context, not a live path.

## Local Reproduction (2026-06-21)

A full 80,000-mission, 8-source dataset was independently generated and
merged on this machine — same methodology (calibrated nominals, seed=42,
35% success bias, early-exit 0.4, downsample 10), not a copy of the
teammate's data. Mercury/Venus/Mars/Jupiter/Saturn/Uranus via the original
`build_database.py`; Neptune via a validated Numba JIT fast path
(`experiments/numba_jit/`, see `docs/NUMBA_JIT_PROPAGATOR.md`, 26-37x
speedup, validated against real production Jupiter data before trusting it).
Stored at `/media/Data/Coding/gmat-pred/data/merged_all_v2/` (71 GB, NTFS
drive — too large for the repo's Linux partition).

- Missions: 80,000 — success rate 32.0% (25,568 success / 54,432 failure),
  matching the teammate's 0.320 almost exactly. Good sanity check that the
  independently-generated dataset reproduces the same statistical profile.

### Verified random-split baselines (own data, not the teammate's)

Same config as before: early exit 0.4, downsample 10, seed 42.

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority | 67.74% | 0.000 | 0.500 |
| Energy threshold | 44.09% | 0.536 | 0.535 |
| XGBoost summary | 99.44% | 0.991 | 1.000 |
| XGBoost endpoints | 99.14% | 0.987 | 0.999 |
| XGBoost initial | 98.29% | 0.974 | 0.998 |
| XGBoost initial no context | 98.33% | 0.974 | 0.998 |

Within ~1% of the teammate's reported numbers across the board — confirms
the local pipeline reproduces the same separability pattern.
Full table: `docs/STATISTICAL_AUDIT_SUMMARY_LOCAL.md`.

### Calibration audit (new — not in the teammate's original run)

Extended `grouped_baselines.py` / `parameter_holdout_baselines.py` to also
report PR-AUC, Brier score, ECE, and a confusion matrix per held-out
target/bin (`src/ml/calibration_utils.py`). This surfaced a distinction the
original F1/AUC-only audit could not make: **some "weak" targets are
calibration failures, not generalization failures.**

Ranking-works-but-threshold-fails cases (AUC >= 0.80, F1 collapses at 0.5):

- Uranus (LOTO): AUC=0.992, F1@0.5=0.000, ECE=0.203 — ranking is excellent,
  the 0.5 threshold is just wrong for this target's probability scale.
- Venus (LOTO): AUC=0.856, F1@0.5=0.000, ECE=0.335 — same pattern, weaker.
- TOI_V bin 4 (corridor holdout): AUC=0.978, F1@0.5=0.675, ECE=0.002 — low
  ECE here, so this one is closer to a genuine sparse-success edge case.
- AOP bin 1 (corridor holdout): AUC=0.885, F1@0.5=0.388, ECE=0.410 — highest
  ECE of any case; likely a mix of calibration failure and genuine boundary
  confusion (57% success rate, near the success/failure decision boundary).

Mars, Mercury, and Moon remain genuinely weak (AUC < 0.6, not just
miscalibrated) — see error analysis below for why.

### Multi-seed stability (new)

`src/ml/multi_seed_grouped.py` and `src/ml/multi_seed_parameter_holdout.py`
re-run every LOTO target and every corridor-holdout bin across seeds
[0,1,2,3,4]. Result: **std=0.000 across all seeds, for every target and
every bin.** The XGBoost configuration used here has no row/column
subsampling, so the model fit is deterministic given a fixed train/test
split — and since LOTO/corridor-holdout splits are themselves deterministic
(defined by target identity or quantile bin, not by seed), there is no
seed-to-seed variance to measure. This is a valid finding: it rules out
training randomness as an explanation for the weak Mars/Mercury/Moon/Venus
results — they are structural, not noise.

Separately, the **random-split** formal ablation (`src/ml/formal_ablation.py`)
genuinely varies the train/test partition by seed, and shows real (if tiny)
variance:

| Model | F1 (mean ± std) | AUC (mean ± std) | ECE (mean ± std) |
|---|---|---|---|
| XGBoost initial | 0.974 ± 0.001 | 0.998 ± 0.000 | 0.012 ± 0.000 |
| XGBoost summary | 0.992 ± 0.001 | 1.000 ± 0.000 | 0.003 ± 0.000 |

Transformer-sequential is reported as a single-run point estimate in the
same artifact, not a 5-seed CI — retraining the Transformer 5x on this 80K
dataset would take ~20+ hours on the available hardware (one run is already
~4-5 hours). This asymmetry is stated explicitly rather than silently
treating the two legs as equivalent evidence.

### Error analysis (new — `src/ml/error_analysis.py`)

For every weak LOTO target and corridor-holdout bin, computed standardized
train/test feature-distribution shift across the 13 input features. Result:
Mars, Mercury, Moon, and Venus all show large shifts in `dist_ratio`,
`earth_rmag`, `rel_x`, and `norm_target_dist` (0.8-43 standard deviations) —
features that encode transfer distance and target-body scale. Confusion
matrices confirm near-total collapse to the majority class for these
targets (e.g. Mercury: 0 predicted successes out of 2,155 actual successes).

**Interpretation:** full unseen-target transfer fails because each target
occupies a categorically different physical regime (distance, SOI scale),
not because the launch-parameter corridor is merely "harder." This is a
stronger, more specific claim than "generalization is mixed" — it points at
*why*. Within-target corridor holdout (TOI_V/AOP bins) avoids this because
the physical regime stays in-distribution; only the parameter values shift.
Full report: `docs/ERROR_ANALYSIS.md`.

### Multi-planet Transformer (complete)

`models/transformer_multiplanet/` — 50 total epochs (30 + a warm-started
continuation, since loss had not plateaued at epoch 30). Final test:
Accuracy 87.67%, F1 0.838, ROC-AUC 0.984. Per-target breakdown (random
split, all targets present in training) shows Mercury (PR-AUC 0.547) and
Moon (PR-AUC 0.851) as the hardest targets even in-distribution — full
detail in `docs/PAPER_READY_SUMMARY.md` §5b.

### Domain generalization baseline (complete — mixed result)

`--upweight-targets mars mercury moon venus --upweight-factor 2.0` via a
`WeightedRandomSampler` in `create_dataloaders`/`train.py`. Compared
against the unbalanced model's own 30-epoch checkpoint (not its extended
50-epoch result, which would have been an unfair comparison).

Result: oversampling weak targets is not a silver bullet. Moon improved
substantially (PR-AUC 0.294→0.654, F1 0.577→0.826) and Mars improved
slightly, but **Venus collapsed entirely** (F1 0.902→0.000), dragging
every aggregate metric down (accuracy 87.14%→80.00%, F1 0.826→0.756).
First implementation attempt was a pure inverse-target-count weighting,
which is a near no-op on this dataset since all 8 targets have exactly
10,000 missions each — caught via a standalone sampler test before
wasting a full training run, fixed by adding explicit per-target weight
overrides. Full numbers in `docs/PAPER_READY_SUMMARY.md` §6.

## Regime-Split Production Models (2026-07-18)

### Motivation

The single multi-planet Transformer showed AUC 0.984 on random splits but the
LOTO audit revealed it completely fails on unseen targets (Mercury, Mars, Moon
AUC ≈ 0.5 — worse than random). The root cause identified in the error analysis:
`dist_ratio`, `earth_rmag`, and `norm_target_dist` shift by 10–43 standard
deviations between inner and outer planets. A single model cannot handle both
physical regimes.

### Architecture

Two specialist Transformers, same hyperparams (d_model=128, nhead=8, 4 Pre-LN
layers, CLS token, early-exit=0.4), different downsample factors to match the
raw data cadence per regime:

| Regime | Planets | ds | Model path |
|--------|---------|---|---|
| Inner | Mercury, Venus, Mars | 15 | `models/inner_production/` |
| Outer | Jupiter, Saturn, Uranus, Neptune | 50 | `models/outer_production/` |

Pre-downsampling for outer planets was done first to avoid writing ~43 GB of
temp files to the NTFS FUSE mount during training (`src/data_collection/presample.py`).

### Training results

Inner (Mercury/Venus/Mars, ds=15):
- Accuracy: ~98%, F1: 0.990, AUC: 0.997

Outer (Jupiter/Saturn/Uranus/Neptune, ds=50, trained on `data/outer_ds50.parquet`):
- Accuracy: ~99%, F1: 0.990, AUC: 0.996

### RegimeRouter

`src/ml/regime_router.py` — loads both models at startup, selects the correct
one at inference time based on target body name, and applies the per-target
calibrated threshold. Falls back to inner model if target is unrecognised.

### Calibration bug found and fixed (2026-07-18)

The initial calibration used `f1_score(labels, preds, zero_division=0)` with
default `pos_label=1` (success). This optimises for predicting *successes*
correctly, not failures — so the optimal strategy is a threshold above any
P(fail) the model actually outputs, meaning the abort system never fires.

**Fix applied to `src/ml/per_target_calibration.py`:**
- Changed to `pos_label=0` to optimise failure-class F1
- Sweep now starts from 0.005 (was 0.02) for finer granularity near zero
- AUC was also computed inverted (passed P(fail) as score for positive class);
  fixed to negate probs: `roc_auc_score(labels, [-p for p in probs])`
- `load_missions` now uses per-regime downsample automatically (ds=15 inner,
  ds=50 outer) — the old `--downsample` CLI arg is removed since it was a
  single value applied to all targets

**Action required:** re-run calibration after this fix to update
`models/thresholds.json`:

```bash
/home/haise/Coding/venvs/gmat-pred/bin/python3 -m src.ml.per_target_calibration \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --models-dir models \
  --early-exit 0.4 \
  --output models/thresholds.json
```

The thresholds currently in `models/thresholds.json` were computed with the
wrong metric and should not be trusted for abort decisions until re-calibration
completes.

## Live Simulator (2026-07-18)

### Physics trajectory generator (`src/api/trajectory_gen.py`)

Two-body solar gravity RK4 propagator with circular planetary orbits. Given a
target planet, launch energy (C3 in km²/s²), and departure phase angle, it
computes a complete synthetic interplanetary trajectory and extracts all 13
training features at each timestep. Feature normalization matches the training
schema exactly:
- `norm_target_dist = dist / SOI` (not initial dist)
- `soi_ratio = SOI / dist`
- `dist_ratio = dist / AU`

`hohmann_c3(target)` and `optimal_phase_deg(target)` expose the minimum-energy
Hohmann transfer parameters for each planet. The success criterion is
`dist_to_target < 1.5 × SOI`.

### Stream endpoint changes

- `step_delay_ms` hardcoded to 10 ms server-side; playback speed is
  frontend-controlled via the buffered playback system
- ML inference stride: `total_steps // 150` (was `// 60`) — 2.5× more inference
  calls emitted per mission for smoother spacecraft animation
- Calibrated threshold applied automatically; sent in SSE `info` header as
  `calibrated_threshold`

### Simulator frontend (`frontend/src/panels/Simulator.jsx`)

- Buffered SSE playback: all steps stream into `bufferRef` at server speed;
  `playTimerRef` drives display at user-selected rate
- Speed controls: 4×/2×/1×/½×/¼×/STEP frame-by-frame
- Pause/Resume during active stream
- Orbital map hover-scrub (after completion only — disabled during live playback
  to prevent position interference)
- Abort detail row: shows elapsed %, P(fail) at abort, calibrated threshold used,
  and excess above threshold
- Regime and calibrated threshold displayed in probability panel during stream

## Code/Branch State

- GitHub branch pushed for teammate review:
  `codex/fix-baseline-split-cadence`
- Pushed branch includes calibrated generation, targeting, adaptive propagation,
  training/model fixes, and baseline split/cadence fixes.
- Latest local audit additions are not pushed yet:
  - `docs/AI_CONTEXT.md`
  - `docs/PAPER_VALIDITY_AUDIT.md`
  - `docs/RESEARCH_LEDGER.md`
  - `src/ml/baselines.py`
  - `src/ml/dataset.py`
- Reason: final commit/push was blocked by tool approval usage limit.

## Final Transformer Result

Configuration:

- Data: `data/merged_through_neptune_15min/missions.parquet`
- Model: Transformer binary classifier
- Early exit: 0.4
- Downsample factor: 10
- Seed: 42
- Batch size: 32
- Training request: 30 epochs
- Completed: interrupted by laptop shutdown after epoch 27
- Best validation epoch by loss: 23

Held-out test result from saved best checkpoint:

- Accuracy: 79.7250%
- Loss: 0.3926
- F1: 0.7447
- ROC-AUC: 0.9363
- Confusion: TP=3549, FP=2064, FN=369, TN=6018

Paper interpretation:

- This checkpoint is usable because it was selected by validation loss and
  evaluated on an untouched deterministic test split.
- Do not claim uninterrupted 30-epoch training.
- Do not claim the Transformer is the strongest classifier.

## Baseline and Validity Audit

Same split/config as Transformer: early exit 0.4, downsample factor 10, seed 42.

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority class | 67.35% | 0.0000 | 0.5000 |
| Energy threshold | 35.81% | 0.4964 | 0.5233 |
| Transformer | 79.73% | 0.7447 | 0.9363 |
| XGBoost full summary | 99.34% | 0.9899 | 0.9998 |
| XGBoost first/last only | 99.33% | 0.9897 | 0.9997 |
| XGBoost initial row only | 98.42% | 0.9760 | 0.9987 |
| XGBoost initial row without context | 98.42% | 0.9760 | 0.9987 |

Confirmed:

- Model inputs are the 13 intended physics/context features only.
- Forbidden columns are not model inputs:
  `label`, `failure_type`, `min_target_rmag`, `mission_id`,
  `source_body`, `target_body`.
- Removing summary statistics does not reduce XGBoost performance.
- Removing context features (`mu_ratio`, `soi_ratio`, `dist_ratio`) from the
  initial-only XGBoost model does not reduce performance.

Scientific conclusion:

- The current random mission split is highly predictable from initial
  dynamical state.
- This is not explicit label leakage.
- It is likely calibrated synthetic-corridor separability.
- Random-split results alone are not enough for strong generalization claims.

## Paper-Safe Claims

Safe:

- The dataset and pipeline support multi-target synthetic trajectory generation
  through Neptune.
- The current feature pipeline avoids explicit target/outcome leakage columns.
- On a random mission split, early outcome prediction is highly accurate.
- XGBoost is the strongest current classifier on this random split.
- The Transformer is a sequential neural baseline with useful ROC-AUC but is
  not state-of-the-art for this dataset.

Unsafe unless further validated:

- Claiming Transformer superiority.
- Claiming generalization to unseen planets.
- Claiming real-world mission readiness.
- Claiming early-exit intelligence beyond calibrated initial-corridor
  separability.

## Next Required Experiment

Run a harder generalization audit before final paper claims:

1. Leave-one-planet-out evaluation:
   train on all targets except one, test on the held-out target.
2. Parameter-corridor holdout:
   train on some launch-parameter regions, test on unseen regions.

Acceptance:

- Report random-split results and grouped-split results separately.
- If grouped-split performance drops, state it clearly and frame the result as
  synthetic-corridor dependence.
- If grouped-split remains strong, the paper becomes much stronger.

## Leave-One-Target-Out Result

Completed for XGBoost summary and initial-no-context baselines.

Strong held-out transfer:

- Jupiter: summary F1 0.997, AUC 0.999
- Saturn: summary F1 1.000, AUC 1.000
- Uranus: summary F1 0.984, AUC 0.992

Weak or failed operational transfer:

- Mars: summary F1 0.000, AUC 0.436
- Mercury: summary F1 0.000, AUC 0.607
- Moon: summary F1 0.000, AUC 0.262
- Neptune: summary F1 0.000, AUC 1.000
- Venus: summary F1 0.000, AUC 0.324

Interpretation:

- Random-split results overstate deployment-level generalization.
- Some held-out targets have good ranking but bad probability calibration.
- Full unseen-target generalization is not solved.
- This is useful, not bad: it tells us exactly what claim reviewers could
  attack and how to strengthen the paper.

Updated next experiment:

- Prefer parameter-corridor holdout within each target over more zero-shot
  planet experiments. The operational claim is mission-family screening, not
  necessarily transfer to a completely unseen planet.

## Parameter-Corridor Holdout Result

Completed for `TOI_V` and `AOP` quintile bands within each target.

Mean summary-XGBoost performance:

- `TOI_V` holdout: Accuracy 97.97%, F1 0.798, AUC 0.986
- `AOP` holdout: Accuracy 90.10%, F1 0.684, AUC 0.978

Mean initial-no-context performance:

- `TOI_V` holdout: Accuracy 95.51%, F1 0.587, AUC 0.939
- `AOP` holdout: Accuracy 88.68%, F1 0.503, AUC 0.864

Interpretation:

- This supports in-family parameter interpolation better than zero-shot
  unseen-planet transfer.
- Summary features generalize better than initial-only features, so temporal
  trajectory information adds value for corridor holdout.
- Edge bins with very low success rates have weak F1 despite high accuracy/AUC.
- `AOP` bin 1 is a real failure case and must be disclosed.

Current strongest paper framing:

- OrbitGuard's synthetic mission-family screening is promising under
  random-split and parameter-corridor holdout.
- Full unseen-target generalization remains unsolved.
- XGBoost trajectory summaries are the strongest current model; Transformer is
  a neural sequential baseline, not the leading classifier.

## Reviewer Risk Register

See `docs/REVIEWER_RISK_REGISTER.md`.

See `docs/STATISTICAL_AUDIT_SUMMARY.md` for paper-ready tables generated from
the random-split, leave-one-target-out, and parameter-corridor JSON artifacts.

Highest-risk claims to avoid:

- Transformer superiority.
- Unseen-planet robustness.
- Flight-readiness or operational cancellation readiness.
- 15-minute interplanetary cadence.

Highest-value current claim:

- Within calibrated synthetic mission families, trajectory-summary features
  provide strong early ranking performance under unseen launch-parameter
  corridor holdout, while full unseen-target transfer remains mixed.

## Critique Policy

Do not optimize for making the numbers look good. Optimize for claims that can
survive reviewer scrutiny.

When a result is suspiciously strong, audit leakage and split design before
celebrating it.

When a model underperforms a simpler baseline, report that honestly and adjust
the contribution framing.
