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

## Critique Policy

Do not optimize for making the numbers look good. Optimize for claims that can
survive reviewer scrutiny.

When a result is suspiciously strong, audit leakage and split design before
celebrating it.

When a model underperforms a simpler baseline, report that honestly and adjust
the contribution framing.
