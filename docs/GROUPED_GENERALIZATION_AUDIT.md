# OrbitGuard Grouped Generalization Audit

## Purpose

Random mission splits are not enough for paper-grade claims because the current
synthetic generator creates highly separable initial-condition corridors. This
audit tests whether XGBoost baselines transfer to an unseen target body.

## Setup

- Dataset: `data/merged_through_neptune_15min/missions.parquet`
- Missions: 80,000
- Targets: Moon, Mars, Mercury, Venus, Jupiter, Saturn, Uranus, Neptune
- Split: leave one target body out
- Train: 70,000 missions from seven targets
- Test: 10,000 missions from the held-out target
- Early exit: 0.4
- Downsample factor: 10
- Seed: 42
- Feature modes:
  - `summary`: mean/std/min/max/first/last plus length
  - `initial_no_context`: first telemetry row only, excluding `mu_ratio`,
    `soi_ratio`, and `dist_ratio`

## Results

| Held out | Test success | Summary Acc | Summary F1 | Summary AUC | Initial-no-context Acc | Initial-no-context F1 | Initial-no-context AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jupiter | 0.349 | 99.81% | 0.997 | 0.999 | 99.79% | 0.997 | 0.998 |
| Mars | 0.263 | 73.72% | 0.000 | 0.436 | 73.72% | 0.000 | 0.782 |
| Mercury | 0.216 | 78.44% | 0.000 | 0.607 | 78.44% | 0.000 | 0.862 |
| Moon | 0.355 | 64.47% | 0.000 | 0.262 | 64.47% | 0.000 | 0.496 |
| Neptune | 0.350 | 65.00% | 0.000 | 1.000 | 65.00% | 0.000 | 1.000 |
| Saturn | 0.350 | 99.99% | 1.000 | 1.000 | 99.99% | 1.000 | 1.000 |
| Uranus | 0.339 | 98.89% | 0.984 | 0.992 | 98.89% | 0.984 | 0.992 |
| Venus | 0.338 | 66.24% | 0.000 | 0.324 | 66.24% | 0.000 | 0.980 |

## Interpretation

The grouped result is mixed and materially weaker than the random split.

Strong transfer:

- Jupiter
- Saturn
- Uranus

Weak or failed operational transfer:

- Mars
- Mercury
- Moon
- Neptune
- Venus

For Neptune and Venus, initial-no-context AUC is very high but F1 is zero at
the training-derived threshold. This means the model can rank positives above
negatives inside the held-out target, but its probability scale does not
transfer. It predicts all held-out examples below the success threshold.

For Moon and Mars, AUC is weak, which indicates poor ranking transfer, not only
threshold miscalibration.

## Paper Consequence

Do not claim robust target-held-out generalization from the current model.

Safe claim:

- Random-split early outcome prediction is highly accurate.
- Some outer-planet held-out targets transfer well.
- Several held-out targets expose calibration and generalization failures.

Unsafe claim:

- OrbitGuard generalizes across unseen target bodies without additional
  domain adaptation, calibration, or grouped training objectives.

## Next Research Direction

Strengthen the paper by adding one of:

1. Per-target calibration: calibrate probability thresholds using a small
   validation sample from each target.
2. Planet-family grouping: train/evaluate inner planets and outer planets
   separately.
3. Target-conditioned model: include target identity/context explicitly but
   evaluate with grouped splits and calibrated thresholds.
4. Parameter-corridor holdout: hold out launch-parameter regions within each
   target, which is more directly relevant than full target holdout if the
   paper claims in-distribution mission screening.

The most defensible next step is parameter-corridor holdout because the paper's
operational goal is screening future samples from calibrated mission families,
not necessarily zero-shot transfer to entirely unseen planets.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.ml.grouped_baselines `
  --data data\merged_through_neptune_15min\missions.parquet `
  --early-exit 0.4 `
  --downsample-factor 10 `
  --seed 42 `
  --feature-modes summary initial_no_context `
  --output reports\baselines\leave_one_target_out_exit40_ds10.json
```
