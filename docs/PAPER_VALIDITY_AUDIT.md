# OrbitGuard Paper-Grade Validity Audit

## Configuration

- Dataset: `data/merged_through_neptune_15min/missions.parquet`
- Missions: 80,000
- Success/failure: 25,590 / 54,410
- Interplanetary telemetry cadence: 54,000 seconds = 15 hours
- Early-exit fraction: 0.4
- Downsample factor: 10
- Random seed: 42
- Split: 70% train, 15% validation, 15% test by mission ID

The directory name is historical. The interplanetary data is not sampled at
15-minute cadence.

## Leakage Audit

Both the Transformer and baselines consume exactly these 13 features:

`rel_x`, `rel_y`, `rel_z`, `spec_energy`, `fpa_deg`,
`norm_target_dist`, `radial_vel`, `vel_mag`, `earth_rmag`, `ecc`,
`mu_ratio`, `soi_ratio`, and `dist_ratio`.

The following columns are not model inputs: `label`, `failure_type`,
`min_target_rmag`, `mission_id`, `source_body`, and `target_body`.

There is no explicit outcome-column leakage. However, `mu_ratio`, `soi_ratio`,
and `dist_ratio` encode the mission regime, and initial dynamical state
features encode the narrow calibrated success corridors.

## Results

| Model | Input available | Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|
| Majority class | Training class frequency | 67.35% | 0.0000 | 0.5000 |
| Energy threshold | Last observed specific energy | 35.81% | 0.4964 | 0.5233 |
| Transformer | First 40% sequential telemetry | 79.73% | 0.7447 | 0.9363 |
| XGBoost full | Mean/std/min/max/first/last + length | 99.34% | 0.9899 | 0.9998 |
| XGBoost endpoints | First and last values only | 99.33% | 0.9898 | 0.9998 |
| XGBoost initial | First telemetry row only | 98.42% | 0.9760 | 0.9987 |
| XGBoost initial, no context | First row excluding all three context features | 98.42% | 0.9760 | 0.9987 |

The restricted baselines demonstrate that XGBoost performance is not caused
by min/max/std aggregation over the 40% observation window. A single initial
telemetry row is already strongly predictive.

Removing `mu_ratio`, `soi_ratio`, and `dist_ratio` from the initial-only model
does not reduce accuracy. The result is therefore not primarily caused by
indirect target identity. Initial orbital state and calibrated targeting
corridors remain sufficient for near-perfect random-split classification.

## Feature Interpretation

For the full-summary model, the strongest features include `rel_z_std`,
`fpa_deg_last`, `fpa_deg_std`, `rel_y_min`, and `rel_z_mean`.

For the endpoint model, `rel_z_first` dominates, followed by
`fpa_deg_last`, `spec_energy_first`, and `radial_vel_last`.

For the initial-only model, the leading features are:

1. `rel_z_first`
2. `vel_mag_first`
3. `mu_ratio_first`
4. `ecc_first`
5. `norm_target_dist_first`

Without context features, the leading features remain `rel_z_first`,
`vel_mag_first`, `ecc_first`, `norm_target_dist_first`, and `rel_y_first`.

These are physically meaningful initial-condition and mission-regime signals.
They are also evidence that the calibrated synthetic generator creates highly
separable success and failure corridors.

## Defensible Paper Claims

- The dataset contains no explicit target columns in model inputs.
- Early mission outcomes are highly predictable on a random mission split.
- A tabular XGBoost model is substantially stronger than the current
  Transformer on this split.
- The Transformer is a sequential neural baseline, not the top-performing
  classifier.

Do not claim that the current random split demonstrates generalization to
unseen planets, unseen targeting corridors, or real mission telemetry.

## Required Next Experiment

Before final paper claims, evaluate at least one harder grouped split:

1. Leave-one-planet-out evaluation, training on all but one target and testing
   on the held-out target.
2. Parameter-corridor holdout, where contiguous launch-parameter regions are
   absent from training.

Report both random-split and grouped-split results. A large grouped-split
performance drop should be interpreted as synthetic-corridor dependence, not
hidden or omitted.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.ml.baselines `
  --data data\merged_through_neptune_15min\missions.parquet `
  --exit-fracs 0.4 `
  --downsample-factor 10 `
  --seed 42 `
  --output reports\baselines\baseline_through_neptune_exit40_ds10_audit.json
```
