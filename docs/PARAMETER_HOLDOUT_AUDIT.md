# OrbitGuard Parameter-Corridor Holdout Audit

> **SUPERSEDED — generation G1 (2026-06-10 to 06-16).**
> Describes the XGBoost audit pass and the original multi-planet
> Transformer (79.73% accuracy, F1 0.745). The current models are the
> per-planet G3 rebuild (F1 0.9981) and none of the model numbers below
> describe them. The *split-design* findings — that random splits are too
> easy, and that unseen-target transfer fails — do carry over.
> See [`README.md`](README.md) for the generation map.

## Purpose

This audit tests the paper-relevant generalization claim: can a model trained
on calibrated mission families generalize to unseen launch-parameter bands
inside those same families?

This is more operationally relevant than full unseen-planet transfer because
OrbitGuard's near-term use case is screening future Monte Carlo samples from
known mission families.

## Setup

- Dataset: `data/merged_through_neptune_15min/missions.parquet`
- Parameters: `data/merged_through_neptune_15min/mission_params.parquet`
- Missions: 80,000
- Early exit: 0.4
- Downsample factor: 10
- Seed: 42
- Holdout design: per-target quintile bins
- Held-out variables: `TOI_V` and `AOP`
- Feature modes:
  - `summary`: mean/std/min/max/first/last plus length
  - `initial_no_context`: first telemetry row excluding context features

Each run holds out one quantile band within every target body, trains on the
remaining bands, and tests on the held-out band.

## Mean Results

| Holdout variable | Feature mode | Mean accuracy | Mean F1 | Mean ROC-AUC |
|---|---|---:|---:|---:|
| TOI_V | Summary | 97.97% | 0.798 | 0.986 |
| TOI_V | Initial no context | 95.51% | 0.587 | 0.939 |
| AOP | Summary | 90.10% | 0.684 | 0.978 |
| AOP | Initial no context | 88.68% | 0.503 | 0.864 |

## Per-Bin Results

| Variable | Bin | Test success | Summary Acc | Summary F1 | Summary AUC | Initial-no-context Acc | Initial-no-context F1 | Initial-no-context AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TOI_V | 0 | 0.018 | 98.33% | 0.521 | 0.990 | 97.63% | 0.144 | 0.916 |
| TOI_V | 1 | 0.322 | 97.88% | 0.968 | 0.990 | 93.77% | 0.901 | 0.982 |
| TOI_V | 2 | 0.907 | 97.11% | 0.984 | 0.982 | 92.05% | 0.957 | 0.906 |
| TOI_V | 3 | 0.344 | 97.14% | 0.960 | 0.995 | 95.19% | 0.934 | 0.988 |
| TOI_V | 4 | 0.008 | 99.37% | 0.555 | 0.973 | 98.89% | 0.000 | 0.905 |
| AOP | 0 | 0.010 | 99.12% | 0.472 | 0.981 | 98.54% | 0.086 | 0.937 |
| AOP | 1 | 0.570 | 55.98% | 0.387 | 0.924 | 54.45% | 0.375 | 0.483 |
| AOP | 2 | 0.602 | 98.37% | 0.987 | 0.999 | 96.45% | 0.971 | 0.994 |
| AOP | 3 | 0.195 | 98.51% | 0.963 | 0.999 | 96.83% | 0.921 | 0.993 |
| AOP | 4 | 0.021 | 98.54% | 0.614 | 0.988 | 97.11% | 0.162 | 0.914 |

## Interpretation

The parameter-corridor test is materially stronger than the random split and
more favorable than full leave-one-target-out.

Strong evidence:

- Summary XGBoost generalizes well across held-out `TOI_V` bands.
- Summary XGBoost has high ROC-AUC across held-out `AOP` bands.
- Initial-only no-context remains useful, but weaker than trajectory summaries.

Limitations:

- F1 is weak in sparse-success edge bins because few positives exist there.
- `AOP` bin 1 is a real failure case: high AUC for summary features, but poor
  classification accuracy/F1 and poor initial-only ranking.
- These results support in-family interpolation better than zero-shot
  cross-planet transfer.

## Paper Consequence

This is currently the strongest defensible generalization result.

Safe claim:

- Within calibrated mission families, trajectory-summary baselines retain
  strong ranking performance on unseen `TOI_V` and `AOP` parameter bands.

Required caveat:

- Operational thresholds and some corridors remain fragile, especially in
  sparse-success bins and `AOP` bin 1.

Unsafe claim:

- Universal robust generalization across all unseen parameter regimes.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.ml.parameter_holdout_baselines `
  --data data\merged_through_neptune_15min\missions.parquet `
  --params data\merged_through_neptune_15min\mission_params.parquet `
  --early-exit 0.4 `
  --downsample-factor 10 `
  --seed 42 `
  --bins 5 `
  --variables TOI_V AOP `
  --feature-modes summary initial_no_context `
  --output reports\baselines\parameter_holdout_exit40_ds10.json
```
