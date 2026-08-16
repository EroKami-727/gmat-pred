# OrbitGuard Statistical Audit Summary

> **SUPERSEDED — generation G2 (2026-06-21 to 06-23).**
> Describes the local 80K reproduction and the multi-planet Transformer
> at 87.67% accuracy / F1 0.838. The current models are the per-planet G3
> rebuild (F1 0.9981), trained differently and evaluated on a different
> split. Retained for provenance and for the analyses that still stand
> (feature-shift error analysis, target-upweighting experiment).
> See [`README.md`](README.md) for the generation map.

Generated from existing audit JSON artifacts. This file is for paper writing and reviewer response drafting.

Configuration: early exit = 40%, downsample factor = 10 interplanetary records. For interplanetary missions, one source record is 54,000 seconds = 15 hours.

## Random-Split Baselines

| Model | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| Majority | 67.74% | 0.000 | 0.500 |
| Energy threshold | 44.09% | 0.536 | 0.535 |
| XGBoost summary | 99.44% | 0.991 | 1.000 |
| XGBoost endpoints | 99.14% | 0.987 | 0.999 |
| XGBoost initial | 98.29% | 0.974 | 0.998 |
| XGBoost initial no context | 98.33% | 0.974 | 0.998 |

## Leave-One-Target-Out Audit

| Held-out target | Success rate | Summary F1 | Summary AUC | Summary PR-AUC | Summary Brier | Summary ECE | Initial-no-context F1 | Initial-no-context AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jupiter | 34.87% | 0.997 | 0.998 | 0.995 | 0.002 | 0.002 | 0.997 | 0.998 |
| mars | 26.28% | 0.000 | 0.509 | 0.253 | 0.261 | 0.257 | 0.000 | 0.817 |
| mercury | 21.55% | 0.000 | 0.496 | 0.202 | 0.213 | 0.210 | 0.000 | 0.761 |
| moon | 35.32% | 0.000 | 0.296 | 0.278 | 0.346 | 0.339 | 0.000 | 0.566 |
| neptune | 35.00% | 1.000 | 1.000 | 1.000 | 0.000 | 0.003 | 1.000 | 1.000 |
| saturn | 35.01% | 1.000 | 1.000 | 1.000 | 0.008 | 0.055 | 1.000 | 1.000 |
| uranus | 33.89% | 0.000 | 0.992 | 0.968 | 0.126 | 0.203 | 0.984 | 0.992 |
| venus | 33.76% | 0.000 | 0.856 | 0.734 | 0.335 | 0.335 | 0.000 | 0.979 |

Aggregate across held-out targets:

| Feature mode | Accuracy mean +/- std | F1 mean +/- std | AUC mean +/- std | Worst F1 |
| --- | ---: | ---: | ---: | ---: |
| Summary | 81.12% +/- 16.22% | 0.375 +/- 0.517 | 0.768 +/- 0.288 | 0.000 |
| Initial no context | 85.22% +/- 16.02% | 0.498 +/- 0.532 | 0.889 +/- 0.161 | 0.000 |

Aggregate calibration (summary mode only):

| Metric | Mean +/- std | Worst |
| --- | ---: | ---: |
| PR-AUC | 0.679 +/- 0.371 | 0.202 |
| Brier score | 0.161 +/- 0.148 | 0.346 |
| ECE | 0.175 +/- 0.139 | 0.339 |

## Parameter-Corridor Holdout Audit

| Variable | Bin | Success rate | Summary F1 | Summary AUC | Summary PR-AUC | Summary Brier | Summary ECE | Initial-no-context F1 | Initial-no-context AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TOI_V | 0 | 1.69% | 0.904 | 0.998 | 0.963 | 0.003 | 0.002 | 0.070 | 0.903 |
| TOI_V | 1 | 32.24% | 0.975 | 0.990 | 0.945 | 0.015 | 0.013 | 0.895 | 0.981 |
| TOI_V | 2 | 90.77% | 0.984 | 0.992 | 0.999 | 0.024 | 0.019 | 0.960 | 0.912 |
| TOI_V | 3 | 34.48% | 0.967 | 0.995 | 0.984 | 0.020 | 0.026 | 0.930 | 0.985 |
| TOI_V | 4 | 0.62% | 0.675 | 0.978 | 0.744 | 0.003 | 0.002 | 0.000 | 0.884 |
| AOP | 0 | 0.92% | 0.838 | 0.987 | 0.856 | 0.003 | 0.001 | 0.055 | 0.930 |
| AOP | 1 | 57.03% | 0.388 | 0.885 | 0.912 | 0.392 | 0.410 | 0.374 | 0.440 |
| AOP | 2 | 59.89% | 0.990 | 1.000 | 1.000 | 0.008 | 0.007 | 0.970 | 0.994 |
| AOP | 3 | 19.60% | 0.972 | 0.998 | 0.991 | 0.008 | 0.007 | 0.919 | 0.994 |
| AOP | 4 | 2.15% | 0.881 | 0.994 | 0.927 | 0.004 | 0.004 | 0.087 | 0.931 |

Aggregate across parameter bins:

| Feature mode | Accuracy mean +/- std | F1 mean +/- std | AUC mean +/- std | Worst F1 |
| --- | ---: | ---: | ---: | ---: |
| Summary | 94.58% +/- 13.45% | 0.857 +/- 0.191 | 0.982 +/- 0.034 | 0.388 |
| Initial no context | 92.02% +/- 13.44% | 0.526 +/- 0.442 | 0.895 +/- 0.165 | 0.000 |

Aggregate calibration (summary mode only):

| Metric | Mean +/- std | Worst |
| --- | ---: | ---: |
| PR-AUC | 0.932 +/- 0.080 | 0.744 |
| Brier score | 0.048 +/- 0.121 | 0.392 |
| ECE | 0.049 +/- 0.127 | 0.410 |

## Paper Claim Guidance

Defensible:

- Random-split prediction is in-distribution and highly separable.
- XGBoost trajectory-summary baselines are stronger than the current Transformer checkpoint.
- Parameter-corridor holdout is the strongest current generalization evidence.
- Full unseen-target transfer remains mixed and should be reported as a limitation.

Weak grouped targets with summary F1 < 0.5:

- mars: F1=0.000, AUC=0.509
- mercury: F1=0.000, AUC=0.496
- moon: F1=0.000, AUC=0.296
- uranus: F1=0.000, AUC=0.992
- venus: F1=0.000, AUC=0.856

Weak parameter bins with summary F1 < 0.7:

- TOI_V bin 4: success=0.62%, F1=0.675, AUC=0.978
- AOP bin 1: success=57.03%, F1=0.388, AUC=0.885

Ranking-works-but-threshold-fails cases (AUC >= 0.80 but F1 collapses at 0.5):
These are calibration problems, not generalization failures — Platt/isotonic
recalibration or threshold tuning should fix them without retraining.

- uranus: AUC=0.992, F1@0.5=0.000, ECE=0.203
- venus: AUC=0.856, F1@0.5=0.000, ECE=0.335
- TOI_V bin 4: AUC=0.978, F1@0.5=0.675, ECE=0.002
- AOP bin 1: AUC=0.885, F1@0.5=0.388, ECE=0.410
