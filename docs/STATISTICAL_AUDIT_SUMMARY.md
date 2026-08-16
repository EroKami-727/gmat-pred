# OrbitGuard Statistical Audit Summary

> **SUPERSEDED — generation G1 (2026-06-10 to 06-16).**
> Describes the XGBoost audit pass and the original multi-planet
> Transformer (79.73% accuracy, F1 0.745). The current models are the
> per-planet G3 rebuild (F1 0.9981) and none of the model numbers below
> describe them. The *split-design* findings — that random splits are too
> easy, and that unseen-target transfer fails — do carry over.
> See [`README.md`](README.md) for the generation map.

Generated from existing audit JSON artifacts. This file is for paper writing and reviewer response drafting.

Configuration: early exit = 40%, downsample factor = 10 interplanetary records. For interplanetary missions, one source record is 54,000 seconds = 15 hours.

## Random-Split Baselines

| Model | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| Majority | 67.35% | 0.000 | 0.500 |
| Energy threshold | 35.81% | 0.496 | 0.523 |
| XGBoost summary | 99.34% | 0.990 | 1.000 |
| XGBoost endpoints | 99.33% | 0.990 | 1.000 |
| XGBoost initial | 98.42% | 0.976 | 0.999 |
| XGBoost initial no context | 98.42% | 0.976 | 0.999 |

## Leave-One-Target-Out Audit

| Held-out target | Success rate | Summary F1 | Summary AUC | Initial-no-context F1 | Initial-no-context AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| jupiter | 34.87% | 0.997 | 0.999 | 0.997 | 0.998 |
| mars | 26.28% | 0.000 | 0.436 | 0.000 | 0.782 |
| mercury | 21.56% | 0.000 | 0.607 | 0.000 | 0.862 |
| moon | 35.53% | 0.000 | 0.262 | 0.000 | 0.496 |
| neptune | 35.00% | 0.000 | 1.000 | 0.000 | 1.000 |
| saturn | 35.01% | 1.000 | 1.000 | 1.000 | 1.000 |
| uranus | 33.89% | 0.984 | 0.992 | 0.984 | 0.992 |
| venus | 33.76% | 0.000 | 0.324 | 0.000 | 0.980 |

Aggregate across held-out targets:

| Feature mode | Accuracy mean +/- std | F1 mean +/- std | AUC mean +/- std | Worst F1 |
| --- | ---: | ---: | ---: | ---: |
| Summary | 80.82% +/- 16.22% | 0.373 +/- 0.514 | 0.702 +/- 0.331 | 0.000 |
| Initial no context | 80.82% +/- 16.21% | 0.373 +/- 0.514 | 0.889 +/- 0.178 | 0.000 |

## Parameter-Corridor Holdout Audit

| Variable | Bin | Success rate | Summary F1 | Summary AUC | Initial-no-context F1 | Initial-no-context AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TOI_V | 0 | 1.81% | 0.521 | 0.990 | 0.144 | 0.916 |
| TOI_V | 1 | 32.19% | 0.968 | 0.990 | 0.901 | 0.982 |
| TOI_V | 2 | 90.73% | 0.984 | 0.982 | 0.957 | 0.906 |
| TOI_V | 3 | 34.42% | 0.960 | 0.995 | 0.934 | 0.988 |
| TOI_V | 4 | 0.78% | 0.555 | 0.973 | 0.000 | 0.905 |
| AOP | 0 | 0.96% | 0.472 | 0.981 | 0.086 | 0.937 |
| AOP | 1 | 56.98% | 0.387 | 0.924 | 0.375 | 0.483 |
| AOP | 2 | 60.19% | 0.987 | 0.999 | 0.971 | 0.994 |
| AOP | 3 | 19.54% | 0.963 | 0.999 | 0.921 | 0.993 |
| AOP | 4 | 2.14% | 0.614 | 0.988 | 0.162 | 0.914 |

Aggregate across parameter bins:

| Feature mode | Accuracy mean +/- std | F1 mean +/- std | AUC mean +/- std | Worst F1 |
| --- | ---: | ---: | ---: | ---: |
| Summary | 94.04% +/- 13.39% | 0.741 +/- 0.251 | 0.982 +/- 0.022 | 0.387 |
| Initial no context | 92.09% +/- 13.39% | 0.545 +/- 0.424 | 0.902 +/- 0.152 | 0.000 |

## Paper Claim Guidance

Defensible:

- Random-split prediction is in-distribution and highly separable.
- XGBoost trajectory-summary baselines are stronger than the current Transformer checkpoint.
- Parameter-corridor holdout is the strongest current generalization evidence.
- Full unseen-target transfer remains mixed and should be reported as a limitation.

Weak grouped targets with summary F1 < 0.5:

- mars: F1=0.000, AUC=0.436
- mercury: F1=0.000, AUC=0.607
- moon: F1=0.000, AUC=0.262
- neptune: F1=0.000, AUC=1.000
- venus: F1=0.000, AUC=0.324

Weak parameter bins with summary F1 < 0.7:

- TOI_V bin 0: success=1.81%, F1=0.521, AUC=0.990
- TOI_V bin 4: success=0.78%, F1=0.555, AUC=0.973
- AOP bin 0: success=0.96%, F1=0.472, AUC=0.981
- AOP bin 1: success=56.98%, F1=0.387, AUC=0.924
- AOP bin 4: success=2.14%, F1=0.614, AUC=0.988
