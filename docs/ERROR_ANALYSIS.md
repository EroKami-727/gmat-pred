# OrbitGuard Error Analysis — Weak Held-Out Cases

Generated from calibrated leave-one-target-out and parameter-corridor-holdout
audit artifacts. Each case below has summary-mode F1 < 0.7 at the default 0.5
decision threshold. For every case: success rate, predicted-probability
distribution, confusion matrix, and the 5 input features whose train/test
distributions shifted the most (standardized mean shift).

## Leave One Target Out — mars

- Test missions: 10000
- Success rate: 26.3%
- F1@0.5: 0.000  |  AUC: 0.509  |  PR-AUC: 0.253
- Brier score: 0.261  |  ECE: 0.257

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 7370 | 2 |
| Actual Success | 2628 | 0 |

**Predicted probability distribution:**

min=0.001  p25=0.002  median=0.003  p75=0.005  max=0.716

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| dist_ratio | 18.2392 | 0.5240 | -1.547 |
| rel_x | -1759076480.0000 | -39861644.0000 | +1.117 |
| earth_rmag | 1072513216.0000 | 41504148.0000 | -1.053 |
| norm_target_dist | 28.5283 | 74.2756 | +0.854 |
| rel_y | 336292416.0000 | 11098728.0000 | -0.815 |

## Leave One Target Out — mercury

- Test missions: 10000
- Success rate: 21.6%
- F1@0.5: 0.000  |  AUC: 0.496  |  PR-AUC: 0.202
- Brier score: 0.213  |  ECE: 0.210

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 7845 | 0 |
| Actual Success | 2155 | 0 |

**Predicted probability distribution:**

min=0.002  p25=0.004  median=0.005  p75=0.006  max=0.015

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| norm_target_dist | 25.7846 | 803.8253 | +42.762 |
| dist_ratio | 18.1288 | 0.6130 | -1.523 |
| rel_x | -1748193664.0000 | -87944656.0000 | +1.078 |
| earth_rmag | 1066196928.0000 | 19757898.0000 | -1.069 |
| rel_y | 334375872.0000 | -14015551.0000 | -0.874 |

## Leave One Target Out — moon

- Test missions: 10000
- Success rate: 35.3%
- F1@0.5: 0.000  |  AUC: 0.296  |  PR-AUC: 0.278
- Brier score: 0.346  |  ECE: 0.339

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 6468 | 0 |
| Actual Success | 3532 | 0 |

**Predicted probability distribution:**

min=0.005  p25=0.007  median=0.012  p75=0.015  max=0.055

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| soi_ratio | 0.0246 | 0.1722 | +10.641 |
| dist_ratio | 21.9242 | 0.0026 | -2.502 |
| vel_mag | 9.0788 | 2.2005 | -1.656 |
| rel_x | -2114362496.0000 | -133997.8281 | +1.461 |
| fpa_deg | 87.9067 | 71.1246 | -1.459 |

## Leave One Target Out — uranus

- Test missions: 10000
- Success rate: 33.9%
- F1@0.5: 0.000  |  AUC: 0.992  |  PR-AUC: 0.968
- Brier score: 0.126  |  ECE: 0.203

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 6611 | 0 |
| Actual Success | 3389 | 0 |

**Predicted probability distribution:**

min=0.000  p25=0.002  median=0.005  p75=0.395  max=0.395

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| ecc | 5.8930 | 72102.9062 | +1299.763 |
| spec_energy | 19.2307 | 112.2730 | +5.065 |
| vel_mag | 5.7265 | 14.9251 | +3.366 |
| radial_vel | -3.6232 | -6.8000 | -1.384 |
| earth_rmag | 828947776.0000 | 1830737408.0000 | +1.206 |

## Leave One Target Out — venus

- Test missions: 10000
- Success rate: 33.8%
- F1@0.5: 0.000  |  AUC: 0.856  |  PR-AUC: 0.734
- Brier score: 0.335  |  ECE: 0.335

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 6624 | 0 |
| Actual Success | 3376 | 0 |

**Predicted probability distribution:**

min=0.000  p25=0.002  median=0.003  p75=0.003  max=0.017

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| dist_ratio | 18.1576 | 0.2770 | -1.556 |
| radial_vel | -4.4062 | 3.4279 | +1.250 |
| rel_x | -1750993280.0000 | -44898348.0000 | +1.108 |
| earth_rmag | 1067853312.0000 | 11599064.0000 | -1.079 |
| norm_target_dist | 28.7335 | 75.8763 | +0.878 |

## Parameter Corridor Holdout — TOI_V bin 4/4

- Test missions: 16000
- Success rate: 0.6%
- F1@0.5: 0.675  |  AUC: 0.978  |  PR-AUC: 0.744
- Brier score: 0.003  |  ECE: 0.002

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 15894 | 7 |
| Actual Success | 45 | 54 |

**Predicted probability distribution:**

min=0.000  p25=0.000  median=0.000  p75=0.001  max=0.999

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| ecc | 20473.9473 | 1992.6586 | -0.159 |
| fpa_deg | 84.4986 | 86.7288 | +0.142 |
| spec_energy | 40.4999 | 42.3152 | +0.042 |
| vel_mag | 7.8367 | 7.9793 | +0.031 |
| rel_y | 331614784.0000 | 338204864.0000 | +0.016 |

## Parameter Corridor Holdout — AOP bin 1/4

- Test missions: 24112
- Success rate: 57.0%
- F1@0.5: 0.388  |  AUC: 0.885  |  PR-AUC: 0.912
- Brier score: 0.392  |  ECE: 0.410

**Confusion matrix (@0.5):**

| | Pred Fail | Pred Success |
| --- | ---: | ---: |
| Actual Fail | 10269 | 91 |
| Actual Success | 10423 | 3329 |

**Predicted probability distribution:**

min=0.000  p25=0.000  median=0.003  p75=0.154  max=1.000

**Top feature shifts (train -> test, standardized):**

| Feature | Train mean | Test mean | Standardized shift |
| --- | ---: | ---: | ---: |
| ecc | 1230.5297 | 39403.1328 | +2.031 |
| soi_ratio | 0.0604 | 0.0364 | -0.376 |
| dist_ratio | 16.2073 | 20.7498 | +0.374 |
| rel_x | -1557204608.0000 | -2009539456.0000 | -0.291 |
| earth_rmag | 958558912.0000 | 1212362240.0000 | +0.255 |
