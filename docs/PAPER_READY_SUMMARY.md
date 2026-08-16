# OrbitGuard — Paper-Ready Summary (Local Reproduction, 2026-06-21)

> **SUPERSEDED — generation G2 (2026-06-21 to 06-23).**
> Describes the local 80K reproduction and the multi-planet Transformer
> at 87.67% accuracy / F1 0.838. The current models are the per-planet G3
> rebuild (F1 0.9981), trained differently and evaluated on a different
> split. Retained for provenance and for the analyses that still stand
> (feature-shift error analysis, target-upweighting experiment).
> See [`README.md`](README.md) for the generation map.

Consolidates every artifact produced during the local 80K-mission
reproduction pass. All numbers below are from this machine's independently
generated dataset (`/media/Data/Coding/gmat-pred/data/merged_all_v2/`),
not copied from the teammate's prior run — see `docs/RESEARCH_LEDGER.md`
for the full narrative and `docs/AI_CONTEXT.md` for pipeline details.

## 1. Dataset

80,000 missions across 8 source/target pairs (Mercury, Venus, Mars,
Jupiter, Saturn, Uranus, Neptune, Moon). 32.0% success rate — matches the
teammate's previously reported 0.320 closely, despite being an independent
generation run (different machine, same seed/methodology).

## 2. Random-split baselines

Early exit 0.4, downsample 10, seed 42. Full table:
`docs/STATISTICAL_AUDIT_SUMMARY_LOCAL.md`.

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority | 67.74% | 0.000 | 0.500 |
| Energy threshold | 44.09% | 0.536 | 0.535 |
| XGBoost summary | 99.44% | 0.991 | 1.000 |
| XGBoost initial | 98.29% | 0.974 | 0.998 |

**5-seed confidence intervals** (`src/ml/formal_ablation.py`, genuinely
seed-dependent splits unlike the grouped audits below):

| Model | F1 | AUC | ECE |
|---|---|---|---|
| XGBoost initial | 0.974 ± 0.001 | 0.998 ± 0.000 | 0.012 ± 0.000 |
| XGBoost summary | 0.992 ± 0.001 | 1.000 ± 0.000 | 0.003 ± 0.000 |
| Transformer-sequential | single run — see §5 | | |

## 3. Leave-one-target-out (LOTO)

`reports/baselines/leave_one_target_out_exit40_ds10_calibrated.json`,
multi-seed confirmation in `..._multiseed.json`.

| Held-out | Success rate | F1@0.5 | AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Jupiter | 34.9% | 0.997 | 0.998 | 0.995 | 0.002 | 0.002 |
| Saturn | 35.0% | 1.000 | 1.000 | 1.000 | 0.008 | 0.055 |
| Neptune | 35.0% | 1.000 | 1.000 | 1.000 | 0.000 | 0.003 |
| **Uranus** | 33.9% | **0.000** | 0.992 | 0.968 | 0.126 | 0.203 |
| **Venus** | 33.8% | **0.000** | 0.856 | 0.734 | 0.335 | 0.335 |
| Mars | 26.3% | 0.000 | 0.509 | 0.253 | 0.261 | 0.257 |
| Mercury | 21.6% | 0.000 | 0.496 | 0.202 | 0.213 | 0.210 |
| Moon | 35.3% | 0.000 | 0.296 | 0.278 | 0.346 | 0.339 |

Stable across 5 seeds (std=0.000 everywhere) — the splits are deterministic
by target identity, so this rules out training-noise as an explanation.

**Two distinct failure modes, not one:**
- **Uranus & Venus** (bold): high AUC, F1 collapses only because the 0.5
  threshold is wrong for these targets' probability scale. Calibration
  problem, fixable without retraining (isotonic/Platt scaling).
- **Mars, Mercury, Moon**: AUC near or below 0.5 — genuine generalization
  failure. `docs/ERROR_ANALYSIS.md` shows these targets occupy a
  categorically different physical regime (`dist_ratio`, `earth_rmag`
  shift by 1-43 standard deviations between train and test), not just a
  harder parameter corridor.

## 4. Parameter-corridor holdout (target-family split)

`reports/baselines/parameter_holdout_exit40_ds10_calibrated.json`,
multi-seed confirmation in `..._multiseed.json`. This already implements
the "train on inner corridor, test on unseen corridor within same target"
design from the original upgrade notes — no separate script was needed.

| Variable | Bin | Success rate | F1@0.5 | AUC | ECE |
|---|---:|---:|---:|---:|---:|
| TOI_V | 0 | 1.7% | 0.904 | 0.998 | 0.002 |
| TOI_V | 4 | 0.6% | 0.675 | 0.978 | 0.002 |
| AOP | 0 | 0.9% | 0.838 | 0.987 | 0.001 |
| **AOP** | **1** | 57.0% | **0.388** | 0.885 | **0.410** |
| AOP | 4 | 2.2% | 0.881 | 0.994 | 0.004 |

AOP bin 1 has the worst calibration of any case in the entire audit
(ECE=0.410) — sits near the 50/50 success boundary, where both ranking and
calibration genuinely struggle. Edge bins (TOI_V 0/4, AOP 0/4) have very
low success rates but the model still ranks/calibrates them well — PR-AUC
is the honest metric here, not raw accuracy.

## 5. Multi-planet Transformer (sequential)

`models/transformer_multiplanet/` — single training run, early exit 0.4,
downsample 10, seed 42, 50 total epochs (30 + a warm-started continuation
of 20 more, since the loss curve had not plateaued at epoch 30 —
`--resume-from`/`--resume-best-val-loss`/`--epoch-offset` in `src/ml/train.py`).

| Epochs | Test Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| 30 | 87.14% | 0.826 | 0.970 |
| **50 (final)** | **87.67%** | **0.838** | **0.984** |

- Best epoch by val_loss was epoch 50 itself — AUC was still climbing in
  the final few epochs (0.975 → 0.981 → 0.985), so further training beyond
  50 epochs may still help. Not pursued further without explicit direction,
  given each 20-epoch block costs ~3 hours on available hardware.
- Notably stronger than the teammate's prior 80K-dataset result (79.73%
  accuracy, F1 0.745, AUC 0.936) on an independently generated dataset.
- Still well below XGBoost-summary (F1=0.992) — consistent with the
  established finding that this task is highly separable from initial/
  summary trajectory statistics, and the Transformer is a baseline, not
  the leading classifier.
- **Not retrained across 5 seeds** — a single run takes ~7.5 hours total on
  available hardware (30+20 epochs); 5 seeds would be a multi-day
  commitment. Reported as a point estimate, explicitly flagged as such in
  `formal_ablation.json`.

## 5b. Calibration of the multi-planet Transformer

`src/ml/calibration_eval.py` against the final (epoch-50) checkpoint.
Note: `calibration_eval.py`/`multi_seed_calibration.py` needed two fixes for
this dataset scale — `tempfile.TemporaryDirectory()` defaults to `/tmp`,
which is RAM-backed tmpfs (12 GB) on this machine and overflowed on the
71 GB dataset's ~10 GB val/test splits (`--tmp-dir` flag added to redirect
to real disk), and the default `--batch-size 128` caused a CUDA OOM at
this dataset's max sequence length of 859 (reduced to 32, matching what
training itself used).

| Metric | @0.5 | @best-threshold (0.557) | Isotonic-calibrated |
|---|---:|---:|---:|
| Accuracy | 87.68% | 94.60% | 94.54% |
| F1 | 0.838 | 0.921 | 0.919 |
| ROC-AUC | 0.984 | 0.984 | 0.986 |
| PR-AUC | 0.966 | 0.966 | 0.961 |
| Brier score | 0.064 | 0.064 | 0.039 |
| ECE | 0.052 | 0.052 | **0.0045** |

Same pattern as the XGBoost LOTO audit: the default 0.5 threshold leaves
real performance on the table (F1 0.838→0.921 just from threshold tuning),
and isotonic calibration cuts ECE by ~11.5x.

Per-target breakdown (random-split test set, all targets present — not a
LOTO setup):

| Target | N | Success% | PR-AUC | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Jupiter | 1518 | 36.0% | 1.000 | 0.995 | 0.003 | 0.003 |
| Saturn | 1523 | 35.7% | 1.000 | 0.999 | 0.001 | 0.001 |
| Neptune | 1462 | 35.0% | 0.998 | 0.999 | 0.001 | 0.001 |
| Uranus | 1485 | 35.0% | 0.929 | 0.980 | 0.014 | 0.011 |
| Mars | 1474 | 26.8% | 0.982 | 0.911 | 0.033 | 0.023 |
| Venus | 1470 | 33.1% | 0.838 | 0.903 | 0.071 | 0.094 |
| **Moon** | 1493 | 34.2% | **0.851** | 0.853 | 0.270 | 0.215 |
| **Mercury** | 1575 | 22.8% | **0.547** | 0.704 | 0.120 | 0.081 |

Even within a random split (every target represented in training), Mercury
and Moon are the hardest targets for the Transformer — consistent with the
XGBoost LOTO finding that these targets are structurally harder, not just
harder to transfer to zero-shot.

## 6. Domain generalization baseline

`--balance-targets`/`--upweight-targets`/`--upweight-factor` added to
`src/ml/train.py`/`src/ml/dataset.py` (`WeightedRandomSampler` over
target_body; val/test stay unweighted so metrics remain honest).

**Design correction made during this run:** the first implementation
weighted purely by inverse target *count*. Since this dataset has exactly
10,000 missions per target (8 targets, perfectly balanced by construction),
that weighting was a near no-op — verified by a standalone sampler test
showing <1% deviation from uniform when all base counts are equal. Fixed
by adding `target_weight_overrides`/`--upweight-targets`, which explicitly
multiplies the sampling weight for named targets. Verified with a
synthetic test (1000 missions/target, 4 targets at 2x override) that the
override targets are sampled almost exactly 2x as often as the rest
before launching the real run — caught and fixed after ~3 minutes into
the first attempt, not after the full ~4.5 hour run.

Trained with `--upweight-targets mars mercury moon venus --upweight-factor
2.0` for 30 epochs (`models/transformer_balanced/`), matched against the
unbalanced model's **own 30-epoch checkpoint** (not its final 50-epoch
result — that comparison would have been unfair, since only the unbalanced
model was extended).

**Aggregate (both @ 30 epochs):**

| Metric | Unbalanced | Balanced (2x upweight) |
|---|---:|---:|
| Accuracy | 87.14% | 80.00% |
| F1 | 0.826 | 0.756 |
| ROC-AUC | 0.970 | 0.950 |
| PR-AUC | 0.940 | 0.918 |
| Brier | 0.067 | 0.093 |
| ECE | 0.045 | 0.074 |

Aggregate metrics all got worse. But the aggregate hides a real, mixed
per-target story:

| Target | PR-AUC (unbal → bal) | F1 (unbal → bal) | Verdict |
|---|---|---|---|
| **Moon** | 0.294 → **0.654** | 0.577 → **0.826** | Large improvement |
| Mars | 0.982 → 0.985 | 0.899 → 0.923 | Small improvement |
| Jupiter | 0.907 → 0.975 | 0.965 → 0.976 | Improved (not even upweighted) |
| Mercury | 0.503 → 0.544 | 0.531 → 0.500 | Mixed — better ranking, worse @0.5 |
| Saturn | 0.995 → 0.998 | 0.997 → 0.997 | Unchanged |
| Neptune | 0.998 → 0.998 | 0.998 → 0.999 | Unchanged |
| Uranus | 0.990 → 0.976 | 0.981 → 0.977 | Tiny regression |
| **Venus** | 0.773 → **0.477** | 0.902 → **0.000** | Collapsed |

**Conclusion: oversampling weak targets is not a silver bullet.** It
produced a large, genuine improvement on Moon and a real one on Mars, with
Jupiter improving as a side effect — but caused Venus to collapse entirely
(F1 0.902 → 0.000) and dragged every aggregate metric down. This is a
defensible, honest finding for the paper: target-balanced sampling helps
some weak targets and actively harms others, and the net effect at the
aggregate level is negative with this approach (uniform 2x upweight, 30
epochs, no per-target tuning). A follow-up worth flagging but not pursued
here: per-target upweight factors tuned individually, rather than one
fixed 2x for all four targets, might avoid the Venus regression while
keeping the Moon/Mars gains — but that is future work, not a result
already in hand.

## 7. Error analysis

Full detail in `docs/ERROR_ANALYSIS.md`. Headline finding: every weak LOTO
target shows large standardized feature shift in distance/scale features
(`dist_ratio`, `earth_rmag`, `rel_x`, `norm_target_dist`), and confusion
matrices show near-total collapse to the majority class (e.g. Mercury: 0
predicted successes out of 2,155 actual successes). This is a regime-shift
explanation, not a vague "generalization is hard" statement.

## What changed in the codebase this session

- `src/ml/calibration_utils.py` — shared ECE/PR-AUC/Brier/confusion-matrix
  helpers, now used by `grouped_baselines.py`, `parameter_holdout_baselines.py`,
  `calibration_eval.py`, `multi_seed_calibration.py`.
- `src/ml/multi_seed_grouped.py`, `src/ml/multi_seed_parameter_holdout.py` —
  multi-seed wrappers for the grouped audits.
- `src/ml/formal_ablation.py` — 5-seed XGBoost CIs + single-run Transformer
  reference for the random-split three-way comparison.
- `src/ml/error_analysis.py` — feature-shift + confusion-matrix tables for
  every weak held-out case.
- `src/ml/dataset.py`, `src/ml/train.py` — added `track_target_body`,
  `balance_targets`/`--balance-targets` (domain generalization), and
  `--resume-from`/`--resume-best-val-loss`/`--epoch-offset` (checkpoint
  continuation).
- `experiments/numba_jit/` — validated 26-37x speedup fast path used to
  generate the Neptune dataset (see `docs/NUMBA_JIT_PROPAGATOR.md`).
