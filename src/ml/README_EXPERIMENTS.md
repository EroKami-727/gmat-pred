# ML Experiment Scripts

Use these scripts for paper-grade validation. Generated reports are written
under `reports/` and are ignored by Git.

## Random Split Baselines

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.ml.baselines `
  --data data\merged_through_neptune_15min\missions.parquet `
  --exit-fracs 0.4 `
  --downsample-factor 10 `
  --seed 42 `
  --output reports\baselines\baseline_through_neptune_exit40_ds10_audit.json
```

## Leave-One-Target-Out Baselines

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.ml.grouped_baselines `
  --data data\merged_through_neptune_15min\missions.parquet `
  --early-exit 0.4 `
  --downsample-factor 10 `
  --seed 42 `
  --feature-modes summary initial_no_context `
  --output reports\baselines\leave_one_target_out_exit40_ds10.json
```

## Parameter-Corridor Holdout Baselines

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

Both grouped_baselines and parameter_holdout_baselines now report PR-AUC,
Brier score, ECE, and a confusion matrix per held-out target/bin (via
`src/ml/calibration_utils.py`), not just accuracy/F1/AUC.

## Multi-Seed Confidence Intervals

Random-split: see `src/ml/multi_seed.py` (Transformer) and
`src/ml/multi_seed_calibration.py` (calibration metrics on saved
multi-seed checkpoints, no retraining).

Grouped audits — the train/test partition for LOTO and corridor holdout
is deterministic (defined by target identity or quantile bin, not by
seed), so these only vary the XGBoost model's random_state:

```bash
python -m src.ml.multi_seed_grouped \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --seeds 0 1 2 3 4 \
  --output reports/baselines/leave_one_target_out_multiseed.json

python -m src.ml.multi_seed_parameter_holdout \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --params /media/Data/Coding/gmat-pred/data/merged_all_v2/mission_params.parquet \
  --seeds 0 1 2 3 4 \
  --output reports/baselines/parameter_holdout_multiseed.json
```

Formal three-way ablation (XGBoost-initial vs XGBoost-summary vs
Transformer-sequential) with genuine random-split multi-seed CIs for the
two cheap legs:

```bash
python -m src.ml.formal_ablation \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --seeds 0 1 2 3 4 \
  --transformer-metrics models/transformer_production/metrics_transformer_binary.json \
  --output reports/baselines/formal_ablation.json
```

## Calibration Evaluation

PR-AUC, Brier score, ECE, best-F1 threshold, isotonic recalibration, and
per-target breakdown for any trained Transformer/LSTM checkpoint:

```bash
python -m src.ml.calibration_eval \
  --model-path models/transformer_production/best_model_transformer_binary.pt \
  --scaler-path models/transformer_production/scaler_transformer_binary.pkl \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --arch transformer --early-exit 0.4 --downsample-factor 10 \
  --tmp-dir /path/to/disk/with/space \
  --output-dir reports/calibration
```

`--tmp-dir` matters for large datasets — the default temp directory is
often RAM-backed tmpfs, which can run out of "disk" space well before
the real disk does. `--batch-size` may also need lowering for datasets
with long max sequence lengths (Transformer attention memory scales
with sequence length squared).

## Error Analysis

For every weak held-out target/bin already flagged by the calibrated
grouped/holdout JSONs above (F1 below a threshold), reports success rate,
predicted-probability distribution, confusion matrix, and the input
features whose train/test distributions shifted the most:

```bash
python -m src.ml.error_analysis \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --params /media/Data/Coding/gmat-pred/data/merged_all_v2/mission_params.parquet \
  --grouped reports/baselines/leave_one_target_out_exit40_ds10.json \
  --parameter reports/baselines/parameter_holdout_exit40_ds10.json \
  --out docs/ERROR_ANALYSIS.md
```

## Domain Generalization Baseline

`src/ml/train.py` supports oversampling specific target bodies during
training via a `WeightedRandomSampler`, without using target identity as
a model input feature (val/test stay unweighted):

```bash
python -m src.ml.train \
  --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
  --task binary --model transformer \
  --early-exit 0.4 --downsample-factor 10 \
  --upweight-targets mars mercury moon venus --upweight-factor 2.0 \
  --output-dir models/transformer_balanced
```

Note: if all targets have equal mission counts (true for this project's
generated datasets), plain `--balance-targets` alone is a near no-op —
inverse-count weighting has nothing to balance. Use `--upweight-targets`
to explicitly oversample targets known to be empirically weak.

`--resume-from`/`--resume-best-val-loss`/`--epoch-offset` allow
warm-starting a continuation run from an existing checkpoint without
losing the best result already achieved, e.g. to extend training past
the originally requested epoch count once the loss curve shows it hasn't
plateaued yet.
