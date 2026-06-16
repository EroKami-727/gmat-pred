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
