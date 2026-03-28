# OrbitGuard — Changelog

## 2026-03-28 — ML Fixes + Frontend + Ablation Study

### Bug Fixes

#### `src/ml/dataset.py` — PyArrow Filter Bug (Critical)
- **Problem:** `batch.filter(np.isin(...))` passed a raw NumPy boolean array to PyArrow's `.filter()`, which expects a `pyarrow.BooleanArray`. This caused incorrect or silently wrong train/val/test splits depending on PyArrow version.
- **Fix:** Changed to `batch.filter(pa.array(np.isin(...)))` with an explicit `import pyarrow as pa`.
- **Impact:** All downstream training was using corrupted data splits. This is now correct.

#### `src/ml/train.py` — Gradient Clipping
- **Problem:** No gradient clipping on the LSTM — susceptible to exploding gradients on long sequences.
- **Fix:** Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` in `train_one_epoch`.

### Improvements

#### `src/ml/train.py` — Learning Rate Scheduler
- Added `ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)`.
- LR halves when val_loss stops improving for 3 consecutive epochs.
- Current LR printed each epoch: `lr=X.XXe-XX`.

#### `src/ml/train.py` — Metrics History Saved to Disk
- After each training run, a JSON file is written to `{output_dir}/metrics_{model}_{task}.json`.
- Format: list of dicts with `epoch`, `train_loss`, `val_loss`, `val_acc`, `f1`, `auc`, `lr`, `elapsed_s`.
- Used by the Streamlit dashboard to render training curves without retraining.

### New Files

#### `src/ml/ablation.py` — Early-Exit Ablation Sweep
- Sweeps `early_exit_frac` over a configurable list (default: `[0.1, 0.2, 0.3, 0.4, 0.6, 1.0]`).
- Trains a full LSTM for each fraction, evaluates on test set, saves confusion matrix.
- Results saved incrementally to `reports/ablation/ablation_results.json` (dashboard reads this live).
- Addresses the **core research question**: how early in a trajectory can the model predict failure?

Usage:
```bash
python3 -m src.ml.ablation \
  --data data/merged/missions.parquet \
  --epochs 30 \
  --output-dir models/ablation \
  --fractions 0.1 0.2 0.3 0.4 0.6 1.0
```

#### `src/frontend/app.py` — Mission Control Dashboard (Full Rewrite)
- Previous state: 2-line stub (`st.title` + `st.write`).
- Rewritten as a 4-tab Streamlit research dashboard.
- Visual style: dark background `#0a0a0f`, monospace font, green `#00ff88` / blue `#0066ff` accents.

**Tab 1 — OVERVIEW**
- System status header (GPU, dataset presence, model loaded state).
- 4 metric tiles: Total Missions, Dataset Size, Success Rate, Best AUC.
- Model loaded / not loaded indicator.
- Single-mission prediction: upload a CSV → get FAIL/SUCCESS + confidence.

**Tab 2 — TRAINING**
- Configurable sidebar: model type, task, epochs, batch size, LR, hidden dim, layers, early-exit fraction, output dir.
- "Launch Training" button streams subprocess stdout in real time.
- Loads `metrics_{model}_{task}.json` and renders:
  - Plotly loss curve (train vs val)
  - Plotly AUC + F1 curve

**Tab 3 — ABLATION**
- Launch ablation sweep from UI with configurable fractions.
- Loads `reports/ablation/ablation_results.json` and renders:
  - Primary Plotly chart: AUC vs Early-Exit % (the key research figure)
  - Key finding callout: "Model achieves X% AUC using only Y% of trajectory"
  - Full results table with confusion matrix columns

**Tab 4 — DATASET EXPLORER**
- Feature documentation (all 13 physics-invariant features explained)
- Outcome class reference
- PNG grid of pre-generated EDA charts from `reports/eda/figures/`

Run with:
```bash
streamlit run src/frontend/app.py
```

---

## Previous History

| Commit | Summary |
|--------|---------|
| a64eec0 | Made visualizations for the database (EDA report) |
| f9775d5 | Remade database with generalization approach, better data |
| 8e60f8c | Updated scripts to 3-body sims + RK4, stored ~2000 sims |
| da054ac | Formalized research plan, cleaned up project overview |
| 34ceb53 | Ran small 200-sim data generation run |
