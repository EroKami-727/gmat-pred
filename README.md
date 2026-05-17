# OrbitGuard — NASA GMAT Early Exit

Predicts failing spacecraft trajectories early. Cancels doomed Monte Carlo simulations before they waste GPU compute. Transformer watches telemetry stream in realtime and outputs Go/No-Go per mission.

**Test results at 40% trajectory seen:** AUC 0.997 · F1 0.948 · Acc 96.5%

---

## Quick start

```bash
# Activate venv
source /home/haise/Coding/venvs/gmat-pred/bin/activate

# Start API backend (from project root)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend
cd frontend && npm run dev
```

---

## Train a model

```bash
# Transformer (recommended) — 40% early exit
python3 -m src.ml.train \
  --data data/merged/missions.parquet \
  --model transformer \
  --epochs 30 \
  --early-exit 0.4 \
  --seed 42 \
  --output-dir models/transformer_production

# Full ablation sweep (feeds Ablation panel with real numbers)
python3 -m src.ml.ablation \
  --data data/merged/missions.parquet \
  --model transformer \
  --epochs 30
```

---

## Paper experiments

```bash
# 1. Baselines — XGBoost, energy threshold heuristic, majority class
python3 -m src.ml.baselines \
  --data data/merged/missions.parquet \
  --exit-fracs 0.1 0.2 0.3 0.4 0.6 1.0 \
  --output reports/baselines/baseline_results.json

# 2. Multi-seed robustness — 5 seeds, reports mean ± std
python3 -m src.ml.multi_seed \
  --data data/merged/missions.parquet \
  --early-exit 0.4 \
  --epochs 30 \
  --output-dir reports/multi_seed

# 3. Architecture ablation — CLS token, pos encoding, context features, LSTM
python3 -m src.ml.arch_ablation \
  --data data/merged/missions.parquet \
  --early-exit 0.4 \
  --epochs 30 \
  --output-dir reports/arch_ablation

# 4. Generate publication tables (Markdown + LaTeX) from all experiment outputs
python3 -m src.ml.results_table \
  --ablation   reports/ablation/ablation_results.json \
  --baselines  reports/baselines/baseline_results.json \
  --arch       reports/arch_ablation/arch_ablation_results.json \
  --multi-seed reports/multi_seed/summary.json \
  --output-dir reports/tables
```

---

## Generate dataset

```bash
# Run 5000 missions
python3 -m src.data_collection.build_database \
  --num-missions 5000 --output-dir data/production

# Merge two runs into master
python3 -m src.data_collection.merge_datasets \
  --base data/production --new data/run2 --out data/merged

# EDA report
python3 -m src.data_collection.eda_report \
  --data data/merged/missions.parquet --out reports/eda/
```

---

## Ablation results

| Trajectory seen | AUC   | F1    | Acc    | Compute saved |
|-----------------|-------|-------|--------|---------------|
| 10%             | 0.889 | 0.778 | 83.9%  | 90%           |
| 20%             | 0.964 | 0.854 | 90.7%  | 80%           |
| 30%             | 0.982 | 0.893 | 92.8%  | 70%           |
| **40%** ← prod  | **0.997** | **0.948** | **96.5%** | **60%** |
| 60%             | 0.998 | 0.971 | 98.0%  | 40%           |
| 100%            | 1.000 | 0.989 | 99.3%  | 0%            |

---

## Dataset

- 10,000 missions · 85.3M rows · 13.44 GB Parquet
- 35.3% success rate · 64.7% failure
- 13 physics-invariant features (synodic coords, orbital energy, eccentricity, etc.)
- 8 failure classes: success, surface_impact, orbit_too_high, missed_target, source_impact, hyperbolic_flyby, degenerate_orbit, unknown

---

## Docs

- [`docs/AI_CONTEXT.md`](docs/AI_CONTEXT.md) — full technical context for AI assistants
