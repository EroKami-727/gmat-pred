# NASA GMAT Early Exit

This project implements an "Early Exit" monitoring system for NASA GMAT missions using Machine Learning (LSTM/Transformer).

It establishes a high-performance **Monte Carlo dispersion analysis pipeline** that generates physically realistic interplanetary transfers using full 3-body physics (Earth + Moon gravity gradients) and Fourth-Order Runge-Kutta numerical integration.

---

## 🛰️ Current Build Status (Production)
- **Dataset:** 5,000 missions (42.6M rows) generated on 20 cores.
- **Success Rate:** Balanced **34.8%** (Success vs. Surface Impact / Orbit Too High).
- **GPU Optimized:** Training verified on **RTX 4060** using `BCEWithLogitsLoss` and `pos_weight`.
- **Mesh Topology:** Support for Earth-Moon, Earth-Mars, and Earth-Jupiter transfers.

---

## 📂 Documentation (handover-ready)
- **[Project Overview](docs/project_overview.md):** High-level goals and "The Why."
- **[ML Roadmap (AI-Ready)](docs/ML_ROADMAP_AI_READY.md):** Detailed technical specification for another AI to understand the physics-invariants and context vectors.
- **[ML Pipeline Summary](docs/ml_pipeline_summary.md):** Feature engineering and transformation logic.

---

## 🚀 Quick Start (Fish Terminal)

### 1. Production Runtime (5000 Missions)
```fish
source /home/haise/Coding/venvs/gmat-pred/bin/activate.fish
python3 -m src.data_collection.build_database --num-missions 5000 --success-ratio 0.35 --output-dir data/production --batch-size 500
```

### 2. View Database & Stats
```fish
# Quality report (Table 1 stats)
python3 -m src.data_collection.analyze_dataset --data data/production/missions.parquet

# CLI inspector
python3 -m src.data_collection.view_database --data-dir data/production
```

### 3. Model Training
```fish
python3 -m src.ml.train --data data/production/missions.parquet --epochs 50 --model lstm --output-dir models/production
```
