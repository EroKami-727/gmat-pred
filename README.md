# NASA GMAT Early Exit

This project implements an "Early Exit" monitoring system for NASA GMAT missions using Machine Learning (LSTM/Transformer).

It establishes a high-performance **Monte Carlo dispersion analysis pipeline** that generates physically realistic interplanetary transfers using full 3-body physics (Earth + Moon gravity gradients) and Fourth-Order Runge-Kutta numerical integration.

---

## 🛰️ Current Build Status (Production)
- **Dataset:** 10,000 missions (85.3M rows) generated on 20 cores.
- **Success Rate:** Balanced **35.3%** (Success vs. Surface Impact / Orbit Too High).
- **Storage:** 13.44 GB (Zstandard/Snappy compressed Parquet).
- **GPU Optimized:** Training verified on **RTX 4060** using `BCEWithLogitsLoss`.
- **Mesh Topology:** Support for Earth-Moon, Earth-Mars, and Earth-Jupiter transfers.

---

## 📂 Documentation (handover-ready)
- **[Project Overview](docs/project_overview.md):** High-level goals and "The Why."
- **[ML Roadmap (AI-Ready)](docs/ML_ROADMAP_AI_READY.md):** Detailed technical specification for another AI to understand the physics-invariants and context vectors.
- **[ML Pipeline Summary](docs/ml_pipeline_summary.md):** Feature engineering and transformation logic.

---

## 🚀 Quick Start (Fish Terminal)

### 1. Production Runtime (10,000 Missions)
```fish
# Run 5000 missions (Run 1)
python3 -m src.data_collection.build_database --num-missions 5000 --output-dir data/production

# Run 5000 missions (Run 2 - different seed)
python3 -m src.data_collection.build_database --num-missions 5000 --output-dir data/run2 --seed 100

# Merge into 10k Master Dataset (Streaming)
python3 -m src.data_collection.merge_datasets --base data/production --new data/run2 --out data/merged
```

### 2. View Database & EDA
```fish
# Terminal CLI inspector
python3 -m src.data_collection.view_database --data-dir data/merged

# Generate Visual EDA Report (9 Premium Charts)
python3 -m src.data_collection.eda_report --data data/merged/missions.parquet --out reports/eda/

# Open HTML Report
xdg-open reports/eda/eda_report.html
```

### 3. Model Training
```fish
python3 -m src.ml.train --data data/merged/missions.parquet --epochs 50 --model lstm --output-dir models/production
```
