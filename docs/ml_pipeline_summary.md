# ML Pipeline Technical Summary & Roadmap

This document details the feature engineering, data transformation, and model evaluation strategy for the `OrbitGuard` Early-Exit system.

---

## 1. Feature Engineering: Physics-Invariants
To solve the **Domain Adaptation** problem (transferring knowledge from Moon to Mars), the model never sees raw Cartesian coordinates. It operates on 13 derived features.

### A. Target-Centric Synodic Frame (rel_x, rel_y, rel_z)
We transform raw position into a rotating frame where:
- **Origin:** The center of the target body.
- **X-axis:** Points from Source to Target.
- **Z-axis:** Aligned with the target's orbital angular momentum.
- **Result:** The model learns "Relative Approach Geometry" regardless of the absolute position in the solar system.

### B. Energy & Flight Geometry
- **Specific Orbital Energy ($E$):** Calculated as $v^2/2 - \mu/r$. This is the single most important predictor of capture.
- **Flight Path Angle ($\gamma$):** Velocity angle relative to the local horizon. Critical for identifying "too shallow" or "too steep" arrivals.
- **Normalized Target Distance:** Distance divided by the Target's Sphere of Influence (SOI). This scales all planets to a `[0, 1]` range near the encounter.

### C. Dimensionless Context Vectors
Crucial for zero-shot generalization:
- `mu_ratio`: Target mass scale.
- `soi_ratio`: Gravity well "tightness."
- `dist_ratio`: Mission spatial scale.

---

## 2. Preprocessing & Temporal Logic
Located in `src/ml/dataset.py`.

### A. Temporal Downsampling
Raw 60s telemetry is downsampled to **15-minute intervals** (1 data point every 15 steps).
- **Reason:** Reduces sequence length from ~8,000 to ~500.
- **Benefit:** Prevents gradient vanishing/explosion in LSTMs and fits within Transformer attention windows.

### B. Robust Scaling
We use `sklearn.RobustScaler` (Median/IQR) instead of Standard Scaling.
- **Reason:** Orbital data often has extreme outliers (e.g., initial high-speed TOI burn vs. slow cruise). Robust scaling prevents outliers from "collapsing" the feature variance.

### C. Dynamic Slicing (Early-Exit)
The dataloader supports `early_exit_frac`. If set to `0.2`, the model only sees the first 20% of the flight. This simulates the real-time "Pruning" scenario.

---

## 3. Training & Metrics Strategy
Located in `src/ml/train.py`.

### A. Imbalance Handling
With a ~35% success rate, the model might still over-predict failures. We use `nn.BCEWithLogitsLoss(pos_weight=2.2)`. This forces the model to care 2.2x more about missing a success (False Negative) than it does about a False Positive.

### B. Evaluation Metrics
- **ROC-AUC:** Measures the model's ability to rank success vs. failure.
- **F1-Score:** Measures the balance between Precision (not wasting propellant on good runs) and Recall (catching all bad runs).

---

## 4. Current Results & Next Steps
- **Production Dataset:** `data/production/missions.parquet` (5000 missions).
- **GPU Status:** Verified on NVIDIA RTX 4060.
- **Baseline AUC:** 0.87 (2-epoch LSTM smoke test).

### Roadmap for the "Next AI"
1. **Ablation:** Sweeping `early-exit` from 0.1 to 0.4.
2. **Zero-Shot:** Train on Earth-Moon, evaluate on Earth-Mars to test Mesh Topology.
3. **Transformer:** Implement the self-attention architecture in `model.py` to compare with LSTM.
