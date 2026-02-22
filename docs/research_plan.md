# Research Enhancement Plan
**Project Title:** Early-Exit Neural Prediction for Trajectory Simulation Pruning in Earth–Moon Transfers

## 1. Comparative Architecture Study
**Objective:** Evaluate the optimal deep learning architecture for early trajectory failure detection.

| Architecture | Rationale |
|--------------|-----------|
| **LSTM** | Baseline for sequential time-series; captures temporal dependencies. |
| **Transformer Encoder** | Uses self-attention; captures global relationships across trajectory windows. |
| **1D CNN** | Computationally efficient; strong at extracting features from structured physical signals. |

**Evaluation Metrics:** Accuracy, Precision, Recall, False Positive Rate (FPR), False Negative Rate (FNR), Inference Time, Training Time, and Memory Footprint.

---

## 2. Early-Exit Timing Ablation Study
**Objective:** Determine the minimum trajectory data required for reliable pruning.

- **Observation Windows:** Systematic testing at 10%, 20%, 30%, 40%, and 50% of total transfer time.
- **Goal:** Plot the "Early-Exit Frontier" (Accuracy vs. Time Saved vs. FPR) to find the earliest reliable decision point.

---

## 3. Dataset Scaling & Variety
**Objective:** Improve generalization and statistical confidence.

- **Target Size:** 50,000 to 100,000 simulated trajectories.
- **Parameter Variation:** Randomized TOI ΔV, RAAN, AOP, Launch Epoch (1 full year), altitude, inclination, and C3 energy levels.
- **Success Criteria:** Clear binary classification based on LOI Eccentricity (<1), Periapsis altitude, B-plane tolerance, and stability.

---

## 4. Statistical Rigor
**Objective:** Ensure reproducibility and scientific stability.

- **Method:** Run every experiment 5–10 times with different random seeds.
- **Reporting:** Report mean results with standard deviation.
- **Impact:** Demonstrates that the "Early-Exit" capability is a robust physical property, not a result of "lucky" weight initialization.

---

## Research Questions Answered
1. Can early trajectory trends reliably predict terminal mission status?
2. Which neural architecture best balances accuracy and inference speed?
3. What is the earliest possible decision point for simulation pruning?
4. Are these results statistically stable across randomized orbital geometries?
