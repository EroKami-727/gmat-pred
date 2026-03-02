# OrbitGuard — Technical Specification & Research Roadmap

---

## 1. System Architecture: 3-Body RK4 Engine
The project uses a custom **3-Body Synthetic Propagator** in `src/data_collection/gmat_runner.py`.

### Mathematical Foundation
- **Integrator:** Fourth-Order Runge-Kutta (RK4).
- **Physics:** Restricted Three-Body Problem (R3BP). The engine calculates acceleration from both the **Source Body** (e.g., Earth) and the **Target Body** (e.g., Moon) simultaneously.
- **Relativistic Accuracy:** Uses physical constants (G * M) from a centralized `PLANET_REGISTRY` in `generator.py`.
- **Nominal Optimization:** The nominal Earth-Moon TOI burn was precisely tuned to **3.240 km/s** to ensure parity between GMAT's high-fidelity model and our custom RK4 integrator.

---

## 2. Dataset Engineering: The Mesh Topology
To achieve **Interplanetary Generalization**, we implemented a "Mesh Topology" architecture.

### Dimensionless Context Vectors
Every mission row contains 3 context features that allow the model to recognize the "type" of mission regardless of units or distances:
1. **`mu_ratio`**: $\mu_{target} / \mu_{sun}$. Defines the mass scale of the target.
2. **`soi_ratio`**: $\text{SOI}_{target} / \text{Distance}_{transfer}$. Defines how "focused" the gravity well is.
3. **`dist_ratio`**: $\text{Distance}_{transfer} / 1 \text{AU}$. Defines the spatial scale.

### Physics-Invariant Input (13 Features)
The input tensor to the ML models (`lstm` or `transformer`) is 13-dimensional at every timestep:
- **Synodic Coordinates (3):** `rel_x, rel_y, rel_z` (Rotating frame).
- **Energy Metrics (2):** `spec_energy`, `vel_mag`.
- **Geometric Metrics (3):** `fpa_deg`, `norm_target_dist`, `radial_vel`.
- **Contextual/Orbital (5):** `earth_rmag`, `ecc`, `mu_ratio`, `soi_ratio`, `dist_ratio`.

---

## 3. Production Scaling: Memory-Optimized Batching
Generating 5000 missions produced **42,676,043 rows** (~6.5GB).

### Incremental Save Strategy
In `src/data_collection/build_database.py`, we solved the "RAM Exhaustion" problem (24GB limit) via:
1. **Multiprocessing Simulation:** 20 cores (i7) generate DataFrames in parallel.
2. **Batch Queue:** Every 500 missions, the accumulated DataFrames are concatenated and dumped to a **Temporary Parquet Batch** on disk.
3. **PyArrow Consolidation:** Once all 5000 missions are done, `pyarrow.parquet.ParquetWriter` streams the batches into a single master file without loading them into memory.

---

## 4. Current Research Results
Verified on **NVIDIA RTX 4060 Laptop GPU**:
- **Class Balance:** Achieved **34.8% success rate** using `success_ratio` stratification.
- **LSTM Performance:** Initial 2-epoch smoke test achieved **0.87 AUC** on binary failure prediction.
- **Data Quality:** Zero NaNs/Infs in 42.6M rows (verified via `analyze_dataset.py`).

---

## 5. Future Research Roadmap (The AI Handover)

### Phase 1: Early-Exit Ablation Study
The project's primary research question: *"How early can we predict failure without losing precision?"*
- Run training with `--early-exit 0.1`, `0.2`, `0.4`.
- Objective: Generate the **Ablation Curve** (Accuracy vs. Telemetry Fraction).

### Phase 2: Mesh Topology Training (Zero-Shot)
1. Generate 1000 missions each for **Earth-Moon** and **Earth-Mars**.
2. Train on Earth-Moon ONLY.
3. Test on Earth-Mars.
4. **Research Goal:** Prove that `mu_ratio` and `soi_ratio` enable the model to transfer knowledge to unseen planetary systems.

### Phase 3: Transformer Transition
Swap the `TrajectoryLSTM` for `TrajectoryTransformer`.
- Transformers are expected to handle the long-sequence attention better for interplanetary transfers where the "failure signal" might hide in subtle energy shifts days before impact.

---
**Status:** `READY_FOR_MODELING`
**Primary Data:** `data/production/missions.parquet`
