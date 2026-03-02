# Research Enhancement Plan
**Project Title:** OrbitGuard: Physics-Invariant Sequence Modeling for Trajectory Simulation Pruning

## 1. Domain Adaptation & Generalization (The "Holy Grail")
**Objective:** Prove that an ML model trained on one mission profile can generalize to an entirely unseen interplanetary topology (e.g., train on Earth-Moon, test on Earth-Mars).

**The Ephemeris Problem:**
If we train a model using standard J2000 coordinates (Absolute X, Y, Z), the model learns *where the Moon was on that specific launch date*. If we launch a month later, the physics of the trajectory might be identical, but the Moon has moved. An absolute-coordinate model will fail instantly because it doesn't "know" where the target is.

**The Solution: Target-Centric Rotating Frames & Physics Invariants:**
To make the model generalize across different dates (or entirely different planetary systems), we must transform the data *before* the LSTM sees it. The LSTM will never see raw X,Y,Z positions. Instead, it will see:

1. **Target-Centric Coordinates:** We place the target body (Moon/Mars) at `(0,0,0)`. The spacecraft's position becomes a *relative distance vector* ($\vec{r}_{rel}$). The model doesn't care where the target is in the solar system; it only cares where the ship is *relative to the target*.
2. **Rotating (Synodic) Frame:** We rotate the coordinate system so that the X-axis always points from the central body (Earth) to the target body. This means the "geometry" of the transfer always looks identical to the LSTM, regardless of the calendar date.
3. **Specific Orbital Energy ($E$) & Angular Momentum ($h$):** Instead of raw velocity, we feed the model $E = v^2 / 2 - \mu / r$. Because we normalize by the target's gravity constant ($\mu$), an energy curve approaching the Moon looks mathematically identical to an energy curve approaching Mars (just scaled by the Sphere of Influence).

**Hypothesis:** By feeding the LSTM a continuous stream of *Target-Relative Energy states* instead of *Absolute Geometries*, the network learns the universal Newtonian pattern of "missing a gravity well," allowing it to prune failing Mars simulations despite only ever being trained on the Moon.

---

## 2. Advanced Prediction Targets
**Objective:** Move beyond simple binary classification (`Landed=True/False`) to provide deep astrodynamic insights.
- **Multi-class Classification:** Predict *how* the mission will fail (`surface_impact`, `orbit_too_high`, `missed_target`, `hyperbolic_escape`).
- **Regression (Closest Approach):** Predict the exact continuous value of the closest-approach distance (e.g., "This trajectory will miss the target by 43,205 km").
- **Survival Analysis (Time-to-Failure):** Predict *when* the trajectory will violate mathematical safety bounds.

---

## 3. Comparative Architecture Study
**Objective:** Evaluate the optimal deep learning architecture for processing early-flight physics telemetry.
- **LSTM / GRU:** Baseline for sequential time-series modeling.
- **Time-Series Transformers (Autoformer / Informer):** Superior at handling long-context sequences through self-attention mechanisms.
- **1D CNN (ResNet):** Computationally lighter and highly effective at localized pattern extraction.

---

## 4. Early-Exit Timing Ablation Study
**Objective:** the minimum trajectory data required for reliable simulation pruning.
- Plotting the "Early-Exit Frontier" (Accuracy vs. Time Observed vs. False Positive Rate). How early can we exit? 10% of flight? 25%? 50%?

---

## Research Questions Addressed
1. **Generalizability:** Can a model trained on normalized Earth-Moon orbital energy flows accurately predict orbital failures in an Earth-Mars system?
2. **Predictive Granularity:** Is ML capable of classifying the *exact physical mode* of failure, or predicting the exact miss-distance via regression, using only the first 20% of flight telemetry?
3. **Architecture Optimization:** Which recurrent or attention-based network best captures astrodynamic invariants?
