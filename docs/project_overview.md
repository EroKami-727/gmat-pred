# Project Overview: NASA GMAT "Early Exit" Optimization

## The Problem
NASA runs millions of space flight simulations on massive supercomputers. However, universities and small research teams do not have supercomputers; we only have standard lab laptops and PCs. 

When running thousands of Monte Carlo simulations (randomized trial-and-error runs), many simulations are doomed to fail right from the start due to bad launch speeds or bad angles. Standard physics tools, like NASA's GMAT, don't know this. They will spend hours calculating the entire path of a crashing spacecraft, second-by-second, all the way until it hits the ground or gets lost in space. This is a massive waste of computer power.

## Our Solution
We are building an "Early Exit" system using Machine Learning (ML). 

Instead of replacing GMAT, we are giving it a "smart spectator." As GMAT starts calculating a simulation, our ML model watches the first 20% to 40% of the flight. If the ML recognizes the mathematical pattern of a doomed flight (one that will miss the Moon or crash), it steps in and says: **"Stop calculating. This is going to crash. Terminate the simulation immediately."**

> [!IMPORTANT]
> This research focuses on **ballistic trajectories**. A "successful" mission is one that naturally encounters the Moon at the correct altitude. We are essentially predicting if the initial launch "aim" was accurate enough to hit the target corridor.

By trading a tiny bit of accuracy, we can skip the remaining 60% to 80% of the math for failing missions. 

### Example of the Time Saved:
* **Person A (Standard GMAT):** Runs 100,000 simulations. It takes 100 hours.
* **Person B (Our System):** Runs 100,000 simulations. Our ML correctly predicts that 20,000 runs will fail shortly after launch. It cancels those runs early, saving 20+ hours of compute time. Person B finishes in less than 80 hours.

## Tech Stack
* **Simulation Engine:** NASA GMAT (Python API)
* **Data Handling:** Python, Pandas, Parquet (for fast saving/loading)
* **Machine Learning:** PyTorch or TensorFlow (LSTM / Transformer networks)
* **Frontend Visualization:** Streamlit (to visualize the orbits and ML predictions in a web browser)
* **Hardware Optimization:** Multiprocessing (using all 14 cores of an i7 processor)

## Project Goals
2. **Data Generation [✓ COMPLETE]:** Built a high-performance, multi-threaded pipeline using 3-body RK4 physics to generate thousands of physically accurate Earth-Moon transfers via Monte Carlo dispersion analysis.
3. **Behavioral Prediction:** Train deep learning models to predict failure from early-flight telemetry.
4. **Execution Pipeline:** Implement a real-time monitor that stops GMAT simulations upon failure detection.
4. **Research & Optimization:** Conduct comparative studies on architectures and early-exit thresholds.

> [!TIP]
> See the detailed [Research Enhancement Plan](docs/research_plan.md) for specifics on architecture comparison, ablation studies, and statistical validation.