# NASA GMAT Early Exit

This project aims to implement an "Early Exit" monitoring system for NASA GMAT missions using machine learning.

The completed Phase 1 has established a high-performance **Monte Carlo dispersion analysis pipeline**. It generates thousands of physically realistic Earth-to-Moon transfers using full 3-body physics (Earth + Moon gravity gradients) and Fourth-Order Runge-Kutta numerical integration. 

Failure states are classified based on physical outcomes (Surface Impact, Missed Moon, Orbit Too High) rather than geometrical bounds. The generated dataset tracks full state telemetry alongside dispersion offsets from a known "perfect" nominal trajectory.

## Project Structure

- `data/`: Parquet databases of the generated missions. (Local only, ignored by Git).
- `docs/`: Project documentation, ML plans, and architectural overview.
- `gmat_scripts/`: GMAT mission scripts for the actual binary runner.
- `notebooks/`: Data exploration and model testing.
- `src/`: Python source code containing the multi-threaded data collectors.

## Quick Start
The project uses `multiprocessing` to utilize all CPU cores natively for dataset generation.

### Bash
```bash
# Generate 2000 missions
source /home/haise/Coding/venvs/gmat-pred/bin/activate.fish
python3 -m src.data_collection.build_database --num-missions 2000

# View database summary
python3 -m src.data_collection.view_database
```

### Fish
```fish
# Generate 2000 missions
python3 -m src.data_collection.build_database --num-missions 2000

# View database summary
python3 -m src.data_collection.view_database
```
