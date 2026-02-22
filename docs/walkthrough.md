# Walkthrough: GMAT Simulation Database

## What Was Built

A complete data collection pipeline that generates orbital mechanics simulation data matching the schema from your project docs: time-series snapshots of spacecraft position, velocity, Moon position, fuel remaining, and mission outcome.

### Files Created/Modified

| File | Purpose |
|------|---------|
| [requirements.txt](file:///home/haise/Coding/Projects/gmat-pred/requirements.txt) | Dependencies for data collection phase |
| [base_mission.script](file:///home/haise/Coding/Projects/gmat-pred/gmat_scripts/base_mission.script) | GMAT script template (ready for when GMAT is installed) |
| [generator.py](file:///home/haise/Coding/Projects/gmat-pred/src/data_collection/generator.py) | Monte Carlo parameter generator (Keplerian elements, mass, duration) |
| [gmat_runner.py](file:///home/haise/Coding/Projects/gmat-pred/src/data_collection/gmat_runner.py) | Dual-mode runner: synthetic (two-body physics) + real GMAT |
| [build_database.py](file:///home/haise/Coding/Projects/gmat-pred/src/data_collection/build_database.py) | Orchestrator — generates, simulates, saves to Parquet |
| [view_database.py](file:///home/haise/Coding/Projects/gmat-pred/src/data_collection/view_database.py) | CLI viewer for inspecting the database |

### Virtual Environment

Created at `/home/haise/Coding/venvs/gmat-pred` with: pandas, numpy, scipy, tqdm, pyarrow, fastparquet, matplotlib.

---

## Database Output

| Metric | Value |
|--------|-------|
| Total missions | **200** |
| Successes | **137 (68.5%)** |
| Failures | **63 (31.5%)** |
| Total rows | **977,012** |
| Columns | 13 |
| File size | **60.1 MB** |
| NaN values | **0** |
| Generation time | **~19 seconds** |

### Columns

`mission_id`, `elapsed_secs`, `pos_x`, `pos_y`, `pos_z`, `vel_x`, `vel_y`, `vel_z`, `moon_x`, `moon_y`, `moon_z`, `fuel_remaining`, `outcome`

---

## How to Use

```bash
# Activate venv
source /home/haise/Coding/venvs/gmat-pred/bin/activate.fish

# Re-generate with different params
python -m src.data_collection.build_database --num-missions 500 --seed 123

# View database
cd ~/Coding/Projects/gmat-pred
/home/haise/Coding/venvs/gmat-pred/bin/python -m src.data_collection.view_database

# Filter to failures only
python -m src.data_collection.view_database --filter 0
```

## Verification Results

- ✅ 200 missions generated, 31.5% failure rate (target: 30–40%)
- ✅ Zero NaN values across 977K rows
- ✅ Zero placeholder (-1) outcomes — all rows have consistent 0 or 1
- ✅ Position values range ±500K km (consistent with trans-lunar orbits)
- ✅ Velocity values range ±14 km/s (physically realistic)
- ✅ Parquet files load correctly via pandas

## When GMAT is Installed

The `gmat_runner.py` module has a `run_gmat()` function ready. It generates customized `.script` files from the template, runs the GMAT binary, and parses its `ReportFile` output into the same DataFrame schema — so the rest of the pipeline works unchanged.
