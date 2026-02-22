"""
Database Builder — Orchestration Script
=========================================
Generates N Monte Carlo missions, runs them through the synthetic propagator,
collects all time-series data, and saves to Parquet in data/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import generate_inputs
from src.data_collection.gmat_runner import run_synthetic, COLUMNS


def build_database(
    num_missions: int = 200,
    time_step: float = 60.0,
    output_dir: str = "data",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full mission database.

    Parameters
    ----------
    num_missions : int
        Number of Monte Carlo missions to simulate.
    time_step : float
        Time between snapshots in seconds (60 = one per minute).
    output_dir : str
        Directory to save Parquet files.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Concatenated time-series data for all missions.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GMAT Mission Database Builder")
    print("=" * 60)
    print(f"  Missions      : {num_missions}")
    print(f"  Time step     : {time_step}s")
    print(f"  Output dir    : {out_path.resolve()}")
    print(f"  Seed          : {seed}")
    print(f"  Mode          : synthetic (two-body)")
    print("=" * 60)
    print()

    # Generate parameter sets
    print("▸ Generating mission parameters...")
    missions = generate_inputs(num_missions=num_missions, seed=seed)

    # Save parameters for reference
    params_df = pd.DataFrame([m.to_dict() for m in missions])
    params_path = out_path / "mission_params.parquet"
    params_df.to_parquet(params_path, index=False)
    print(f"  ✓ Saved parameters → {params_path}")

    # Run all missions
    print(f"\n▸ Running {num_missions} simulations...")
    t0 = time.time()

    all_dfs = []
    successes = 0
    failures = 0

    for params in tqdm(missions, desc="  Simulating", unit="mission"):
        df = run_synthetic(params, time_step=time_step)
        all_dfs.append(df)

        # Count outcomes
        outcome = df["outcome"].iloc[0]
        if outcome == 1:
            successes += 1
        else:
            failures += 1

    elapsed = time.time() - t0

    # Combine all data
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Save to Parquet
    missions_path = out_path / "missions.parquet"
    full_df.to_parquet(missions_path, index=False)

    # Print summary
    total = successes + failures
    print()
    print("=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total missions   : {total}")
    print(f"  ✓ Successes      : {successes}  ({100*successes/total:.1f}%)")
    print(f"  ✗ Failures       : {failures}  ({100*failures/total:.1f}%)")
    print(f"  Total rows       : {len(full_df):,}")
    print(f"  Columns          : {list(full_df.columns)}")
    print(f"  Data file        : {missions_path}  ({missions_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Params file      : {params_path}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print("=" * 60)

    return full_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build GMAT simulation database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-missions", type=int, default=200,
                        help="Number of Monte Carlo missions")
    parser.add_argument("--time-step", type=float, default=60.0,
                        help="Snapshot interval in seconds")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for Parquet files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    build_database(
        num_missions=args.num_missions,
        time_step=args.time_step,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
