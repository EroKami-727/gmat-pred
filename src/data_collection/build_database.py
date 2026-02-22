"""
Database Builder — Monte Carlo Orchestrator
===========================================
Generates N dispersed Monte Carlo scenarios from the nominal solution,
runs them through the 3-body physics propagator, and builds the
Parquet database.
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
from src.data_collection.gmat_runner import run_synthetic

from multiprocessing import Pool, cpu_count
from functools import partial


def build_database(
    num_missions: int = 200,
    time_step: float = 60.0,
    output_dir: str = "data",
    seed: int = 42,
) -> pd.DataFrame:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GMAT Monte Carlo Database Builder")
    print("=" * 60)
    print(f"  Missions      : {num_missions}")
    print(f"  Time step     : {time_step}s")
    print(f"  Output dir    : {out_path.resolve()}")
    print("=" * 60)
    print()

    print("▸ Generating dispersed parameters...")
    missions = generate_inputs(num_missions=num_missions, seed=seed)

    params_df = pd.DataFrame([m.to_dict() for m in missions])
    params_path = out_path / "mission_params.parquet"
    params_df.to_parquet(params_path, index=False)
    print(f"  ✓ Saved parameters → {params_path}")

    cores = cpu_count()
    print(f"\n▸ Running {num_missions} simulations using {cores} cores (3-Body RK4)...")
    t0 = time.time()

    all_dfs = []
    
    run_func = partial(run_synthetic, time_step=time_step)

    with Pool(processes=cores) as pool:
        for df in tqdm(pool.imap_unordered(run_func, missions), total=num_missions, desc="  Simulating", unit="mission"):
            all_dfs.append(df)

    elapsed = time.time() - t0

    full_df = pd.concat(all_dfs, ignore_index=True)
    missions_path = out_path / "missions.parquet"
    full_df.to_parquet(missions_path, index=False)

    summary = full_df.groupby("mission_id").first()
    total = len(summary)
    successes = int((summary["label"] == 1).sum())

    print()
    print("=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total missions   : {total}")
    print(f"  ✓ Successes      : {successes}  ({100*successes/total:.1f}%)")
    print(f"  ✗ Failures       : {total - successes}  ({100*(total-successes)/total:.1f}%)")
    print()
    print("  Failure Breakdown:")
    for ftype, count in summary["failure_type"].value_counts().items():
        if ftype != "success":
            print(f"    - {ftype:<18}: {count} ({100*count/total:.1f}%)")
            
    print()
    print(f"  Total rows       : {len(full_df):,}")
    print(f"  Data file        : {missions_path}  ({missions_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print("=" * 60)

    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-missions", type=int, default=200)
    parser.add_argument("--time-step", type=float, default=60.0)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_database(
        num_missions=args.num_missions,
        time_step=args.time_step,
        output_dir=args.output_dir,
        seed=args.seed,
    )
