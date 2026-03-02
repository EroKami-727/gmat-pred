"""
Database Builder — Monte Carlo Orchestrator (Memory Optimized)
==============================================================
Generates N dispersed Monte Carlo scenarios for any planet pair,
runs them through the 3-body physics propagator, and builds the
Parquet database.

Optimized for high-volume runs (5000+ missions) by saving results
in batches to disk, avoiding RAM exhaustion during concatenation.

Supports multi-planet mesh topology via --source / --target flags.
"""

from __future__ import annotations

import argparse
import sys
import time
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import generate_inputs, PLANET_REGISTRY
from src.data_collection.gmat_runner import run_synthetic

from multiprocessing import Pool, cpu_count
from functools import partial


def build_database(
    num_missions: int = 200,
    time_step: float = 60.0,
    output_dir: str = "data",
    seed: int = 42,
    source: str = "earth",
    target: str = "moon",
    success_ratio: float = 0.0,
    batch_size: int = 500,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Temporary directory for batches
    tmp_path = out_path / "_tmp_batches"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    print("=" * 60)
    print("  GMAT Monte Carlo Database Builder (Memory-Optimized)")
    print("=" * 60)
    print(f"  Mission       : {source.capitalize()} → {target.capitalize()}")
    print(f"  Missions      : {num_missions}")
    print(f"  Success ratio : {success_ratio:.0%}" if success_ratio > 0 else "  Success ratio : Pure MC (unbiased)")
    print(f"  Batch size    : {batch_size}")
    print(f"  Output dir    : {out_path.resolve()}")
    print("=" * 60)
    print()

    print("▸ Generating dispersed parameters...")
    missions = generate_inputs(
        num_missions=num_missions,
        seed=seed,
        source=source,
        target=target,
        success_ratio=success_ratio,
    )

    params_df = pd.DataFrame([m.to_dict() for m in missions])
    params_path = out_path / "mission_params.parquet"
    params_df.to_parquet(params_path, index=False)
    print(f"  ✓ Saved parameters → {params_path}")

    cores = cpu_count()
    print(f"\n▸ Running {num_missions} simulations using {cores} cores (3-Body RK4)...")
    t0 = time.time()

    current_batch = []
    batch_count = 0
    total_rows = 0
    successes = 0

    run_func = partial(run_synthetic, time_step=time_step)

    with Pool(processes=cores) as pool:
        pbar = tqdm(pool.imap_unordered(run_func, missions), total=num_missions, desc="  Simulating", unit="mission")
        for df in pbar:
            current_batch.append(df)
            
            # Update summary stats from the first row of each mission
            successes += int(df["label"].iloc[0] == 1)
            
            if len(current_batch) >= batch_size:
                # Save batch to disk
                batch_df = pd.concat(current_batch, ignore_index=True)
                total_rows += len(batch_df)
                batch_file = tmp_path / f"batch_{batch_count:04d}.parquet"
                batch_df.to_parquet(batch_file, index=False)
                
                # Clear RAM
                current_batch = []
                batch_count += 1
                pbar.set_postfix({"batch": batch_count, "saved_rows": f"{total_rows/1e6:.1f}M"})

        # Save final partial batch
        if current_batch:
            batch_df = pd.concat(current_batch, ignore_index=True)
            total_rows += len(batch_df)
            batch_file = tmp_path / f"batch_{batch_count:04d}.parquet"
            batch_df.to_parquet(batch_file, index=False)
            batch_count += 1

    elapsed = time.time() - t0
    print(f"\n▸ Simulations complete. Consolidating {batch_count} batches...")

    # Consolidate using pyarrow (efficient memory mapping)
    missions_path = out_path / "missions.parquet"
    
    # Read all batches and write to one file
    # We use pyarrow.parquet.ParquetWriter to merge without loading everything into Pandas
    all_batch_files = sorted(list(tmp_path.glob("*.parquet")))
    
    # Get schema from first file
    first_table = pq.read_table(all_batch_files[0])
    schema = first_table.schema
    
    with pq.ParquetWriter(missions_path, schema, compression='snappy') as writer:
        for bfile in tqdm(all_batch_files, desc="  Consolidating", unit="batch"):
            table = pq.read_table(bfile)
            writer.write_table(table)

    # Cleanup temp batches
    shutil.rmtree(tmp_path)

    # 4. Generate Summary for fast viewing
    print(f"▸ Generating summary index...")
    summary_path = out_path / "summary.parquet"
    # We can use the missions_path we just wrote to extract one row per mission
    # Using the same streaming logic as repair_database.py
    sum_pf = pq.ParquetFile(missions_path)
    sum_rows = []
    last_mid = None
    cols = ["mission_id", "label", "failure_type", "min_target_rmag"]
    cols = [c for c in cols if c in sum_pf.schema_arrow.names]
    
    for batch in sum_pf.iter_batches(batch_size=500000, columns=cols):
        df = batch.to_pandas(categories=["failure_type"])
        mids = df["mission_id"].values
        changes = mids[1:] != mids[:-1]
        indices = [0] + (np.where(changes)[0] + 1).tolist()
        for i in indices:
            mid = int(mids[i])
            if mid != last_mid:
                sum_rows.append(df.iloc[i].to_dict())
                last_mid = mid
    
    summary_df = pd.DataFrame(sum_rows)
    summary_df.to_parquet(summary_path, index=False)

    print()
    print("=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"  Mission type   : {source.capitalize()} → {target.capitalize()}")
    print(f"  Total missions : {num_missions}")
    print(f"  ✓ Successes    : {successes}  ({100*successes/num_missions:.1f}%)")
    print(f"  ✗ Failures     : {num_missions - successes}  ({100*(num_missions-successes)/num_missions:.1f}%)")
    print()
    print(f"  Total rows     : {total_rows:,}")
    print(f"  Data file      : {missions_path}  ({missions_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Time elapsed   : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    available = list(PLANET_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Build Monte Carlo trajectory database for any planet pair."
    )
    parser.add_argument("--num-missions", type=int, default=200)
    parser.add_argument("--time-step", type=float, default=60.0)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str, default="earth", choices=available,
                        help="Source body for the transfer")
    parser.add_argument("--target", type=str, default="moon", choices=available,
                        help="Target body for the transfer")
    parser.add_argument("--success-ratio", type=float, default=0.0,
                        help="Fraction of missions to bias toward success (0.0-1.0)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Save results to disk every N missions to save RAM")
    args = parser.parse_args()

    build_database(
        num_missions=args.num_missions,
        time_step=args.time_step,
        output_dir=args.output_dir,
        seed=args.seed,
        source=args.source,
        target=args.target,
        success_ratio=args.success_ratio,
        batch_size=args.batch_size,
    )
