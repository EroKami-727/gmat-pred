"""
OrbitGuard Dataset Quality Analyzer (Memory Optimized)
======================================================
Generates a comprehensive statistics report for the trajectory database.
Optimized for 10M+ rows — calculates statistics incrementally via streaming
batches to maintain constant RAM usage.

Usage:
    python -m src.data_collection.analyze_dataset --data data/missions.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# Feature columns to analyze
# ═══════════════════════════════════════════════════════════════════════════

PHYSICS_FEATURES = [
    "rel_x", "rel_y", "rel_z",
    "spec_energy", "fpa_deg",
    "norm_target_dist", "radial_vel", "vel_mag",
]

CONTEXT_FEATURES = [
    "mu_ratio", "soi_ratio", "dist_ratio",
]

DERIVED_FEATURES = [
    "earth_rmag", "luna_rmag", "ecc", "sma",
]


def analyze(data_path: str) -> None:
    """Run full analysis using streaming batches for memory efficiency."""
    path = Path(data_path)
    if not path.exists():
        print(f"✗ Data file not found: {data_path}")
        return

    pf = pq.ParquetFile(data_path)
    meta = pf.metadata
    n_rows = meta.num_rows

    print("=" * 60)
    print("  OrbitGuard Dataset Analysis (Memory-Optimized)")
    print("=" * 60)
    print(f"  Data file         : {data_path}")
    print(f"  Total rows        : {n_rows:,}")
    print(f"  File size         : {path.stat().st_size / 1e6:.1f} MB")
    print("-" * 60)

    # 1. Initialize stats accumulators
    cols_to_stat = [c for c in PHYSICS_FEATURES if c in pf.schema_arrow.names]
    
    stats = {
        c: {
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float('inf'),
            "max": float('-inf'),
            "count": 0,
            "nan": 0,
            "inf": 0
        } for c in cols_to_stat
    }

    # Tracking mission-level stats
    # Since we can't load all mission IDs into RAM if they are millions,
    # we'll assume mission_id is sorted/contiguous to count them efficiently.
    unique_mission_ids = set()
    label_counts = {0: 0, 1: 0}
    failure_type_counts = {}
    last_mission_id = None
    seq_lengths = [] # For min/max/avg we can just track these if missions < 100k
    
    # We'll also track context features (they are redundant per mission)
    context_values = set()

    # 2. Iterate in batches
    print("▸ Calculating statistics (Streaming)...")
    
    # Columns needed for full analysis
    needed_cols = list(set(["mission_id", "label", "failure_type"] + cols_to_stat + CONTEXT_FEATURES))
    needed_cols = [c for c in needed_cols if c in pf.schema_arrow.names]

    # Batching loop
    batch_iter = pf.iter_batches(batch_size=100000, columns=needed_cols)
    pbar = tqdm(total=n_rows, desc="  Processing", unit="row")

    for batch in batch_iter:
        # Categorical strings save massive RAM for failure_type during batches
        df = batch.to_pandas(categories=["failure_type"])
        batch_len = len(df)
        pbar.update(batch_len)

        # A. Mission-level stats
        # Efficiently track unique IDs and their outcomes without expensive full-batch groupby
        # We assume mission_ids are mostly contiguous within batches
        current_mids = df["mission_id"].values
        # Find indices where mission_id changes
        change_idx = np.where(current_mids[:-1] != current_mids[1:])[0]
        # Always check the very first row of the batch
        start_indices = np.concatenate([[0], change_idx + 1])
        
        for idx in start_indices:
            mid = int(current_mids[idx])
            if mid != last_mission_id:
                unique_mission_ids.add(mid)
                lbl = int(df["label"].iloc[idx])
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                
                ftype = df["failure_type"].iloc[idx]
                failure_type_counts[ftype] = failure_type_counts.get(ftype, 0) + 1
                
                # Context features (take first)
                if all(c in df.columns for c in CONTEXT_FEATURES):
                    ctx_tuple = tuple(df[CONTEXT_FEATURES].iloc[idx].values)
                    context_values.add(ctx_tuple)
                
                last_mission_id = mid

        # B. Physics feature stats (column-wise sums)
        for col in cols_to_stat:
            series = df[col]
            # Handle non-finite
            valid = series[np.isfinite(series)]
            stats[col]["nan"] += series.isna().sum()
            stats[col]["inf"] += len(series) - len(valid) - series.isna().sum()
            
            if len(valid) > 0:
                stats[col]["sum"] += valid.sum()
                stats[col]["sum_sq"] += (valid ** 2).sum()
                stats[col]["min"] = min(stats[col]["min"], valid.min())
                stats[col]["max"] = max(stats[col]["max"], valid.max())
                stats[col]["count"] += len(valid)

    pbar.close()

    n_missions = len(unique_mission_ids)
    print(f"\n  Total Missions    : {n_missions:,}")
    print(f"  Rows / Mission    : {n_rows / n_missions:.1f} (avg)")

    # 3. Finalize and Print Results
    
    # ── Class Distribution ──
    n_success = label_counts.get(1, 0)
    n_failure = label_counts.get(0, 0)
    print(f"\n  ─── Class Distribution ───")
    print(f"    Success (1)     : {n_success:,}  ({100*n_success/n_missions:.1f}%)")
    print(f"    Failure (0)     : {n_failure:,}  ({100*n_failure/n_missions:.1f}%)")

    # ── Failure Type Breakdown ──
    print(f"\n  ─── Failure Type Breakdown ───")
    for ftype, count in failure_type_counts.items():
        marker = "✓" if ftype == "success" else "✗"
        print(f"    {marker} {ftype:<20}: {count:>5} ({100*count/n_missions:.1f}%)")

    # ── Data Integrity ──
    print(f"\n  ─── Data Integrity ───")
    total_nan = sum(s["nan"] for s in stats.values())
    total_inf = sum(s["inf"] for s in stats.values())
    print(f"    NaN values      : {total_nan}")
    print(f"    Inf values      : {total_inf}")
    if total_nan == 0 and total_inf == 0:
        print(f"    ✓ Dataset is clean — no NaN or Inf values")

    # ── Physics Stats ──
    print(f"\n  ─── Physics Feature Statistics ───")
    for feat in cols_to_stat:
        s = stats[feat]
        if s["count"] > 0:
            mean = s["sum"] / s["count"]
            # Variance = E[X^2] - (E[X])^2
            var = (s["sum_sq"] / s["count"]) - (mean ** 2)
            std = np.sqrt(max(var, 0))
            print(f"    {feat:<20}: μ={mean:>12.4f}  σ={std:>12.4f}  "
                  f"[{s['min']:>12.4f}, {s['max']:>12.4f}]")

    # ── Context Features ──
    if context_values:
        print(f"\n  ─── Context Features (uniques found) ───")
        for ctx in context_values:
            val_str = "  ".join(f"{c}={v:.6e}" for c, v in zip(CONTEXT_FEATURES, ctx))
            print(f"    {val_str}")

    print(f"\n{'=' * 60}")
    print(f"  ✓ Analysis complete (Memory-Safe)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trajectory database quality (Memory-Safe)")
    parser.add_argument("--data", type=str, default="data/missions.parquet",
                        help="Path to missions.parquet file")
    args = parser.parse_args()

    analyze(args.data)
