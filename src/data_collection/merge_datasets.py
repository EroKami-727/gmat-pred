"""
merge_datasets.py — Stream-merge two simulation runs into one
==============================================================
Combines data/production/ and data/run2/ (or any two dirs) into
a single missions.parquet with globally unique mission_ids.

The second dataset's mission_ids are offset by the count of the
first dataset. Uses streaming PyArrow writes — never loads the
full 13 GB into RAM.

Usage
-----
    python -m src.data_collection.merge_datasets \
        --base data/production \
        --new  data/run2 \
        --out  data/merged
"""

from __future__ import annotations
import argparse, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def merge(base_dir: Path, new_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    base_missions = base_dir / "missions.parquet"
    new_missions  = new_dir  / "missions.parquet"

    if not base_missions.exists():
        sys.exit(f"ERROR: {base_missions} not found.")
    if not new_missions.exists():
        sys.exit(f"ERROR: {new_missions} not found.")

    # ── Determine ID offset ───────────────────────────────────────────────
    base_summary = pd.read_parquet(base_dir / "summary.parquet", columns=["mission_id"])
    id_offset = int(base_summary["mission_id"].max()) + 1
    n_base = len(base_summary)
    del base_summary

    new_summary = pd.read_parquet(new_dir / "summary.parquet", columns=["mission_id"])
    n_new = len(new_summary)
    del new_summary

    n_total = n_base + n_new
    print("=" * 60)
    print("  Dataset Merger (Streaming)")
    print("=" * 60)
    print(f"  Base     : {base_missions}  ({n_base} missions)")
    print(f"  New      : {new_missions}   ({n_new} missions, IDs → +{id_offset})")
    print(f"  Output   : {out_dir}")
    print(f"  Total    : {n_total} missions")
    print("=" * 60)

    merged_path = out_dir / "missions.parquet"

    base_pf = pq.ParquetFile(base_missions)
    schema  = base_pf.schema_arrow

    total_rows = 0
    with pq.ParquetWriter(merged_path, schema, compression="snappy") as writer:
        # 1. Stream base (unchanged)
        for batch in tqdm(base_pf.iter_batches(batch_size=500_000),
                          desc="  Base   ", unit="batch"):
            writer.write_batch(batch)
            total_rows += len(batch)

        # 2. Stream new (re-index mission_id)
        new_pf = pq.ParquetFile(new_missions)
        for batch in tqdm(new_pf.iter_batches(batch_size=500_000),
                          desc="  New    ", unit="batch"):
            df = batch.to_pandas()
            df["mission_id"] = df["mission_id"] + id_offset
            new_batch = pa.RecordBatch.from_pandas(df, schema=schema, preserve_index=False)
            writer.write_batch(new_batch)
            total_rows += len(batch)

    print(f"\n  ✓ Merged missions.parquet  ({total_rows:,} rows, {merged_path.stat().st_size/1e9:.2f} GB)")

    # ── Merge mission_params ──────────────────────────────────────────────
    base_params = base_dir / "mission_params.parquet"
    new_params  = new_dir  / "mission_params.parquet"
    if base_params.exists() and new_params.exists():
        p1 = pd.read_parquet(base_params)
        p2 = pd.read_parquet(new_params)
        p2["sim_id"] = p2["sim_id"] + id_offset
        merged_params = pd.concat([p1, p2], ignore_index=True)
        merged_params.to_parquet(out_dir / "mission_params.parquet", index=False)
        print(f"  ✓ Merged mission_params.parquet  ({len(merged_params)} rows)")

    # ── Rebuild summary ───────────────────────────────────────────────────
    print("  Rebuilding summary.parquet ...")
    summary_path = out_dir / "summary.parquet"
    pf = pq.ParquetFile(merged_path)
    cols = ["mission_id", "label", "failure_type", "min_target_rmag"]
    cols = [c for c in cols if c in pf.schema_arrow.names]

    sum_rows = []
    last_mid = None
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas()
        mids = df["mission_id"].values
        changes = np.where(np.diff(mids) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes])
        for idx in boundaries:
            mid = int(mids[idx])
            if mid != last_mid:
                sum_rows.append(df.iloc[idx].to_dict())
                last_mid = mid

    summary_df = pd.DataFrame(sum_rows)
    summary_df.to_parquet(summary_path, index=False)

    n_success = int(summary_df["label"].sum())
    n_fail    = n_total - n_success
    print(f"  ✓ summary.parquet  ({len(summary_df)} missions)")
    print()
    print("=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    print(f"  Total missions : {n_total}")
    print(f"  ✓ Successes    : {n_success}  ({100*n_success/n_total:.1f}%)")
    print(f"  ✗ Failures     : {n_fail}    ({100*n_fail/n_total:.1f}%)")
    if "failure_type" in summary_df.columns:
        print(f"\n  Failure breakdown:")
        for ft, cnt in summary_df["failure_type"].value_counts().items():
            print(f"    {ft:20s}: {cnt:5d}  ({100*cnt/n_total:.1f}%)")
    print(f"\n  Output file    : {merged_path}  ({merged_path.stat().st_size/1e9:.2f} GB)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two simulation datasets")
    parser.add_argument("--base", type=str, default="data/production",
                        help="Directory containing the base missions.parquet")
    parser.add_argument("--new",  type=str, default="data/run2",
                        help="Directory containing the new missions.parquet to append")
    parser.add_argument("--out",  type=str, default="data/merged",
                        help="Output directory for the merged dataset")
    args = parser.parse_args()
    merge(Path(args.base), Path(args.new), Path(args.out))
