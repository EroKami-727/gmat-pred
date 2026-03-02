"""
Database Repair Tool — Summary Generator
========================================
Generates a small 'summary.parquet' (one row per mission) from a massive 
'missions.parquet' file, enabling instant metadata views without RAM overhead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def generate_summary(data_dir: str):
    data_path = Path(data_dir)
    missions_file = data_path / "missions.parquet"
    summary_file = data_path / "summary.parquet"
    
    if not missions_file.exists():
        print(f"✗ Missions file not found: {missions_file}")
        return

    print("=" * 60)
    print("  OrbitGuard Database Repair (Summary Generator)")
    print("=" * 60)
    print(f"  Source: {missions_file}")
    
    pf = pq.ParquetFile(missions_file)
    n_rows = pf.metadata.num_rows
    
    # Columns needed for the summary
    cols = ["mission_id", "label", "failure_type", "min_target_rmag"]
    cols = [c for c in cols if c in pf.schema_arrow.names]

    rows = []
    last_mid = None
    
    print(f"▸ Scanning {n_rows:,} rows...")
    
    # Batch size of 500k keeps RAM usage < 1GB
    for batch in tqdm(pf.iter_batches(batch_size=500000, columns=cols), 
                      total=n_rows // 500000 + 1, unit="batch"):
        
        # Use Categorical for failure_type to save RAM
        df = batch.to_pandas(categories=["failure_type"])
        
        # Group by mission_id but keep it simple
        # Since missions are contiguous, we just take the first of each ID
        mids = df["mission_id"].values
        # Find ID changes
        changes = mids[1:] != mids[:-1]
        change_idx = [0] + (pd.Series(changes).index[changes] + 1).tolist()
        
        for idx in change_idx:
            mid = int(mids[idx])
            if mid != last_mid:
                # Store one row
                rows.append(df.iloc[idx].to_dict())
                last_mid = mid

    summary_df = pd.DataFrame(rows)
    summary_df.to_parquet(summary_file, index=False)
    
    print("-" * 60)
    print(f"✓ Summary generated: {summary_file}")
    print(f"  Total Missions   : {len(summary_df)}")
    print(f"  Summary size     : {summary_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/production")
    args = parser.parse_args()
    generate_summary(args.data_dir)
