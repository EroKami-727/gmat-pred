"""
Database Viewer — CLI tool to inspect the generated database
==============================================================
Loads Parquet files from data/ and displays summary statistics,
sample rows, and outcome distributions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def view_database(data_dir: str = "data", show_rows: int = 10, 
                  filter_outcome: int | None = None, 
                  mission_id: int | None = None,
                  full_width: bool = False):
    """Display database summary and sample rows."""
    data_path = Path(data_dir)
    missions_file = data_path / "missions.parquet"
    params_file = data_path / "mission_params.parquet"

    if not missions_file.exists():
        print(f"✗ No database found at {missions_file}")
        print(f"  Run:  python -m src.data_collection.build_database")
        return

    # Load data
    df = pd.read_parquet(missions_file)
    print()
    print("=" * 70)
    print("  GMAT Mission Database")
    print("=" * 70)
    print(f"  File           : {missions_file}")
    print(f"  Size           : {missions_file.stat().st_size / 1e6:.2f} MB")
    print(f"  Total rows     : {len(df):,}")
    print(f"  Unique missions: {df['mission_id'].nunique()}")
    print(f"  Columns ({len(df.columns)})   : {list(df.columns)}")
    print()

    # Outcome distribution
    outcomes = df.groupby("mission_id")["label"].first()
    failure_types = df.groupby("mission_id")["failure_type"].first()
    
    success_count = int((outcomes == 1).sum())
    failure_count = int((outcomes == 0).sum())
    total = success_count + failure_count
    
    print("  ─── Outcome Distribution ───")
    print(f"  ✓ Successes : {success_count:>5}  ({100*success_count/total:.1f}%)")
    print(f"  ✗ Failures  : {failure_count:>5}  ({100*failure_count/total:.1f}%)")
    
    print("\n  ─── Failure Breakdown ───")
    for ftype, count in failure_types.value_counts().items():
        if ftype != "success":
            print(f"    - {ftype:<18}: {count:>5} ({100*count/total:.1f}%)")
    print()

    # Data statistics
    print("  ─── Column Statistics ───")
    stats = df[["elapsed_secs", "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z", "ecc", "sma"]].describe()
    # Transpose for readability
    with pd.option_context("display.max_columns", 20, "display.width", 120,
                           "display.float_format", "{:.2f}".format):
        print(stats.T.to_string())
    print()

    # Rows per mission stats
    rows_per_mission = df.groupby("mission_id").size()
    print(f"  ─── Rows per Mission ───")
    print(f"  Min   : {rows_per_mission.min():,}")
    print(f"  Max   : {rows_per_mission.max():,}")
    print(f"  Mean  : {rows_per_mission.mean():,.0f}")
    print(f"  Median: {rows_per_mission.median():,.0f}")
    print()

    # Filter by Outcome
    if filter_outcome is not None:
        filter_ids = outcomes[outcomes == filter_outcome].index
        df = df[df["mission_id"].isin(filter_ids)]
        label = "successes" if filter_outcome == 1 else "failures"
        print(f"  (Filtered to {label} only: {len(filter_ids)} missions)")
        print()

    # Filter by Mission ID
    if mission_id is not None:
        if mission_id not in df["mission_id"].values:
            print(f"⚠ Mission ID {mission_id} not found.")
        else:
            df = df[df["mission_id"] == mission_id]
            show_rows = len(df) # Show all rows for this mission
            print(f"  (Filtered to Mission ID {mission_id}: {show_rows} rows)")
            print()

    # Display Options
    pd_opts = [
        "display.max_columns", 20, 
        "display.width", 200 if full_width else 140,
        "display.float_format", "{:.4f}".format
    ]
    if full_width:
        pd_opts.extend(["display.max_rows", None, "display.expand_frame_repr", False])

    # Sample rows
    print(f"  ─── View Data ({'Full' if mission_id else 'First ' + str(show_rows) + ' rows'}) ───")
    with pd.option_context(*pd_opts):
        print(df.head(show_rows).to_string(index=False))
    print()

    # Show params if available
    if params_file.exists():
        params_df = pd.read_parquet(params_file)
        if mission_id is not None:
            params_df = params_df[params_df["sim_id"] == mission_id]
        
        print(f"  ─── Mission Parameters ({'ID ' + str(mission_id) if mission_id else 'First 5'}) ───")
        with pd.option_context("display.max_columns", 20, "display.width", 140,
                               "display.float_format", "{:.4f}".format):
            print(params_df.head(5).to_string(index=False))
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="View GMAT simulation database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing Parquet files")
    parser.add_argument("--rows", type=int, default=10,
                        help="Number of sample rows to show")
    parser.add_argument("--filter", type=int, choices=[0, 1], default=None,
                        dest="filter_outcome",
                        help="Filter by outcome (0=failures, 1=successes)")
    parser.add_argument("--mission", type=int, default=None,
                        help="View all rows for a specific mission_id")
    parser.add_argument("--full", action="store_true",
                        help="Disable row/column truncation for full view")
    args = parser.parse_args()

    view_database(
        data_dir=args.data_dir,
        show_rows=args.rows,
        filter_outcome=args.filter_outcome,
        mission_id=args.mission,
        full_width=args.full
    )


if __name__ == "__main__":
    main()
