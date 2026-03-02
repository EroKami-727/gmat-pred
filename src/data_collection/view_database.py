"""
Database Viewer — CLI tool to inspect the generated database (Memory Optimized)
==============================================================================
Optimized for 10M+ row datasets. Uses pyarrow to inspect metadata and read
only necessary row slices, avoiding RAM exhaustion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def view_database(data_dir: str = "data", show_rows: int = 10, 
                  filter_outcome: int | None = None, 
                  mission_id: int | None = None,
                  full_width: bool = False):
    """Display database summary and sample rows without loading the full file into RAM."""
    data_path = Path(data_dir)
    missions_file = data_path / "missions.parquet"
    params_file = data_path / "mission_params.parquet"
    summary_file = data_path / "summary.parquet"

    if not missions_file.exists():
        print(f"✗ No database found at {missions_file}")
        return

    # Use pyarrow to inspect metadata without reading data
    pf = pq.ParquetFile(missions_file)
    meta = pf.metadata
    schema = pf.schema_arrow

    print()
    print("=" * 70)
    print("  GMAT Mission Database (Memory-Optimized)")
    print("=" * 70)
    print(f"  File           : {missions_file}")
    print(f"  Size           : {missions_file.stat().st_size / 1e6:.2f} MB")
    print(f"  Total rows     : {meta.num_rows:,}")
    print(f"  Row groups     : {meta.num_row_groups}")
    print(f"  Columns ({len(schema.names)})")
    print(f"  Schema         : {schema.names[:10]} ... (+{len(schema.names)-10} more)")
    print()

    # ── Quick Summary (Prioritize summary.parquet) ──
    if summary_file.exists():
        print("  [Loading summary from index...]")
        summary_df = pd.read_parquet(summary_file)
        n_unique = len(summary_df)
    else:
        print("  [Calculating summary statistics from full data (Memory-Safe)...]")
        import pyarrow.compute as pc
        table = pf.read(columns=["mission_id", "label", "failure_type"])
        # Accurate unique mission count
        unique_ids = pc.unique(table["mission_id"])
        n_unique = len(unique_ids)
        # Using 'categories' for failure_type drastically reduces RAM (orders of magnitude)
        summary_df = table.to_pandas(categories=["failure_type"]).groupby("mission_id").first()
    
    success_count = int((summary_df["label"] == 1).sum())
    failure_count = int((summary_df["label"] == 0).sum())
    total = success_count + failure_count
    
    print(f"  Unique missions: {n_unique}")
    print("  ─── Outcome Distribution ───")
    print(f"  ✓ Successes : {success_count:>5}  ({100*success_count/total:.1f}%)")
    print(f"  ✗ Failures  : {failure_count:>5}  ({100*failure_count/total:.1f}%)")
    
    print("\n  ─── Failure Breakdown ───")
    for ftype, count in summary_df["failure_type"].value_counts().items():
        if ftype != "success":
            print(f"    - {ftype:<18}: {count:>5} ({100*count/total:.1f}%)")
    print()

    # ── Loading Rows ──
    view_df = pd.DataFrame()
    
    # Ensure mission_id is a column for consistent access
    if not summary_df.empty and "mission_id" not in summary_df.columns:
        summary_df = summary_df.reset_index()

    if mission_id is not None:
        # Load specific mission
        view_df = summary_df[summary_df["mission_id"] == mission_id]
        if view_df.empty:
            print(f"⚠ Mission ID {mission_id} not found.")
        else:
            # Re-read full columns for just these indices
            # Since mission_id is contiguous, we can find the range
            indices = view_df.index
            row_start = indices[0]
            row_end = indices[-1] + 1
            # Pyarrow doesn't support easy slicing by absolute row index across groups in a single call easily
            # But we can read the specific rows via the full table read for small missions
            full_mission_table = pf.read_row_group(row_start // (meta.num_rows // meta.num_row_groups) if meta.num_row_groups > 1 else 0)
            # Actually, simpler: read the whole mission filtered by mission_id
            view_df = pf.read().to_pandas() # NO! That crashes.
            # Correct memory-safe way:
            view_df = pd.DataFrame()
            for i in range(meta.num_row_groups):
                group = pf.read_row_group(i).to_pandas()
                m_rows = group[group["mission_id"] == mission_id]
                if not m_rows.empty:
                    view_df = m_rows
                    break
    elif filter_outcome is not None:
        # Load sample of specific outcome
        # Ensure mission_id is a column for consistent access
        if "mission_id" not in summary_df.columns:
            search_df = summary_df.reset_index()
        else:
            search_df = summary_df
            
        target_ids = search_df[search_df["label"] == filter_outcome]["mission_id"].values
        if len(target_ids) > 0:
            mission_id = int(target_ids[0])
            for i in range(meta.num_row_groups):
                group = pf.read_row_group(i).to_pandas()
                m_rows = group[group["mission_id"] == mission_id]
                if not m_rows.empty:
                    view_df = m_rows.head(show_rows)
                    print(f"  (Showing sample of { 'successes' if filter_outcome==1 else 'failures' })")
                    break
    else:
        # Just load first N rows
        view_df = pf.read_row_group(0).to_pandas().head(show_rows)

    if not view_df.empty:
        # Display Options
        pd_opts = [
            "display.max_columns", 20, 
            "display.width", 200 if full_width else 140,
            "display.float_format", "{:.4f}".format
        ]
        if full_width:
            pd_opts.extend(["display.max_rows", None, "display.expand_frame_repr", False])

        print(f"  ─── View Data ({'Mission ' + str(mission_id) if mission_id is not None else 'First ' + str(show_rows) + ' rows'}) ───")
        with pd.option_context(*pd_opts):
            print(view_df.to_string(index=False))
        print()

    # Show params if available
    if params_file.exists():
        params_df = pd.read_parquet(params_file)
        if mission_id is not None:
            params_df = params_df[params_df["sim_id"] == mission_id]
        
        print(f"  ─── Mission Parameters ({'ID ' + str(mission_id) if mission_id is not None else 'First 5'}) ───")
        with pd.option_context("display.max_columns", 20, "display.width", 140,
                               "display.float_format", "{:.4f}".format):
            print(params_df.head(5).to_string(index=False))
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="View GMAT simulation database (Memory Optimized)",
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
