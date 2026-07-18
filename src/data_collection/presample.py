"""
Pre-downsample a filtered subset of the missions parquet.

Streams the source parquet in small batches (no full-file scan), keeps only
the requested target bodies, and takes every Nth row per mission. Writes a
small local parquet so training temp-split files are tiny instead of 43 GB.

Usage:
    python -m src.data_collection.presample \
        --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
        --planets Jupiter Saturn Uranus Neptune \
        --downsample 50 \
        --out data/outer_ds50.parquet
"""

from __future__ import annotations
import argparse
import gc
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--planets",    nargs="+", required=True)
    parser.add_argument("--downsample", type=int, default=50)
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()

    src        = Path(args.data)
    dst        = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    planet_set = pa.array([p.lower() for p in args.planets])
    ds         = args.downsample

    pf     = pq.ParquetFile(src)
    schema = pf.schema_arrow

    # Single streaming pass — never loads more than one batch into RAM.
    # Filter by target_body using Arrow compute (no pandas needed for filter).
    print(f"Streaming & downsampling {args.planets} (every {ds}th row) ...")
    writer        = pq.ParquetWriter(dst, schema)
    rows_written  = 0
    missions_seen = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=100_000)):
        # Arrow-native filter — no pandas, minimal allocations
        lower_tb = pc.utf8_lower(batch.column("target_body"))
        mask     = pc.is_in(lower_tb, value_set=planet_set)
        filtered = batch.filter(mask)

        if filtered.num_rows == 0:
            del batch, filtered, lower_tb, mask
            gc.collect()
            continue

        # Downsample per mission in pandas (groupby is fast enough here)
        df     = filtered.to_pandas()
        pieces = []
        for _mid, grp in df.groupby("mission_id", sort=False):
            pieces.append(grp.iloc[::ds])
            missions_seen += 1

        out    = pd.concat(pieces, ignore_index=True)
        rows_written += len(out)
        writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))

        del batch, filtered, lower_tb, mask, df, out, pieces
        gc.collect()

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}: {missions_seen:,} missions, {rows_written:,} rows written")

    writer.close()
    size_mb = dst.stat().st_size / 1e6
    print(f"\nDone. {missions_seen:,} missions, {rows_written:,} rows → {dst} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
