"""Stream-downsample a mission parquet dataset by timestep stride.

This keeps every Nth row within each mission after sorting by elapsed time,
while copying mission-level companion files unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def downsample_dataset(src_dir: Path, out_dir: Path, factor: int) -> None:
    if factor < 1:
        raise ValueError("--factor must be >= 1")

    missions_path = src_dir / "missions.parquet"
    if not missions_path.exists():
        raise FileNotFoundError(f"{missions_path} not found")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "missions.parquet"

    pf = pq.ParquetFile(missions_path)
    schema = pf.schema_arrow
    total_in = pf.metadata.num_rows
    total_out = 0

    current_mid = None
    current_parts: list[pd.DataFrame] = []

    def flush_current(writer: pq.ParquetWriter) -> None:
        nonlocal current_parts, total_out
        if not current_parts:
            return
        mission_df = pd.concat(current_parts, ignore_index=True)
        if "elapsed_secs" in mission_df.columns:
            mission_df = mission_df.sort_values("elapsed_secs")
        mission_df = mission_df.iloc[::factor].reset_index(drop=True)
        writer.write_table(pa.Table.from_pandas(mission_df, schema=schema, preserve_index=False))
        total_out += len(mission_df)
        current_parts = []

    with pq.ParquetWriter(out_path, schema, compression="snappy") as writer:
        for batch in tqdm(pf.iter_batches(batch_size=500_000), total=None, desc="  Downsample", unit="batch"):
            df = batch.to_pandas()
            for mid, group in df.groupby("mission_id", sort=False):
                if current_mid is None:
                    current_mid = mid
                if mid != current_mid:
                    flush_current(writer)
                    current_mid = mid
                current_parts.append(group)
        flush_current(writer)

    for name in ("summary.parquet", "mission_params.parquet"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    print(f"Downsampled {src_dir} -> {out_dir}")
    print(f"Rows: {total_in:,} -> {total_out:,} ({total_out / total_in:.2%})")
    print(f"Output: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample a mission dataset by per-mission timestep stride")
    parser.add_argument("--src", required=True, help="Source dataset directory")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--factor", type=int, default=15, help="Keep every Nth timestep per mission")
    args = parser.parse_args()

    downsample_dataset(Path(args.src), Path(args.out), args.factor)


if __name__ == "__main__":
    main()
