"""
Database Builder — Monte Carlo Orchestrator (Memory Optimized)
==============================================================
Generates N dispersed Monte Carlo scenarios for any planet pair,
runs them through the 3-body physics propagator, and builds the
Parquet database.

Optimized for high-volume runs (5000+ missions) by saving results
in batches to disk, avoiding RAM exhaustion during concatenation.

Supports multi-planet mesh topology via --source / --target flags.

Append Mode (--append)
-----------------------
When --append is passed and an existing missions.parquet is found
in --output-dir, new missions are generated with sim_ids starting
from the current max + 1, then merged into the existing file using
streaming PyArrow writes (no RAM blowup). summary.parquet is
regenerated afterwards.

Time Step & Integration Accuracy
----------------------------------
``--time-step`` controls recorded telemetry cadence, not the RK4 sub-step.
Moon propagation uses sub-steps up to 10 seconds. Interplanetary propagation
uses 30-second sub-steps near either body, 300 seconds during target approach,
and 900 seconds during deep-space cruise.

The CLI default is 1800 seconds for exploratory runs. Production datasets must
set their validated cadence explicitly. The current research corpus records
Moon telemetry at 900 seconds and interplanetary telemetry at 54000 seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import generate_inputs, PLANET_REGISTRY
from src.data_collection.gmat_runner import run_synthetic

from multiprocessing import Pool, cpu_count
from functools import partial


# ---------------------------------------------------------------------------
# Summary index builder (shared between build and append paths)
# ---------------------------------------------------------------------------

def _build_summary(missions_path: Path, summary_path: Path) -> None:
    """Stream-scan missions.parquet to extract one row per mission."""
    pf = pq.ParquetFile(missions_path)
    cols = ["mission_id", "label", "failure_type", "min_target_rmag"]
    cols = [c for c in cols if c in pf.schema_arrow.names]

    sum_rows = []
    last_mid = None
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas(categories=["failure_type"])
        mids = df["mission_id"].values
        changes = mids[1:] != mids[:-1]
        indices = [0] + (np.where(changes)[0] + 1).tolist()
        for i in indices:
            mid = int(mids[i])
            if mid != last_mid:
                sum_rows.append(df.iloc[i].to_dict())
                last_mid = mid

    pd.DataFrame(sum_rows).to_parquet(summary_path, index=False)
    print(f"  ✓ summary.parquet written  ({len(sum_rows)} missions)")


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def build_database(
    num_missions: int = 200,
    time_step: float = 1800.0,
    output_dir: str = "data",
    seed: int = 42,
    source: str = "earth",
    target: str = "moon",
    success_ratio: float = 0.0,
    batch_size: int = 500,
    append: bool = False,
    allow_dense_interplanetary: bool = False,
    workers: int | None = None,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    missions_path = out_path / "missions.parquet"
    params_path   = out_path / "mission_params.parquet"
    summary_path  = out_path / "summary.parquet"

    if target != "moon" and time_step < 900.0 and not allow_dense_interplanetary:
        raise ValueError(
            "Refusing dense interplanetary telemetry. Earth-Mars at 60s creates "
            "~447k rows per mission. Use --time-step 900, 1800, or 3600, or pass "
            "--allow-dense-interplanetary for a small validation-only run."
        )

    # ── Determine starting sim_id for append mode ──────────────────────────
    id_offset = 0
    if append and missions_path.exists():
        existing_summary = out_path / "summary.parquet"
        if existing_summary.exists():
            existing_count = len(pd.read_parquet(existing_summary, columns=["mission_id"]))
        else:
            # Slow path: scan the big file
            pf = pq.ParquetFile(missions_path)
            last_id = 0
            for batch in pf.iter_batches(batch_size=500_000, columns=["mission_id"]):
                last_id = max(last_id, batch.column("mission_id").to_pylist()[-1])
            existing_count = last_id + 1
        id_offset = existing_count
        print(f"  ▸ Append mode: existing missions = {id_offset}, new IDs start at {id_offset}")
    elif append and not missions_path.exists():
        print("  ⚠  --append specified but no missions.parquet found. Running in fresh mode.")
        append = False

    # Temporary directory for batches
    # Only needed for append mode, where we must stage new data before merging.
    tmp_path = out_path / "_tmp_batches"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if append:
        tmp_path.mkdir(parents=True)

    print("=" * 60)
    print("  GMAT Monte Carlo Database Builder (Memory-Optimized)")
    print("=" * 60)
    print(f"  Mode          : {'APPEND' if append else 'FRESH'}")
    print(f"  Mission       : {source.capitalize()} → {target.capitalize()}")
    print(f"  Missions      : {num_missions}  (IDs {id_offset} – {id_offset + num_missions - 1})")
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

    # Override sim_ids for append mode so they remain globally unique
    if id_offset > 0:
        for m in missions:
            m.sim_id += id_offset

    params_df = pd.DataFrame([m.to_dict() for m in missions])
    if append and params_path.exists():
        # Append new rows to existing params file
        old_params = pd.read_parquet(params_path)
        params_df = pd.concat([old_params, params_df], ignore_index=True)
    params_df.to_parquet(params_path, index=False)
    print(f"  ✓ Saved parameters → {params_path}")

    cores = workers if workers is not None else cpu_count()
    cores = max(1, int(cores))
    print(f"\n▸ Running {num_missions} simulations using {cores} cores (3-Body RK4)...")
    t0 = time.time()

    current_batch = []
    batch_count = 0
    total_rows = 0
    successes = 0
    writer = None

    run_func = partial(run_synthetic, time_step=time_step)

    def consume_results(results_iter):
        nonlocal current_batch, batch_count, total_rows, successes, writer
        for df in results_iter:
            current_batch.append(df)
            successes += int(df["label"].iloc[0] == 1)

            if len(current_batch) >= batch_size:
                batch_df = pd.concat(current_batch, ignore_index=True)
                total_rows += len(batch_df)

                if append:
                    batch_file = tmp_path / f"batch_{batch_count:04d}.parquet"
                    batch_df.to_parquet(batch_file, index=False)
                else:
                    table = pa.Table.from_pandas(batch_df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(missions_path, table.schema, compression="snappy")
                    writer.write_table(table)

                current_batch = []
                batch_count += 1
                pbar.set_postfix({"batch": batch_count, "saved_rows": f"{total_rows/1e6:.1f}M"})

        if current_batch:
            batch_df = pd.concat(current_batch, ignore_index=True)
            total_rows += len(batch_df)

            if append:
                batch_file = tmp_path / f"batch_{batch_count:04d}.parquet"
                batch_df.to_parquet(batch_file, index=False)
            else:
                table = pa.Table.from_pandas(batch_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(missions_path, table.schema, compression="snappy")
                writer.write_table(table)

            batch_count += 1

    if cores == 1:
        pbar = tqdm((run_func(m) for m in missions), total=num_missions, desc="  Simulating", unit="mission")
        consume_results(pbar)
    else:
        try:
            with Pool(processes=cores) as pool:
                pbar = tqdm(pool.imap_unordered(run_func, missions), total=num_missions, desc="  Simulating", unit="mission")
                consume_results(pbar)
        except PermissionError as exc:
            print(f"  ⚠ Multiprocessing unavailable ({exc}). Falling back to one worker.")
            pbar = tqdm((run_func(m) for m in missions), total=num_missions, desc="  Simulating", unit="mission")
            consume_results(pbar)

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    all_batch_files = sorted(list(tmp_path.glob("*.parquet"))) if append else []

    # ── NEW: Consolidate (merge with existing if appending) ────────────────
    if append and missions_path.exists():
        print(f"\n▸ Merging new {batch_count} batches into existing missions.parquet...")
        merged_path = out_path / "_missions_merged.parquet"

        # Get schema from first batch (new data schema must match existing)
        first_table = pq.read_table(all_batch_files[0])
        schema = first_table.schema

        with pq.ParquetWriter(merged_path, schema, compression="snappy") as writer:
            # 1. Stream the existing file
            existing_pf = pq.ParquetFile(missions_path)
            for batch in tqdm(
                existing_pf.iter_batches(batch_size=1_000_000),
                desc="  Copying existing",
                unit="batch",
            ):
                writer.write_table(batch.cast(schema))

            # 2. Append new batches
            for bfile in tqdm(all_batch_files, desc="  Appending new  ", unit="batch"):
                writer.write_table(pq.read_table(bfile))

        # Atomic replace
        missions_path.unlink()
        merged_path.rename(missions_path)
        print(f"  ✓ Merged → {missions_path}")

    else:
        print(f"\n▸ Simulations complete. Data streamed directly to missions.parquet in {batch_count} batches.")

    if append and tmp_path.exists():
        shutil.rmtree(tmp_path)

    print(f"▸ Generating summary index...")
    _build_summary(missions_path, summary_path)

    # ── Final stats ────────────────────────────────────────────────────────
    total_missions_now = id_offset + num_missions if append else num_missions

    print()
    print("=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)
    print(f"  Mode           : {'APPEND' if append else 'FRESH'}")
    print(f"  Mission type   : {source.capitalize()} → {target.capitalize()}")
    print(f"  New missions   : {num_missions}")
    print(f"  Total missions : {total_missions_now}")
    print(f"  ✓ New successes: {successes}  ({100*successes/num_missions:.1f}%)")
    print(f"  ✗ New failures : {num_missions - successes}  ({100*(num_missions-successes)/num_missions:.1f}%)")
    print()
    print(f"  New rows added : {total_rows:,}")
    print(f"  Data file      : {missions_path}  ({missions_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Time elapsed   : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    available = list(PLANET_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Build Monte Carlo trajectory database for any planet pair."
    )
    parser.add_argument("--num-missions",  type=int,   default=200)
    parser.add_argument("--time-step",     type=float, default=1800.0,
                        help="Seconds between recorded data points (default 1800s = 30 min). "
                             "Does not affect RK4 integration accuracy — adaptive sub-stepping "
                             "uses finer steps near the source and target bodies.")
    parser.add_argument("--output-dir",    type=str,   default="data")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--source",        type=str,   default="earth", choices=available,
                        help="Source body for the transfer")
    parser.add_argument("--target",        type=str,   default="moon",  choices=available,
                        help="Target body for the transfer")
    parser.add_argument("--success-ratio", type=float, default=0.0,
                        help="Fraction of missions to bias toward success (0.0-1.0)")
    parser.add_argument("--batch-size",    type=int,   default=500,
                        help="Save results to disk every N missions to save RAM")
    parser.add_argument("--append",        action="store_true",
                        help="Append new missions to an existing missions.parquet in --output-dir")
    parser.add_argument("--allow-dense-interplanetary", action="store_true",
                        help="Allow <900s telemetry for non-Moon targets. Use only for small validation runs.")
    parser.add_argument("--workers", "--cores", dest="workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores). "
                             "Reduce for interplanetary targets to limit peak RAM.")
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
        append=args.append,
        allow_dense_interplanetary=args.allow_dense_interplanetary,
        workers=args.workers,
    )
