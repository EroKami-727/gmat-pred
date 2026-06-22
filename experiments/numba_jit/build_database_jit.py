"""
Numba-JIT Database Builder — Monte Carlo Orchestrator
=======================================================
Near-identical copy of src/data_collection/build_database.py, with the
single substantive change being the propagation function:

    run_func = partial(run_synthetic_numba, time_step=time_step)

instead of

    run_func = partial(run_synthetic, time_step=time_step)

run_synthetic_numba (experiments/numba_jit/runner_numba.py) reuses every
other piece of production logic (parameter generation, outcome
classification, row schema) unchanged — only the RK4 substep loop is
JIT-compiled. See experiments/numba_jit/README.md for validation evidence
before trusting output from this script for paper-grade data.

Usage
-----
  python -m experiments.numba_jit.build_database_jit \\
      --source earth --target neptune \\
      --num-missions 10000 --output-dir data/neptune \\
      --seed 42 --success-ratio 0.35 --batch-size 10 \\
      --workers 14 --time-step 54000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import generate_inputs, PLANET_REGISTRY
from experiments.numba_jit.runner_numba import run_synthetic_numba


def _build_summary(missions_path: Path, summary_path: Path) -> None:
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


def build_database_jit(
    num_missions: int = 200,
    time_step: float = 1800.0,
    output_dir: str = "data",
    seed: int = 42,
    source: str = "earth",
    target: str = "moon",
    success_ratio: float = 0.0,
    batch_size: int = 500,
    allow_dense_interplanetary: bool = False,
    workers: int | None = None,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    missions_path = out_path / "missions.parquet"
    params_path = out_path / "mission_params.parquet"
    summary_path = out_path / "summary.parquet"

    if target != "moon" and time_step < 900.0 and not allow_dense_interplanetary:
        raise ValueError(
            "Refusing dense interplanetary telemetry. Use --time-step 900, "
            "1800, 3600, or 54000, or pass --allow-dense-interplanetary."
        )

    print("=" * 60)
    print("  Numba-JIT Monte Carlo Database Builder")
    print("=" * 60)
    print(f"  Mission       : {source.capitalize()} → {target.capitalize()}")
    print(f"  Missions      : {num_missions}")
    print(f"  Success ratio : {success_ratio:.0%}" if success_ratio > 0 else "  Success ratio : Pure MC (unbiased)")
    print(f"  Batch size    : {batch_size}")
    print(f"  Output dir    : {out_path.resolve()}")
    print(f"  Engine        : Numba JIT (experiments/numba_jit)")
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
    params_df.to_parquet(params_path, index=False)
    print(f"  ✓ Saved parameters → {params_path}")

    cores = workers if workers is not None else cpu_count()
    cores = max(1, int(cores))
    print(f"\n▸ Running {num_missions} simulations using {cores} cores (Numba JIT 3-Body RK4)...")

    print("▸ Warming up JIT compilation on main process...")
    warmup_params = missions[0]
    run_synthetic_numba(warmup_params, time_step=time_step)
    print("  Done.")

    t0 = time.time()

    current_batch = []
    batch_count = 0
    total_rows = 0
    successes = 0
    writer = None

    run_func = partial(run_synthetic_numba, time_step=time_step)

    def consume_results(results_iter):
        nonlocal current_batch, batch_count, total_rows, successes, writer
        for df in results_iter:
            current_batch.append(df)
            successes += int(df["label"].iloc[0] == 1)

            if len(current_batch) >= batch_size:
                batch_df = pd.concat(current_batch, ignore_index=True)
                total_rows += len(batch_df)
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
    print(f"\n▸ Simulations complete. Data streamed directly to missions.parquet in {batch_count} batches.")

    print("▸ Generating summary index...")
    _build_summary(missions_path, summary_path)

    print()
    print("=" * 60)
    print("  BUILD COMPLETE (Numba JIT)")
    print("=" * 60)
    print(f"  Mission type   : {source.capitalize()} → {target.capitalize()}")
    print(f"  Missions       : {num_missions}")
    print(f"  ✓ Successes    : {successes}  ({100*successes/num_missions:.1f}%)")
    print(f"  ✗ Failures     : {num_missions - successes}  ({100*(num_missions-successes)/num_missions:.1f}%)")
    print()
    print(f"  Rows written   : {total_rows:,}")
    print(f"  Data file      : {missions_path}  ({missions_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Time elapsed   : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    available = list(PLANET_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Build Monte Carlo trajectory database using the Numba JIT propagator."
    )
    parser.add_argument("--num-missions", type=int, default=200)
    parser.add_argument("--time-step", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str, default="earth", choices=available)
    parser.add_argument("--target", type=str, default="moon", choices=available)
    parser.add_argument("--success-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--allow-dense-interplanetary", action="store_true")
    parser.add_argument("--workers", "--cores", dest="workers", type=int, default=None)
    args = parser.parse_args()

    build_database_jit(
        num_missions=args.num_missions,
        time_step=args.time_step,
        output_dir=args.output_dir,
        seed=args.seed,
        source=args.source,
        target=args.target,
        success_ratio=args.success_ratio,
        batch_size=args.batch_size,
        allow_dense_interplanetary=args.allow_dense_interplanetary,
        workers=args.workers,
    )
