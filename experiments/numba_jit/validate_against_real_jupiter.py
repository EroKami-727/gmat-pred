"""
Validate the Numba JIT runner against the REAL, already-generated Jupiter
dataset (produced by the production gmat_runner.run_synthetic pipeline).

Rather than re-deriving mission parameters via generate_inputs (which can
diverge in RNG branch count when num_missions differs — success_ratio
biasing depends on n_biased = int(num_missions * success_ratio)), this
loads the EXACT MissionParams that produced the real dataset and replays
them through run_synthetic_numba directly. This is the strongest possible
check: same physics inputs, same code paths except the JIT'd substep loop.

Usage
-----
  python -m experiments.numba_jit.validate_against_real_jupiter \\
      --real-dir "$ORBITGUARD_RAW/jupiter" \\
      --n-sample 50 --time-step 54000

--real-dir defaults to <ORBITGUARD_DATA>/../jupiter, i.e. the per-planet raw
generation output that sits alongside the merged dataset. Pass it explicitly if
your layout differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import MissionParams
from src.paths import data_root
from experiments.numba_jit.runner_numba import run_synthetic_numba

# Raw per-planet generation output sits next to the merged dataset rather than
# inside it, so derive it from the same env-configurable root (src/paths.py)
# instead of hardcoding one machine's absolute path.
DEFAULT_REAL_DIR = data_root().parent / "jupiter"


NUMERIC_COLS = [
    "elapsed_secs", "elapsed_days",
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
    "earth_rmag", "luna_rmag", "ecc", "sma",
    "rel_x", "rel_y", "rel_z", "spec_energy", "fpa_deg",
    "norm_target_dist", "radial_vel", "vel_mag",
]


def load_real_rows(real_dir: Path, sim_ids: list[int]) -> dict[int, pd.DataFrame]:
    """Stream missions.parquet and pull out rows for the requested sim_ids."""
    pf = pq.ParquetFile(real_dir / "missions.parquet")
    wanted = set(sim_ids)
    found: dict[int, list] = {sid: [] for sid in sim_ids}

    for batch in pf.iter_batches(batch_size=200_000):
        df = batch.to_pandas()
        df = df[df["mission_id"].isin(wanted)]
        if len(df):
            for mid, group in df.groupby("mission_id"):
                found[int(mid)].append(group)

    return {sid: pd.concat(rows).sort_values("elapsed_secs").reset_index(drop=True)
            for sid, rows in found.items() if rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--time-step", type=float, default=54000.0)
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    print("=" * 70)
    print("  VALIDATE NUMBA JIT AGAINST REAL JUPITER DATASET")
    print("=" * 70)

    print(f"\n▸ Loading real mission_params from {real_dir}...")
    params_df = pd.read_parquet(real_dir / "mission_params.parquet")
    n_total = len(params_df)
    print(f"  {n_total} missions available")

    # Sample spread across the full range (covers both biased and full-MC regimes)
    rng = np.random.default_rng(99)
    sample_ids = sorted(rng.choice(params_df["sim_id"].values, size=args.n_sample, replace=False).tolist())
    print(f"▸ Sampling {len(sample_ids)} sim_ids spread across the dataset")

    print(f"▸ Loading real telemetry rows for sampled missions (streaming scan)...")
    real_rows = load_real_rows(real_dir, sample_ids)
    print(f"  Found {len(real_rows)} of {len(sample_ids)} requested missions")

    print(f"\n▸ Running each sampled mission through run_synthetic_numba (time_step={args.time_step})...")
    n_match = 0
    n_total_checked = 0
    max_err_overall = 0.0
    mismatches = []

    for sim_id in sample_ids:
        if sim_id not in real_rows:
            continue
        row = params_df[params_df["sim_id"] == sim_id].iloc[0].to_dict()
        params = MissionParams(**row)

        df_jit = run_synthetic_numba(params, time_step=args.time_step)
        df_real = real_rows[sim_id]

        n_total_checked += 1
        rows_match = len(df_jit) == len(df_real)
        label_real = int(df_real["label"].iloc[0])
        label_jit = int(df_jit["label"].iloc[0])
        ft_real = str(df_real["failure_type"].iloc[0])
        ft_jit = str(df_jit["failure_type"].iloc[0])

        label_match = label_real == label_jit
        ft_match = ft_real == ft_jit

        max_err = float("nan")
        if rows_match and len(df_jit) > 0:
            max_err = 0.0
            for col in NUMERIC_COLS:
                if col in df_jit.columns and col in df_real.columns:
                    diff = np.abs(df_jit[col].values - df_real[col].values)
                    max_err = max(max_err, float(np.max(diff)) if len(diff) else 0.0)
            max_err_overall = max(max_err_overall, max_err)

        ok = rows_match and label_match and ft_match
        if ok:
            n_match += 1
        else:
            mismatches.append({
                "sim_id": sim_id, "rows_real": len(df_real), "rows_jit": len(df_jit),
                "label_real": label_real, "label_jit": label_jit,
                "ft_real": ft_real, "ft_jit": ft_jit,
            })

        status = "OK" if ok else "MISMATCH"
        print(
            f"  [{status}] sim_id={sim_id:5d}  rows: {len(df_real)}=={len(df_jit)}  "
            f"label: {label_real}=={label_jit}  failure: {ft_real}=={ft_jit}  "
            f"max_abs_err={max_err:.4e}"
        )

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {n_match}/{n_total_checked} missions match real Jupiter data exactly")
    print(f"  Max absolute numeric error: {max_err_overall:.6e}")
    print(f"{'=' * 70}")

    if mismatches:
        print("\n  Mismatches:")
        for m in mismatches:
            print(f"    {m}")

    if n_match == n_total_checked and max_err_overall < 1e-3:
        print("\n  ✓ PASSED — JIT runner reproduces the real production Jupiter dataset.")
    elif n_match == n_total_checked:
        print("\n  ~ Outcomes match but numeric drift larger than expected — inspect.")
    else:
        print("\n  ✗ FAILED — outcome mismatches found vs real production data.")


if __name__ == "__main__":
    main()
