"""
Validation — confirm the Numba JIT runner produces identical results to
the production gmat_runner.run_synthetic, before trusting it for any
real dataset generation.

Usage
-----
  python experiments/numba_jit/validate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import generate_inputs
from src.data_collection.gmat_runner import run_synthetic
from experiments.numba_jit.runner_numba import run_synthetic_numba


NUMERIC_COLS = [
    "elapsed_secs", "elapsed_days",
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
    "earth_rmag", "luna_rmag", "ecc", "sma",
    "rel_x", "rel_y", "rel_z", "spec_energy", "fpa_deg",
    "norm_target_dist", "radial_vel", "vel_mag",
]


def compare_one(params, time_step: float) -> dict:
    df_orig = run_synthetic(params, time_step=time_step)
    df_jit = run_synthetic_numba(params, time_step=time_step)

    result = {
        "sim_id": params.sim_id,
        "source": params.source,
        "target": params.target,
        "n_rows_orig": len(df_orig),
        "n_rows_jit": len(df_jit),
        "label_orig": int(df_orig["label"].iloc[0]),
        "label_jit": int(df_jit["label"].iloc[0]),
        "failure_type_orig": str(df_orig["failure_type"].iloc[0]),
        "failure_type_jit": str(df_jit["failure_type"].iloc[0]),
        "min_rmag_orig": float(df_orig["min_target_rmag"].iloc[0]),
        "min_rmag_jit": float(df_jit["min_target_rmag"].iloc[0]),
    }

    result["rows_match"] = result["n_rows_orig"] == result["n_rows_jit"]
    result["label_match"] = result["label_orig"] == result["label_jit"]
    result["failure_type_match"] = result["failure_type_orig"] == result["failure_type_jit"]

    if result["rows_match"] and len(df_orig) > 0:
        max_abs_err = 0.0
        for col in NUMERIC_COLS:
            if col in df_orig.columns and col in df_jit.columns:
                diff = np.abs(df_orig[col].values - df_jit[col].values)
                max_abs_err = max(max_abs_err, float(np.max(diff)) if len(diff) else 0.0)
        result["max_abs_error"] = max_abs_err
    else:
        result["max_abs_error"] = float("nan")

    return result


def main():
    planets = ["moon", "mars", "venus", "mercury"]
    n_per_planet = 8
    time_step = 54000.0

    print(f"{'=' * 70}")
    print("  NUMBA JIT VALIDATION — comparing against production run_synthetic")
    print(f"{'=' * 70}")

    all_results = []
    for planet in planets:
        print(f"\n▸ Generating {n_per_planet} test missions: earth → {planet}")
        params_list = generate_inputs(
            source="earth", target=planet,
            num_missions=n_per_planet, seed=123, success_ratio=0.5,
        )
        for params in params_list:
            r = compare_one(params, time_step)
            all_results.append(r)
            status = "OK" if (r["rows_match"] and r["label_match"] and r["failure_type_match"]) else "MISMATCH"
            print(
                f"  [{status}] sim_id={r['sim_id']:5d}  "
                f"rows: {r['n_rows_orig']}=={r['n_rows_jit']}  "
                f"label: {r['label_orig']}=={r['label_jit']}  "
                f"failure: {r['failure_type_orig']}=={r['failure_type_jit']}  "
                f"max_abs_err={r['max_abs_error']:.6e}"
            )

    n_total = len(all_results)
    n_mismatch = sum(
        1 for r in all_results
        if not (r["rows_match"] and r["label_match"] and r["failure_type_match"])
    )
    max_err_overall = max(r["max_abs_error"] for r in all_results if not np.isnan(r["max_abs_error"]))

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {n_total - n_mismatch}/{n_total} missions match exactly")
    print(f"  Max absolute numeric error across all missions: {max_err_overall:.6e}")
    print(f"{'=' * 70}")

    # Outcome (label/failure_type) match is what matters for dataset validity.
    # Sub-millimeter position drift (<1e-3 km) is expected IEEE754 reordering
    # noise from JIT compilation over million-substep trajectories, not a bug.
    if n_mismatch == 0 and max_err_overall < 1e-3:
        print("  ✓ VALIDATION PASSED — identical outcomes, negligible numeric drift.")
    elif n_mismatch == 0:
        print("  ~ OUTCOMES MATCH but numeric drift is larger than expected — inspect further.")
    else:
        print("  ✗ VALIDATION FAILED — outcome mismatches found, do not use for real data.")


if __name__ == "__main__":
    main()
