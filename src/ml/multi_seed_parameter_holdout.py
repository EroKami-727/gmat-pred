"""
Multi-seed confidence intervals for the parameter-corridor holdout audit.

Mirrors multi_seed_grouped.py's reasoning: the bin assignment (quantile-based
per target) is deterministic, not seed-dependent — only the XGBoost model's
random_state varies across seeds. Loads mission/parameter data once and
reuses it across all seeds x bins x feature-modes.

Usage
-----
  python -m src.ml.multi_seed_parameter_holdout \\
      --data $ORBITGUARD_DATA/missions.parquet \\
      --params $ORBITGUARD_DATA/mission_params.parquet \\
      --seeds 0 1 2 3 4 \\
      --variables TOI_V AOP \\
      --output reports/baselines/parameter_holdout_multiseed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.baselines import majority_class_baseline
from src.ml.grouped_baselines import _xgboost, load_grouped_data
from src.ml.parameter_holdout_baselines import _assign_bins


def run(args: argparse.Namespace) -> list[dict]:
    missions = load_grouped_data(args.data, args.early_exit, args.downsample_factor)
    params = pd.read_parquet(args.params)
    params = params[["sim_id", "target", *args.variables]].copy()
    params["mission_id"] = params["sim_id"].astype(int)
    params = params[params["mission_id"].isin(missions.keys())].copy()

    for variable in args.variables:
        params[f"{variable}_bin"] = _assign_bins(params, variable, args.bins)

    print(f"  Missions loaded: {len(missions)}")
    print(f"  Seeds: {args.seeds}")

    results: list[dict] = []

    for variable in args.variables:
        bin_col = f"{variable}_bin"
        available_bins = sorted(params[bin_col].dropna().astype(int).unique().tolist())

        for heldout_bin in available_bins:
            test_ids = set(params.loc[params[bin_col] == heldout_bin, "mission_id"].astype(int))
            train_items = [(mid, row) for mid, row in missions.items() if mid not in test_ids]
            test_items = [(mid, row) for mid, row in missions.items() if mid in test_ids]

            X_train = [row["seq"] for _, row in train_items]
            y_train = np.array([row["label"] for _, row in train_items], dtype=np.int64)
            X_test = [row["seq"] for _, row in test_items]
            y_test = np.array([row["label"] for _, row in test_items], dtype=np.int64)

            print(
                f"\nHoldout {variable} bin {heldout_bin}/{args.bins - 1} "
                f"test={len(y_test)} success={y_test.mean():.3f}"
            )

            bin_result: dict = {
                "variable": variable,
                "heldout_bin": int(heldout_bin),
                "bins": int(args.bins),
                "test_missions": int(len(y_test)),
                "test_success_rate": float(y_test.mean()),
                "majority_class": majority_class_baseline(y_train, y_test),
                "seeds": args.seeds,
            }

            for mode in args.feature_modes:
                per_seed = []
                for seed in args.seeds:
                    metric = _xgboost(X_train, y_train, X_test, y_test, mode, seed)
                    if "error" in metric:
                        print(f"  [{mode} seed={seed}] ERROR: {metric['error']}")
                        continue
                    per_seed.append(metric)

                if per_seed:
                    agg = {}
                    for key in ("acc", "f1", "auc", "pr_auc", "brier_score", "ece"):
                        vals = [float(m[key]) for m in per_seed]
                        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
                    bin_result[f"xgboost_{mode}"] = agg
                    print(
                        f"  [{mode}] AGGREGATE: F1={agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}  "
                        f"AUC={agg['auc']['mean']:.3f}+/-{agg['auc']['std']:.3f}  "
                        f"ECE={agg['ece']['mean']:.3f}+/-{agg['ece']['std']:.3f}"
                    )

            results.append(bin_result)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed parameter-corridor holdout audit")
    parser.add_argument("--data", required=True, help="Path to missions.parquet")
    parser.add_argument("--params", required=True, help="Path to mission_params.parquet")
    parser.add_argument("--early-exit", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--variables", nargs="+", default=["TOI_V", "AOP"])
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["summary", "initial_no_context"],
        choices=["summary", "endpoints", "initial", "initial_no_context"],
    )
    parser.add_argument(
        "--output",
        default="reports/baselines/parameter_holdout_multiseed.json",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
