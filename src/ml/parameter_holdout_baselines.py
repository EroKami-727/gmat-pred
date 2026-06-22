"""
Parameter-corridor holdout audit for OrbitGuard.

This tests whether baselines generalize to unseen launch-parameter bands within
each target body. Unlike leave-one-target-out, this matches the operational
mission-family screening claim: train on calibrated mission families, then test
nearby but held-out launch corridors from the same families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.grouped_baselines import _metrics, _xgboost, load_grouped_data
from src.ml.baselines import majority_class_baseline


def _assign_bins(params: pd.DataFrame, variable: str, n_bins: int) -> pd.Series:
    bins = pd.Series(index=params.index, dtype="int64")
    for target, group in params.groupby("target", sort=False):
        try:
            target_bins = pd.qcut(
                group[variable],
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            target_bins = pd.Series(np.zeros(len(group), dtype=int), index=group.index)
        bins.loc[group.index] = target_bins.astype(int)
    return bins.astype(int)


def _summarize_labels(y: np.ndarray) -> dict:
    return {
        "missions": int(len(y)),
        "success": int(y.sum()),
        "failure": int(len(y) - y.sum()),
        "success_rate": float(y.mean()) if len(y) else 0.0,
    }


def run(args: argparse.Namespace) -> list[dict]:
    missions = load_grouped_data(args.data, args.early_exit, args.downsample_factor)
    params = pd.read_parquet(args.params)
    params = params[["sim_id", "target", *args.variables]].copy()
    params["mission_id"] = params["sim_id"].astype(int)
    params = params[params["mission_id"].isin(missions.keys())].copy()

    for variable in args.variables:
        params[f"{variable}_bin"] = _assign_bins(params, variable, args.bins)

    rows_by_mid = params.set_index("mission_id").to_dict(orient="index")
    results: list[dict] = []

    for variable in args.variables:
        bin_col = f"{variable}_bin"
        available_bins = sorted(params[bin_col].dropna().astype(int).unique().tolist())
        for heldout_bin in available_bins:
            test_ids = set(params.loc[params[bin_col] == heldout_bin, "mission_id"].astype(int))
            train_items = [
                (mid, row) for mid, row in missions.items()
                if mid not in test_ids and mid in rows_by_mid
            ]
            test_items = [
                (mid, row) for mid, row in missions.items()
                if mid in test_ids and mid in rows_by_mid
            ]

            X_train = [row["seq"] for _, row in train_items]
            y_train = np.array([row["label"] for _, row in train_items], dtype=np.int64)
            X_test = [row["seq"] for _, row in test_items]
            y_test = np.array([row["label"] for _, row in test_items], dtype=np.int64)

            by_target = {}
            for _, row in test_items:
                target = row["target"]
                by_target.setdefault(target, []).append(row["label"])
            by_target = {
                target: _summarize_labels(np.array(labels, dtype=np.int64))
                for target, labels in sorted(by_target.items())
            }

            result: dict = {
                "variable": variable,
                "heldout_bin": int(heldout_bin),
                "bins": int(args.bins),
                "train": _summarize_labels(y_train),
                "test": _summarize_labels(y_test),
                "test_by_target": by_target,
                "majority_class": majority_class_baseline(y_train, y_test),
            }

            print(
                f"\nHoldout {variable} bin {heldout_bin}/{args.bins - 1} "
                f"test={len(y_test)} success={y_test.mean():.3f}"
            )
            for mode in args.feature_modes:
                print(f"  XGBoost {mode}...")
                metric = _xgboost(X_train, y_train, X_test, y_test, mode, args.seed)
                result[f"xgboost_{mode}"] = metric
                if "error" in metric:
                    print(f"    ERROR: {metric['error']}")
                    continue
                tuned = metric["test_at_train_threshold"]
                print(
                    f"    @0.5 Acc={metric['acc']:.2%} F1={metric['f1']:.3f} "
                    f"AUC={metric['auc']:.3f} PR-AUC={metric['pr_auc']:.3f} "
                    f"Brier={metric['brier_score']:.3f} ECE={metric['ece']:.3f} | train-thr "
                    f"Acc={tuned['acc']:.2%} F1={tuned['f1']:.3f}"
                )

            results.append(result)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter-corridor holdout baseline audit")
    parser.add_argument("--data", required=True, help="Path to missions.parquet")
    parser.add_argument("--params", required=True, help="Path to mission_params.parquet")
    parser.add_argument("--early-exit", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--variables", nargs="+", default=["TOI_V", "AOP"])
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["summary", "initial_no_context"],
        choices=["summary", "endpoints", "initial", "initial_no_context"],
    )
    parser.add_argument(
        "--output",
        default="reports/baselines/parameter_holdout_exit40_ds10.json",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
