"""
Multi-seed confidence intervals for the leave-one-target-out audit.

The train/test partition for LOTO is deterministic (defined by target
identity, not by seed) — only the XGBoost model's random_state varies.
Looping seeds here answers a different question than multi_seed.py does
for the Transformer: not "does the split matter" but "is this held-out
result stable across model-training randomness, or a training-noise
artifact." Loads the grouped mission data once and reuses it across all
seeds x targets x feature-modes to avoid repeated expensive streaming.

Usage
-----
  python -m src.ml.multi_seed_grouped \\
      --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \\
      --seeds 0 1 2 3 4 \\
      --feature-modes summary initial_no_context \\
      --output reports/baselines/leave_one_target_out_multiseed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ml.baselines import majority_class_baseline
from src.ml.grouped_baselines import _xgboost, load_grouped_data


def run(args: argparse.Namespace) -> dict:
    missions = load_grouped_data(args.data, args.early_exit, args.downsample_factor)
    targets = sorted({row["target"] for row in missions.values()})
    print(f"  Missions loaded: {len(missions)}")
    print(f"  Targets: {', '.join(targets)}")
    print(f"  Seeds: {args.seeds}")

    per_target: dict[str, dict] = {}

    for heldout in targets:
        train = [row for row in missions.values() if row["target"] != heldout]
        test = [row for row in missions.values() if row["target"] == heldout]

        X_train = [row["seq"] for row in train]
        y_train = np.array([row["label"] for row in train], dtype=np.int64)
        X_test = [row["seq"] for row in test]
        y_test = np.array([row["label"] for row in test], dtype=np.int64)

        print(f"\nHeld out: {heldout}  test_success={y_test.mean():.3f}")

        target_result: dict = {
            "heldout_target": heldout,
            "train_missions": int(len(y_train)),
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
                print(
                    f"  [{mode} seed={seed}] F1={metric['f1']:.3f} AUC={metric['auc']:.3f} "
                    f"PR-AUC={metric['pr_auc']:.3f} Brier={metric['brier_score']:.3f} ECE={metric['ece']:.3f}"
                )

            if per_seed:
                agg = {}
                for key in ("acc", "f1", "auc", "pr_auc", "brier_score", "ece"):
                    vals = [float(m[key]) for m in per_seed]
                    agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
                target_result[f"xgboost_{mode}"] = agg
                print(
                    f"  [{mode}] AGGREGATE: F1={agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}  "
                    f"AUC={agg['auc']['mean']:.3f}+/-{agg['auc']['std']:.3f}"
                )

        per_target[heldout] = target_result
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(per_target, indent=2), encoding="utf-8")

    return per_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed leave-one-target-out audit")
    parser.add_argument("--data", required=True, help="Path to missions.parquet")
    parser.add_argument("--early-exit", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["summary", "initial_no_context"],
        choices=["summary", "endpoints", "initial", "initial_no_context"],
    )
    parser.add_argument(
        "--output",
        default="reports/baselines/leave_one_target_out_multiseed.json",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
