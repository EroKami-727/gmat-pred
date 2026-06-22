"""
Leave-one-target-out baseline audit for OrbitGuard.

This evaluates whether high random-split scores generalize across target
regimes. It trains on seven target bodies and tests on the held-out body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

from src.ml.baselines import (
    extract_endpoint_features,
    extract_initial_features,
    extract_initial_no_context_features,
    extract_summary_features,
    majority_class_baseline,
)
from src.ml.calibration_utils import best_f1_threshold, metrics_block
from src.ml.dataset import FEATURE_COLS


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Backward-compatible thin wrapper — y_pred is unused, kept for call-site
    compatibility; threshold is implied 0.5 to match prior behaviour, now also
    carrying pr_auc/brier_score/ece/confusion_matrix via metrics_block."""
    return metrics_block(y_true, y_prob, threshold=0.5)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    return best_f1_threshold(y_true, y_prob)


def _process_group(rows: list, early_exit_frac: float, downsample_factor: int) -> dict:
    import pandas as pd

    df = pd.concat(rows).sort_values("elapsed_secs")
    df = df.iloc[::downsample_factor].reset_index(drop=True)
    if early_exit_frac < 1.0:
        keep_n = max(1, int(len(df) * early_exit_frac))
        df = df.iloc[:keep_n]

    first = df.iloc[0]
    return {
        "seq": df[FEATURE_COLS].values.astype(np.float32),
        "label": int(first["label"]),
        "target": str(first["target_body"]),
    }


def load_grouped_data(
    parquet_path: str | Path,
    early_exit_frac: float,
    downsample_factor: int,
) -> dict[int, dict]:
    pf = pq.ParquetFile(parquet_path)
    needed = ["mission_id", "elapsed_secs", "target_body", "label"] + FEATURE_COLS
    needed = [col for col in needed if col in pf.schema_arrow.names]

    missions: dict[int, dict] = {}
    current_mid = None
    current_rows: list = []

    print(f"  [Streaming {pf.metadata.num_rows:,} rows @ {early_exit_frac:.0%} exit]")
    for batch in pf.iter_batches(batch_size=500_000, columns=needed):
        df = batch.to_pandas()
        for mid, group in df.groupby("mission_id", sort=False):
            if mid != current_mid:
                if current_rows and current_mid is not None:
                    missions[int(current_mid)] = _process_group(
                        current_rows, early_exit_frac, downsample_factor
                    )
                current_mid = mid
                current_rows = [group]
            else:
                current_rows.append(group)

    if current_rows and current_mid is not None:
        missions[int(current_mid)] = _process_group(
            current_rows, early_exit_frac, downsample_factor
        )

    return missions


def _xgboost(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    mode: str,
    seed: int,
) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "xgboost not installed"}

    extractors = {
        "summary": extract_summary_features,
        "endpoints": extract_endpoint_features,
        "initial": extract_initial_features,
        "initial_no_context": extract_initial_no_context_features,
    }
    extractor = extractors[mode]

    X_train_tab = extractor(X_train)
    X_test_tab = extractor(X_test)

    scaler = RobustScaler()
    X_train_tab = scaler.fit_transform(X_train_tab)
    X_test_tab = scaler.transform(X_test_tab)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train_tab, y_train)
    train_prob = clf.predict_proba(X_train_tab)[:, 1]
    y_prob = clf.predict_proba(X_test_tab)[:, 1]

    threshold, train_f1 = _best_f1_threshold(y_train, train_prob)

    result = metrics_block(y_test, y_prob, threshold=0.5)
    result["train_tuned_threshold"] = threshold
    result["train_f1_at_threshold"] = train_f1
    result["test_at_train_threshold"] = metrics_block(y_test, y_prob, threshold=threshold)
    result["test_probability_summary"] = {
        "min": float(np.min(y_prob)),
        "p25": float(np.quantile(y_prob, 0.25)),
        "median": float(np.median(y_prob)),
        "p75": float(np.quantile(y_prob, 0.75)),
        "max": float(np.max(y_prob)),
    }
    result["feature_mode"] = mode
    return result


def run(args: argparse.Namespace) -> list[dict]:
    missions = load_grouped_data(args.data, args.early_exit, args.downsample_factor)
    targets = sorted({row["target"] for row in missions.values()})
    results = []

    print(f"  Missions loaded: {len(missions)}")
    print(f"  Targets: {', '.join(targets)}")

    for heldout in targets:
        train = [row for row in missions.values() if row["target"] != heldout]
        test = [row for row in missions.values() if row["target"] == heldout]

        X_train = [row["seq"] for row in train]
        y_train = np.array([row["label"] for row in train], dtype=np.int64)
        X_test = [row["seq"] for row in test]
        y_test = np.array([row["label"] for row in test], dtype=np.int64)

        row: dict = {
            "heldout_target": heldout,
            "train_missions": int(len(y_train)),
            "test_missions": int(len(y_test)),
            "train_success_rate": float(y_train.mean()),
            "test_success_rate": float(y_test.mean()),
            "majority_class": majority_class_baseline(y_train, y_test),
        }

        print(f"\nHeld out: {heldout}  test_success={y_test.mean():.3f}")
        for mode in args.feature_modes:
            print(f"  XGBoost {mode}...")
            row[f"xgboost_{mode}"] = _xgboost(
                X_train, y_train, X_test, y_test, mode, args.seed
            )
            r = row[f"xgboost_{mode}"]
            if "error" in r:
                print(f"    ERROR: {r['error']}")
            else:
                tuned = r["test_at_train_threshold"]
                print(
                    f"    @0.5 Acc={r['acc']:.2%} F1={r['f1']:.3f} AUC={r['auc']:.3f} "
                    f"PR-AUC={r['pr_auc']:.3f} Brier={r['brier_score']:.3f} ECE={r['ece']:.3f} | "
                    f"train-thr={r['train_tuned_threshold']:.4f} "
                    f"Acc={tuned['acc']:.2%} F1={tuned['f1']:.3f}"
                )

        results.append(row)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-target-out baseline audit")
    parser.add_argument("--data", required=True, help="Path to missions.parquet")
    parser.add_argument("--early-exit", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["summary", "initial_no_context"],
        choices=["summary", "endpoints", "initial", "initial_no_context"],
    )
    parser.add_argument(
        "--output",
        default="reports/baselines/leave_one_target_out_exit40_ds10.json",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
