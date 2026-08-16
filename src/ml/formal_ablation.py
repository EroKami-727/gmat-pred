"""
Formal three-way ablation: XGBoost-initial vs XGBoost-summary vs
Transformer-sequential, on identical random splits across multiple seeds.

XGBoost-initial and XGBoost-summary are cheap (seconds per fit) so they run
across 5 seeds here for genuine confidence intervals. The Transformer leg is
NOT retrained 5x in this script — on the full 80K-mission, 8-planet dataset
a single training run takes ~4-5 hours on this hardware, making a 5-seed
sweep a ~20+ hour commitment. Instead, this script reports the existing
single-run Transformer result (from src/ml/train.py's metrics JSON) as a
point estimate alongside the XGBoost confidence intervals, with that
asymmetry stated explicitly in the output — not silently treated as
equivalent evidence.

Streams the dataset once and re-splits in memory per seed (mirrors
multi_seed_calibration.py's approach) rather than re-reading the 71GB
parquet from disk for every seed.

Usage
-----
  python -m src.ml.formal_ablation \\
      --data $ORBITGUARD_DATA/missions.parquet \\
      --seeds 0 1 2 3 4 \\
      --transformer-metrics models/transformer_multiplanet/metrics_transformer_binary.json \\
      --output reports/baselines/formal_ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.ml.baselines import (
    _process_group,
    extract_initial_features,
    extract_summary_features,
)
from src.ml.calibration_utils import metrics_block
from src.ml.dataset import FEATURE_COLS


def load_all_missions(parquet_path: str, early_exit_frac: float, downsample_factor: int) -> dict[int, dict]:
    pf = pq.ParquetFile(parquet_path)
    needed = ["mission_id", "elapsed_secs", "label"] + FEATURE_COLS
    needed = [c for c in needed if c in pf.schema_arrow.names]

    missions: dict[int, dict] = {}
    current_mid = None
    current_rows: list = []

    print(f"  [Streaming {pf.metadata.num_rows:,} rows @ {early_exit_frac:.0%} exit]")
    for batch in pf.iter_batches(batch_size=500_000, columns=needed):
        df = batch.to_pandas()
        for mid, group in df.groupby("mission_id", sort=False):
            if mid != current_mid:
                if current_rows and current_mid is not None:
                    missions[int(current_mid)] = _process_group(current_rows, early_exit_frac, downsample_factor)
                current_mid = mid
                current_rows = [group]
            else:
                current_rows.append(group)
    if current_rows and current_mid is not None:
        missions[int(current_mid)] = _process_group(current_rows, early_exit_frac, downsample_factor)
    return missions


def _xgboost_fit(X_train, y_train, X_test, y_test, mode: str, seed: int) -> dict:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import RobustScaler

    extractor = extract_initial_features if mode == "initial" else extract_summary_features
    X_train_tab = extractor(X_train)
    X_test_tab = extractor(X_test)

    scaler = RobustScaler()
    X_train_tab = scaler.fit_transform(X_train_tab)
    X_test_tab = scaler.transform(X_test_tab)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        random_state=seed, n_jobs=-1, eval_metric="logloss", verbosity=0,
    )
    clf.fit(X_train_tab, y_train)
    y_prob = clf.predict_proba(X_test_tab)[:, 1]
    return metrics_block(y_test, y_prob, threshold=0.5)


def run(args: argparse.Namespace) -> dict:
    missions = load_all_missions(args.data, args.early_exit, args.downsample_factor)
    all_ids = np.array(sorted(missions.keys()))
    print(f"  Missions loaded: {len(missions)}")
    print(f"  Seeds: {args.seeds}")

    per_mode: dict[str, list[dict]] = {"initial": [], "summary": []}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        ids = all_ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * args.train_ratio)
        n_test_start = n_train + int(n * args.val_ratio)

        train_ids = ids[:n_train]
        test_ids = ids[n_test_start:]

        X_train = [missions[i]["seq"] for i in train_ids]
        y_train = np.array([missions[i]["label"] for i in train_ids], dtype=np.int64)
        X_test = [missions[i]["seq"] for i in test_ids]
        y_test = np.array([missions[i]["label"] for i in test_ids], dtype=np.int64)

        print(f"\n  Seed {seed}: train={len(train_ids)} test={len(test_ids)} success={y_test.mean():.3f}")
        for mode in ("initial", "summary"):
            metric = _xgboost_fit(X_train, y_train, X_test, y_test, mode, seed)
            per_mode[mode].append(metric)
            print(
                f"    [{mode}] F1={metric['f1']:.3f} AUC={metric['auc']:.3f} "
                f"PR-AUC={metric['pr_auc']:.3f} Brier={metric['brier_score']:.3f} ECE={metric['ece']:.3f}"
            )

    summary: dict = {"seeds": args.seeds, "xgboost_initial": {}, "xgboost_summary": {}}
    for mode, key in (("initial", "xgboost_initial"), ("summary", "xgboost_summary")):
        agg = {}
        for metric_name in ("acc", "f1", "auc", "pr_auc", "brier_score", "ece"):
            vals = [float(m[metric_name]) for m in per_mode[mode]]
            agg[metric_name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        summary[key] = agg

    # Transformer: single-run reference, not multi-seed (cost-prohibitive at this dataset size)
    if args.transformer_metrics and Path(args.transformer_metrics).exists():
        with open(args.transformer_metrics) as f:
            history = json.load(f)
        if history:
            best_epoch = min(history, key=lambda e: e["val_loss"])
            summary["transformer_sequential_single_run"] = {
                "note": "Single training run, NOT multi-seed — retraining 5x on this "
                        "80K-mission dataset would take ~20+ hours on this hardware. "
                        "Reported as a point estimate, not a confidence interval.",
                "best_epoch": best_epoch,
            }

    print(f"\n{'=' * 70}")
    print("  FORMAL ABLATION SUMMARY (5-seed XGBoost CIs; Transformer = single run)")
    print(f"{'=' * 70}")
    for key, label in (("xgboost_initial", "XGBoost initial"), ("xgboost_summary", "XGBoost summary")):
        agg = summary[key]
        print(
            f"  {label:<20} F1={agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}  "
            f"AUC={agg['auc']['mean']:.3f}+/-{agg['auc']['std']:.3f}  "
            f"ECE={agg['ece']['mean']:.3f}+/-{agg['ece']['std']:.3f}"
        )
    print(f"{'=' * 70}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal 3-way ablation with multi-seed CIs")
    parser.add_argument("--data", required=True)
    parser.add_argument("--early-exit", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--transformer-metrics", type=str, default=None,
                        help="Path to a train.py metrics_*.json for single-run reference")
    parser.add_argument("--output", default="reports/baselines/formal_ablation.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
