"""
OrbitGuard Baselines — Non-Neural Reference Classifiers
=======================================================
Establishes a performance floor for the paper comparison table.

Baselines
---------
1. MajorityClass      — always predicts the majority class (failure)
2. EnergyThreshold    — if spec_energy at exit point is above a fitted
                        threshold, predict failure (pure physics heuristic)
3. XGBoostSummary     — XGBoost on hand-crafted trajectory summary statistics
                        (mean/std/min/max/first/last × 13 features + length)

Usage
-----
  python -m src.ml.baselines \\
      --data data/merged/missions.parquet \\
      --exit-fracs 0.1 0.2 0.3 0.4 0.6 1.0 \\
      --output reports/baselines/baseline_results.json

Output
------
  reports/baselines/baseline_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.ml.dataset import FEATURE_COLS

SPEC_ENERGY_IDX = FEATURE_COLS.index("spec_energy")


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def _process_group(rows: list, early_exit_frac: float, downsample_factor: int) -> dict:
    import pandas as pd
    df = pd.concat(rows).sort_values("elapsed_secs")
    label = int(df.iloc[0]["label"])
    df = df.iloc[::downsample_factor].reset_index(drop=True)
    if early_exit_frac < 1.0:
        keep_n = max(1, int(len(df) * early_exit_frac))
        df = df.iloc[:keep_n]
    seq = df[FEATURE_COLS].values.astype(np.float32)
    return {"seq": seq, "label": label}


def load_split_data(
    parquet_path: str | Path,
    early_exit_frac: float,
    downsample_factor: int = 15,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list, np.ndarray, list, np.ndarray, list, np.ndarray]:
    """
    Stream-load trajectories from Parquet and split into train/val/test.

    Uses the same mission_id-based split strategy as create_dataloaders to
    ensure results are comparable with the transformer evaluation.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    where X_* are lists of (T, 13) float32 arrays (variable-length trajectories).
    """
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
                    missions[current_mid] = _process_group(current_rows, early_exit_frac, downsample_factor)
                current_mid = mid
                current_rows = [group]
            else:
                current_rows.append(group)
    if current_rows and current_mid is not None:
        missions[current_mid] = _process_group(current_rows, early_exit_frac, downsample_factor)

    # Match create_dataloaders exactly: use Parquet mission_id unique order,
    # then apply the same seeded shuffle. Sorting here changes the split.
    all_ids = pf.read(columns=["mission_id"])["mission_id"].unique().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    def _gather(ids):
        seqs = [missions[m]["seq"] for m in ids]
        labels = np.array([missions[m]["label"] for m in ids], dtype=np.int64)
        return seqs, labels

    return (
        *_gather(all_ids[:n_train]),
        *_gather(all_ids[n_train:n_train + n_val]),
        *_gather(all_ids[n_train + n_val:]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering for Tabular Baselines
# ═══════════════════════════════════════════════════════════════════════════

def extract_summary_features(sequences: list[np.ndarray]) -> np.ndarray:
    """
    Convert variable-length trajectories into fixed-size tabular vectors.

    For each of the 13 physics features: mean, std, min, max, first, last.
    Plus total sequence length. Total: 13 × 6 + 1 = 79 features per mission.
    """
    rows = []
    for seq in sequences:
        row = []
        for col_idx in range(seq.shape[1]):
            col = seq[:, col_idx]
            row.extend([col.mean(), col.std(), col.min(), col.max(), col[0], col[-1]])
        row.append(float(seq.shape[0]))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def extract_endpoint_features(sequences: list[np.ndarray]) -> np.ndarray:
    """
    Convert trajectories into first/last values only.

    This deliberately excludes mean/std/min/max and sequence length so the
    restricted baseline cannot exploit aggregate trajectory-shape statistics.
    Total: 13 features x 2 endpoints = 26 features per mission.
    """
    return np.array(
        [
            np.concatenate([seq[0], seq[-1]])
            for seq in sequences
        ],
        dtype=np.float32,
    )


def extract_initial_features(sequences: list[np.ndarray]) -> np.ndarray:
    """Use only the first observed telemetry row for each mission."""
    return np.array([seq[0] for seq in sequences], dtype=np.float32)


def extract_initial_no_context_features(sequences: list[np.ndarray]) -> np.ndarray:
    """Use the first row without mu_ratio, soi_ratio, or dist_ratio."""
    context_start = FEATURE_COLS.index("mu_ratio")
    return np.array([seq[0, :context_start] for seq in sequences], dtype=np.float32)


def _feature_names(mode: str) -> list[str]:
    if mode == "initial_no_context":
        return [f"{name}_first" for name in FEATURE_COLS[:FEATURE_COLS.index("mu_ratio")]]
    if mode == "initial":
        return [f"{name}_first" for name in FEATURE_COLS]
    if mode == "endpoints":
        return (
            [f"{name}_first" for name in FEATURE_COLS]
            + [f"{name}_last" for name in FEATURE_COLS]
        )
    stats = ["mean", "std", "min", "max", "first", "last"]
    return [f"{name}_{stat}" for name in FEATURE_COLS for stat in stats] + ["length"]


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Classifiers
# ═══════════════════════════════════════════════════════════════════════════

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "acc":  float(accuracy_score(y_true, y_pred)),
        "f1":   float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":  float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.5,
    }


def majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Always predicts whichever class is most common in the training set."""
    majority = int(np.bincount(y_train).argmax())
    y_pred = np.full_like(y_test, majority)
    y_prob = np.full(len(y_test), float(majority))
    result = _metrics(y_test, y_pred, y_prob)
    result["majority_class"] = majority
    return result


def energy_threshold_baseline(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
) -> dict:
    """
    Physics heuristic: predict success (label=1) when specific orbital energy
    at the last observed timestep is below a fitted threshold.

    Rationale: spec_energy > 0 indicates a hyperbolic (escape) orbit, which
    is a strong failure signal for lunar/planetary capture missions.

    The threshold is chosen to maximise F1 on the training set.
    """
    train_energies = np.array([seq[-1, SPEC_ENERGY_IDX] for seq in X_train])
    test_energies  = np.array([seq[-1, SPEC_ENERGY_IDX] for seq in X_test])

    thresholds = np.percentile(train_energies, np.linspace(5, 95, 100))
    best_thresh, best_f1 = 0.0, -1.0
    for t in thresholds:
        preds = (train_energies <= t).astype(int)
        score = f1_score(y_train, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    y_pred = (test_energies <= best_thresh).astype(int)
    sigma = float(np.std(train_energies)) + 1e-8
    y_prob = 1.0 / (1.0 + np.exp(-(best_thresh - test_energies) / sigma))
    result = _metrics(y_test, y_pred, y_prob)
    result["best_threshold"] = best_thresh
    return result


def xgboost_baseline(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    seed: int = 42,
    feature_mode: str = "summary",
) -> dict:
    """
    XGBoost trained on trajectory summary statistics (79 tabular features).

    Serves as the strongest non-sequential baseline — it sees the same
    trajectory data but cannot model temporal ordering.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "xgboost not installed — pip install xgboost"}

    extractors = {
        "summary": extract_summary_features,
        "endpoints": extract_endpoint_features,
        "initial": extract_initial_features,
        "initial_no_context": extract_initial_no_context_features,
    }
    extractor = extractors[feature_mode]
    X_train_tab = extractor(X_train)
    X_test_tab  = extractor(X_test)

    scaler = RobustScaler()
    X_train_tab = scaler.fit_transform(X_train_tab)
    X_test_tab  = scaler.transform(X_test_tab)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train_tab, y_train)

    y_prob = clf.predict_proba(X_test_tab)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    result = _metrics(y_test, y_pred, y_prob)
    names = _feature_names(feature_mode)
    top_indices = np.argsort(clf.feature_importances_)[-15:][::-1]
    result["feature_mode"] = feature_mode
    result["top_features"] = [
        {"feature": names[i], "importance": float(clf.feature_importances_[i])}
        for i in top_indices
    ]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OrbitGuard — Baseline Classifiers")
    parser.add_argument("--data",             type=str,   default="data/merged/missions.parquet")
    parser.add_argument("--exit-fracs",       type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.6, 1.0])
    parser.add_argument("--downsample-factor",type=int,   default=15)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output",           type=str,   default="reports/baselines/baseline_results.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for frac in sorted(args.exit_fracs):
        pct = int(frac * 100)
        print(f"\n{'═' * 60}")
        print(f"  EXIT FRACTION = {frac:.0%}")
        print(f"{'═' * 60}")

        X_train, y_train, _, _, X_test, y_test = load_split_data(
            args.data, frac, args.downsample_factor, args.seed
        )
        n_pos = int(y_train.sum())
        print(f"  Train: {len(y_train)} missions ({n_pos} success / {len(y_train)-n_pos} failure)")

        row: dict = {"early_exit_frac": frac, "early_exit_pct": pct}

        print("  Running MajorityClass...")
        row["majority_class"]    = majority_class_baseline(y_train, y_test)

        print("  Running EnergyThreshold...")
        row["energy_threshold"]  = energy_threshold_baseline(X_train, y_train, X_test, y_test)

        print("  Running XGBoost...")
        row["xgboost"]           = xgboost_baseline(X_train, y_train, X_test, y_test, args.seed)

        print("  Running XGBoostEndpoints...")
        row["xgboost_endpoints"] = xgboost_baseline(
            X_train, y_train, X_test, y_test, args.seed, feature_mode="endpoints"
        )

        print("  Running XGBoostInitial...")
        row["xgboost_initial"] = xgboost_baseline(
            X_train, y_train, X_test, y_test, args.seed, feature_mode="initial"
        )

        print("  Running XGBoostInitialNoContext...")
        row["xgboost_initial_no_context"] = xgboost_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            args.seed,
            feature_mode="initial_no_context",
        )

        all_results.append(row)

        for name in [
            "majority_class",
            "energy_threshold",
            "xgboost",
            "xgboost_endpoints",
            "xgboost_initial",
            "xgboost_initial_no_context",
        ]:
            r = row[name]
            if "error" in r:
                print(f"  [{pct}%] {name:<22} ERROR: {r['error']}")
            else:
                print(f"  [{pct}%] {name:<22} AUC={r['auc']:.3f}  F1={r['f1']:.3f}  Acc={r['acc']:.2%}")

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n▸ Baseline results saved to {out_path}")


if __name__ == "__main__":
    main()
