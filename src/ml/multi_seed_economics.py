"""
Multi-seed confidence intervals for pruning economics.

Calculates compute savings, false prune rates, and failure recall across
seeds [0, 1, 2, 3, 4] to report mean +/- std for all headline economic figures.

Usage:
    python -m src.ml.multi_seed_economics
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.data_collection.gmat_runner import MissionConfig
from src.ml.planet_config import OPERATING_FRAC, PLANETS
from src.ml.splits import train_val_test
from src.paths import params_parquet, require, summary_parquet

LAUNCH_COLS = ["dv_V_offset", "dv_N_offset", "dv_B_offset",
               "RAAN_offset", "AOP_offset", "INC_offset"]
TARGET_RECALL = 0.99
FP_BUDGET = 0.01
CASCADE_QUANTILES = [0.99, 0.995, 0.999, 1.0]


def prop_days(planet: str) -> float:
    try:
        return float(MissionConfig("earth", planet).prop_days)
    except Exception:
        return 1.0


def achieved(pred_abort: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    n_fail = (y_true == 0).sum()
    rec = float((pred_abort & (y_true == 0)).sum() / max(n_fail, 1))
    n_good = (y_true == 1).sum()
    fp = float((pred_abort & (y_true == 1)).sum() / max(n_good, 1))
    return rec, fp


def run_seed(seed: int) -> dict:
    params_df = pd.read_parquet(require(params_parquet()))
    summary_df = pd.read_parquet(require(summary_parquet()))

    joined = (params_df[["sim_id", "target"] + LAUNCH_COLS]
              .merge(summary_df[["mission_id", "label"]],
                     left_on="sim_id", right_on="mission_id",
                     how="inner", validate="one_to_one"))
    joined = joined[joined["target"].isin(PLANETS)].copy()

    tot = {"t0_saved": 0.0, "t40_saved": 0.0, "casc_saved": 0.0,
           "total": 0.0, "t0_fp": 0, "t40_fp": 0, "casc_fp": 0,
           "n_good": 0, "t0_caught": 0, "t40_caught": 0, "casc_caught": 0,
           "n_fail": 0}

    for planet in PLANETS:
        p_df = joined[joined["target"] == planet].sort_values("sim_id").reset_index(drop=True)
        N = len(p_df)
        d = prop_days(planet)
        tr, va, te = train_val_test(N, seed)

        X = p_df[LAUNCH_COLS].to_numpy(dtype=np.float32)
        ys = p_df["label"].to_numpy(dtype=np.int64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                random_state=seed, eval_metric="logloss", n_jobs=4)
        clf.fit(X_scaled[tr], ys[tr])

        p_succ_tr = clf.predict_proba(X_scaled[tr])[:, 1]
        p_succ_va = clf.predict_proba(X_scaled[va])[:, 1]
        p_succ_te = clf.predict_proba(X_scaled[te])[:, 1]

        pf_tr_t0 = 1.0 - p_succ_tr
        pf_va_t0 = 1.0 - p_succ_va
        pf_te_t0 = 1.0 - p_succ_te

        # Validation threshold targeting recall
        fail_va = ys[va] == 0
        sorted_scores = np.sort(pf_va_t0[fail_va])
        idx = max(0, int((1.0 - TARGET_RECALL) * len(sorted_scores)))
        thr_t0 = float(sorted_scores[idx]) if len(sorted_scores) else 0.5

        prune_t0 = pf_te_t0 >= thr_t0
        n_test = len(te)
        total_cost = n_test * d
        t0_cost = (float(prune_t0.sum()) * 0.0 + float((~prune_t0).sum()) * d)
        t0_saved = 1.0 - t0_cost / total_cost

        rec_t0, fp_t0 = achieved(prune_t0, ys[te])

        good, fail = ys[te] == 1, ys[te] == 0
        tot["t0_saved"] += total_cost - t0_cost
        tot["total"] += total_cost
        tot["t0_fp"] += int((prune_t0 & good).sum())
        tot["n_good"] += int(good.sum())
        tot["t0_caught"] += int((prune_t0 & fail).sum())
        tot["n_fail"] += int(fail.sum())

    weighted = {
        "compute_saved_t0": tot["t0_saved"] / tot["total"],
        "false_prune_rate_t0": tot["t0_fp"] / max(tot["n_good"], 1),
        "fail_recall_t0": tot["t0_caught"] / max(tot["n_fail"], 1),
    }
    return weighted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="reports/prune_economics_multiseed.json")
    args = parser.parse_args()

    results = []
    print(f"Running multi-seed economics evaluation across seeds: {args.seeds}")
    for s in args.seeds:
        res = run_seed(s)
        results.append(res)
        print(f"  Seed {s}: T0 Saved={res['compute_saved_t0']:.1%}, FP={res['false_prune_rate_t0']:.2%}, Recall={res['fail_recall_t0']:.2%}")

    keys = list(results[0].keys())
    summary = {}
    for k in keys:
        vals = [r[k] for r in results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": args.seeds, "summary": summary, "per_seed": results}, indent=2))
    print(f"Saved multi-seed economics summary to {out_path}")


if __name__ == "__main__":
    main()
