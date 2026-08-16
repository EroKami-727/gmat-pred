"""
Error analysis tables for weak held-out targets/bins.

For every target/bin already flagged as weak in the calibrated
leave-one-target-out and parameter-corridor-holdout audits, this combines:
  - success rate (from the existing audit JSON)
  - predicted-probability distribution (from the existing audit JSON's
    test_probability_summary)
  - confusion matrix (from the existing audit JSON's metrics_block)
  - top feature shift: which of the 13 input features differ most between
    train and test distributions (new computation — standardized mean
    shift per feature, flattened across all timesteps and missions)

This does not retrain anything. It reads the calibrated JSON artifacts
produced by grouped_baselines.py / parameter_holdout_baselines.py (Phase 3)
and adds the one missing diagnostic: feature distribution shift.

Usage
-----
  python -m src.ml.error_analysis \\
      --data $ORBITGUARD_DATA/missions.parquet \\
      --params $ORBITGUARD_DATA/mission_params.parquet \\
      --grouped reports/baselines/leave_one_target_out_exit40_ds10_calibrated.json \\
      --parameter reports/baselines/parameter_holdout_exit40_ds10_calibrated.json \\
      --out docs/ERROR_ANALYSIS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.dataset import FEATURE_COLS
from src.ml.grouped_baselines import load_grouped_data
from src.ml.parameter_holdout_baselines import _assign_bins


def _feature_shift(train_seqs: list[np.ndarray], test_seqs: list[np.ndarray]) -> list[dict]:
    """
    Standardized mean shift per feature: (test_mean - train_mean) / train_std,
    flattened across all timesteps and missions. Ranks features by |shift|.
    """
    train_flat = np.vstack(train_seqs) if train_seqs else np.empty((0, len(FEATURE_COLS)))
    test_flat = np.vstack(test_seqs) if test_seqs else np.empty((0, len(FEATURE_COLS)))

    rows = []
    for i, col in enumerate(FEATURE_COLS):
        train_mean = float(train_flat[:, i].mean()) if len(train_flat) else 0.0
        train_std = float(train_flat[:, i].std()) if len(train_flat) else 1.0
        test_mean = float(test_flat[:, i].mean()) if len(test_flat) else 0.0
        shift = (test_mean - train_mean) / train_std if train_std > 1e-9 else 0.0
        rows.append({
            "feature": col,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "standardized_shift": shift,
        })
    rows.sort(key=lambda r: abs(r["standardized_shift"]), reverse=True)
    return rows


def analyze_grouped(missions: dict, grouped_json: list[dict], weak_f1_threshold: float = 0.7) -> list[dict]:
    out = []
    for row in grouped_json:
        summary = row.get("xgboost_summary", {})
        if "error" in summary or float(summary.get("f1", 1.0)) >= weak_f1_threshold:
            continue
        heldout = row["heldout_target"]
        train_seqs = [m["seq"] for m in missions.values() if m["target"] != heldout]
        test_seqs = [m["seq"] for m in missions.values() if m["target"] == heldout]
        out.append({
            "kind": "leave_one_target_out",
            "identifier": heldout,
            "success_rate": row["test_success_rate"],
            "test_missions": row["test_missions"],
            "metrics": summary,
            "probability_summary": summary.get("test_probability_summary", {}),
            "confusion_matrix": summary.get("confusion_matrix", {}),
            "top_feature_shift": _feature_shift(train_seqs, test_seqs)[:5],
        })
    return out


def analyze_parameter(
    missions: dict, params_path: str, parameter_json: list[dict],
    bins: int, weak_f1_threshold: float = 0.7,
) -> list[dict]:
    out = []
    weak_rows = [
        row for row in parameter_json
        if "error" not in row.get("xgboost_summary", {})
        and float(row["xgboost_summary"].get("f1", 1.0)) < weak_f1_threshold
    ]
    if not weak_rows:
        return out

    params = pd.read_parquet(params_path)
    variables = sorted({row["variable"] for row in weak_rows})
    params = params[["sim_id", "target", *variables]].copy()
    params["mission_id"] = params["sim_id"].astype(int)
    params = params[params["mission_id"].isin(missions.keys())].copy()
    for variable in variables:
        params[f"{variable}_bin"] = _assign_bins(params, variable, bins)

    for row in weak_rows:
        variable = row["variable"]
        heldout_bin = row["heldout_bin"]
        bin_col = f"{variable}_bin"
        test_ids = set(params.loc[params[bin_col] == heldout_bin, "mission_id"].astype(int))

        train_seqs = [m["seq"] for mid, m in missions.items() if mid not in test_ids]
        test_seqs = [m["seq"] for mid, m in missions.items() if mid in test_ids]

        summary = row["xgboost_summary"]
        out.append({
            "kind": "parameter_corridor_holdout",
            "identifier": f"{variable} bin {heldout_bin}/{row['bins'] - 1}",
            "success_rate": row["test"]["success_rate"],
            "test_missions": row["test"]["missions"],
            "metrics": summary,
            "probability_summary": summary.get("test_probability_summary", {}),
            "confusion_matrix": summary.get("confusion_matrix", {}),
            "top_feature_shift": _feature_shift(train_seqs, test_seqs)[:5],
        })
    return out


def render_markdown(cases: list[dict]) -> str:
    lines = [
        "# OrbitGuard Error Analysis — Weak Held-Out Cases",
        "",
        "Generated from calibrated leave-one-target-out and parameter-corridor-holdout",
        "audit artifacts. Each case below has summary-mode F1 < 0.7 at the default 0.5",
        "decision threshold. For every case: success rate, predicted-probability",
        "distribution, confusion matrix, and the 5 input features whose train/test",
        "distributions shifted the most (standardized mean shift).",
        "",
    ]
    for case in cases:
        m = case["metrics"]
        cm = case["confusion_matrix"]
        prob = case["probability_summary"]
        lines.extend([
            f"## {case['kind'].replace('_', ' ').title()} — {case['identifier']}",
            "",
            f"- Test missions: {case['test_missions']}",
            f"- Success rate: {case['success_rate']:.1%}",
            f"- F1@0.5: {m.get('f1', float('nan')):.3f}  |  AUC: {m.get('auc', float('nan')):.3f}  |  "
            f"PR-AUC: {m.get('pr_auc', float('nan')):.3f}",
            f"- Brier score: {m.get('brier_score', float('nan')):.3f}  |  ECE: {m.get('ece', float('nan')):.3f}",
            "",
            "**Confusion matrix (@0.5):**",
            "",
            f"| | Pred Fail | Pred Success |",
            f"| --- | ---: | ---: |",
            f"| Actual Fail | {cm.get('tn', '-')} | {cm.get('fp', '-')} |",
            f"| Actual Success | {cm.get('fn', '-')} | {cm.get('tp', '-')} |",
            "",
            "**Predicted probability distribution:**",
            "",
            f"min={prob.get('min', float('nan')):.3f}  p25={prob.get('p25', float('nan')):.3f}  "
            f"median={prob.get('median', float('nan')):.3f}  p75={prob.get('p75', float('nan')):.3f}  "
            f"max={prob.get('max', float('nan')):.3f}",
            "",
            "**Top feature shifts (train -> test, standardized):**",
            "",
            "| Feature | Train mean | Test mean | Standardized shift |",
            "| --- | ---: | ---: | ---: |",
        ])
        for f in case["top_feature_shift"]:
            lines.append(
                f"| {f['feature']} | {f['train_mean']:.4f} | {f['test_mean']:.4f} | "
                f"{f['standardized_shift']:+.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis tables for weak held-out cases")
    parser.add_argument("--data", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--grouped", required=True, help="Calibrated LOTO JSON (Phase 3 output)")
    parser.add_argument("--parameter", required=True, help="Calibrated parameter-holdout JSON (Phase 3 output)")
    parser.add_argument("--early-exit", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--weak-f1-threshold", type=float, default=0.7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("▸ Loading mission data...")
    missions = load_grouped_data(args.data, args.early_exit, args.downsample_factor)

    with open(args.grouped) as f:
        grouped_json = json.load(f)
    with open(args.parameter) as f:
        parameter_json = json.load(f)

    print("▸ Analyzing leave-one-target-out weak cases...")
    grouped_cases = analyze_grouped(missions, grouped_json, args.weak_f1_threshold)
    print(f"  Found {len(grouped_cases)} weak target(s)")

    print("▸ Analyzing parameter-corridor-holdout weak cases...")
    parameter_cases = analyze_parameter(missions, args.params, parameter_json, args.bins, args.weak_f1_threshold)
    print(f"  Found {len(parameter_cases)} weak bin(s)")

    all_cases = grouped_cases + parameter_cases
    md = render_markdown(all_cases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"▸ Wrote {args.out}")


if __name__ == "__main__":
    main()
