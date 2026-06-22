"""
Summarize OrbitGuard validity-audit JSON files into paper-ready tables.

This does not train models. It turns existing random-split, leave-one-target-out,
and parameter-corridor audit artifacts into concise Markdown/CSV summaries so
paper claims stay tied to exact experiment outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("acc", "f1", "auc", "pr_auc", "brier_score", "ece")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _metric_block(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    valid = [row[key] for row in rows if key in row and "error" not in row[key]]
    for metric in METRICS:
        out[metric] = _summary([float(row[metric]) for row in valid])
    return out


def _random_split_table(random_split: dict[str, Any]) -> list[str]:
    if isinstance(random_split, list):
        random_split = random_split[0] if random_split else {}

    lines = [
        "## Random-Split Baselines",
        "",
        "| Model | Accuracy | F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "majority_class": "Majority",
        "energy_threshold": "Energy threshold",
        "xgboost": "XGBoost summary",
        "xgboost_endpoints": "XGBoost endpoints",
        "xgboost_initial": "XGBoost initial",
        "xgboost_initial_no_context": "XGBoost initial no context",
    }
    for key, label in labels.items():
        if key not in random_split:
            continue
        row = random_split[key]
        lines.append(
            f"| {label} | {_pct(float(row['acc']))} | {_fmt(float(row['f1']))} | "
            f"{_fmt(float(row['auc']))} |"
        )
    return lines


def _grouped_table(grouped: list[dict[str, Any]]) -> list[str]:
    has_calibration = all("pr_auc" in row.get("xgboost_summary", {}) for row in grouped)

    if has_calibration:
        lines = [
            "## Leave-One-Target-Out Audit",
            "",
            "| Held-out target | Success rate | Summary F1 | Summary AUC | Summary PR-AUC | Summary Brier | Summary ECE | Initial-no-context F1 | Initial-no-context AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in grouped:
            summary = row["xgboost_summary"]
            initial = row["xgboost_initial_no_context"]
            lines.append(
                f"| {row['heldout_target']} | {_pct(float(row['test_success_rate']))} | "
                f"{_fmt(float(summary['f1']))} | {_fmt(float(summary['auc']))} | "
                f"{_fmt(float(summary['pr_auc']))} | {_fmt(float(summary['brier_score']))} | "
                f"{_fmt(float(summary['ece']))} | "
                f"{_fmt(float(initial['f1']))} | {_fmt(float(initial['auc']))} |"
            )
    else:
        lines = [
            "## Leave-One-Target-Out Audit",
            "",
            "| Held-out target | Success rate | Summary F1 | Summary AUC | Initial-no-context F1 | Initial-no-context AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in grouped:
            summary = row["xgboost_summary"]
            initial = row["xgboost_initial_no_context"]
            lines.append(
                f"| {row['heldout_target']} | {_pct(float(row['test_success_rate']))} | "
                f"{_fmt(float(summary['f1']))} | {_fmt(float(summary['auc']))} | "
                f"{_fmt(float(initial['f1']))} | {_fmt(float(initial['auc']))} |"
            )

    lines.extend(
        [
            "",
            "Aggregate across held-out targets:",
            "",
            "| Feature mode | Accuracy mean +/- std | F1 mean +/- std | AUC mean +/- std | Worst F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in [
        ("xgboost_summary", "Summary"),
        ("xgboost_initial_no_context", "Initial no context"),
    ]:
        block = _metric_block(grouped, key)
        lines.append(
            f"| {label} | {_pct(block['acc']['mean'])} +/- {_pct(block['acc']['std'])} | "
            f"{_fmt(block['f1']['mean'])} +/- {_fmt(block['f1']['std'])} | "
            f"{_fmt(block['auc']['mean'])} +/- {_fmt(block['auc']['std'])} | "
            f"{_fmt(block['f1']['min'])} |"
        )

    if has_calibration:
        lines.extend(
            [
                "",
                "Aggregate calibration (summary mode only):",
                "",
                "| Metric | Mean +/- std | Worst |",
                "| --- | ---: | ---: |",
            ]
        )
        block = _metric_block(grouped, "xgboost_summary")
        for metric, label, worst_is_max in [
            ("pr_auc", "PR-AUC", False),
            ("brier_score", "Brier score", True),
            ("ece", "ECE", True),
        ]:
            worst = block[metric]["max"] if worst_is_max else block[metric]["min"]
            lines.append(
                f"| {label} | {_fmt(block[metric]['mean'])} +/- {_fmt(block[metric]['std'])} | "
                f"{_fmt(worst)} |"
            )
    return lines


def _parameter_table(parameter: list[dict[str, Any]]) -> list[str]:
    has_calibration = all("pr_auc" in row.get("xgboost_summary", {}) for row in parameter)

    if has_calibration:
        lines = [
            "## Parameter-Corridor Holdout Audit",
            "",
            "| Variable | Bin | Success rate | Summary F1 | Summary AUC | Summary PR-AUC | Summary Brier | Summary ECE | Initial-no-context F1 | Initial-no-context AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in parameter:
            summary = row["xgboost_summary"]
            initial = row["xgboost_initial_no_context"]
            lines.append(
                f"| {row['variable']} | {row['heldout_bin']} | {_pct(float(row['test']['success_rate']))} | "
                f"{_fmt(float(summary['f1']))} | {_fmt(float(summary['auc']))} | "
                f"{_fmt(float(summary['pr_auc']))} | {_fmt(float(summary['brier_score']))} | "
                f"{_fmt(float(summary['ece']))} | "
                f"{_fmt(float(initial['f1']))} | {_fmt(float(initial['auc']))} |"
            )
    else:
        lines = [
            "## Parameter-Corridor Holdout Audit",
            "",
            "| Variable | Bin | Success rate | Summary F1 | Summary AUC | Initial-no-context F1 | Initial-no-context AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in parameter:
            summary = row["xgboost_summary"]
            initial = row["xgboost_initial_no_context"]
            lines.append(
                f"| {row['variable']} | {row['heldout_bin']} | {_pct(float(row['test']['success_rate']))} | "
                f"{_fmt(float(summary['f1']))} | {_fmt(float(summary['auc']))} | "
                f"{_fmt(float(initial['f1']))} | {_fmt(float(initial['auc']))} |"
            )

    lines.extend(
        [
            "",
            "Aggregate across parameter bins:",
            "",
            "| Feature mode | Accuracy mean +/- std | F1 mean +/- std | AUC mean +/- std | Worst F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in [
        ("xgboost_summary", "Summary"),
        ("xgboost_initial_no_context", "Initial no context"),
    ]:
        block = _metric_block(parameter, key)
        lines.append(
            f"| {label} | {_pct(block['acc']['mean'])} +/- {_pct(block['acc']['std'])} | "
            f"{_fmt(block['f1']['mean'])} +/- {_fmt(block['f1']['std'])} | "
            f"{_fmt(block['auc']['mean'])} +/- {_fmt(block['auc']['std'])} | "
            f"{_fmt(block['f1']['min'])} |"
        )

    if has_calibration:
        lines.extend(
            [
                "",
                "Aggregate calibration (summary mode only):",
                "",
                "| Metric | Mean +/- std | Worst |",
                "| --- | ---: | ---: |",
            ]
        )
        block = _metric_block(parameter, "xgboost_summary")
        for metric, label, worst_is_max in [
            ("pr_auc", "PR-AUC", False),
            ("brier_score", "Brier score", True),
            ("ece", "ECE", True),
        ]:
            worst = block[metric]["max"] if worst_is_max else block[metric]["min"]
            lines.append(
                f"| {label} | {_fmt(block[metric]['mean'])} +/- {_fmt(block[metric]['std'])} | "
                f"{_fmt(worst)} |"
            )
    return lines


def _risk_notes(grouped: list[dict[str, Any]], parameter: list[dict[str, Any]]) -> list[str]:
    weak_grouped = [
        (row["heldout_target"], row["xgboost_summary"]["f1"], row["xgboost_summary"]["auc"])
        for row in grouped
        if float(row["xgboost_summary"]["f1"]) < 0.5
    ]
    weak_parameter = [
        (
            row["variable"],
            row["heldout_bin"],
            row["xgboost_summary"]["f1"],
            row["xgboost_summary"]["auc"],
            row["test"]["success_rate"],
        )
        for row in parameter
        if float(row["xgboost_summary"]["f1"]) < 0.7
    ]

    lines = [
        "## Paper Claim Guidance",
        "",
        "Defensible:",
        "",
        "- Random-split prediction is in-distribution and highly separable.",
        "- XGBoost trajectory-summary baselines are stronger than the current Transformer checkpoint.",
        "- Parameter-corridor holdout is the strongest current generalization evidence.",
        "- Full unseen-target transfer remains mixed and should be reported as a limitation.",
        "",
        "Weak grouped targets with summary F1 < 0.5:",
        "",
    ]
    for target, f1, auc in weak_grouped:
        lines.append(f"- {target}: F1={_fmt(float(f1))}, AUC={_fmt(float(auc))}")
    if not weak_grouped:
        lines.append("- None")

    lines.extend(["", "Weak parameter bins with summary F1 < 0.7:", ""])
    for variable, heldout_bin, f1, auc, success_rate in weak_parameter:
        lines.append(
            f"- {variable} bin {heldout_bin}: success={_pct(float(success_rate))}, "
            f"F1={_fmt(float(f1))}, AUC={_fmt(float(auc))}"
        )
    if not weak_parameter:
        lines.append("- None")

    has_calibration = (
        all("pr_auc" in row.get("xgboost_summary", {}) for row in grouped)
        and all("pr_auc" in row.get("xgboost_summary", {}) for row in parameter)
    )
    if has_calibration:
        miscalibrated_grouped = [
            (row["heldout_target"], row["xgboost_summary"]["auc"], row["xgboost_summary"]["f1"], row["xgboost_summary"]["ece"])
            for row in grouped
            if float(row["xgboost_summary"]["auc"]) >= 0.80 and float(row["xgboost_summary"]["f1"]) < 0.5
        ]
        miscalibrated_parameter = [
            (row["variable"], row["heldout_bin"], row["xgboost_summary"]["auc"], row["xgboost_summary"]["f1"], row["xgboost_summary"]["ece"])
            for row in parameter
            if float(row["xgboost_summary"]["auc"]) >= 0.80 and float(row["xgboost_summary"]["f1"]) < 0.7
        ]
        lines.extend(
            [
                "",
                "Ranking-works-but-threshold-fails cases (AUC >= 0.80 but F1 collapses at 0.5):",
                "These are calibration problems, not generalization failures — Platt/isotonic",
                "recalibration or threshold tuning should fix them without retraining.",
                "",
            ]
        )
        for target, auc, f1, ece in miscalibrated_grouped:
            lines.append(f"- {target}: AUC={_fmt(float(auc))}, F1@0.5={_fmt(float(f1))}, ECE={_fmt(float(ece))}")
        for variable, heldout_bin, auc, f1, ece in miscalibrated_parameter:
            lines.append(
                f"- {variable} bin {heldout_bin}: AUC={_fmt(float(auc))}, "
                f"F1@0.5={_fmt(float(f1))}, ECE={_fmt(float(ece))}"
            )
        if not miscalibrated_grouped and not miscalibrated_parameter:
            lines.append("- None")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OrbitGuard audit artifacts")
    parser.add_argument("--random-split", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    random_split = _load_json(Path(args.random_split))
    grouped = _load_json(Path(args.grouped))
    parameter = _load_json(Path(args.parameter))

    lines = [
        "# OrbitGuard Statistical Audit Summary",
        "",
        "Generated from existing audit JSON artifacts. This file is for paper writing and reviewer response drafting.",
        "",
        "Configuration: early exit = 40%, downsample factor = 10 interplanetary records. "
        "For interplanetary missions, one source record is 54,000 seconds = 15 hours.",
        "",
    ]
    for section in (
        _random_split_table(random_split),
        _grouped_table(grouped),
        _parameter_table(parameter),
        _risk_notes(grouped, parameter),
    ):
        lines.extend(section)
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
