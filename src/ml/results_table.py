"""
OrbitGuard Results Table Generator
====================================
Reads all experiment result JSON files and produces publication-ready
comparison tables in Markdown and LaTeX formats.

Tables Generated
----------------
1. main_comparison  — Transformer vs baselines per early-exit fraction
2. arch_ablation    — Component contribution at 40% exit
3. multi_seed       — Mean ± std robustness across seeds

Usage
-----
  # After running all experiments:
  python -m src.ml.results_table \\
      --ablation   reports/ablation/ablation_results.json \\
      --baselines  reports/baselines/baseline_results.json \\
      --arch       reports/arch_ablation/arch_ablation_results.json \\
      --multi-seed reports/multi_seed/summary.json \\
      --output-dir reports/tables

  # Generate only what's available (missing files are silently skipped):
  python -m src.ml.results_table --output-dir reports/tables

Output
------
  reports/tables/main_comparison.md   / .tex
  reports/tables/arch_ablation.md     / .tex
  reports/tables/multi_seed.md        / .tex
  reports/tables/summary.json         (all raw data in one place)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════

def _pct(v: float) -> str:
    return f"{v * 100:.1f}\\%"

def _f3(v: float) -> str:
    return f"{v:.3f}"

def _pct_plain(v: float) -> str:
    return f"{v * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# Table 1 — Main Comparison
# ═══════════════════════════════════════════════════════════════════════════

def build_main_comparison(
    ablation_data: Optional[list],
    baseline_data: Optional[list],
) -> tuple[str, str]:
    """
    Builds the primary results table with one row per (exit%, model).

    Transformer numbers come from the early-exit ablation sweep.
    Baseline numbers come from baselines.py output.
    """
    exit_pcts = [10, 20, 30, 40, 60, 100]

    transformer_by_pct: dict[int, dict] = {}
    if ablation_data:
        for entry in ablation_data:
            pct = entry.get("early_exit_pct", int(entry.get("early_exit_frac", 0) * 100))
            transformer_by_pct[pct] = {
                "auc": entry["test_auc"],
                "f1":  entry["test_f1"],
                "acc": entry["test_acc"],
            }

    baselines_by_pct: dict[int, dict] = {}
    if baseline_data:
        for entry in baseline_data:
            pct = entry.get("early_exit_pct", int(entry.get("early_exit_frac", 0) * 100))
            baselines_by_pct[pct] = entry

    models = [
        ("Transformer (ours)", "transformer"),
        ("XGBoost",            "xgboost"),
        ("Energy Threshold",   "energy_threshold"),
        ("Majority Class",     "majority_class"),
    ]

    # ── Markdown ──
    md = ["## Table 1: Baseline Comparison\n",
          "| Exit % | Model | AUC | F1 | Acc | Compute Saved |",
          "|--------|-------|-----|----|----|---------------|"]

    for pct in exit_pcts:
        saved = f"{100 - pct}%"
        for label, key in models:
            if key == "transformer":
                d = transformer_by_pct.get(pct)
                bold = key == "transformer"
                if d:
                    md.append(
                        f"| **{pct}%** | {'**' if bold else ''}{label}{'**' if bold else ''} "
                        f"| **{_f3(d['auc'])}** | **{_f3(d['f1'])}** | **{_pct_plain(d['acc'])}** | {saved} |"
                    )
                else:
                    md.append(f"| {pct}% | {label} | — | — | — | {saved} |")
            else:
                d = baselines_by_pct.get(pct, {}).get(key, {})
                if d and "error" not in d:
                    md.append(f"| {pct}% | {label} | {_f3(d['auc'])} | {_f3(d['f1'])} | {_pct_plain(d['acc'])} | {saved} |")
                else:
                    md.append(f"| {pct}% | {label} | — | — | — | {saved} |")

    # ── LaTeX ──
    tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Performance comparison by early-exit fraction}",
        r"\label{tab:main_comparison}",
        r"\begin{tabular}{llcccr}",
        r"\toprule",
        r"Exit \% & Model & AUC & F1 & Acc & Compute Saved \\",
        r"\midrule",
    ]

    for i, pct in enumerate(exit_pcts):
        saved = f"{100 - pct}\\%"
        for label, key in models:
            if key == "transformer":
                d = transformer_by_pct.get(pct)
                if d:
                    tex.append(
                        rf"{pct}\% & \textbf{{{label}}} & \textbf{{{_f3(d['auc'])}}} "
                        rf"& \textbf{{{_f3(d['f1'])}}} & \textbf{{{_pct(d['acc'])}}} & {saved} \\"
                    )
                else:
                    tex.append(rf"{pct}\% & {label} & -- & -- & -- & {saved} \\")
            else:
                d = baselines_by_pct.get(pct, {}).get(key, {})
                if d and "error" not in d:
                    tex.append(rf"{pct}\% & {label} & {_f3(d['auc'])} & {_f3(d['f1'])} & {_pct(d['acc'])} & {saved} \\")
                else:
                    tex.append(rf"{pct}\% & {label} & -- & -- & -- & {saved} \\")
        if i < len(exit_pcts) - 1:
            tex.append(r"\midrule")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md), "\n".join(tex)


# ═══════════════════════════════════════════════════════════════════════════
# Table 2 — Architecture Ablation
# ═══════════════════════════════════════════════════════════════════════════

def build_arch_table(arch_data: Optional[list]) -> tuple[str, str]:
    """Builds ablation table showing delta AUC for each removed component."""
    if not arch_data:
        return "*(arch ablation results not yet available)*", ""

    ref = next((r for r in arch_data if r["variant"] == "full_transformer"), None)
    ref_auc = ref["test_auc"] if ref else 0.0

    md = [
        "## Table 2: Architecture Ablation (exit=40%)\n",
        "| Variant | AUC | F1 | Acc | ΔAUC | Params |",
        "|---------|-----|----|----|------|--------|",
    ]

    tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Architecture ablation at 40\% early-exit}",
        r"\label{tab:arch_ablation}",
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"Variant & AUC & F1 & Acc & $\Delta$AUC & Params \\",
        r"\midrule",
    ]

    for r in arch_data:
        is_ref = r["variant"] == "full_transformer"
        delta_str = "ref" if is_ref else f"{r['test_auc'] - ref_auc:+.3f}"

        md.append(
            f"| {'**' if is_ref else ''}{r['label']}{'**' if is_ref else ''} "
            f"| {_f3(r['test_auc'])} | {_f3(r['test_f1'])} | {_pct_plain(r['test_acc'])} "
            f"| {delta_str} | {r['total_params']:,} |"
        )

        label_tex = r"\textbf{" + r["label"] + "}" if is_ref else r["label"]
        tex.append(
            rf"{label_tex} & {_f3(r['test_auc'])} & {_f3(r['test_f1'])} "
            rf"& {_pct(r['test_acc'])} & {delta_str} & {r['total_params']:,} \\"
        )

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md), "\n".join(tex)


# ═══════════════════════════════════════════════════════════════════════════
# Table 3 — Multi-Seed Robustness
# ═══════════════════════════════════════════════════════════════════════════

def build_multi_seed_table(ms_data: Optional[dict]) -> tuple[str, str]:
    """Builds mean ± std table for statistical robustness reporting."""
    if not ms_data:
        return "*(multi-seed results not yet available)*", ""

    exit_pct = int(ms_data.get("early_exit_frac", 0.4) * 100)
    n = ms_data["n_seeds"]

    md = [
        f"## Table 3: Multi-Seed Robustness (exit={exit_pct}%, n={n} seeds)\n",
        "| Metric | Mean ± Std | Per-Seed Values |",
        "|--------|------------|-----------------|",
    ]
    tex = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Statistical robustness across ${n}$ random seeds (exit={exit_pct}\%)}}",
        r"\label{tab:multi_seed}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Mean & Std \\",
        r"\midrule",
    ]

    for metric in ["auc", "f1", "acc"]:
        d = ms_data[metric]
        vals_str = ", ".join(f"{v:.4f}" for v in d["values"])
        md.append(f"| {metric.upper()} | {d['mean']:.4f} ± {d['std']:.4f} | {vals_str} |")
        tex.append(rf"{metric.upper()} & ${d['mean']:.4f}$ & $\pm{d['std']:.4f}$ \\")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md), "\n".join(tex)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] Not found: {path}")
        return None
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="OrbitGuard Results Table Generator")
    parser.add_argument("--ablation",   type=str, default="reports/ablation/ablation_results.json")
    parser.add_argument("--baselines",  type=str, default="reports/baselines/baseline_results.json")
    parser.add_argument("--arch",       type=str, default="reports/arch_ablation/arch_ablation_results.json")
    parser.add_argument("--multi-seed", type=str, default="reports/multi_seed/summary.json")
    parser.add_argument("--output-dir", type=str, default="reports/tables")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_data = _load_json(args.ablation)
    baseline_data = _load_json(args.baselines)
    arch_data     = _load_json(args.arch)
    ms_data       = _load_json(args.multi_seed)

    md1, tex1 = build_main_comparison(ablation_data, baseline_data)
    (out_dir / "main_comparison.md").write_text(md1)
    (out_dir / "main_comparison.tex").write_text(tex1)
    print(f"▸ Table 1  →  {out_dir}/main_comparison.{{md,tex}}")

    md2, tex2 = build_arch_table(arch_data)
    (out_dir / "arch_ablation.md").write_text(md2)
    (out_dir / "arch_ablation.tex").write_text(tex2)
    print(f"▸ Table 2  →  {out_dir}/arch_ablation.{{md,tex}}")

    md3, tex3 = build_multi_seed_table(ms_data)
    (out_dir / "multi_seed.md").write_text(md3)
    (out_dir / "multi_seed.tex").write_text(tex3)
    print(f"▸ Table 3  →  {out_dir}/multi_seed.{{md,tex}}")

    summary = {
        "ablation":     ablation_data,
        "baselines":    baseline_data,
        "arch_ablation":arch_data,
        "multi_seed":   ms_data,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"▸ Summary  →  {out_dir}/summary.json")

    print(f"\n{'═' * 60}")
    print(md1)
    print(f"\n{md2}")
    print(f"\n{md3}")


if __name__ == "__main__":
    main()
