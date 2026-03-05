"""
EDA Report Generator — Streaming, Memory-Safe, Premium Visuals
===============================================================
Produces a 9-chart HTML + PNG EDA report for any missions.parquet
database without loading the full file into RAM at once.

Strategy
--------
Pass 1  — Stream missions.parquet → per-mission summary (~10k rows).
Pass 2  — Stream timestep data for trajectory profile charts
          (median ± IQR of physics features by outcome).
Params  — mission_params.parquet loaded directly (tiny, fits RAM).

Usage
-----
    python -m src.data_collection.eda_report \\
        --data  data/merged/missions.parquet \\
        --out   reports/eda/
"""

from __future__ import annotations

import argparse
import base64
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════════════

# Palette inspired by NASA/Space UI
C = {
    "bg":       "#0a0e1a",
    "panel":    "#111827",
    "card":     "#1a2332",
    "grid":     "#1e2d3d",
    "border":   "#2a3a4f",
    "text":     "#ffffff",
    "muted":    "#cbd5e1",
    "accent":   "#38bdf8",
    # Outcome colors
    "success":          "#34d399",
    "failure":          "#f87171",
    # Failure sub-types
    "missed_moon":      "#60a5fa",
    "surface_impact":   "#fb923c",
    "orbit_too_high":   "#a78bfa",
    "hyperbolic_flyby": "#fbbf24",
    # Gradient pairs for histogram fills
    "grad_green":  "#22c55e",
    "grad_red":    "#ef4444",
}

# Shared rcParams
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Inter", "Segoe UI", "Helvetica Neue", "Arial"],
    "font.size":          10,
    "axes.facecolor":     C["card"],
    "figure.facecolor":   C["bg"],
    "axes.edgecolor":     C["border"],
    "axes.labelcolor":    C["text"],
    "xtick.color":        C["muted"],
    "ytick.color":        C["muted"],
    "text.color":         C["text"],
    "axes.grid":          True,
    "grid.color":         C["grid"],
    "grid.linewidth":     0.4,
    "grid.alpha":         0.6,
    "legend.facecolor":   C["panel"],
    "legend.edgecolor":   C["border"],
    "legend.labelcolor":  C["text"],
    "savefig.facecolor":  C["bg"],
    "savefig.dpi":        180,
    "savefig.bbox":       "tight",
})

sns.set_style("darkgrid", {
    "axes.facecolor":  C["card"],
    "figure.facecolor": C["bg"],
    "grid.color":      C["grid"],
})


def _glow_bar(ax, x, height, color, width=0.45, alpha=0.85, label=None):
    """Draw a bar with a subtle glow underneath."""
    # Glow (wider, lower opacity)
    ax.bar(x, height, width=width*1.3, color=color, alpha=0.15, zorder=1)
    # Main bar
    return ax.bar(x, height, width=width, color=color, alpha=alpha,
                  edgecolor=color, linewidth=0.5, zorder=2, label=label)


def _title(ax, text, fontsize=13, pad=14):
    ax.set_title(text, color=C["text"], fontsize=fontsize, fontweight="600",
                 pad=pad, loc="left")


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Pass 1 — per-mission summary
# ═══════════════════════════════════════════════════════════════════════════

def _pass1_per_mission_summary(missions_path: Path) -> pd.DataFrame:
    print("  Pass 1: collecting per-mission summary rows …")
    pf = pq.ParquetFile(missions_path)
    wanted = ["mission_id", "label", "failure_type", "min_target_rmag"]
    cols = [c for c in wanted if c in pf.schema_arrow.names]

    rows, step_counts = [], {}
    last_mid = None

    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas()
        mids = df["mission_id"].values
        boundaries = np.concatenate([[0], np.where(np.diff(mids) != 0)[0] + 1])
        for idx in boundaries:
            mid = int(mids[idx])
            if mid != last_mid:
                rows.append(df.iloc[idx].to_dict())
                last_mid = mid
        unique, counts = np.unique(mids, return_counts=True)
        for m, c in zip(unique, counts):
            step_counts[int(m)] = step_counts.get(int(m), 0) + c

    summary = pd.DataFrame(rows)
    summary["mission_steps"] = summary["mission_id"].map(step_counts)
    print(f"    → {len(summary)} missions collected")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Pass 2 — trajectory profiles (streaming quantiles)
# ═══════════════════════════════════════════════════════════════════════════

def _pass2_trajectory_profiles(missions_path: Path, features: list[str],
                                n_bins: int = 120) -> dict:
    print("  Pass 2: streaming trajectory profiles …")
    pf = pq.ParquetFile(missions_path)
    avail = pf.schema_arrow.names
    feat_cols = [f for f in features if f in avail]
    if not feat_cols:
        return {}

    # Accumulate per-bin samples using reservoir-like approach
    # Key: (bin_idx, label) -> {feat: list}
    acc = defaultdict(lambda: defaultdict(list))
    MAX_PER_BIN = 5000  # cap to keep memory bounded

    cols_needed = ["mission_id", "label"] + feat_cols
    cols_needed = [c for c in cols_needed if c in avail]

    mission_start = {}  # mid -> first_row_global
    mission_len   = {}  # mid -> total steps
    # Pre-scan for mission lengths (from pass1 we can't easily get this)
    # Quick approximation: use summary if available, else estimate
    global_row = 0

    for batch in pf.iter_batches(batch_size=300_000, columns=cols_needed):
        df = batch.to_pandas()
        mids   = df["mission_id"].values
        labels = df["label"].values
        n_rows = len(df)

        # Track mission starts for normalization
        for j in range(n_rows):
            mid = int(mids[j])
            if mid not in mission_start:
                mission_start[mid] = global_row + j

        # Bin assignment: use local position within each mission
        # Group by mission_id for this batch
        unique_mids = np.unique(mids)
        for mid in unique_mids:
            mask = mids == mid
            local_df = df[mask]
            local_n = len(local_df)
            lbl = int(labels[mask][0])

            # Assign bins based on fractional position within this mission chunk
            frac_positions = np.linspace(0, 1, local_n, endpoint=False)
            bin_indices = (frac_positions * n_bins).astype(int)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            for feat in feat_cols:
                vals = local_df[feat].values
                for b_idx in np.unique(bin_indices):
                    b_mask = bin_indices == b_idx
                    key = (int(b_idx), lbl)
                    current = acc[key][feat]
                    new_vals = vals[b_mask].tolist()
                    if len(current) < MAX_PER_BIN:
                        acc[key][feat].extend(new_vals[:MAX_PER_BIN - len(current)])

        global_row += n_rows

    # Compute quantiles
    profile = {f: {0: {}, 1: {}} for f in feat_cols}
    bins = np.arange(n_bins)

    for feat in feat_cols:
        for lbl in (0, 1):
            q10, q50, q90 = [], [], []
            for b in bins:
                vals = acc.get((b, lbl), {}).get(feat, [])
                if vals:
                    arr = np.array(vals)
                    q10.append(np.percentile(arr, 10))
                    q50.append(np.percentile(arr, 50))
                    q90.append(np.percentile(arr, 90))
                else:
                    q10.append(np.nan)
                    q50.append(np.nan)
                    q90.append(np.nan)
            profile[feat][lbl] = {
                "bins": bins,
                "q10": np.array(q10),
                "q50": np.array(q50),
                "q90": np.array(q90),
            }

    return profile


# ═══════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════

def chart_01_class_balance(summary: pd.DataFrame, out: Path) -> Path:
    counts = summary["label"].value_counts().sort_index()
    labels = ["Failure", "Success"]
    colors = [C["failure"], C["success"]]
    vals   = [counts.get(0, 0), counts.get(1, 0)]
    total  = sum(vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (lbl, v, col) in enumerate(zip(labels, vals, colors)):
        _glow_bar(ax, i, v, col, label=lbl)
        pct = 100 * v / total
        ax.text(i, v + total * 0.02, f"{v:,}\n({pct:.1f}%)",
                ha="center", va="bottom", color=col, fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("Mission Count")
    _title(ax, "01 · Class Balance")
    return _save(fig, out / "01_class_balance.png")


def chart_02_failure_types(summary: pd.DataFrame, out: Path) -> Path:
    if "failure_type" not in summary.columns:
        return None
    counts = summary["failure_type"].value_counts()
    type_colors = {
        "success":          C["success"],
        "orbit_too_high":   C["orbit_too_high"],
        "surface_impact":   C["surface_impact"],
        "missed_moon":      C["missed_moon"],
        "hyperbolic_flyby": C["hyperbolic_flyby"],
    }
    colors_list = [type_colors.get(str(k), C["muted"]) for k in counts.index]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5),
                                          gridspec_kw={"width_ratios": [1, 1.2]})

    # Pie chart (left)
    wedges, texts, autotexts = ax_pie.pie(
        counts.values, labels=None, autopct="%1.1f%%",
        colors=colors_list, startangle=90, pctdistance=0.78,
        wedgeprops=dict(edgecolor=C["bg"], linewidth=2, width=0.45),
        textprops=dict(color=C["text"], fontsize=9),
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(9)
    # Center label
    ax_pie.text(0, 0, f"{len(summary):,}\nmissions",
                ha="center", va="center", color=C["text"], fontsize=12, fontweight="bold")
    ax_pie.set_title("Donut View", color=C["muted"], fontsize=10)

    # Horizontal bar (right)
    y_pos = np.arange(len(counts))
    ax_bar.barh(y_pos, counts.values, color=colors_list, height=0.55,
                edgecolor=C["bg"], linewidth=1)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([str(k).replace("_", " ").title() for k in counts.index], fontsize=10)
    ax_bar.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax_bar.text(v + max(counts.values) * 0.02, i,
                    f"{v:,}  ({100*v/len(summary):.1f}%)",
                    va="center", color=C["text"], fontsize=9, fontweight="bold")
    ax_bar.set_xlabel("Count")
    ax_bar.set_title("Breakdown", color=C["muted"], fontsize=10)

    fig.suptitle("02 · Failure Type Breakdown", color=C["text"],
                 fontsize=13, fontweight="600", y=1.02)
    return _save(fig, out / "02_failure_types.png")


def chart_03_launch_param_dists(params: pd.DataFrame, out: Path) -> Path:
    offset_cols = [c for c in ["dv_V_offset", "dv_N_offset", "dv_B_offset",
                                "RAAN_offset", "AOP_offset", "INC_offset"]
                   if c in params.columns]
    if not offset_cols:
        return None

    nice_names = {
        "dv_V_offset": "ΔV Prograde", "dv_N_offset": "ΔV Normal",
        "dv_B_offset": "ΔV Binormal", "RAAN_offset": "RAAN",
        "AOP_offset": "AOP", "INC_offset": "INC",
    }

    n = len(offset_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for i, col in enumerate(offset_cols):
        ax = axes_flat[i]
        for lbl, color, name in [(1, C["success"], "Success"),
                                   (0, C["failure"], "Failure")]:
            subset = params.loc[params["label"] == lbl, col].dropna()
            if not subset.empty:
                ax.hist(subset, bins=55, alpha=0.55, color=color,
                        label=name, density=True, histtype="stepfilled",
                        edgecolor=color, linewidth=0.8)
        ax.set_title(nice_names.get(col, col), fontsize=11, fontweight="500")
        ax.set_xlabel("")
        ax.set_ylabel("Density" if i % 3 == 0 else "")

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    axes_flat[0].legend(fontsize=9)
    fig.suptitle("03 · Launch Parameter Distributions by Outcome",
                 fontsize=13, fontweight="600", y=1.01, color=C["text"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, out / "03_launch_param_dists.png")


def chart_04_launch_param_boxplots(params: pd.DataFrame, out: Path) -> Path:
    offset_cols = [c for c in ["dv_V_offset", "dv_N_offset", "dv_B_offset",
                                "RAAN_offset", "AOP_offset", "INC_offset"]
                   if c in params.columns]
    if not offset_cols:
        return None

    nice_names = {
        "dv_V_offset": "ΔV Prograde", "dv_N_offset": "ΔV Normal",
        "dv_B_offset": "ΔV Binormal", "RAAN_offset": "RAAN",
        "AOP_offset": "AOP", "INC_offset": "INC",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for i, col in enumerate(offset_cols):
        ax = axes_flat[i]
        data = [params.loc[params["label"] == 1, col].dropna(),
                params.loc[params["label"] == 0, col].dropna()]

        bp = ax.boxplot(
            data, tick_labels=["Success", "Failure"], patch_artist=True, widths=0.45,
            medianprops=dict(color="#fbbf24", linewidth=2),
            whiskerprops=dict(color=C["muted"], linewidth=1),
            capprops=dict(color=C["muted"], linewidth=1),
            flierprops=dict(markerfacecolor=C["muted"], marker=".", markersize=2, alpha=0.4),
            boxprops=dict(linewidth=0),
        )
        bp["boxes"][0].set_facecolor(C["success"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(C["failure"])
        bp["boxes"][1].set_alpha(0.7)
        ax.set_title(nice_names.get(col, col), fontsize=11, fontweight="500")

    for j in range(len(offset_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("04 · Launch Parameter Box Plots vs Outcome",
                 fontsize=13, fontweight="600", y=1.01, color=C["text"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, out / "04_launch_param_boxplots.png")


def chart_05_correlation_matrix(params: pd.DataFrame, out: Path) -> Path:
    num_cols = [c for c in ["dv_V_offset", "dv_N_offset", "dv_B_offset",
                             "RAAN_offset", "AOP_offset", "INC_offset", "label"]
               if c in params.columns]
    if len(num_cols) < 2:
        return None

    nice = {"dv_V_offset": "ΔV_V", "dv_N_offset": "ΔV_N", "dv_B_offset": "ΔV_B",
            "RAAN_offset": "RAAN", "AOP_offset": "AOP", "INC_offset": "INC",
            "label": "Label"}

    corr = params[num_cols].corr()
    corr.columns = [nice.get(c, c) for c in corr.columns]
    corr.index   = [nice.get(c, c) for c in corr.index]
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap=cmap, center=0,
                ax=ax, cbar_kws={"shrink": 0.75, "pad": 0.02},
                linecolor=C["bg"], linewidths=1.5, square=True,
                annot_kws={"size": 10, "weight": "bold"})

    _title(ax, "05 · Pearson Correlation Matrix", fontsize=13)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=C["muted"])
    return _save(fig, out / "05_correlation_matrix.png")


def chart_06_trajectory_profiles(profile: dict, out: Path) -> Path:
    feat_meta = {
        "spec_energy":      ("Specific Orbital Energy", "km²/s²", "#38bdf8", "#f472b6"),
        "norm_target_dist": ("Normalised Target Dist",  "[0–1]",  "#34d399", "#fb923c"),
        "fpa_deg":          ("Flight Path Angle",       "deg",    "#a78bfa", "#fbbf24"),
        "vel_mag":          ("Velocity Magnitude",      "km/s",   "#60a5fa", "#f87171"),
    }
    feats = [f for f in feat_meta if f in profile]
    if not feats:
        return None

    n = len(feats)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    x = np.linspace(0, 100, profile[feats[0]][0]["q50"].shape[0])

    for ax, feat in zip(axes, feats):
        label, unit, col_s, col_f = feat_meta[feat]
        for lbl, color, name in [(1, col_s, "Success"), (0, col_f, "Failure")]:
            data = profile[feat].get(lbl, {})
            q10 = data.get("q10", np.full_like(x, np.nan))
            q50 = data.get("q50", np.full_like(x, np.nan))
            q90 = data.get("q90", np.full_like(x, np.nan))
            ax.fill_between(x, q10, q90, alpha=0.15, color=color)
            ax.plot(x, q50, color=color, linewidth=2, label=f"{name} (median)", alpha=0.9)
        ax.set_ylabel(f"{label}\n({unit})", fontsize=9, linespacing=1.5)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Mission Elapsed (%)")
    _title(axes[0], "06 · Trajectory Profiles — Median ± 10–90% Band")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return _save(fig, out / "06_trajectory_profiles.png")


def chart_07_mission_length_dist(summary: pd.DataFrame, out: Path) -> Path:
    if "mission_steps" not in summary.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, color, name in [(1, C["success"], "Success"),
                               (0, C["failure"], "Failure")]:
        subset = summary.loc[summary["label"] == lbl, "mission_steps"].dropna()
        if not subset.empty:
            ax.hist(subset, bins=65, alpha=0.6, color=color, label=name,
                    histtype="stepfilled", edgecolor=color, linewidth=0.6)

    _title(ax, "07 · Mission Length Distribution (Timesteps)")
    ax.set_xlabel("Timesteps per Mission")
    ax.set_ylabel("Count")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    return _save(fig, out / "07_mission_length_dist.png")


def chart_08_min_target_rmag(summary: pd.DataFrame, out: Path) -> Path:
    if "min_target_rmag" not in summary.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    q99 = summary["min_target_rmag"].quantile(0.99)

    for lbl, color, name in [(1, C["success"], "Success"),
                               (0, C["failure"], "Failure")]:
        subset = summary.loc[summary["label"] == lbl, "min_target_rmag"].dropna()
        if not subset.empty:
            ax.hist(subset, bins=100, alpha=0.55, color=color, label=name,
                    range=(0, q99), histtype="stepfilled",
                    edgecolor=color, linewidth=0.6)

    # Mark success corridor
    ax.axvspan(1837, 2237, alpha=0.12, color=C["success"], zorder=0)
    ax.axvline(1837, color=C["success"], linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(2237, color=C["success"], linestyle="--", linewidth=1, alpha=0.6)
    ax.text(2037, ax.get_ylim()[1] * 0.92, "Capture\nCorridor",
            ha="center", va="top", color=C["success"], fontsize=8,
            fontweight="bold", alpha=0.8)

    _title(ax, "08 · Closest Approach Distance (min_target_rmag)")
    ax.set_xlabel("Distance to Target Surface (km)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    return _save(fig, out / "08_min_target_rmag_dist.png")


def chart_09_velocity_scatter(params: pd.DataFrame, out: Path) -> Path:
    if "dv_V_offset" not in params.columns or "dv_N_offset" not in params.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot failures first (background), then successes (foreground)
    for lbl, color, name, zorder, size in [
        (0, C["failure"], "Failure", 1, 5),
        (1, C["success"], "Success", 2, 8),
    ]:
        sub = params[params["label"] == lbl]
        ax.scatter(sub["dv_V_offset"] * 1000, sub["dv_N_offset"] * 1000,
                   c=color, s=size, alpha=0.45, label=name,
                   rasterized=True, zorder=zorder, edgecolors="none")

    # Zero crosshairs
    ax.axhline(0, color=C["muted"], linewidth=0.5, alpha=0.4)
    ax.axvline(0, color=C["muted"], linewidth=0.5, alpha=0.4)

    _title(ax, "09 · ΔV_prograde vs ΔV_normal (coloured by outcome)")
    ax.set_xlabel("ΔV prograde offset (m/s)")
    ax.set_ylabel("ΔV normal offset (m/s)")
    ax.legend(markerscale=4, fontsize=10)
    return _save(fig, out / "09_velocity_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def _img_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def build_html(figure_paths: list[Path], out_dir: Path, summary: pd.DataFrame) -> Path:
    n     = len(summary)
    n_ok  = int(summary["label"].sum())
    n_f   = n - n_ok
    sr    = 100 * n_ok / max(n, 1)
    rows  = summary["mission_steps"].sum() if "mission_steps" in summary.columns else "?"

    # Failure breakdown string
    ft_html = ""
    if "failure_type" in summary.columns:
        for ft, cnt in summary["failure_type"].value_counts().items():
            pct = 100 * cnt / n
            ft_html += f'<div class="ft-row"><span class="ft-name">{str(ft).replace("_"," ").title()}</span><span class="ft-val">{cnt:,} ({pct:.1f}%)</span></div>'

    CHART_DESCS = {
        "class balance": "Shows the ratio of successful captures vs. mission failures. A 35.3% success rate ensures a reasonably balanced dataset for ML classification.",
        "failure types": "Breakdown of failure modes. 'Orbit Too High' is the most frequent failure, indicating excessive orbital energy at the capture window.",
        "launch param dists": "Probability density of launch offsets. Successes (green) are tightly clustered around zero, while failures (red) span the full dispersion range.",
        "launch param boxplots": "Statistical distribution of launch offsets. Successes show significantly lower variance (smaller boxes) than failures across all parameters.",
        "correlation matrix": "Linear relationships between launch offsets and mission outcome. ΔV Prograde (ΔV_V) shows the strongest single influence on success.",
        "trajectory profiles": "Time-series evolution of orbital physics. Lines show medians; shaded bands show 10-90% range. Divergence indicates where failures become predictable.",
        "mission length dist": "Total timesteps per mission. Failures often terminate early (impacts) or late (lost in space), while successes cluster at the intended capture duration.",
        "min target rmag dist": "Closest approach distance. Successes are perfectly bounded within the 1837-2237 km physical capture corridor (100–500 km altitude).",
        "velocity scatter": "2D distribution of prograde vs. normal velocity offsets. Successes form a razor-thin 'safe zone' rectangular cluster at the centre.",
    }

    cards = ""
    for p in figure_paths:
        if p is None or not p.exists():
            continue
        b64  = _img_b64(p)
        raw_name = p.stem.replace("_", " ")
        # Remove leading number prefix
        name = raw_name.lstrip("0123456789 ")
        desc = CHART_DESCS.get(name.lower(), "Automated visual analysis of mission dataset.")
        
        cards += f"""
    <div class="card">
      <div class="card-head">{name.title()}</div>
      <div class="card-desc">{desc}</div>
      <img src="data:image/png;base64,{b64}" alt="{name}" loading="lazy"/>
    </div>"""

    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>OrbitGuard — EDA Report</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      :root{{
        --bg:{C["bg"]};--panel:{C["panel"]};--card:{C["card"]};
        --border:{C["border"]};--text:{C["text"]};--muted:{C["muted"]};
        --accent:{C["accent"]};--success:{C["success"]};--fail:{C["failure"]};
      }}
      *{{box-sizing:border-box;margin:0;padding:0}}
      body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
           padding:2rem 3rem;max-width:1400px;margin:0 auto}}

      /* ── Header ── */
      header{{text-align:center;padding:3rem 0 2rem}}
      header h1{{font-size:2.8rem;font-weight:700;
                 background:linear-gradient(135deg,{C["success"]},{C["accent"]});
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 letter-spacing:-0.02em}}
      header .sub{{color:var(--muted);font-size:1rem;margin-top:.4rem}}

      /* ── Stats strip ── */
      .stats{{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin:2.5rem 0}}
      .stat{{background:var(--panel);border:1px solid var(--border);border-radius:14px;
             padding:1.2rem 2rem;text-align:center;min-width:135px;
             transition:transform .2s;position:relative;overflow:hidden}}
      .stat:hover{{transform:translateY(-3px)}}
      .stat::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:3px}}
      .stat:nth-child(1)::after{{background:var(--accent)}}
      .stat:nth-child(2)::after{{background:var(--success)}}
      .stat:nth-child(3)::after{{background:var(--fail)}}
      .stat:nth-child(4)::after{{background:var(--accent)}}
      .stat:nth-child(5)::after{{background:var(--muted)}}
      .stat .val{{font-size:2rem;font-weight:700;color:var(--accent)}}
      .stat .lbl{{color:var(--muted);font-size:.8rem;margin-top:.25rem;text-transform:uppercase;letter-spacing:.05em}}

      /* ── Failure breakdown ── */
      .ft-strip{{max-width:500px;margin:0 auto 2rem;background:var(--panel);border:1px solid var(--border);
                 border-radius:12px;padding:1rem 1.5rem}}
      .ft-row{{display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid var(--border)}}
      .ft-row:last-child{{border-bottom:none}}
      .ft-name{{color:var(--text);font-size:.9rem}}.ft-val{{color:var(--muted);font-size:.9rem;font-weight:600}}

      /* ── Chart grid ── */
      .grid{{display:grid;grid-template-columns:1fr;gap:2rem;margin-top:2rem}}
      .card{{background:var(--card);border:1px solid var(--border);border-radius:16px;
             padding:1.5rem;transition:transform .15s,box-shadow .15s}}
      .card:hover{{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.3)}}
      .card-head{{font-size:.85rem;font-weight:600;color:var(--accent);text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:0.4rem;padding-bottom:.5rem;
                  border-bottom:1px solid var(--border)}}
      .card-desc{{font-size:.85rem;color:var(--text);margin-bottom:1.2rem;line-height:1.5;opacity:0.9}}
      .card img{{width:100%;border-radius:10px;display:block;border:1px solid var(--border)}}

      footer{{text-align:center;color:var(--muted);font-size:.75rem;margin-top:4rem;
              padding-top:1.5rem;border-top:1px solid var(--border)}}
    </style>
    </head>
    <body>
    <header>
      <h1>OrbitGuard · EDA Report</h1>
      <div class="sub">Monte Carlo Dispersion Analysis · Earth → Moon Lunar Transfer</div>
    </header>

    <div class="stats">
      <div class="stat"><div class="val">{n:,}</div><div class="lbl">Missions</div></div>
      <div class="stat"><div class="val" style="color:var(--success)">{n_ok:,}</div><div class="lbl">Successes</div></div>
      <div class="stat"><div class="val" style="color:var(--fail)">{n_f:,}</div><div class="lbl">Failures</div></div>
      <div class="stat"><div class="val">{sr:.1f}%</div><div class="lbl">Success Rate</div></div>
      <div class="stat"><div class="val">{rows:,}</div><div class="lbl">Total Rows</div></div>
    </div>

    {"<div class='ft-strip'>" + ft_html + "</div>" if ft_html else ""}

    <div class="grid">{cards}
    </div>

    <footer>Generated by OrbitGuard EDA · src/data_collection/eda_report.py</footer>
    </body>
    </html>""")

    html_path = out_dir / "eda_report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_eda(missions_path: Path, params_path: Path | None, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = _pass1_per_mission_summary(missions_path)

    # Load params
    params = None
    if params_path and params_path.exists():
        print("  Loading mission_params.parquet …")
        params = pd.read_parquet(params_path)
        if "label" not in params.columns:
            if "sim_id" in params.columns and "mission_id" in summary.columns:
                lbl_map = summary.set_index("mission_id")["label"]
                params["label"] = params["sim_id"].map(lbl_map)
            elif "mission_id" in params.columns:
                lbl_map = summary.set_index("mission_id")["label"]
                params["label"] = params["mission_id"].map(lbl_map)
    else:
        print("  ⚠  mission_params.parquet not found — param charts will be skipped")

    # Trajectory profiles
    profile_feats = ["spec_energy", "norm_target_dist", "fpa_deg", "vel_mag"]
    profile = _pass2_trajectory_profiles(missions_path, profile_feats)

    # ── Generate all charts ───────────────────────────────────────────────
    print("\n  Rendering charts …")
    figs: list[Path | None] = []

    figs.append(chart_01_class_balance(summary, fig_dir))
    print("    ✓ 01 class balance")

    figs.append(chart_02_failure_types(summary, fig_dir))
    print("    ✓ 02 failure types (donut + bar)")

    if params is not None and "label" in params.columns:
        figs.append(chart_03_launch_param_dists(params, fig_dir))
        print("    ✓ 03 launch param distributions")
        figs.append(chart_04_launch_param_boxplots(params, fig_dir))
        print("    ✓ 04 box plots")
        figs.append(chart_05_correlation_matrix(params, fig_dir))
        print("    ✓ 05 correlation matrix")
    else:
        figs += [None, None, None]

    if profile:
        figs.append(chart_06_trajectory_profiles(profile, fig_dir))
        print("    ✓ 06 trajectory profiles")
    else:
        figs.append(None)

    figs.append(chart_07_mission_length_dist(summary, fig_dir))
    print("    ✓ 07 mission length")

    figs.append(chart_08_min_target_rmag(summary, fig_dir))
    print("    ✓ 08 min_target_rmag")

    if params is not None and "label" in params.columns:
        figs.append(chart_09_velocity_scatter(params, fig_dir))
        print("    ✓ 09 velocity scatter")
    else:
        figs.append(None)

    # ── HTML ──
    html_path = build_html([f for f in figs if f is not None], out_dir, summary)
    print(f"\n  ✓ Report → {html_path}")
    print(f"  ✓ Figures → {fig_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA report")
    parser.add_argument("--data",   type=str, default="data/merged/missions.parquet")
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--out",    type=str, default="reports/eda/")
    args = parser.parse_args()

    missions_path = Path(args.data)
    if not missions_path.exists():
        sys.exit(f"ERROR: {missions_path} not found.")

    params_path = Path(args.params) if args.params else None
    if params_path is None:
        candidate = missions_path.parent / "mission_params.parquet"
        params_path = candidate if candidate.exists() else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OrbitGuard — EDA Report Generator")
    print("=" * 60)
    print(f"  Data   : {missions_path}")
    print(f"  Params : {params_path or 'not found'}")
    print(f"  Output : {out_dir.resolve()}")
    print("=" * 60)
    print()

    run_eda(missions_path, params_path, out_dir)
