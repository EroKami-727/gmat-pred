"""
Per-Target Threshold Calibration
=================================
For each interplanetary target body, find the P(fail) threshold that maximises F1
on the full dataset (post-training operating-point search — not leaking into train
weights since the model is frozen).

Saves results to models/thresholds.json, which the RegimeRouter reads at load time.

Usage:
    python -m src.ml.per_target_calibration \\
        --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \\
        --models-dir models \\
        --early-exit 0.4 \\
        --output models/thresholds.json

Regime models must exist before running this (train inner/outer first):
    models/inner_production/best_model_transformer_binary.pt
    models/outer_production/best_model_transformer_binary.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.ml.dataset import FEATURE_COLS
from src.ml.regime_router import RegimeRouter, INNER_PLANETS, OUTER_PLANETS

ALL_TARGETS = sorted(INNER_PLANETS | OUTER_PLANETS)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_missions(
    parquet_path: str,
    targets: list[str],
    early_exit: float = 0.4,
) -> dict[str, list[tuple[np.ndarray, int]]]:
    """
    Stream the parquet once and collect downsampled, early-exit-truncated
    feature arrays per target body.

    Downsample factors match training: ds=15 for inner planets, ds=50 for outer.
    Returns: {target_name: [(features_array, label), ...]}
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import pandas as pd
    import gc

    # Per-target downsample to match training regime
    ds_map = {t: (50 if t in OUTER_PLANETS else 15) for t in targets}
    print(f"  Downsample map: { {t: ds_map[t] for t in targets} }")

    pf = pq.ParquetFile(parquet_path)
    planet_set = pa.array([t.lower() for t in targets])

    wanted = ["mission_id", "label", "target_body", "elapsed_secs"] + FEATURE_COLS
    available = [c for c in wanted if c in pf.schema_arrow.names]

    print(f"  Streaming missions for {targets} ...")
    raw: dict[int, list] = {}          # mission_id -> list of downsampled DataFrames
    target_of: dict[int, str] = {}
    row_offsets: dict[int, int] = {}   # tracks cross-batch downsample phase per mission

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=100_000, columns=available)):
        lower_tb = pc.utf8_lower(batch.column("target_body"))
        mask     = pc.is_in(lower_tb, value_set=planet_set)
        filtered = batch.filter(mask)
        if filtered.num_rows == 0:
            del batch, filtered, lower_tb, mask
            gc.collect()
            continue

        df = filtered.to_pandas()
        for mid, grp in df.groupby("mission_id", sort=False):
            body   = grp["target_body"].iloc[0].lower()
            ds     = ds_map.get(body, 15)
            offset = row_offsets.get(mid, 0)
            n      = len(grp)
            # keep only rows whose global position is divisible by ds
            kept_idx = [i for i in range(n) if (i + offset) % ds == 0]
            if kept_idx:
                raw.setdefault(mid, []).append(grp.iloc[kept_idx])
            row_offsets[mid] = (offset + n) % ds
            if mid not in target_of:
                target_of[mid] = body

        del batch, filtered, lower_tb, mask, df
        gc.collect()

        if batch_idx % 100 == 0:
            print(f"    batch {batch_idx}: {len(raw):,} missions buffered")

    print(f"  Found {len(raw)} missions across {targets}")

    # Assemble per-target lists (already downsampled — just truncate to early-exit)
    missions: dict[str, list[tuple[np.ndarray, int]]] = {t: [] for t in targets}
    for mid, frames in raw.items():
        mdf = pd.concat(frames).reset_index(drop=True)
        keep_n = max(1, int(len(mdf) * early_exit))
        mdf = mdf.iloc[:keep_n]
        features = mdf[FEATURE_COLS].values.astype(np.float32)
        label    = int(mdf["label"].iloc[0])
        body     = target_of.get(mid)
        if body and body in missions:
            missions[body].append((features, label))

    for t, lst in missions.items():
        print(f"    {t:<12}: {len(lst)} missions")
    return missions


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    cache: dict,
    data: list[tuple[np.ndarray, int]],
) -> tuple[list[float], list[int]]:
    """Run a model cache on all (features, label) pairs. Returns (P_fail_list, labels)."""
    model  = cache["model"]
    scaler = cache["scaler"]
    device = cache["device"]
    arch   = cache["arch"]
    probs, labels = [], []
    for features, label in data:
        scaled = scaler.transform(features)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            if arch == "transformer":
                mask = torch.zeros(1, len(features), dtype=torch.bool, device=device)
                logit = model(x, mask)
            else:
                lengths = torch.tensor([len(features)], dtype=torch.long)
                logit = model(x, lengths)
        p_success = float(torch.sigmoid(logit).item())
        probs.append(1.0 - p_success)
        labels.append(label)
    return probs, labels


# ── Threshold sweep ───────────────────────────────────────────────────────────

def optimal_threshold(
    probs: list[float],
    labels: list[int],
    step: float = 0.01,
) -> tuple[float, float, float]:
    """
    Sweep P(fail) thresholds and return (best_threshold, best_f1, auc).
    Abort prediction: P(fail) >= threshold → predict failure (label=0).
    """
    try:
        # labels: 1=success, 0=fail. probs: P(fail). Negate so higher=more likely success.
        auc = roc_auc_score(labels, [-p for p in probs])
    except ValueError:
        auc = 0.0

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.005, 0.99, step):
        # Below threshold → predict success (1); above → predict failure (0)
        preds = [1 if p < thr else 0 for p in probs]
        # pos_label=0: optimise for catching FAILURES (the abort use-case)
        f1 = f1_score(labels, preds, pos_label=0, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return round(best_thr, 4), round(best_f1, 4), round(auc, 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-target threshold calibration for OrbitGuard")
    parser.add_argument("--data",        required=True,           help="Path to merged missions.parquet")
    parser.add_argument("--models-dir",  default="models",        help="Root models directory")
    parser.add_argument("--early-exit",  type=float, default=0.4, help="Same fraction used during training")
    parser.add_argument("--output",      default="models/thresholds.json")
    parser.add_argument("--targets",     nargs="+",  default=None,
                        help="Subset of targets to calibrate (default: all interplanetary)")
    args = parser.parse_args()

    targets = args.targets or ALL_TARGETS
    print(f"\n[ OrbitGuard Per-Target Calibration ]")
    print(f"  Targets  : {targets}")
    print(f"  Data     : {args.data}")
    print(f"  Models   : {args.models_dir}")
    print(f"  EarlyExit: {args.early_exit:.0%}\n")

    router = RegimeRouter(args.models_dir)
    print(f"  Router   : {router.status()}\n")

    if not router.is_available():
        print("ERROR: No model found. Train at least one model first.")
        return

    missions = load_missions(args.data, targets, args.early_exit)

    thresholds: dict[str, float] = {}
    results: list[dict] = []

    print(f"\n{'TARGET':<14} {'REGIME':<10} {'N':>6} {'AUC':>7} {'F1':>7} {'THR':>7}")
    print("─" * 56)

    for target in sorted(missions.keys()):
        data = missions[target]
        if not data:
            print(f"{target:<14} {'—':<10} {'0':>6}  (no data)")
            continue

        regime = router.regime_for(target)
        cache  = router._caches.get(regime) or router._caches.get("fallback")
        if cache is None:
            print(f"{target:<14} {'no model':<10} {len(data):>6}  (skip)")
            continue

        n_pos = sum(l for _, l in data)
        n_neg = len(data) - n_pos
        if n_pos == 0 or n_neg == 0:
            # Only one class — use default
            print(f"{target:<14} {regime:<10} {len(data):>6}  (single class, using default)")
            thresholds[target] = 0.443
            continue

        probs, labels = run_inference(cache, data)
        thr, f1, auc  = optimal_threshold(probs, labels)
        thresholds[target] = thr

        print(f"{target:<14} {regime:<10} {len(data):>6} {auc:>7.4f} {f1:>7.4f} {thr:>7.4f}")
        results.append({"target": target, "regime": regime, "n": len(data),
                        "auc": auc, "f1_at_optimal_thr": f1, "threshold": thr})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n  Saved thresholds → {out_path}")
    print(json.dumps(thresholds, indent=2))

    summary_path = out_path.with_name("calibration_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved full results → {summary_path}")


if __name__ == "__main__":
    main()
