"""
OrbitGuard Interpretability Analysis
=====================================
Three complementary analyses that explain *why* the model works:

1. Attention Heatmaps   — which timesteps does each transformer layer focus on?
2. Feature Attribution  — which of the 13 physics features drive predictions?
3. Failure-Type Breakdown — which failure modes are easiest/hardest to detect early?

Methods
-------
- Attention: extract CLS→sequence weights from each encoder layer via
  forward_with_attention(), average across heads.
- Attribution: Gradient × Input saliency on a random test sample.
  Fast and sufficient for ranking features; IG would be 50× slower.
- Failure breakdown: run inference on the test split (same seed as training),
  stratify predictions by failure_type label.

Usage
-----
  python -m src.ml.interpretability \\
      --data data/merged/missions.parquet \\
      --model-path models/transformer_production/best_model_transformer_binary.pt \\
      --scaler-path models/transformer_production/scaler_transformer_binary.pkl \\
      --n-sample-missions 10 \\
      --n-attr-missions 200 \\
      --seed 42 \\
      --output-dir reports/interpretability

Output
------
  reports/interpretability/summary.json   (all results, consumed by dashboard)
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from src.ml.dataset import FEATURE_COLS, FAILURE_TYPE_MAP
from src.ml.model import TrajectoryTransformer


# ═══════════════════════════════════════════════════════════════════════════
# Data Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_mission_raw(
    parquet_path: Path,
    mission_id: int,
    downsample_factor: int = 15,
    early_exit_frac: float = 1.0,
) -> tuple[np.ndarray, int, str]:
    """Stream-load a single mission's features, label, and failure_type."""
    import pandas as pd
    needed = ["mission_id", "elapsed_secs", "label", "failure_type"] + FEATURE_COLS
    pf = pq.ParquetFile(parquet_path)
    needed = [c for c in needed if c in pf.schema_arrow.names]

    frames = []
    for batch in pf.iter_batches(batch_size=500_000, columns=needed):
        df = batch.to_pandas()
        chunk = df[df["mission_id"] == mission_id]
        if len(chunk):
            frames.append(chunk)

    if not frames:
        return None, None, None

    import pandas as pd
    mission = pd.concat(frames).sort_values("elapsed_secs").reset_index(drop=True)
    mission = mission.iloc[::downsample_factor].reset_index(drop=True)
    if early_exit_frac < 1.0:
        keep = max(1, int(len(mission) * early_exit_frac))
        mission = mission.iloc[:keep]

    features    = mission[FEATURE_COLS].values.astype(np.float32)
    label       = int(mission["label"].iloc[0])
    failure_type = str(mission["failure_type"].iloc[0]) if "failure_type" in mission else "unknown"
    return features, label, failure_type


def _stream_missions(
    parquet_path: Path,
    mission_ids: set,
    downsample_factor: int = 15,
    early_exit_frac: float = 1.0,
) -> dict[int, dict]:
    """Stream-load a set of missions by ID."""
    import pandas as pd
    needed = ["mission_id", "elapsed_secs", "label", "failure_type"] + FEATURE_COLS
    pf = pq.ParquetFile(parquet_path)
    needed = [c for c in needed if c in pf.schema_arrow.names]

    current_mid = None
    current_rows: list = []
    results: dict[int, dict] = {}

    def _flush(mid, rows):
        if mid not in mission_ids:
            return
        df = pd.concat(rows).sort_values("elapsed_secs").reset_index(drop=True)
        df = df.iloc[::downsample_factor].reset_index(drop=True)
        if early_exit_frac < 1.0:
            keep = max(1, int(len(df) * early_exit_frac))
            df = df.iloc[:keep]
        results[mid] = {
            "seq":          df[FEATURE_COLS].values.astype(np.float32),
            "label":        int(df["label"].iloc[0]),
            "failure_type": str(df["failure_type"].iloc[0]) if "failure_type" in df else "unknown",
        }

    for batch in pf.iter_batches(batch_size=500_000, columns=needed):
        df = batch.to_pandas()
        for mid, group in df.groupby("mission_id", sort=False):
            if mid != current_mid:
                if current_rows and current_mid is not None:
                    _flush(current_mid, current_rows)
                current_mid = mid
                current_rows = [group]
            else:
                current_rows.append(group)
    if current_rows and current_mid is not None:
        _flush(current_mid, current_rows)

    return results


def _split_mission_ids(parquet_path: Path, seed: int = 42,
                       train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Return test-split mission IDs using the same deterministic split as training."""
    pf = pq.ParquetFile(parquet_path)
    all_ids = pf.read(columns=["mission_id"])["mission_id"].unique().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return set(all_ids[n_train + n_val:])


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1: Attention Heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def analyze_attention(
    model: TrajectoryTransformer,
    scaler,
    missions: dict[int, dict],
    device: torch.device,
    n_display_steps: int = 60,
) -> list[dict]:
    """
    Extract CLS→sequence attention weights for each sample mission.

    Attention is averaged across heads per layer, then downsampled to
    n_display_steps bins for frontend rendering.

    Returns list of per-mission dicts ready to serialize as JSON.
    """
    model.eval()
    results = []

    for mission_id, data in missions.items():
        seq = data["seq"]
        scaled = scaler.transform(seq)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.zeros(1, len(seq), dtype=torch.bool, device=device)

        with torch.no_grad():
            logit, attn_list = model.forward_with_attention(x, mask)

        prob_fail = 1.0 - float(torch.sigmoid(logit).item())

        # attn_list: list of (1, nhead, seq+1, seq+1)
        # CLS row (index 0) attending to sequence (indices 1:)
        attention_by_layer = []
        for attn_w in attn_list:
            # (1, nhead, seq+1, seq+1) → average heads → (seq_len,)
            cls_attn = attn_w[0, :, 0, 1:].mean(dim=0).cpu().numpy()  # (seq_len,)

            # Downsample to n_display_steps by averaging buckets
            T = len(cls_attn)
            buckets = np.array_split(cls_attn, min(n_display_steps, T))
            downsampled = [float(b.mean()) for b in buckets]
            attention_by_layer.append(downsampled)

        # Elapsed-percent axis for frontend
        n_steps = min(n_display_steps, len(seq))
        elapsed_pcts = [round((i + 1) / n_steps, 3) for i in range(n_steps)]

        results.append({
            "mission_id":        int(mission_id),
            "true_label":        data["label"],
            "failure_type":      data["failure_type"],
            "prob_fail":         round(prob_fail, 4),
            "n_timesteps":       len(seq),
            "elapsed_pcts":      elapsed_pcts,
            "attention_by_layer": attention_by_layer,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2: Feature Attribution (Gradient × Input Saliency)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_feature_attribution(
    model: TrajectoryTransformer,
    scaler,
    missions: dict[int, dict],
    device: torch.device,
) -> dict:
    """
    Compute Gradient × Input saliency for each mission and aggregate
    mean absolute attribution per feature across all missions.

    Returns a dict with sorted feature importance for JSON output.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    all_attributions = []  # list of (n_features,) arrays

    for mission_id, data in missions.items():
        seq = data["seq"]
        scaled = scaler.transform(seq)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.zeros(1, len(seq), dtype=torch.bool, device=device)
        x.requires_grad_(True)

        with torch.enable_grad():
            logit = model(x, mask)
            prob = torch.sigmoid(logit)
            prob.backward()

        # Gradient × Input — sum over sequence → per-feature importance
        saliency = (x.grad * x).abs().squeeze(0)       # (seq_len, n_features)
        feature_attr = saliency.mean(dim=0).detach().cpu().numpy()  # (n_features,)
        all_attributions.append(feature_attr)

    # Re-enable gradients for model params
    for p in model.parameters():
        p.requires_grad_(True)

    if not all_attributions:
        return {}

    mean_attr = np.stack(all_attributions).mean(axis=0)  # (n_features,)
    # Normalize to [0, 1] for display
    mean_attr_norm = mean_attr / (mean_attr.max() + 1e-8)

    by_feature = {feat: round(float(mean_attr_norm[i]), 4) for i, feat in enumerate(FEATURE_COLS)}
    sorted_features = sorted(by_feature, key=by_feature.get, reverse=True)

    return {
        "method":          "gradient_saliency",
        "n_missions":      len(all_attributions),
        "by_feature":      by_feature,
        "sorted_features": sorted_features,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 3: Failure-Type Stratification
# ═══════════════════════════════════════════════════════════════════════════

def analyze_failure_types(
    model: TrajectoryTransformer,
    scaler,
    test_missions: dict[int, dict],
    device: torch.device,
    early_exit_frac: float = 1.0,
) -> dict:
    """
    Run inference on the test split and stratify AUC/F1/Acc by failure_type.

    Shows which failure modes are easiest/hardest to detect early.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    model.eval()

    # Collect predictions per failure type
    by_type: dict[str, dict] = {}

    for mission_id, data in test_missions.items():
        seq = data["seq"]
        ft  = data["failure_type"]
        lbl = data["label"]

        scaled = scaler.transform(seq)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.zeros(1, len(seq), dtype=torch.bool, device=device)

        with torch.no_grad():
            logit = model(x, mask)
            prob_success = float(torch.sigmoid(logit).item())

        if ft not in by_type:
            by_type[ft] = {"labels": [], "probs": []}
        by_type[ft]["labels"].append(lbl)
        by_type[ft]["probs"].append(prob_success)

    results = {}
    for ft, d in sorted(by_type.items()):
        y_true = np.array(d["labels"], dtype=int)
        y_prob = np.array(d["probs"])
        y_pred = (y_prob > 0.5).astype(int)

        n = len(y_true)
        n_success = int(y_true.sum())
        n_fail    = n - n_success

        has_both = len(np.unique(y_true)) > 1
        results[ft] = {
            "n":        n,
            "n_success": n_success,
            "n_fail":   n_fail,
            "acc":      round(float(accuracy_score(y_true, y_pred)), 4),
            "f1":       round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "auc":      round(float(roc_auc_score(y_true, y_prob)), 4) if has_both else None,
        }
        print(f"  {ft:<20} n={n:4d}  AUC={results[ft]['auc'] or 'N/A':>6}  F1={results[ft]['f1']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OrbitGuard Interpretability Analysis")
    parser.add_argument("--data",              type=str, default="data/merged/missions.parquet")
    parser.add_argument("--model-path",        type=str, default="models/transformer_production/best_model_transformer_binary.pt")
    parser.add_argument("--scaler-path",       type=str, default="models/transformer_production/scaler_transformer_binary.pkl")
    parser.add_argument("--n-sample-missions", type=int, default=10,  help="Missions to use for attention + attribution (half success, half failure)")
    parser.add_argument("--n-attr-missions",   type=int, default=200, help="Missions for aggregated feature attribution")
    parser.add_argument("--n-failure-missions",type=int, default=500, help="Test missions for failure-type breakdown")
    parser.add_argument("--early-exit",        type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=15)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--output-dir",        type=str, default="reports/interpretability")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▸ Device: {device}")

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = TrajectoryTransformer(input_dim=len(FEATURE_COLS), output_dim=1, task="binary")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"▸ Model loaded: {args.model_path}  ({total_params:,} params)")

    # ── Identify test-split mission IDs ──
    data_path = Path(args.data)
    print(f"\n▸ Loading mission index from {data_path}")
    test_ids = _split_mission_ids(data_path, seed=args.seed)
    print(f"  Test split: {len(test_ids)} missions")

    # ── Sample missions for attention/attribution (balanced success/failure) ──
    print(f"\n▸ Sampling {args.n_sample_missions} missions for attention heatmaps...")
    rng = random.Random(args.seed)
    sample_ids_list = rng.sample(list(test_ids), min(args.n_sample_missions * 3, len(test_ids)))

    sample_missions = _stream_missions(
        data_path, set(sample_ids_list[:args.n_sample_missions * 3]),
        downsample_factor=args.downsample_factor,
        early_exit_frac=args.early_exit,
    )
    # Balance success/failure
    successes = {k: v for k, v in sample_missions.items() if v["label"] == 1}
    failures  = {k: v for k, v in sample_missions.items() if v["label"] == 0}
    n_each    = max(1, args.n_sample_missions // 2)
    selected  = dict(list(successes.items())[:n_each])
    selected.update(dict(list(failures.items())[:n_each]))
    print(f"  Selected: {len(successes)} success candidates, {len(failures)} failure candidates → using {len(selected)}")

    # ── Sample missions for attribution (larger pool) ──
    attr_ids = rng.sample(list(test_ids), min(args.n_attr_missions, len(test_ids)))
    print(f"\n▸ Loading {len(attr_ids)} missions for feature attribution...")
    attr_missions = _stream_missions(
        data_path, set(attr_ids),
        downsample_factor=args.downsample_factor,
        early_exit_frac=args.early_exit,
    )

    # ── Sample missions for failure-type breakdown ──
    ft_ids = rng.sample(list(test_ids), min(args.n_failure_missions, len(test_ids)))
    print(f"\n▸ Loading {len(ft_ids)} missions for failure-type breakdown...")
    ft_missions = _stream_missions(
        data_path, set(ft_ids),
        downsample_factor=args.downsample_factor,
        early_exit_frac=args.early_exit,
    )

    # ── Run analyses ──
    print(f"\n▸ Analysis 1/3 — Attention heatmaps...")
    attention_results = analyze_attention(model, scaler, selected, device)

    print(f"\n▸ Analysis 2/3 — Feature attribution (n={len(attr_missions)})...")
    attribution_results = analyze_feature_attribution(model, scaler, attr_missions, device)

    print(f"\n▸ Analysis 3/3 — Failure-type breakdown (n={len(ft_missions)})...")
    failure_results = analyze_failure_types(
        model, scaler, ft_missions, device, args.early_exit
    )

    # ── Save summary ──
    summary = {
        "model_path":          args.model_path,
        "early_exit_frac":     args.early_exit,
        "feature_names":       list(FEATURE_COLS),
        "attention_missions":  attention_results,
        "feature_attribution": attribution_results,
        "failure_type_breakdown": failure_results,
    }

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──
    print(f"\n{'═' * 60}")
    print(f"  INTERPRETABILITY REPORT")
    print(f"{'═' * 60}")
    print(f"  Attention missions  : {len(attention_results)}")
    print(f"  Attribution missions: {attribution_results.get('n_missions', 0)}")
    if attribution_results.get("sorted_features"):
        top3 = attribution_results["sorted_features"][:3]
        print(f"  Top-3 features      : {', '.join(top3)}")
    print(f"  Failure types found : {list(failure_results.keys())}")
    print(f"\n▸ Report saved to {out_path}")


if __name__ == "__main__":
    main()
