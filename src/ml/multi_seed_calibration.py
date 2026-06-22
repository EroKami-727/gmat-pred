"""
OrbitGuard — Multi-Seed Calibration Audit
==========================================
Loads the five pre-trained seed checkpoints from reports/multi_seed/ and
evaluates each with PR-AUC, Brier score, ECE, and isotonic-calibrated
variants.  Does NOT re-train — only inference on the deterministic test splits.

Usage
-----
  python -m src.ml.multi_seed_calibration \\
      --data data/merged/missions.parquet \\
      --checkpoint-dir reports/multi_seed \\
      --seeds 42 123 456 789 1024 \\
      --output-dir reports/calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src.ml.dataset import FEATURE_COLS
from src.ml.model import TrajectoryTransformer


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Helpers (shared with calibration_eval but kept local to avoid cross-import)
# ---------------------------------------------------------------------------

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_prob)) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 199)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        score = f1_score(y_true, (y_prob > thr).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr, best_f1


def _metrics_block(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_prob > threshold).astype(int)
    n_unique = len(np.unique(y_true))
    return {
        "threshold": float(threshold),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_true, y_prob)) if n_unique > 1 else 0.5,
        "pr_auc":      float(average_precision_score(y_true, y_prob)) if n_unique > 1 else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece":         _compute_ece(y_true, y_prob),
        "n_samples":   int(len(y_true)),
        "success_rate": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# In-memory dataset loading (avoids writing split parquets to disk)
# ---------------------------------------------------------------------------

def load_all_missions(
    parquet_path: str | Path,
    downsample_factor: int,
    early_exit: float,
) -> dict[int, dict]:
    """
    Stream parquet, downsample, apply early-exit, return
    {mission_id: {"seq": np.ndarray, "label": int}} for all missions.
    After downsampling+early-exit the full Moon dataset fits in ~100 MB.
    """
    needed = ["mission_id", "elapsed_secs", "label"] + FEATURE_COLS
    pf = pq.ParquetFile(parquet_path)
    needed = [c for c in needed if c in pf.schema_arrow.names]

    missions: dict[int, dict] = {}
    current_mid = None
    current_rows: list = []

    def _process(rows: list, mid: int) -> None:
        import pandas as pd
        df = pd.concat(rows).sort_values("elapsed_secs")
        label = int(df.iloc[0]["label"])
        df = df.iloc[::downsample_factor].reset_index(drop=True)
        if early_exit < 1.0:
            keep_n = max(1, int(len(df) * early_exit))
            df = df.iloc[:keep_n]
        missions[int(mid)] = {
            "seq":   df[FEATURE_COLS].values.astype(np.float32),
            "label": label,
        }

    print(f"  [Streaming {pf.metadata.num_rows:,} rows → in-memory (no disk writes)]")
    for batch in pf.iter_batches(batch_size=500_000, columns=needed):
        df = batch.to_pandas()
        for mid, group in df.groupby("mission_id", sort=False):
            if mid != current_mid:
                if current_rows and current_mid is not None:
                    _process(current_rows, current_mid)
                current_mid = mid
                current_rows = [group]
            else:
                current_rows.append(group)
    if current_rows and current_mid is not None:
        _process(current_rows, current_mid)

    return missions


def _split_mission_ids(
    all_ids: np.ndarray, seed: int,
    train_ratio: float = 0.7, val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = all_ids.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def _pad_and_batch(
    seqs: list[np.ndarray], batch_size: int, max_len: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (padded_batch, lengths) arrays ready for torch."""
    n = len(seqs)
    for start in range(0, n, batch_size):
        batch_seqs = seqs[start:start + batch_size]
        lengths = np.array([s.shape[0] for s in batch_seqs], dtype=np.int64)
        padded = np.zeros((len(batch_seqs), max_len, len(FEATURE_COLS)), dtype=np.float32)
        for i, s in enumerate(batch_seqs):
            l = min(s.shape[0], max_len)
            padded[i, :l] = s[:l]
        yield padded, lengths


def _infer_in_memory(
    model: torch.nn.Module,
    seqs: list[np.ndarray],
    labels: list[int],
    scaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    # Scale sequences
    scaled = []
    for s in seqs:
        scaled.append(scaler.transform(s))
    max_len = max(s.shape[0] for s in scaled)

    probs_out: list[float] = []
    with torch.no_grad():
        for padded, lengths in _pad_and_batch(scaled, batch_size, max_len):
            X = torch.tensor(padded, dtype=torch.float32, device=device)
            lens = lengths.tolist()
            mask = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
            for i, ln in enumerate(lens):
                mask[i, int(ln):] = True
            logits = model(X, mask)
            probs_out.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist())

    return np.array(probs_out, dtype=np.float32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def evaluate_seed(
    seed: int,
    all_missions: dict[int, dict],
    ckpt_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict:
    from sklearn.preprocessing import RobustScaler

    print(f"\n  ── Seed {seed} ──")
    all_ids = np.array(sorted(all_missions.keys()))
    train_ids, val_ids, test_ids = _split_mission_ids(all_ids, seed)
    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    def _gather(ids: np.ndarray) -> tuple[list, list]:
        seqs = [all_missions[m]["seq"] for m in ids]
        labs = [all_missions[m]["label"] for m in ids]
        return seqs, labs

    train_seqs, _ = _gather(train_ids)
    val_seqs,   val_labs  = _gather(val_ids)
    test_seqs,  test_labs = _gather(test_ids)

    # Fit scaler on train only
    scaler = RobustScaler()
    scaler.fit(np.vstack(train_seqs))
    del train_seqs  # free RAM

    # Load model
    model = TrajectoryTransformer(
        input_dim=len(FEATURE_COLS), output_dim=1, task="binary"
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    print(f"  Inference on val ({len(val_seqs)} missions)...")
    val_probs, val_labels = _infer_in_memory(
        model, val_seqs, val_labs, scaler, batch_size, device
    )
    print(f"  Inference on test ({len(test_seqs)} missions)...")
    test_probs, test_labels = _infer_in_memory(
        model, test_seqs, test_labs, scaler, batch_size, device
    )

    # Threshold tuning on val
    best_thr, val_f1 = _best_f1_threshold(val_labels, val_probs)

    # Isotonic calibrator (fit on val, apply to test)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, val_labels)
    test_probs_cal = iso.predict(test_probs).astype(np.float32)

    result = {
        "seed": seed,
        "val_best_threshold":  best_thr,
        "val_f1_at_threshold": val_f1,
        "test_at_0_5":       _metrics_block(test_labels, test_probs, 0.5),
        "test_at_best_thr":  _metrics_block(test_labels, test_probs, best_thr),
        "test_calibrated":   _metrics_block(test_labels, test_probs_cal, 0.5),
    }

    m  = result["test_at_0_5"]
    mc = result["test_calibrated"]
    print(
        f"  AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
        f"F1@0.5={m['f1']:.4f}  F1@thr={result['test_at_best_thr']['f1']:.4f}  "
        f"Brier={m['brier_score']:.4f}→{mc['brier_score']:.4f}  "
        f"ECE={m['ece']:.4f}→{mc['ece']:.4f}  thr={best_thr:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Summary across seeds
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> dict:
    arr = np.array(values)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": values}


def summarise(results: list[dict]) -> dict:
    def _collect(key_path: list[str]) -> list[float]:
        out = []
        for r in results:
            node = r
            for k in key_path:
                node = node[k]
            out.append(float(node))
        return out

    return {
        "seeds":            [r["seed"] for r in results],
        "n_seeds":          len(results),
        # @0.5 threshold
        "roc_auc":   _mean_std(_collect(["test_at_0_5", "roc_auc"])),
        "pr_auc":    _mean_std(_collect(["test_at_0_5", "pr_auc"])),
        "f1":        _mean_std(_collect(["test_at_0_5", "f1"])),
        "accuracy":  _mean_std(_collect(["test_at_0_5", "accuracy"])),
        "brier":     _mean_std(_collect(["test_at_0_5", "brier_score"])),
        "ece":       _mean_std(_collect(["test_at_0_5", "ece"])),
        # best threshold (tuned on val)
        "f1_best_thr": _mean_std(_collect(["test_at_best_thr", "f1"])),
        "thresholds":  _mean_std(_collect(["val_best_threshold"])),
        # isotonic calibrated
        "brier_cal": _mean_std(_collect(["test_calibrated", "brier_score"])),
        "ece_cal":   _mean_std(_collect(["test_calibrated", "ece"])),
        "pr_auc_cal": _mean_std(_collect(["test_calibrated", "pr_auc"])),
    }


def _print_summary(s: dict) -> None:
    print(f"\n{'═' * 70}")
    print(f"  MULTI-SEED CALIBRATION SUMMARY  ({s['n_seeds']} seeds)")
    print(f"{'═' * 70}")
    rows = [
        ("ROC-AUC  (@0.5)",  "roc_auc"),
        ("PR-AUC   (@0.5)",  "pr_auc"),
        ("F1       (@0.5)",  "f1"),
        ("F1   (best-thr)",  "f1_best_thr"),
        ("Accuracy (@0.5)",  "accuracy"),
        ("Brier    (@0.5)",  "brier"),
        ("Brier (isotonic)", "brier_cal"),
        ("ECE      (@0.5)",  "ece"),
        ("ECE  (isotonic)",  "ece_cal"),
        ("Best threshold",   "thresholds"),
    ]
    print(f"  {'Metric':<26} {'Mean':>8} {'± Std':>8}  Values")
    print(f"  {'-'*60}")
    for label, key in rows:
        m = s[key]
        vals = "  " + "  ".join(f"{v:.4f}" for v in m["values"])
        print(f"  {label:<26} {m['mean']:>8.4f} {m['std']:>8.4f}{vals}")
    print(f"{'═' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed calibration audit (no re-training)"
    )
    parser.add_argument("--data",           required=True, help="Path to missions.parquet")
    parser.add_argument("--checkpoint-dir", default="reports/multi_seed",
                        help="Directory containing best_model_seed*.pt files")
    parser.add_argument("--seeds",          type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--early-exit",     type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=15)
    parser.add_argument("--batch-size",     type=int, default=128)
    parser.add_argument("--output-dir",     default="reports/calibration")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▸ Device: {device}")
    print(f"▸ Seeds: {args.seeds}")
    print(f"▸ Checkpoints: {ckpt_dir}")

    # Load all missions once into memory — avoids writing split parquets to disk
    print(f"\n▸ Loading all missions into memory (downsample={args.downsample_factor}, exit={args.early_exit:.0%})...")
    all_missions = load_all_missions(args.data, args.downsample_factor, args.early_exit)
    print(f"  Loaded {len(all_missions)} missions")

    results = []
    for seed in args.seeds:
        ckpt = ckpt_dir / f"best_model_seed{seed}.pt"
        if not ckpt.exists():
            print(f"  [WARNING] Checkpoint not found: {ckpt} — skipping seed {seed}")
            continue
        result = evaluate_seed(
            seed=seed,
            all_missions=all_missions,
            ckpt_path=ckpt,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(result)
        # Save per-seed results incrementally
        (out_dir / f"calib_seed_{seed}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    if not results:
        print("No checkpoints found. Run multi_seed.py first.")
        return

    summary = summarise(results)
    _print_summary(summary)

    summary_path = out_dir / "multi_seed_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"▸ Summary saved: {summary_path}")

    # Also save all per-seed details in one file
    all_path = out_dir / "multi_seed_calibration_all.json"
    all_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"▸ Per-seed details: {all_path}")


if __name__ == "__main__":
    main()
