"""
OrbitGuard — Calibration & Extended Evaluation
===============================================
Computes PR-AUC, Brier score, ECE, best F1 threshold (tuned on val),
per-split confusion matrices, and reliability diagrams for any trained
Transformer or LSTM model.

Usage
-----
  python -m src.ml.calibration_eval \\
      --model-path models/transformer_production/best_model_transformer_binary.pt \\
      --scaler-path models/transformer_production/scaler_transformer_binary.pkl \\
      --data data/merged_through_neptune_15min/missions.parquet \\
      --arch transformer \\
      --output-dir reports/calibration
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.ml.dataset import TrajectoryDataset, FEATURE_COLS
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (uniform-width bins)."""
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return float(ece)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds to find the one that maximises F1 on the given split."""
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 199)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        score = f1_score(y_true, (y_prob > thr).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr, best_f1


def _metrics_block(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    n_unique = len(np.unique(y_true))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if n_unique > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if n_unique > 1 else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": _compute_ece(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "success_rate": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_model(args, device: torch.device):
    input_dim = len(FEATURE_COLS)
    if args.arch == "lstm":
        model = TrajectoryLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=1,
            task="binary",
        )
    else:
        model = TrajectoryTransformer(input_dim=input_dim, output_dim=1, task="binary")
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def _run_inference(
    model,
    parquet_path: str | Path,
    scaler,
    args,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) for all missions in parquet_path."""
    ds = TrajectoryDataset(
        parquet_path=parquet_path,
        target_mode="binary",
        downsample_factor=args.downsample_factor,
        max_seq_len=args.max_seq_len,
        early_exit_frac=args.early_exit,
        scaler=scaler,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            if args.arch == "lstm":
                logits = model(X, lengths)
            else:
                mask = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
                for i, ln in enumerate(lengths):
                    mask[i, int(ln):] = True
                logits = model(X, mask)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())
    return np.array(all_probs, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Mission-level metadata (for per-split breakdown)
# ---------------------------------------------------------------------------

def _load_mission_meta(parquet_path: str | Path) -> dict[int, str]:
    """Return {mission_id: target_body} from first row per mission."""
    pf = pq.ParquetFile(parquet_path)
    cols = [c for c in ["mission_id", "target_body"] if c in pf.schema_arrow.names]
    if "target_body" not in cols:
        return {}

    meta: dict[int, str] = {}
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas()
        first_rows = df.groupby("mission_id", sort=False).first().reset_index()
        for _, row in first_rows.iterrows():
            mid = int(row["mission_id"])
            if mid not in meta:
                meta[mid] = str(row["target_body"])
    return meta


# ---------------------------------------------------------------------------
# Split helpers (mirrors create_dataloaders logic exactly)
# ---------------------------------------------------------------------------

def _split_mission_ids(
    parquet_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[set, set, set]:
    pf = pq.ParquetFile(parquet_path)
    mission_ids = pf.read(columns=["mission_id"])["mission_id"].unique().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(mission_ids)
    n_total = len(mission_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_ids = set(mission_ids[:n_train])
    val_ids = set(mission_ids[n_train:n_train + n_val])
    test_ids = set(mission_ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def _write_split_parquet(
    parquet_path: str | Path,
    keep_ids: set,
    out_path: str | Path,
) -> None:
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    with pq.ParquetWriter(out_path, schema) as writer:
        for batch in pf.iter_batches(batch_size=200_000):
            mask = np.isin(batch["mission_id"].to_numpy(), list(keep_ids))
            writer.write_batch(batch.filter(pa.array(mask)))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def _save_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_cal: np.ndarray | None,
    out_path: Path,
    n_bins: int = 15,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available — skipping reliability diagram]")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ax.plot(mean_pred, frac_pos, "o-", color="#e07b39", label="Uncalibrated")

    if y_prob_cal is not None:
        frac_pos_cal, mean_pred_cal = calibration_curve(
            y_true, y_prob_cal, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred_cal, frac_pos_cal, "s--", color="#4c8eda", label="Isotonic (val-fitted)")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram — OrbitGuard Transformer")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Reliability diagram saved: {out_path}")


def _save_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#e07b39", lw=1.5, label=f"Transformer (PR-AUC={pr_auc:.4f})")
    baseline = float(y_true.mean())
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"No-skill baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — OrbitGuard Transformer (test split)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ PR curve saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▸ Device: {device}")

    # ── Load scaler ──
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"▸ Scaler: {args.scaler_path}")

    # ── Deterministic split (must match training) ──
    print(f"▸ Computing mission splits (seed={args.seed})...")
    train_ids, val_ids, test_ids = _split_mission_ids(
        args.data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # ── Mission-level metadata for per-target breakdown ──
    print("▸ Loading mission metadata (target_body)...")
    mission_meta = _load_mission_meta(args.data)

    # ── Model ──
    print(f"▸ Loading model: {args.model_path}")
    model = _load_model(args, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params:,} parameters")

    with tempfile.TemporaryDirectory(dir=args.tmp_dir, ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        val_parquet = tmp / "val.parquet"
        test_parquet = tmp / "test.parquet"

        print("▸ Writing val split parquet...")
        _write_split_parquet(args.data, val_ids, val_parquet)
        print("▸ Running inference on val split...")
        val_probs, val_labels = _run_inference(model, val_parquet, scaler, args, device)

        print("▸ Writing test split parquet...")
        _write_split_parquet(args.data, test_ids, test_parquet)
        print("▸ Running inference on test split...")
        test_probs, test_labels = _run_inference(model, test_parquet, scaler, args, device)

    # ── Threshold tuning on val ──
    print("\n▸ Tuning decision threshold on val split...")
    best_thr, val_f1_at_thr = _best_f1_threshold(val_labels, val_probs)
    print(f"  Best threshold: {best_thr:.4f}  val-F1: {val_f1_at_thr:.4f}  (vs F1@0.5: "
          f"{f1_score(val_labels, (val_probs > 0.5).astype(int), zero_division=0):.4f})")

    # ── Isotonic calibration (fit on val, apply to test) ──
    print("▸ Fitting isotonic calibrator on val split...")
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(val_probs, val_labels)
    test_probs_cal = iso_cal.predict(test_probs).astype(np.float32)

    # ── Overall test metrics ──
    print("\n▸ Computing overall test metrics...")
    overall_05 = _metrics_block(test_labels, test_probs, threshold=0.5)
    overall_tuned = _metrics_block(test_labels, test_probs, threshold=best_thr)
    overall_cal = _metrics_block(test_labels, test_probs_cal, threshold=0.5)

    # ── Per-target breakdown ──
    test_mission_ids = sorted(test_ids)
    # Map mission index order from test parquet to mission IDs
    # Re-read mission IDs in order from the written file is expensive; instead
    # we match by reading the parquet mission_id column once.
    print("▸ Computing per-target breakdown...")
    per_target: dict[str, dict] = {}

    if mission_meta:
        # Build mission-order mapping from the parquet
        pf_test_tmp = None
        with tempfile.TemporaryDirectory(dir=args.tmp_dir, ignore_cleanup_errors=True) as tmpdir2:
            tmp2 = Path(tmpdir2)
            test_parquet2 = tmp2 / "test2.parquet"
            _write_split_parquet(args.data, test_ids, test_parquet2)

            pf2 = pq.ParquetFile(test_parquet2)
            needed_cols = [c for c in ["mission_id", "target_body"]
                           if c in pf2.schema_arrow.names]
            ordered_mids: list[int] = []
            seen: set = set()
            for batch in pf2.iter_batches(batch_size=500_000, columns=needed_cols):
                for mid in batch["mission_id"].to_numpy():
                    mid_int = int(mid)
                    if mid_int not in seen:
                        seen.add(mid_int)
                        ordered_mids.append(mid_int)

        # test_probs/test_labels are in dataset order (grouped by mission_id as
        # encountered by TrajectoryDataset streaming), which matches ordered_mids
        if len(ordered_mids) == len(test_probs):
            mid_arr = np.array(ordered_mids)
            target_col = np.array([mission_meta.get(m, "unknown") for m in ordered_mids])
            for tgt in sorted(set(target_col)):
                mask = target_col == tgt
                if mask.sum() == 0:
                    continue
                per_target[tgt] = _metrics_block(
                    test_labels[mask], test_probs[mask], threshold=best_thr
                )
                per_target[tgt]["calibrated_brier"] = float(
                    brier_score_loss(test_labels[mask], test_probs_cal[mask])
                )
        else:
            print(f"  [Warning: mission count mismatch {len(ordered_mids)} vs "
                  f"{len(test_probs)} — skipping per-target breakdown]")

    # ── Plots ──
    _save_reliability_diagram(
        test_labels, test_probs, test_probs_cal,
        out_dir / "reliability_diagram.png",
    )
    _save_pr_curve(
        test_labels, test_probs, overall_05["pr_auc"],
        out_dir / "pr_curve.png",
    )

    # ── Print summary ──
    print(f"\n{'=' * 65}")
    print(f"  CALIBRATION EVALUATION — {Path(args.model_path).parent.name}")
    print(f"  Data: {args.data}  |  Early-exit: {args.early_exit:.0%}")
    print(f"{'=' * 65}")
    header = f"  {'Metric':<22} {'@0.5':>10} {'@best-thr':>10} {'Calibrated':>12}"
    print(header)
    print(f"  {'-'*58}")
    rows_to_print = [
        ("ROC-AUC",     "roc_auc"),
        ("PR-AUC",      "pr_auc"),
        ("F1",          "f1"),
        ("Accuracy",    "accuracy"),
        ("Brier score", "brier_score"),
        ("ECE",         "ece"),
    ]
    for label, key in rows_to_print:
        v_05  = overall_05[key]
        v_thr = overall_tuned[key]
        v_cal = overall_cal[key]
        print(f"  {label:<22} {v_05:>10.4f} {v_thr:>10.4f} {v_cal:>12.4f}")
    print(f"  {'Best threshold (val)':<22} {best_thr:>10.4f}")
    print(f"{'=' * 65}\n")

    if per_target:
        print(f"  {'Target':<14} {'N':>5} {'Succ%':>6} {'PR-AUC':>7} {'F1':>7} {'Brier':>7} {'ECE':>7}")
        print(f"  {'-'*55}")
        for tgt, m in sorted(per_target.items()):
            print(
                f"  {tgt:<14} {m['n_samples']:>5} "
                f"{m['success_rate']:>6.1%} "
                f"{m['pr_auc']:>7.4f} "
                f"{m['f1']:>7.4f} "
                f"{m['brier_score']:>7.4f} "
                f"{m['ece']:>7.4f}"
            )
        print()

    # ── Save JSON ──
    report = {
        "model_path": str(args.model_path),
        "data_path": str(args.data),
        "arch": args.arch,
        "early_exit": args.early_exit,
        "downsample_factor": args.downsample_factor,
        "seed": args.seed,
        "split": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "val_best_threshold": best_thr,
        "val_f1_at_best_threshold": val_f1_at_thr,
        "test_at_0_5": overall_05,
        "test_at_best_threshold": overall_tuned,
        "test_isotonic_calibrated": overall_cal,
        "per_target": per_target,
    }
    report_path = out_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"▸ Full report saved: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OrbitGuard — Calibration & extended evaluation"
    )
    parser.add_argument("--model-path",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--scaler-path", required=True, help="Path to scaler .pkl")
    parser.add_argument("--data",        required=True, help="Path to missions.parquet")
    parser.add_argument("--arch", default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--hidden-dim",  type=int, default=128)
    parser.add_argument("--num-layers",  type=int, default=2)
    parser.add_argument("--early-exit",  type=float, default=0.4,
                        help="Fraction of trajectory to use (0.1–1.0)")
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output-dir",  default="reports/calibration",
                        help="Directory for JSON report and PNG plots")
    parser.add_argument("--tmp-dir", default=None,
                        help="Directory for temp val/test split parquets. Defaults to "
                             "the system temp dir (often RAM-backed tmpfs) — point this "
                             "at real disk for large datasets to avoid running out of RAM.")
    args = parser.parse_args()
    main(args)
