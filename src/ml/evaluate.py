"""
OrbitGuard — Model Evaluation & Inference Script
=================================================
Run a trained LSTM/Transformer against any mission Parquet database
and produce a full metrics report.

Usage
-----
  python -m src.ml.evaluate \\
      --model-path models/lstm_production/best_model_lstm_binary.pt \\
      --scaler-path models/lstm_production/scaler_lstm_binary.pkl \\
      --data data/production/missions.parquet \\
      --early-exit 1.0
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.ml.dataset import TrajectoryDataset, FEATURE_COLS
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer
from torch.utils.data import DataLoader


def evaluate(args):
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\n▸ Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"\n▸ Device: cpu")

    # ── Load Scaler ──
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"▸ Scaler loaded: {args.scaler_path}")

    # ── Load Dataset ──
    print(f"▸ Loading dataset: {args.data}")
    print(f"  Early-exit frac  : {args.early_exit:.0%}")
    print(f"  Downsample factor: {args.downsample_factor}x")
    ds = TrajectoryDataset(
        parquet_path=args.data,
        target_mode="binary",
        downsample_factor=args.downsample_factor,
        early_exit_frac=args.early_exit,
        scaler=scaler,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    n_missions = len(ds)
    n_success = int((ds._y == 1).sum().item())
    n_fail = n_missions - n_success
    print(f"  Missions loaded  : {n_missions}  ({n_success} success / {n_fail} failure)")

    # ── Load Model ──
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

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"▸ Model loaded: {args.model_path}  ({total_params:,} params)")

    # ── Inference ──
    all_probs = []
    all_labels = []

    print(f"\n▸ Running inference on {n_missions} missions...")
    with torch.no_grad():
        for X, y, lengths in tqdm(loader, desc="  Evaluating"):
            X = X.to(device)
            if args.arch == "lstm":
                logits = model(X, lengths)
            else:
                mask = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
                for i, l in enumerate(lengths):
                    mask[i, int(l):] = True
                logits = model(X, mask)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    # ── Metrics ──
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        confusion_matrix, classification_report
    )

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, preds)

    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"  Data       : {args.data}")
    print(f"  Model      : {args.model_path}")
    print(f"  Early-exit : {args.early_exit:.0%} of trajectory seen")
    print(f"{'=' * 60}")
    print(f"  Accuracy   : {acc:.4%}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Precision  : {precision:.4f}  (of 'success' calls, how many were right)")
    print(f"  Recall     : {recall:.4f}  (of true successes, how many did we catch)")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred:Fail  Pred:Success")
    print(f"  Actual:Fail    {tn:5d}       {fp:5d}")
    print(f"  Actual:Succ    {fn:5d}       {tp:5d}")
    print(f"\n  Missions that would be SAVED (correctly pruned failures): {tn}")
    print(f"  Good runs incorrectly killed (FP — waste of propellant) : {fp}")
    print(f"  Failed runs that slipped through (FN — wasted compute)  : {fn}")
    print(f"{'=' * 60}\n")

    # Save results to file
    out_path = Path(args.model_path).parent / f"eval_results_exit{int(args.early_exit*100)}pct.txt"
    with open(out_path, "w") as f:
        f.write(f"Data: {args.data}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Early-exit: {args.early_exit:.0%}\n")
        f.write(f"Accuracy:  {acc:.4%}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
    print(f"▸ Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrbitGuard — Evaluate trained LSTM on any mission DB")
    parser.add_argument("--model-path",  type=str, required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--scaler-path", type=str, required=True,  help="Path to scaler .pkl")
    parser.add_argument("--data",        type=str, required=True,  help="Path to missions.parquet")
    parser.add_argument("--arch",        type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--hidden-dim",  type=int, default=128)
    parser.add_argument("--num-layers",  type=int, default=2)
    parser.add_argument("--early-exit",  type=float, default=1.0,  help="Fraction of trajectory (0.1–1.0)")
    parser.add_argument("--downsample-factor", type=int, default=15)
    parser.add_argument("--batch-size",  type=int, default=128)
    args = parser.parse_args()
    evaluate(args)
