"""
OrbitGuard Ablation Study — Early-Exit Fraction Sweep
======================================================
Sweeps `early_exit_frac` across [0.1, 0.2, 0.3, 0.4, 0.6, 1.0] and trains
an LSTM for each value. Saves a summary JSON for dashboard plotting.

Usage:
    python3 -m src.ml.ablation \\
        --data data/merged/missions.parquet \\
        --epochs 30 \\
        --output-dir models/ablation

Output:
    reports/ablation/ablation_results.json
    models/ablation/best_model_lstm_binary_exit{frac}.pt  (per fraction)
"""

import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.ml.dataset import create_dataloaders
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer
from src.ml.train import train_one_epoch, validate, _compute_metrics


ABLATION_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]


def run_single(
    data_path: str,
    early_exit_frac: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    output_dir: Path,
    device: torch.device,
    model_type: str = "transformer",
) -> dict:
    """Train one LSTM for a given early_exit_frac and return metrics dict."""

    label = f"{int(early_exit_frac * 100)}pct"
    print(f"\n{'═' * 60}")
    print(f"  EARLY EXIT = {early_exit_frac:.0%}  ({label})")
    print(f"{'═' * 60}")

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_path,
        target_mode="binary",
        early_exit_frac=early_exit_frac,
        batch_size=batch_size,
    )

    input_dim = train_loader.dataset.num_features
    train_labels = train_loader.dataset._y
    n_pos = (train_labels == 1).sum().item()
    n_neg = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 else torch.tensor([1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if model_type == "transformer":
        model = TrajectoryTransformer(
            input_dim=input_dim,
            output_dim=1,
            task="binary",
        ).to(device)
    else:
        model = TrajectoryLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            task="binary",
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    best_auc = 0.0
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, "binary")
        val_loss, val_acc, val_preds, val_labels_list = validate(model, val_loader, criterion, device, "binary")
        scheduler.step(val_loss)
        dt = time.time() - t0

        f1, auc = 0.0, 0.0
        if val_preds:
            f1, auc = _compute_metrics(val_preds, val_labels_list)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  [{label}] Epoch {epoch:02d} | Loss={val_loss:.4f} Acc={val_acc:.2%} "
              f"F1={f1:.3f} AUC={auc:.3f} lr={current_lr:.2e} | {dt:.1f}s")

        epoch_metrics.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6),
            "f1": round(f1, 6), "auc": round(auc, 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = auc
            model_path = output_dir / f"best_model_{model_type}_binary_exit{label}.pt"
            torch.save(model.state_dict(), model_path)

    # Final test evaluation
    best_model_path = output_dir / f"best_model_{model_type}_binary_exit{label}.pt"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc, test_preds, test_labels_list = validate(model, test_loader, criterion, device, "binary")
    test_f1, test_auc = _compute_metrics(test_preds, test_labels_list) if test_preds else (0.0, 0.0)

    # Confusion matrix
    tp = sum(1 for p, l in zip(test_preds, test_labels_list) if p > 0.5 and l == 1)
    fp = sum(1 for p, l in zip(test_preds, test_labels_list) if p > 0.5 and l == 0)
    fn = sum(1 for p, l in zip(test_preds, test_labels_list) if p <= 0.5 and l == 1)
    tn = sum(1 for p, l in zip(test_preds, test_labels_list) if p <= 0.5 and l == 0)

    result = {
        "early_exit_frac": early_exit_frac,
        "early_exit_pct": int(early_exit_frac * 100),
        "model_type": model_type,
        "test_acc": round(test_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_f1": round(test_f1, 6),
        "test_auc": round(test_auc, 6),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "epoch_history": epoch_metrics,
        "model_path": str(best_model_path),
    }

    print(f"\n  [{label}] TEST → AUC={test_auc:.3f}  F1={test_f1:.3f}  Acc={test_acc:.2%}")
    return result


def main():
    parser = argparse.ArgumentParser(description="OrbitGuard Ablation — Early-Exit Sweep")
    parser.add_argument("--data", type=str, default="data/merged/missions.parquet")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="models/ablation")
    parser.add_argument("--fractions", type=float, nargs="+", default=ABLATION_FRACTIONS,
                        help="List of early-exit fractions to sweep (e.g. 0.1 0.2 0.4 1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n▸ GPU: {gpu_name}")
    else:
        print(f"\n▸ Device: CPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path("reports/ablation")
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▸ Ablation sweep over fractions: {args.fractions}")
    print(f"▸ Model: {args.model}")
    print(f"▸ Epochs per run: {args.epochs}")
    print(f"▸ Data: {args.data}")

    all_results = []
    for frac in sorted(args.fractions):
        result = run_single(
            data_path=args.data,
            early_exit_frac=frac,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            output_dir=output_dir,
            device=device,
            model_type=args.model,
        )
        all_results.append(result)

        # Save incrementally (so dashboard can read partial results during a long run)
        out_path = report_dir / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  ABLATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"{'Exit %':>8}  {'AUC':>6}  {'F1':>6}  {'Acc':>7}")
    print(f"{'─' * 36}")
    for r in all_results:
        print(f"  {r['early_exit_pct']:>5}%  {r['test_auc']:>6.3f}  {r['test_f1']:>6.3f}  {r['test_acc']:>7.2%}")

    print(f"\n▸ Results saved to {out_path}")


if __name__ == "__main__":
    main()
