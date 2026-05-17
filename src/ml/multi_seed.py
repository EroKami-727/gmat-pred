"""
OrbitGuard Multi-Seed Experiment — Statistical Robustness
==========================================================
Trains the production Transformer configuration (binary classification,
40% early exit) across N random seeds and reports mean ± std for AUC, F1,
and Accuracy, turning single-point results into defensible paper claims.

Usage
-----
  python -m src.ml.multi_seed \\
      --data data/merged/missions.parquet \\
      --seeds 42 123 456 789 1024 \\
      --early-exit 0.4 \\
      --output-dir reports/multi_seed

Output
------
  reports/multi_seed/seed_{N}_metrics.json   (one per seed)
  reports/multi_seed/summary.json            (mean ± std across all seeds)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.ml.dataset import create_dataloaders, FEATURE_COLS
from src.ml.model import TrajectoryTransformer
from src.ml.train import train_one_epoch, validate, _compute_metrics


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed: int):
    """Set all relevant RNG states for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_seed(
    data_path: str,
    seed: int,
    early_exit_frac: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """
    Full training and test evaluation for a single random seed.

    Uses the same Transformer config as the production model: binary
    classification, d_model=128, 4 layers, 8 heads, 40% early exit.
    The data split is also seeded so train/val/test missions differ
    between seeds, which is intentional — we want variance across seeds
    to reflect real generalisation uncertainty, not just weight init noise.
    """
    set_seed(seed)
    print(f"\n  {'─' * 50}")
    print(f"  SEED {seed}")
    print(f"  {'─' * 50}")

    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_path,
        target_mode="binary",
        early_exit_frac=early_exit_frac,
        batch_size=batch_size,
        seed=seed,
    )

    input_dim = len(FEATURE_COLS)
    train_labels = train_loader.dataset._y
    n_pos = (train_labels == 1).sum().item()
    n_neg = (train_labels == 0).sum().item()
    pos_weight = (
        torch.tensor([n_neg / n_pos], device=device) if n_pos > 0
        else torch.tensor([1.0], device=device)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = TrajectoryTransformer(input_dim=input_dim, output_dim=1, task="binary").to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    epoch_history = []
    ckpt_path = output_dir / f"best_model_seed{seed}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, "binary")
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, "binary")
        scheduler.step(val_loss)
        dt = time.time() - t0

        f1, auc = _compute_metrics(val_preds, val_labels) if val_preds else (0.0, 0.0)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  [seed={seed}] Epoch {epoch:02d} | Loss={val_loss:.4f} F1={f1:.3f} AUC={auc:.3f} | {dt:.1f}s")

        epoch_history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
            "f1":         round(f1, 6),
            "auc":        round(auc, 6),
            "lr":         lr_now,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device, "binary")
    test_f1, test_auc = _compute_metrics(test_preds, test_labels) if test_preds else (0.0, 0.0)

    result = {
        "seed":            seed,
        "early_exit_frac": early_exit_frac,
        "test_acc":        round(test_acc, 6),
        "test_f1":         round(test_f1, 6),
        "test_auc":        round(test_auc, 6),
        "epoch_history":   epoch_history,
    }
    print(f"  [seed={seed}] TEST → AUC={test_auc:.3f}  F1={test_f1:.3f}  Acc={test_acc:.2%}")
    return result


def main():
    parser = argparse.ArgumentParser(description="OrbitGuard Multi-Seed Robustness Experiment")
    parser.add_argument("--data",        type=str,   default="data/merged/missions.parquet")
    parser.add_argument("--seeds",       type=int,   nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--early-exit",  type=float, default=0.4)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--output-dir",  type=str,   default="reports/multi_seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\n▸ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n▸ Device: CPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▸ Multi-seed experiment: {len(args.seeds)} seeds × {args.epochs} epochs")
    print(f"▸ Config: Transformer | binary | early_exit={args.early_exit:.0%}")
    print(f"▸ Seeds:  {args.seeds}")

    all_results = []
    for seed in args.seeds:
        result = run_one_seed(
            data_path=args.data,
            seed=seed,
            early_exit_frac=args.early_exit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(result)
        with open(output_dir / f"seed_{seed}_metrics.json", "w") as f:
            json.dump(result, f, indent=2)

    aucs = [r["test_auc"] for r in all_results]
    f1s  = [r["test_f1"]  for r in all_results]
    accs = [r["test_acc"] for r in all_results]

    summary = {
        "seeds":           args.seeds,
        "early_exit_frac": args.early_exit,
        "n_seeds":         len(args.seeds),
        "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "values": aucs},
        "f1":  {"mean": float(np.mean(f1s)),  "std": float(np.std(f1s)),  "values": f1s},
        "acc": {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "values": accs},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  MULTI-SEED SUMMARY  ({len(args.seeds)} seeds, exit={args.early_exit:.0%})")
    print(f"{'═' * 60}")
    print(f"  AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"\n▸ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
