"""
OrbitGuard Architecture Ablation — Component Contribution Study
===============================================================
Trains transformer variants at a fixed early-exit fraction (default 40%)
to measure how much each design choice contributes to performance.

Variants
--------
1. full_transformer       — Production config (CLS token, pos encoding, all features)
2. no_cls_token           — Mean pooling over sequence instead of CLS token
3. no_pos_encoding        — No sinusoidal positional encoding
4. no_context_features    — Drop mu_ratio, soi_ratio, dist_ratio (last 3 of 13 features)
5. lstm                   — Bidirectional LSTM baseline at the same exit fraction

Usage
-----
  python -m src.ml.arch_ablation \\
      --data data/merged/missions.parquet \\
      --early-exit 0.4 \\
      --epochs 30 \\
      --output-dir reports/arch_ablation

  # Run a subset only
  python -m src.ml.arch_ablation \\
      --variants full_transformer lstm

Output
------
  reports/arch_ablation/arch_ablation_results.json
  reports/arch_ablation/best_model_{variant}.pt    (checkpoint per variant)
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.ml.dataset import create_dataloaders, FEATURE_COLS
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer
from src.ml.train import _compute_metrics


# Indices of the three context features to zero out in the no_context_features variant
_CONTEXT_IDXS = [
    FEATURE_COLS.index("mu_ratio"),
    FEATURE_COLS.index("soi_ratio"),
    FEATURE_COLS.index("dist_ratio"),
]

VARIANTS = [
    {
        "name":            "full_transformer",
        "label":           "Transformer (full)",
        "arch":            "transformer",
        "use_cls_token":   True,
        "use_pos_encoding":True,
        "mask_context":    False,
    },
    {
        "name":            "no_cls_token",
        "label":           "No CLS token (mean pool)",
        "arch":            "transformer",
        "use_cls_token":   False,
        "use_pos_encoding":True,
        "mask_context":    False,
    },
    {
        "name":            "no_pos_encoding",
        "label":           "No positional encoding",
        "arch":            "transformer",
        "use_cls_token":   True,
        "use_pos_encoding":False,
        "mask_context":    False,
    },
    {
        "name":            "no_context_features",
        "label":           "No context features",
        "arch":            "transformer",
        "use_cls_token":   True,
        "use_pos_encoding":True,
        "mask_context":    True,
    },
    {
        "name":            "lstm",
        "label":           "LSTM",
        "arch":            "lstm",
        "use_cls_token":   None,
        "use_pos_encoding":None,
        "mask_context":    False,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(variant: dict, input_dim: int, device: torch.device) -> nn.Module:
    if variant["arch"] == "lstm":
        return TrajectoryLSTM(input_dim=input_dim, output_dim=1, task="binary").to(device)
    return TrajectoryTransformer(
        input_dim=input_dim,
        output_dim=1,
        task="binary",
        use_cls_token=variant["use_cls_token"],
        use_pos_encoding=variant["use_pos_encoding"],
    ).to(device)


def _make_pad_mask(lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Build (batch, seq_len) bool mask — True where padding exists."""
    mask = torch.zeros(len(lengths), seq_len, dtype=torch.bool, device=device)
    for i, l in enumerate(lengths):
        mask[i, int(l):] = True
    return mask


def _forward(model: nn.Module, X: torch.Tensor, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(model, TrajectoryLSTM):
        return model(X, lengths)
    mask = _make_pad_mask(lengths, X.size(1), device)
    return model(X, mask)


def _apply_context_mask(X: torch.Tensor) -> torch.Tensor:
    X = X.clone()
    X[:, :, _CONTEXT_IDXS] = 0.0
    return X


# ═══════════════════════════════════════════════════════════════════════════
# Single-Variant Training
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(
    variant: dict,
    data_path: str,
    early_exit_frac: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    _set_seed(seed)
    print(f"\n{'═' * 60}")
    print(f"  VARIANT: {variant['label']}")
    print(f"{'═' * 60}")

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

    model = _build_model(variant, input_dim, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    epoch_history = []
    ckpt_path = output_dir / f"best_model_{variant['name']}.pt"
    mask_ctx = variant["mask_context"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y, lengths in train_loader:
            X, y = X.to(device), y.to(device)
            if mask_ctx:
                X = _apply_context_mask(X)
            optimizer.zero_grad()
            logits = _forward(model, X, lengths, device)
            loss = criterion(logits, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += ((torch.sigmoid(logits) > 0.5).long() == y).sum().item()
            total += y.size(0)
        train_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss_sum, val_preds_list, val_labels_list = 0.0, [], []
        with torch.no_grad():
            for X, y, lengths in val_loader:
                X, y = X.to(device), y.to(device)
                if mask_ctx:
                    X = _apply_context_mask(X)
                logits = _forward(model, X, lengths, device)
                val_loss_sum += criterion(logits, y.float()).item()
                val_preds_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                val_labels_list.extend(y.cpu().numpy().tolist())
        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        f1, auc = _compute_metrics(val_preds_list, val_labels_list) if val_preds_list else (0.0, 0.0)
        dt = time.time() - t0
        print(f"  Epoch {epoch:02d} | Loss={val_loss:.4f}  F1={f1:.3f}  AUC={auc:.3f} | {dt:.1f}s")

        epoch_history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "f1":         round(f1, 6),
            "auc":        round(auc, 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # ── Test ──
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    test_preds_list, test_labels_list = [], []
    with torch.no_grad():
        for X, y, lengths in test_loader:
            X, y = X.to(device), y.to(device)
            if mask_ctx:
                X = _apply_context_mask(X)
            logits = _forward(model, X, lengths, device)
            test_preds_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            test_labels_list.extend(y.cpu().numpy().tolist())

    test_pred_bin = [1 if p > 0.5 else 0 for p in test_preds_list]
    test_acc = float(accuracy_score(test_labels_list, test_pred_bin))
    test_f1  = float(f1_score(test_labels_list, test_pred_bin, zero_division=0))
    test_auc = (
        float(roc_auc_score(test_labels_list, test_preds_list))
        if len(set(test_labels_list)) > 1 else 0.5
    )

    result = {
        "variant":          variant["name"],
        "label":            variant["label"],
        "arch":             variant["arch"],
        "use_cls_token":    variant["use_cls_token"],
        "use_pos_encoding": variant["use_pos_encoding"],
        "mask_context":     variant["mask_context"],
        "total_params":     total_params,
        "early_exit_frac":  early_exit_frac,
        "test_acc":         round(test_acc, 6),
        "test_f1":          round(test_f1, 6),
        "test_auc":         round(test_auc, 6),
        "epoch_history":    epoch_history,
    }
    print(f"\n  [{variant['name']}] TEST → AUC={test_auc:.3f}  F1={test_f1:.3f}  Acc={test_acc:.2%}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OrbitGuard Architecture Ablation")
    parser.add_argument("--data",        type=str,   default="data/merged/missions.parquet")
    parser.add_argument("--early-exit",  type=float, default=0.4)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  type=str,   default="reports/arch_ablation")
    parser.add_argument(
        "--variants", type=str, nargs="+",
        default=[v["name"] for v in VARIANTS],
        help="Subset of variant names to run (default: all)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\n▸ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n▸ Device: CPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [v for v in VARIANTS if v["name"] in args.variants]
    print(f"\n▸ Architecture ablation at {args.early_exit:.0%} early exit")
    print(f"▸ Variants: {[v['name'] for v in selected]}")
    print(f"▸ Epochs: {args.epochs}  |  Seed: {args.seed}")

    all_results = []
    out_path = output_dir / "arch_ablation_results.json"

    for variant in selected:
        result = run_variant(
            variant=variant,
            data_path=args.data,
            early_exit_frac=args.early_exit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(result)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Summary table ──
    ref = next((r for r in all_results if r["variant"] == "full_transformer"), None)
    print(f"\n{'═' * 72}")
    print(f"  ARCHITECTURE ABLATION  (exit={args.early_exit:.0%}, seed={args.seed})")
    print(f"{'═' * 72}")
    print(f"  {'Variant':<30}  {'AUC':>6}  {'F1':>6}  {'Acc':>7}  {'ΔAUC':>7}  {'Params':>9}")
    print(f"  {'─' * 68}")
    for r in all_results:
        delta = f"{r['test_auc'] - ref['test_auc']:+.3f}" if ref and r["variant"] != "full_transformer" else "  ref"
        print(
            f"  {r['label']:<30}  {r['test_auc']:>6.3f}  {r['test_f1']:>6.3f}  "
            f"{r['test_acc']:>7.2%}  {delta:>7}  {r['total_params']:>9,}"
        )
    print(f"\n▸ Results saved to {out_path}")


if __name__ == "__main__":
    main()
