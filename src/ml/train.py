"""
OrbitGuard Training Script — Pruning Prediction Phase
======================================================
Orchestrates the training of LSTM/Transformer models on physics-invariant
trajectory data. 

Features:
- Automated train/val/test splitting
- Support for Binary, Multi-class, and Regression tasks
- Early-stopping and model checkpointing
- Precision/Recall/F1/ROC-AUC tracking
- Class imbalance handling via pos_weight
- GPU auto-detection with device logging
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.dataset import create_dataloaders
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer


# ═══════════════════════════════════════════════════════════════════════════
# Training Logic
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, task="binary"):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y, lengths in tqdm(loader, desc="  Training", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        # LSTM needs lengths for packing, Transformer uses mask
        if isinstance(model, TrajectoryLSTM):
            preds = model(X, lengths)
        else:
            # Create mask for Transformer (True where padding exists)
            mask = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
            for i, l in enumerate(lengths):
                mask[i, int(l):] = True
            preds = model(X, mask)
            
        loss = criterion(preds, y.float() if task in ["binary", "regression"] else y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if task == "binary":
            predicted = (torch.sigmoid(preds) > 0.5).long()
            correct += (predicted == y).sum().item()
            total += y.size(0)
        elif task == "multiclass":
            _, predicted = torch.max(preds.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
    avg_loss = total_loss / len(loader)
    acc = (correct / total) if total > 0 else 0
    return avg_loss, acc


def validate(model, loader, criterion, device, task="binary"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y, lengths in tqdm(loader, desc="  Validating", leave=False):
            X, y = X.to(device), y.to(device)
            
            if isinstance(model, TrajectoryLSTM):
                preds = model(X, lengths)
            else:
                mask = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
                for i, l in enumerate(lengths):
                    mask[i, int(l):] = True
                preds = model(X, mask)
                
            loss = criterion(preds, y.float() if task in ["binary", "regression"] else y)
            total_loss += loss.item()
            
            if task == "binary":
                probs = torch.sigmoid(preds)
                predicted = (probs > 0.5).long()
                correct += (predicted == y).sum().item()
                total += y.size(0)
                all_preds.extend(probs.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
            elif task == "multiclass":
                _, predicted = torch.max(preds.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                
    avg_loss = total_loss / len(loader)
    acc = (correct / total) if total > 0 else 0
    return avg_loss, acc, all_preds, all_labels


def _compute_metrics(all_preds, all_labels):
    """Compute F1 and ROC-AUC from prediction probabilities and labels."""
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.0  # Only one class present
        return f1, auc
    except ImportError:
        return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OrbitGuard LSTM/Transformer Trainer")
    parser.add_argument("--data", type=str, default="data/missions.parquet")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass", "regression"], default="binary")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--early-exit", type=float, default=1.0, help="Fraction of trajectory to use (0.1-1.0)")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    # ── GPU Detection ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n▸ Using device: cuda ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"\n▸ Using device: cpu")

    # 1. Create DataLoaders
    print(f"▸ Loading dataset and preparing splits...")
    use_pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        args.data,
        target_mode=args.task,
        early_exit_frac=args.early_exit,
        batch_size=args.batch_size,
    )

    # 2. Initialize Model — pull input_dim dynamically from dataset
    input_dim = train_loader.dataset.num_features
    print(f"  Input features   : {input_dim} ({train_loader.dataset.feature_names})")

    if args.task == "binary":
        output_dim = 1
        # ── Class imbalance handling ──
        # Count successes vs failures in training set
        train_labels = train_loader.dataset._y
        n_pos = (train_labels == 1).sum().item()
        n_neg = (train_labels == 0).sum().item()
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)
            print(f"  Class balance    : {n_pos} success / {n_neg} failure → pos_weight={pos_weight.item():.2f}")
        else:
            pos_weight = torch.tensor([1.0], device=device)
            print(f"  Class balance    : WARNING — no positive samples!")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.task == "multiclass":
        from src.ml.dataset import FAILURE_TYPE_MAP
        output_dim = len(set(FAILURE_TYPE_MAP.values()))
        criterion = nn.CrossEntropyLoss()
    else:  # regression
        output_dim = 1
        criterion = nn.MSELoss()

    if args.model == "lstm":
        model = TrajectoryLSTM(
            input_dim=input_dim, output_dim=output_dim, task=args.task,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        ).to(device)
    else:
        model = TrajectoryTransformer(
            input_dim=input_dim, output_dim=output_dim, task=args.task,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params     : {total_params:,}")
    print(f"\n▸ Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.task)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, args.task)
        dt = time.time() - t0
        
        # Compute extra metrics for binary
        extra = ""
        if args.task == "binary" and val_preds:
            f1, auc = _compute_metrics(val_preds, val_labels)
            extra = f" | F1={f1:.3f} AUC={auc:.3f}"
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} ({train_acc:.2%}) | "
              f"Val Loss: {val_loss:.4f} ({val_acc:.2%}){extra} | {dt:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(args.output_dir) / f"best_model_{args.model}_{args.task}.pt"
            torch.save(model.state_dict(), model_path)
            # Also save the scaler
            scaler_path = Path(args.output_dir) / f"scaler_{args.model}_{args.task}.pkl"
            import pickle
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

    # 4. Final Evaluation on Test Set
    print(f"\n▸ Evaluating best model on test set...")
    best_model_path = Path(args.output_dir) / f"best_model_{args.model}_{args.task}.pt"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device, args.task)

    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Test Accuracy : {test_acc:.2%}")
    print(f"  Test Loss     : {test_loss:.4f}")

    if args.task == "binary" and test_preds:
        f1, auc = _compute_metrics(test_preds, test_labels)
        print(f"  Test F1       : {f1:.3f}")
        print(f"  Test ROC-AUC  : {auc:.3f}")

        # Confusion matrix
        tp = sum(1 for p, l in zip(test_preds, test_labels) if p > 0.5 and l == 1)
        fp = sum(1 for p, l in zip(test_preds, test_labels) if p > 0.5 and l == 0)
        fn = sum(1 for p, l in zip(test_preds, test_labels) if p <= 0.5 and l == 1)
        tn = sum(1 for p, l in zip(test_preds, test_labels) if p <= 0.5 and l == 0)
        print(f"\n  Confusion Matrix:")
        print(f"    TP={tp}  FP={fp}")
        print(f"    FN={fn}  TN={tn}")

    print(f"\n  Best Model    : {best_model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
