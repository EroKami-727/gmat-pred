"""
Shared calibration helpers — ECE, best-F1 threshold search, and a unified
metrics block (accuracy/F1/ROC-AUC/PR-AUC/Brier/ECE/confusion matrix).

Used by calibration_eval.py, multi_seed_calibration.py, grouped_baselines.py,
and parameter_holdout_baselines.py (via grouped_baselines re-exports) so the
same calibration math is computed identically everywhere in the audit suite.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (uniform-width bins)."""
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    if n == 0:
        return 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds to find the one maximising F1 on the given split."""
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 199)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        score = f1_score(y_true, (y_prob > threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metrics_block(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Unified metrics block: accuracy, F1, ROC-AUC, PR-AUC, Brier score, ECE,
    and a confusion matrix, all at the given decision threshold.
    """
    y_pred = (y_prob > threshold).astype(int)
    n_unique = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if n_unique > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if n_unique > 1 else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "success_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }
