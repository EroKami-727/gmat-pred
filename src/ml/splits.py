"""
The canonical train/val/test split.

Every script that touches a per-planet extract must carve it up the *same* way,
or evaluation silently reports on missions the model selected against. This
module is the single definition; nothing should reimplement it.

It exists because the drift already happened. `per_planet_train.py` split
70/15/15 with seed 42, while `prune_economics.py` independently split 70/30 with
seed 42 — the same seed and the same N, so the same permutation, so the
economics script's "test" set was exactly the model's validation set plus its
test set. Half of the 3,000 missions it reported on were the missions the
checkpoint and the abort threshold had been chosen on. `test_ml.py` had a third
copy of the arithmetic which happened to agree, with a comment reading "must
stay in sync with it" — an invariant a comment cannot enforce.

The split is a deterministic function of (n, seed), so any two callers that pass
the same arguments get identical index arrays, and callers that need a different
seed get a consistently different partition.

    tr  70%   fit model parameters
    va  15%   choose checkpoints, thresholds, operating points, hyperparameters
    te  15%   touched once, for reporting only
"""

from __future__ import annotations

import numpy as np

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test is the remainder — writing it as 0.15 and slicing to n_tr + n_va would
# silently drop a mission or two to integer truncation.


def train_val_test(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Indices for a 70/15/15 partition of `n` items.

    Deterministic in (n, seed). Returns (train, val, test).
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(TRAIN_FRAC * n)
    n_va = int(VAL_FRAC * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def test_indices(n: int, seed: int = 42) -> np.ndarray:
    """Just the held-out test split, for report-only consumers."""
    return train_val_test(n, seed)[2]
