"""
Per-Planet Tree Assist
======================
A gradient-boosted classifier that runs alongside the per-planet Transformer at
the abort decision point and is fused with it.

Motivation: the Transformer misses rare failure modes even when the signal is
fully present in its own input. On Uranus, `surface_impact` is 119 of 6,611
failures and the model's recall on it is 0.000 — yet XGBoost on the *identical
per-timestep z-normalised window* separates surface_impact from success at
AUC 1.000. Oversampling the mode up to 45x (mode_alpha=1.0) does not fix it, so
this is an optimisation limit of the sequence model, not missing information.

The Transformer is still the primary model: it streams, works at any prefix
length, and predicts the failure mode. The tree only contributes at the fixed
decision window, where it is strongest.

Usage:
    python -m src.ml.train_assist --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ml.planet_config import FAILURE_NAMES, OPERATING_FRAC, SERVING_TARGETS
from src.ml.splits import train_val_test

SEED = 42


def window_for(lengths: np.ndarray, frac: float = OPERATING_FRAC) -> int:
    """Fixed feature window: the shortest mission's prefix, so every mission fills it."""
    return int(max(2, (lengths * frac).min()))


def build_features(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, W: int) -> np.ndarray:
    """Per-timestep z-normalised prefix, flattened — the same view the model sees."""
    z = (X[:, :W] - mu[:W]) / sd[:W]
    z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(z, -10.0, 10.0).reshape(len(X), -1).astype(np.float32)


def splits(n: int, seed: int = SEED):
    """
    The canonical partition. This was a fourth hand-rolled copy of the same
    arithmetic; it agreed with the trainer, but so did two of the three copies
    that already existed when one of them silently did not.
    """
    return train_val_test(n, seed)


def train_one(planet: str, data_dir: Path, models_root: Path) -> dict | None:
    import xgboost as xgb

    f = data_dir / f"{planet}.npz"
    mdir = models_root / planet
    if not f.exists() or not (mdir / "norm_stats.npz").exists():
        print(f"  {planet:9}: missing extract or model — skipped")
        return None

    z = np.load(f)
    X, y, ft, L = z["X"], z["y"], z["failure_type"], z["lengths"]
    st = np.load(mdir / "norm_stats.npz")
    W = window_for(L)
    F = build_features(X, st["mu"], st["sd"], W)
    tr, va, te = splits(len(y))

    n_pos = int((y[tr] == 1).sum())
    n_neg = int((y[tr] == 0).sum())
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8,
        tree_method="hist", n_jobs=6, verbosity=0, eval_metric="logloss",
        scale_pos_weight=n_neg / max(n_pos, 1), random_state=SEED,
    )
    clf.fit(F[tr], y[tr])

    p_fail_te = 1.0 - clf.predict_proba(F[te])[:, 1]
    auc = roc_auc_score(y[te], -p_fail_te)

    # Per-mode recall at a neutral 0.5 cut, for the record.
    ab = p_fail_te >= 0.5
    per_mode = {}
    for m in sorted(set(ft[te][y[te] == 0].tolist())):
        sel = (ft[te] == m) & (y[te] == 0)
        if sel.sum():
            per_mode[FAILURE_NAMES[int(m)]] = {
                "n": int(sel.sum()), "recall": round(float(ab[sel].mean()), 4)}

    clf.save_model(mdir / "assist.json")
    meta = {"window": int(W), "auc": round(float(auc), 4),
            "operating_frac": OPERATING_FRAC, "per_mode_recall_at_0.5": per_mode}
    (mdir / "assist_meta.json").write_text(json.dumps(meta, indent=2))

    modes = "  ".join(f"{k}={v['recall']:.3f}(n={v['n']})" for k, v in per_mode.items())
    print(f"  {planet:9}: W={W:<3} AUC={auc:.4f}  {modes}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-dir", default="data/per_planet")
    ap.add_argument("--models-root", default="models/per_planet")
    args = ap.parse_args()

    # Serving set, not the study set — the assist is a serving artifact.
    targets = SERVING_TARGETS if args.all else [args.planet]
    if not targets or targets == [None]:
        ap.error("pass --planet <name> or --all")

    print(f"\n[ Tree assist @ {OPERATING_FRAC:.0%} decision window ]")
    for p in targets:
        train_one(p, Path(args.data_dir), Path(args.models_root))
    print()


if __name__ == "__main__":
    main()
