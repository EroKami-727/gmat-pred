"""
Recalibrate per-planet abort thresholds without retraining.

The models are frozen; only the operating point moves. Thresholds are chosen on
the VALIDATION split (never test) as the midpoint of the plateau of thresholds
that maximise failure-class F1, then reported against the untouched test split.

Usage:
    python -m src.ml.recalibrate
    python -m src.ml.recalibrate --planet venus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.ml.per_planet_train import pick_threshold
from src.ml.planet_config import OPERATING_FRAC, PLANETS
from src.ml.planet_router import PlanetRouter

SEED = 42


def splits(n: int, seed: int = SEED):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def p_fail_for(router: PlanetRouter, planet: str, X, lengths, idx, frac):
    out = np.empty(len(idx))
    for j, i in enumerate(idx):
        n = max(2, int(lengths[i] * frac))
        out[j] = router.predict(X[i, :n], planet)["p_fail"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", default=None)
    ap.add_argument("--data-dir", default="data/per_planet")
    ap.add_argument("--models-root", default="models/per_planet")
    ap.add_argument("--limit", type=int, default=0, help="cap missions per split (0 = all)")
    args = ap.parse_args()

    router = PlanetRouter(args.models_root)
    if not router.is_available():
        raise SystemExit("No per-planet models found.")

    data_dir = Path(args.data_dir)
    root = Path(args.models_root)
    targets = [args.planet] if args.planet else PLANETS

    print(f"\n[ Recalibration @ {OPERATING_FRAC:.0%} observed ]")
    print(f"  {'PLANET':10} {'OLD':>7} {'NEW':>7} {'VAL_F1':>7} "
          f"{'TEST_REC':>9} {'TEST_PRE':>9} {'TEST_F1':>8} {'AUC':>7}")
    print(f"  {'-'*74}")

    thresholds = {}
    for planet in targets:
        f = data_dir / f"{planet}.npz"
        if not f.exists() or not router.supports(planet):
            continue
        d = np.load(f)
        X, y, lengths = d["X"], d["y"], d["lengths"]
        _, va, te = splits(len(y))
        if args.limit:
            va, te = va[:args.limit], te[:args.limit]

        old_thr = router.threshold_for(planet)

        pf_va = p_fail_for(router, planet, X, lengths, va, OPERATING_FRAC)
        new_thr, val_f1 = pick_threshold(pf_va, y[va])

        pf_te = p_fail_for(router, planet, X, lengths, te, OPERATING_FRAC)
        aborted = pf_te >= new_thr
        af = y[te] == 0
        tp = int((aborted & af).sum()); fp = int((aborted & ~af).sum())
        fn = int((~aborted & af).sum())
        rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
        f1 = 2 * pre * rec / max(pre + rec, 1e-9)
        try:
            auc = roc_auc_score(y[te], -pf_te)
        except ValueError:
            auc = float("nan")

        print(f"  {planet:10} {old_thr:>7.3f} {new_thr:>7.3f} {val_f1:>7.4f} "
              f"{rec:>9.4f} {pre:>9.4f} {f1:>8.4f} {auc:>7.4f}")

        meta_p = root / planet / "meta.json"
        meta = json.loads(meta_p.read_text())
        meta["threshold"] = round(new_thr, 4)
        meta["threshold_selection"] = "plateau midpoint of failure-class F1 on val"
        meta["recalibrated_test"] = {
            "recall": round(rec, 4), "precision": round(pre, 4),
            "f1": round(f1, 4), "auc": round(float(auc), 4),
        }
        meta_p.write_text(json.dumps(meta, indent=2))
        thresholds[planet] = round(new_thr, 4)

    (root / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"\n  Updated → {root/'thresholds.json'}\n")


if __name__ == "__main__":
    main()
