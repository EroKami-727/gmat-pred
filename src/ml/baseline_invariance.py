"""
C1b — Why a scale-invariant baseline certifies a broken pipeline.

The collapse in `norm_ablation.py` is only half the finding. The other half is
that it is INVISIBLE to the baseline normally used to sanity-check the data:
gradient-boosted trees split on absolute feature values, so they neither need
nor benefit from the amplification that normalisation provides. Run a tree and a
network over the same arrays under the same preprocessing and the tree reports
that the task is solved while the network emits a constant.

That is what makes this a methodological failure rather than a bug. The
practitioner's instinct — "check a simple baseline before trusting the deep
model" — actively misleads here, because the baseline is invariant to precisely
the defect that destroys the deep model.

This script needs no GPU and no training of the sequence model: it fits XGBoost
on the same normalised decision window under each condition and reports AUC.
Pair it with norm_ablation.json, which holds the network's side.

Usage:
    python -m src.ml.baseline_invariance
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ml.norm_ablation import CONDITIONS, DEFAULT_PLANETS, grouped_stats, group_of
from src.ml.per_planet_train import fit_norm_stats
from src.ml.splits import train_val_test
from src.ml.train_assist import build_features, window_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planets", nargs="+", default=DEFAULT_PLANETS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data/per_planet")
    ap.add_argument("--out", default="reports/baseline_invariance.json")
    args = ap.parse_args()

    import xgboost as xgb

    data_dir = Path(args.data_dir)
    rows = []

    print(f"\n{'='*84}")
    print("  C1b — TREE BASELINE UNDER THE SAME NORMALISATION AS THE NETWORK")
    print(f"{'='*84}")
    print(f"  {'PLANET':9} {'CONDITION':13} {'TREE AUC':>9} {'TREE F1':>9}   "
          f"(network AUC in normalisation_ablation.json)")
    print(f"  {'-'*80}")

    for planet in args.planets:
        z = np.load(data_dir / f"{planet}.npz")
        X, y, lengths = z["X"].astype(np.float64), z["y"], z["lengths"]
        tr, va, te = train_val_test(len(y), args.seed)
        W = window_for(lengths)

        for cond in CONDITIONS:
            if cond == "grouped":
                mu, sd = grouped_stats(planet, group_of(planet), data_dir,
                                       args.seed, X.shape[1])
            else:
                mu, sd = fit_norm_stats(X[tr], lengths[tr], cond)

            feats = build_features(X, mu, sd, W)
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=5,
                                    learning_rate=0.08, tree_method="hist",
                                    eval_metric="logloss", n_jobs=6, verbosity=0)
            clf.fit(feats[tr], y[tr])
            p = clf.predict_proba(feats[te])[:, 1]
            auc = float(roc_auc_score(y[te], p))
            pred = (p >= 0.5).astype(int)
            tp = int(((pred == 0) & (y[te] == 0)).sum())
            fp = int(((pred == 0) & (y[te] == 1)).sum())
            fn = int(((pred == 1) & (y[te] == 0)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)

            print(f"  {planet:9} {cond:13} {auc:>9.4f} {f1:>9.4f}")
            rows.append({"planet": planet, "norm_mode": cond,
                         "tree_auc": round(auc, 4), "tree_f1": round(f1, 4),
                         "window_steps": int(W)})

    # The claim: the tree is invariant to the condition that destroys the network.
    spread = {}
    for planet in args.planets:
        aucs = [r["tree_auc"] for r in rows if r["planet"] == planet]
        if aucs:
            spread[planet] = round(max(aucs) - min(aucs), 4)

    print(f"  {'-'*80}")
    print("  Tree AUC spread across the three normalisations (invariance):")
    for k, v in spread.items():
        print(f"    {k:9} {v:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": ("C1b: XGBoost on the same normalised decision window as "
                        "the sequence model, under each normalisation condition"),
        "seed": args.seed,
        "protocol": ("features are the per-condition normalised prefix, flattened "
                     "— build_features() from train_assist, the identical view the "
                     "network receives; split from src/ml/splits.py"),
        "tree_auc_spread_across_conditions": spread,
        "runs": rows,
    }, indent=2))
    print(f"\n  Saved -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
