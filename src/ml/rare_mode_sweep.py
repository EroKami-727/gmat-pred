"""
C3 — A sequence model failing to reach a signal that is present in its own input.

On Uranus, `surface_impact` is a small minority of failures and the per-planet
Transformer's recall on it is zero, while a gradient-boosted tree on the
IDENTICAL per-timestep z-normalised 40% window separates that mode from success
almost perfectly. Oversampling the rare mode does not close the gap.

If both models see the same array and one of them can separate the classes, the
information is there and the other model's failure is an optimisation limit, not
an information limit. That distinction is the contribution, and it is only
credible if the resampling sweep is actually run rather than asserted — this
result previously existed only as a sentence in a docstring.

For each mode_alpha the script trains a per-planet model from scratch (no tree
assist anywhere in the loop) and records:

  per-mode recall     on the held-out split at the operating point
  effective resampling how much the rare mode was actually upweighted
  tree AUC / recall   XGBoost on the same normalised window, same split

Usage:
    python -m src.ml.rare_mode_sweep                       # uranus
    python -m src.ml.rare_mode_sweep --planet mercury --alphas 0 1.0
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.ml.dataset import FEATURE_COLS
from src.ml.model import TrajectoryTransformer
from src.ml.per_planet_train import (
    apply_timestep_norm, fit_norm_stats, mode_sample_weights, predict, train_planet,
)
from src.ml.planet_config import FAILURE_NAMES, N_FAILURE_CLASSES, OPERATING_FRAC
from src.ml.splits import train_val_test
from src.ml.train_assist import build_features, window_for

DEFAULT_ALPHAS = [0.0, 0.5, 1.0]


def per_mode_recall(pred_fail: np.ndarray, ft: np.ndarray) -> dict:
    """Recall of the abort decision, broken out by true failure mode."""
    out = {}
    for code in np.unique(ft):
        if code == 0:                                  # success is not a mode
            continue
        m = ft == code
        out[FAILURE_NAMES.get(int(code), str(code))] = {
            "n": int(m.sum()),
            "recall": round(float(pred_fail[m].mean()), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", default="uranus")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data/per_planet")
    ap.add_argument("--out", default="reports/rare_mode_sweep.json")
    ap.add_argument("--work-root", default="models/_rare_mode_sweep")
    args = ap.parse_args()

    import xgboost as xgb

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planet = args.planet

    z = np.load(data_dir / f"{planet}.npz")
    X, y, ft, lengths = z["X"], z["y"], z["failure_type"], z["lengths"]
    tr, va, te = train_val_test(len(y), args.seed)

    # Which mode is the rare one, among training failures.
    fails = ft[tr][ft[tr] != 0]
    codes, counts = np.unique(fails, return_counts=True)
    rare_code = int(codes[np.argmin(counts)])
    rare_name = FAILURE_NAMES.get(rare_code, str(rare_code))
    rare_n = int(counts.min())

    print(f"\n{'='*94}")
    print(f"  C3 — RARE-MODE SWEEP  ({planet}, rarest training mode: "
          f"{rare_name} n={rare_n} of {len(fails)} failures)")
    print(f"{'='*94}")

    # ── Reference: a tree on the identical normalised window ─────────────────
    mu, sd = fit_norm_stats(X[tr], lengths[tr], "per-timestep")
    W = window_for(lengths)
    feats = build_features(X, mu, sd, W)

    # surface_impact vs success only — the separability question C3 asks.
    sel_tr = tr[(ft[tr] == rare_code) | (y[tr] == 1)]
    sel_te = te[(ft[te] == rare_code) | (y[te] == 1)]
    tree = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                             tree_method="hist", eval_metric="logloss",
                             n_jobs=6, verbosity=0)
    tree.fit(feats[sel_tr], (ft[sel_tr] == rare_code).astype(int))
    p_rare = tree.predict_proba(feats[sel_te])[:, 1]
    tree_auc = float(roc_auc_score((ft[sel_te] == rare_code).astype(int), p_rare))
    tree_recall = float(((p_rare >= 0.5) & (ft[sel_te] == rare_code)).sum()
                        / max((ft[sel_te] == rare_code).sum(), 1))

    print(f"\n  Tree on the identical z-normalised {W}-step window "
          f"({rare_name} vs success):")
    print(f"    AUC {tree_auc:.4f}   recall@0.5 {tree_recall:.4f}   "
          f"n_test={int((ft[sel_te] == rare_code).sum())}")

    # ── Sweep: sequence model at increasing rare-mode oversampling ───────────
    print(f"\n  {'ALPHA':>6} {'effective':>10} {'test F1':>9} {'overall':>9} "
          f"{rare_name[:16]:>17}")
    print(f"  {'':>6} {'resample':>10} {'':>9} {'recall':>9} {'recall':>17}")
    print(f"  {'-'*62}")

    runs = []
    for alpha in args.alphas:
        # How much the rare mode is upweighted relative to uniform sampling.
        # mode_sample_weights returns UNNORMALISED ones at alpha=0 and a
        # normalised distribution above it, so normalise before comparing or
        # alpha=0 reports a spurious N-fold factor.
        w = mode_sample_weights(ft[tr], alpha)
        w = w / w.sum()
        rare_mask = ft[tr] == rare_code
        eff = float(w[rare_mask].mean() * len(w)) if rare_mask.any() else 1.0

        out_dir = Path(args.work_root) / f"{planet}_alpha{alpha}"
        meta = train_planet(planet, data_dir, out_dir, epochs=args.epochs,
                            seed=args.seed, device=device, mode_alpha=alpha)

        model = TrajectoryTransformer(
            input_dim=len(FEATURE_COLS), output_dim=1, task="binary",
            aux_dim=meta.get("aux_dim", N_FAILURE_CLASSES),
        ).to(device)
        model.load_state_dict(
            torch.load(out_dir / "model.pt", map_location=device, weights_only=True))
        model.eval()

        Xn = torch.from_numpy(apply_timestep_norm(X, mu, sd))
        p_succ, _ = predict(model, Xn, torch.from_numpy(lengths), te,
                            OPERATING_FRAC, device)
        pred_fail = (1.0 - p_succ) >= meta["threshold"]
        modes = per_mode_recall(pred_fail, ft[te])
        overall = float(pred_fail[y[te] == 0].mean())
        rare_rec = modes.get(rare_name, {}).get("recall", float("nan"))

        print(f"  {alpha:>6.2f} {eff:>9.1f}x {meta['test_at_operating_point']['f1']:>9.4f} "
              f"{overall:>9.4f} {rare_rec:>17.4f}")

        runs.append({
            "mode_alpha": alpha,
            "effective_resample_factor": round(eff, 2),
            "test_f1": meta["test_at_operating_point"]["f1"],
            "overall_failure_recall": round(overall, 4),
            "rare_mode_recall": rare_rec,
            "per_mode_recall": modes,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_rare = max(r["rare_mode_recall"] for r in runs)
    verdict = ("optimisation limit — the tree separates the mode the sequence "
               "model cannot reach") if tree_auc - best_rare > 0.3 else \
              ("no gap — resampling or the sequence model does reach the mode")

    print(f"\n  Tree AUC {tree_auc:.4f} vs best sequence recall {best_rare:.4f} "
          f"across alpha -> {verdict}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "planet": planet, "seed": args.seed, "epochs": args.epochs,
        "operating_frac": OPERATING_FRAC,
        "rare_mode": rare_name, "rare_mode_train_n": rare_n,
        "n_train_failures": int(len(fails)),
        "window_steps": int(W),
        "tree_reference": {
            "auc": round(tree_auc, 4), "recall_at_0.5": round(tree_recall, 4),
            "n_test_rare": int((ft[sel_te] == rare_code).sum()),
            "note": ("XGBoost on the identical per-timestep z-normalised window "
                     "the sequence model receives, same split"),
        },
        "best_sequence_rare_recall": best_rare,
        "verdict": verdict,
        "runs": runs,
    }, indent=2))
    print(f"  Saved -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
