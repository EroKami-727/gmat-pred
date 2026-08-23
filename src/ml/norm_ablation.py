"""
C1 — Grouped-normalisation collapse, as a controlled ablation.

Same architecture, same seed, same split, same data. The ONLY difference is how
features are normalised:

  per-timestep  each feature standardised against its distribution at that
                timestep index across missions                    (production)
  global        one RobustScaler pooled over every timestep       (superseded)

This script exists because the result was previously recorded only as a prose
table in docs/RESEARCH_LEDGER.md, with no script that reproduced it. A reviewer
asking to see the ablation got a paragraph.

Three quantities are recorded per (planet, mode):

  signal_ratio     mean within-timestep std of the normalised features. This is
                   the mechanism: it is what gradient descent has to work with,
                   and pooling drives it toward zero.
  val_auc/test_f1  whether the model can actually discriminate.
  pred_std         std of P(fail) across held-out missions. The collapse
                   signature is not "bad accuracy" but a CONSTANT output — the
                   network emitting one number per group — so this is the
                   diagnostic that distinguishes collapse from ordinary
                   underfitting.

Usage:
    python -m src.ml.norm_ablation                       # 4 planets, both modes
    python -m src.ml.norm_ablation --planets mars venus --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

from src.ml.per_planet_train import (
    apply_timestep_norm, fit_norm_stats, predict, robust_stats_from_rows,
    train_planet, valid_rows,
)
from src.ml.dataset import FEATURE_COLS
from src.ml.model import TrajectoryTransformer
from src.ml.planet_config import N_FAILURE_CLASSES, OPERATING_FRAC
from src.ml.splits import train_val_test

# The four targets the original (unreproducible) ledger table covered.
DEFAULT_PLANETS = ["venus", "mars", "mercury", "jupiter"]

# The regime groups the superseded models used — the sharing that caused C1.
REGIME_GROUPS = {
    "inner": ["mercury", "venus", "mars"],
    "outer": ["jupiter", "saturn", "uranus", "neptune"],
}

# Three conditions, isolating the two independent kinds of pooling:
#   per-timestep  no pooling                                   (production)
#   global        pooled across timesteps, one planet
#   grouped       pooled across timesteps AND across planets   (the regime models)
CONDITIONS = ["per-timestep", "global", "grouped"]


def group_of(planet: str) -> list[str]:
    for members in REGIME_GROUPS.values():
        if planet in members:
            return members
    return [planet]


def signal_ratio(X: np.ndarray, lengths: np.ndarray, mu, sd) -> float:
    """
    Mean within-timestep standard deviation of the normalised features, measured
    over the OBSERVED window only (up to OPERATING_FRAC of the sequence).

    The window matters. Averaged across the whole flight this metric is
    meaningless: late timesteps carry enormous across-mission spread because
    failing trajectories have physically diverged by then, which swamps the
    early signal and makes every normalisation look equally healthy. The model
    commits at 40%, so what it actually has to work with is the mission-to-
    mission spread inside the prefix it sees.
    """
    Xn = apply_timestep_norm(X, mu, sd)
    L = Xn.shape[1]
    lo, hi = 1, max(2, int(OPERATING_FRAC * L))
    return float(np.mean([Xn[:, t, :].std(axis=0).mean() for t in range(lo, hi)]))


def grouped_stats(planet: str, group: list[str], data_dir: Path,
                  seed: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """
    One RobustScaler fitted over the TRAINING rows of every planet in `group`.

    This is the condition that actually produced the collapse. The regime models
    shared a scaler across 3-4 planets whose feature ranges differ by orders of
    magnitude, so the pooled IQR spans the cross-planet range and within-planet
    mission-to-mission variation is compressed toward zero. A scaler pooled over
    one planet's own timesteps is a much milder manipulation and does not
    reproduce it, which is why this mode exists separately.
    """
    rows = []
    for p in group:
        z = np.load(data_dir / f"{p}.npz")
        tr, _, _ = train_val_test(len(z["y"]), seed)
        rows.append(valid_rows(z["X"][tr].astype(np.float64), z["lengths"][tr]))
    return robust_stats_from_rows(np.concatenate(rows), L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planets", nargs="+", default=DEFAULT_PLANETS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data/per_planet")
    ap.add_argument("--out", default="reports/normalisation_ablation.json")
    ap.add_argument("--work-root", default="models/_norm_ablation",
                    help="scratch dir; these checkpoints are not for serving")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    print(f"\n{'='*92}")
    print("  C1 — NORMALISATION ABLATION  (identical architecture, seed and split)")
    print(f"{'='*92}")
    print(f"  {'PLANET':9} {'MODE':13} {'signal':>10} {'VAL AUC':>9} "
          f"{'TEST F1':>9} {'P(fail) std':>12}  verdict")
    print(f"  {'-'*88}")

    for planet in args.planets:
        z = np.load(data_dir / f"{planet}.npz")
        X, lengths = z["X"], z["lengths"]
        tr, va, te = train_val_test(len(z["y"]), args.seed)

        for mode in CONDITIONS:
            if mode == "grouped":
                group = group_of(planet)
                mu, sd = grouped_stats(planet, group, data_dir, args.seed, X.shape[1])
                injected = (mu, sd)
            else:
                group = [planet]
                mu, sd = fit_norm_stats(X[tr], lengths[tr], mode)
                injected = None
            sig = signal_ratio(X, lengths, mu, sd)

            out_dir = Path(args.work_root) / f"{planet}_{mode}"
            t0 = time.time()
            meta = train_planet(planet, data_dir, out_dir, epochs=args.epochs,
                                seed=args.seed, device=device,
                                norm_mode=("global" if mode == "grouped" else mode),
                                norm_stats=injected)
            secs = time.time() - t0

            # Held-out prediction spread — the collapse signature.
            Xn = torch.from_numpy(apply_timestep_norm(X, mu, sd))
            # model.pt is a state_dict, matching how the serving router loads it.
            model = TrajectoryTransformer(
                input_dim=len(FEATURE_COLS), output_dim=1, task="binary",
                aux_dim=meta.get("aux_dim", N_FAILURE_CLASSES),
            ).to(device)
            model.load_state_dict(
                torch.load(out_dir / "model.pt", map_location=device, weights_only=True))
            model.eval()
            p_succ, _ = predict(model, Xn, torch.from_numpy(lengths), te,
                                OPERATING_FRAC, device)
            pred_std = float(np.std(1.0 - p_succ))

            val_auc = meta["best_val_auc"]
            test_f1 = meta["test_at_operating_point"]["f1"]
            collapsed = pred_std < 1e-3
            verdict = "COLLAPSED — constant output" if collapsed else "discriminating"

            print(f"  {planet:9} {mode:13} {sig:>10.6f} {val_auc:>9.4f} "
                  f"{test_f1:>9.4f} {pred_std:>12.2e}  {verdict}")

            rows.append({
                "planet": planet, "norm_mode": mode,
                "scaler_fitted_over": group,
                "signal_ratio": round(sig, 8),
                "val_auc": val_auc, "test_f1": test_f1,
                "pred_std": pred_std, "collapsed": collapsed,
                "train_seconds": round(secs, 1),
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Per-planet comparison across the three conditions.
    by = {(r["planet"], r["norm_mode"]): r for r in rows}
    paired = []
    for planet in args.planets:
        got = {m: by.get((planet, m)) for m in CONDITIONS}
        if not all(got.values()):
            continue
        base = got["per-timestep"]
        entry = {"planet": planet}
        for m in CONDITIONS:
            entry[f"val_auc_{m}"] = got[m]["val_auc"]
            entry[f"signal_{m}"] = got[m]["signal_ratio"]
            entry[f"pred_std_{m}"] = got[m]["pred_std"]
            entry[f"collapsed_{m}"] = got[m]["collapsed"]
        entry["val_auc_gain_vs_grouped"] = round(
            base["val_auc"] - got["grouped"]["val_auc"], 4)
        entry["signal_compression_grouped"] = round(
            base["signal_ratio"] / max(got["grouped"]["signal_ratio"], 1e-12), 1)
        paired.append(entry)

    print(f"  {'-'*88}")
    for p in paired:
        print(f"  {p['planet']:9} val AUC  grouped {p['val_auc_grouped']:.4f} | "
              f"global {p['val_auc_global']:.4f} | per-timestep "
              f"{p['val_auc_per-timestep']:.4f}   "
              f"(gain {p['val_auc_gain_vs_grouped']:+.4f}, "
              f"signal x{p['signal_compression_grouped']:,.0f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "C1 controlled normalisation ablation",
        "seed": args.seed, "epochs": args.epochs,
        "operating_frac": OPERATING_FRAC,
        "protocol": ("identical architecture, seed and split; only the "
                     "normalisation differs. signal_ratio is the mean "
                     "within-timestep std of normalised features; pred_std is "
                     "the std of P(fail) over the held-out split."),
        "paired": paired, "runs": rows,
    }, indent=2))
    print(f"\n  Saved -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
