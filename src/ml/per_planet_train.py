"""
Per-Planet Multi-Task Trajectory Model
======================================
Trains ONE model per target body. This is the fix for the cross-planet
collapse: when Mercury/Venus/Mars share a scaler, its IQR spans the
cross-planet range and within-planet variation shrinks to ~1e-5 of the
input range — below what gradient descent can learn to amplify. Trees
(XGBoost) are immune because they split on absolute values, which is why
baselines looked healthy while the transformer emitted constant
probabilities.

Three changes versus the old regime models:

1. **Per-planet fit** — scaler/normaliser sees one planet only, so
   within-planet spread occupies the full dynamic range.
2. **Per-timestep z-score** — each feature is standardised against the
   distribution of that feature *at that timestep index* across missions.
   Mission-to-mission deviation becomes O(1) instead of O(1e-5).
   (Mars val AUC 0.939 → 0.998 from this alone.)
3. **Random-prefix training** — every batch samples a prefix fraction, so
   the model is in-distribution at ANY point of the stream rather than only
   at the single early-exit horizon it was trained on.

The model has two heads: mission outcome (binary) and failure mode
(how it fails), sharing one trunk.

Usage:
    python -m src.ml.per_planet_train --planet venus
    python -m src.ml.per_planet_train --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from src.ml.dataset import FEATURE_COLS
from src.ml.model import TrajectoryTransformer
from src.ml.planet_config import (
    FAILURE_NAMES, N_FAILURE_CLASSES, OPERATING_FRAC, PLANETS,
    TARGET_STEPS, downsample_for,
)

# Prefix fractions (of each mission's own length) used for reporting.
EVAL_FRACS = [0.10, 0.20, 0.30, 0.40]


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_timestep_stats(X: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-timestep mean/std over missions, ignoring padded rows.
    X: (N, L, F) float64, lengths: (N,)
    """
    N, L, F = X.shape
    valid = np.arange(L)[None, :] < lengths[:, None]        # (N, L)
    cnt = valid.sum(axis=0).astype(np.float64)              # (L,)
    cnt_safe = np.maximum(cnt, 1.0)[:, None]                # (L, 1)

    vm = valid[:, :, None]
    mu = (X * vm).sum(axis=0) / cnt_safe                     # (L, F)
    var = ((X - mu[None]) ** 2 * vm).sum(axis=0) / cnt_safe
    sd = np.sqrt(var)

    # Features that are constant at a timestep (or unobserved) must not blow up.
    sd[sd < 1e-12] = 1.0
    # Timesteps no training mission reached: fall back to the last observed row.
    unseen = cnt < 1
    if unseen.any():
        last = np.where(~unseen)[0].max() if (~unseen).any() else 0
        mu[unseen] = mu[last]
        sd[unseen] = sd[last]
    return mu, sd


def apply_timestep_norm(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """X: (N, L, F) -> normalised float32, broadcasting (L, F) stats."""
    L = X.shape[1]
    return ((X - mu[None, :L]) / sd[None, :L]).astype(np.float32)


# ── Batching ──────────────────────────────────────────────────────────────────

def pick_threshold(p_fail: np.ndarray, y: np.ndarray,
                   tol: float = 1e-4) -> tuple[float, float]:
    """
    Choose the P(fail) abort threshold maximising failure-class F1.

    Well-separated models have a wide plateau of equally-optimal thresholds.
    Taking the first maximum parks the threshold at the edge of that plateau
    (Mercury landed on 0.010, the sweep's lower bound), leaving no margin if
    the score distribution shifts. Take the plateau's midpoint instead.
    """
    grid = np.arange(0.005, 0.995, 0.005)
    scores = np.array([
        f1_score(y, (p_fail < t).astype(int), pos_label=0, zero_division=0)
        for t in grid
    ])
    best = scores.max()
    plateau = grid[scores >= best - tol]
    return float(np.median(plateau)), float(best)


def make_prefix_batch(Xn: torch.Tensor, lengths: torch.Tensor, idx: np.ndarray,
                      frac: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice each mission to `frac` of its OWN length, right-pad to the batch max,
    and build the key-padding mask.
    """
    lens = torch.clamp((lengths[idx].double() * frac).long(), min=2)
    W = int(lens.max().item())
    x = Xn[idx, :W].to(device)
    mask = torch.arange(W, device=device)[None, :] >= lens.to(device)[:, None]
    # Zero out padded rows so they can't leak through any unmasked path.
    x = x.masked_fill(mask.unsqueeze(-1), 0.0)
    return x, mask


@torch.no_grad()
def predict(model, Xn, lengths, idx, frac, device, bs=512):
    model.eval()
    p_succ, mode = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x, m = make_prefix_batch(Xn, lengths, b, frac, device)
        logit, aux = model.forward_multitask(x, m)
        p_succ.append(torch.sigmoid(logit).float().cpu().numpy())
        if aux is not None:
            mode.append(aux.argmax(dim=1).cpu().numpy())
    return (np.concatenate(p_succ),
            np.concatenate(mode) if mode else None)


# ── Training ──────────────────────────────────────────────────────────────────

def mode_sample_weights(ft: np.ndarray, alpha: float) -> np.ndarray:
    """
    Per-mission sampling weight ~ (1 / count of its failure mode) ** alpha.

    Rare failure modes are otherwise ignored: Uranus has 119 surface_impact
    missions against 5,879 orbit_too_high, and the model learned to never
    predict the rare one (recall 0.067) even though the signal is present —
    XGBoost separates surface_impact from success at AUC 0.998 on the same
    40% window. alpha=0 disables balancing, 1 equalises modes exactly;
    the default trades off between the two so 119 examples are not
    over-fitted.
    """
    if alpha <= 0:
        return np.ones(len(ft), dtype=np.float64)
    vals, counts = np.unique(ft, return_counts=True)
    freq = dict(zip(vals.tolist(), counts.tolist()))
    w = np.array([(1.0 / freq[int(f)]) ** alpha for f in ft], dtype=np.float64)
    return w / w.sum()


def train_planet(planet: str, data_dir: Path, out_dir: Path,
                 epochs: int = 60, batch_size: int = 128, lr: float = 1e-3,
                 seed: int = 42, device=None, mode_alpha: float = 0.5) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(data_dir / f"{planet}.npz")
    X, y, ft, lengths = d["X"], d["y"], d["failure_type"], d["lengths"]
    N, L, F = X.shape

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr, n_va = int(0.70 * N), int(0.15 * N)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    # Stats fitted on TRAIN ONLY — no leakage into val/test.
    mu, sd = fit_timestep_stats(X[tr], lengths[tr])
    Xn = torch.from_numpy(apply_timestep_norm(X, mu, sd))
    lengths_t = torch.from_numpy(lengths)
    y_t = torch.from_numpy(y.astype(np.float32))
    ft_t = torch.from_numpy(ft.astype(np.int64))

    torch.manual_seed(seed)
    model = TrajectoryTransformer(
        input_dim=F, output_dim=1, task="binary", aux_dim=N_FAILURE_CLASSES,
    ).to(device)

    n_pos = int((y[tr] == 1).sum())
    n_neg = int((y[tr] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    crit_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_mode = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n  ── {planet.upper()} ──  N={N} L={L} F={F}  "
          f"fail={n_neg + int((y[va]==0).sum()) + int((y[te]==0).sum())} "
          f"pos_weight={pos_weight.item():.2f}")
    print(f"     train={len(tr)} val={len(va)} test={len(te)}  device={device}")

    best_auc, best_state, best_ep = -1.0, None, -1
    history = []

    # Mode-balanced sampling: draw with replacement so rare failure modes are
    # actually seen. Restricted to the train split; val/test stay untouched.
    w_tr = mode_sample_weights(ft[tr], mode_alpha)
    mode_counts = {int(k): int(v) for k, v in zip(*np.unique(ft[tr], return_counts=True))}
    print(f"     modes(train)={mode_counts}  mode_alpha={mode_alpha}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_rng = np.random.default_rng(seed * 1000 + ep)
        order = (ep_rng.choice(tr, size=len(tr), replace=True, p=w_tr)
                 if mode_alpha > 0 else ep_rng.permutation(tr))
        tot_loss = 0.0
        nb = 0
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            # Random prefix fraction => model works at ANY streaming position.
            frac = float(ep_rng.uniform(0.05, OPERATING_FRAC))
            x, m = make_prefix_batch(Xn, lengths_t, b, frac, device)
            yy = y_t[b].to(device)
            ff = ft_t[b].to(device)

            opt.zero_grad()
            logit, aux = model.forward_multitask(x, m)
            loss = crit_bin(logit, yy) + 0.5 * crit_mode(aux, ff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.item())
            nb += 1
        sched.step()

        p_succ, _ = predict(model, Xn, lengths_t, va, OPERATING_FRAC, device)
        auc = roc_auc_score(y[va], p_succ)
        f1f = f1_score(y[va], (p_succ > 0.5).astype(int), pos_label=0, zero_division=0)
        history.append({"epoch": ep, "loss": round(tot_loss / max(nb, 1), 6),
                        "val_auc": round(float(auc), 6), "val_f1_fail": round(float(f1f), 6)})

        if auc > best_auc:
            best_auc, best_ep = auc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"     ep {ep:3d}  loss={tot_loss/max(nb,1):.4f}  "
                  f"val_AUC={auc:.4f}  val_F1fail={f1f:.4f}", flush=True)

    model.load_state_dict(best_state)
    print(f"     best epoch {best_ep} (val AUC {best_auc:.4f})")

    # ── Held-out test evaluation across the streaming window ──
    per_frac = {}
    for frac in EVAL_FRACS:
        p_succ, mode = predict(model, Xn, lengths_t, te, frac, device)
        pf = 1.0 - p_succ
        auc = roc_auc_score(y[te], p_succ)
        f1f = f1_score(y[te], (p_succ > 0.5).astype(int), pos_label=0, zero_division=0)
        mode_acc = float((mode == ft[te]).mean()) if mode is not None else 0.0
        fail_mask = y[te] == 0
        mode_acc_fail = (float((mode[fail_mask] == ft[te][fail_mask]).mean())
                         if mode is not None and fail_mask.any() else 0.0)
        per_frac[f"{frac:.2f}"] = {
            "auc": round(float(auc), 4),
            "f1_fail": round(float(f1f), 4),
            "mode_acc": round(mode_acc, 4),
            "mode_acc_on_failures": round(mode_acc_fail, 4),
            "p_fail_spread": round(float(pf.max() - pf.min()), 6),
        }
        print(f"     test @{frac:.0%}: AUC={auc:.4f} F1fail={f1f:.4f} "
              f"mode_acc={mode_acc:.4f} spread={pf.max()-pf.min():.3f}")

    # ── Threshold calibrated on VAL at the operating point (failure-class F1) ──
    p_succ_va, _ = predict(model, Xn, lengths_t, va, OPERATING_FRAC, device)
    pf_va = 1.0 - p_succ_va
    best_thr, best_f1 = pick_threshold(pf_va, y[va])

    # Apply that threshold to the untouched test split.
    p_succ_te, _ = predict(model, Xn, lengths_t, te, OPERATING_FRAC, device)
    pf_te = 1.0 - p_succ_te
    aborted = pf_te >= best_thr
    actual_fail = y[te] == 0
    tp = int((aborted & actual_fail).sum())
    fp = int((aborted & ~actual_fail).sum())
    fn = int((~aborted & actual_fail).sum())
    tn = int((~aborted & ~actual_fail).sum())
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1_test = 2 * prec * recall / max(prec + recall, 1e-9)

    print(f"     threshold={best_thr:.3f} → TEST recall={recall:.4f} "
          f"precision={prec:.4f} F1={f1_test:.4f}  (TP={tp} FP={fp} FN={fn} TN={tn})")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(out_dir / "norm_stats.npz", mu=mu, sd=sd)

    meta = {
        "planet": planet, "n_missions": int(N), "seq_len": int(L),
        "n_features": int(F), "feature_cols": list(FEATURE_COLS),
        # Serving must downsample raw telemetry by exactly this factor.
        "downsample": downsample_for(planet), "target_steps": TARGET_STEPS,
        "operating_frac": OPERATING_FRAC, "threshold": round(best_thr, 4),
        "best_epoch": best_ep, "best_val_auc": round(float(best_auc), 4),
        "test": per_frac,
        "test_at_operating_point": {
            "recall": round(recall, 4), "precision": round(prec, 4),
            "f1": round(f1_test, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "mode_alpha": mode_alpha,
        "failure_classes": {str(k): v for k, v in FAILURE_NAMES.items()},
        "aux_dim": N_FAILURE_CLASSES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planet", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-dir", default="data/per_planet")
    ap.add_argument("--out-root", default="models/per_planet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode-alpha", type=float, default=0.5,
                    help="Failure-mode balancing strength for train sampling "
                         "(0 = off, 1 = equalise modes). Rare modes such as "
                         "Uranus surface_impact are otherwise never learned.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    targets = PLANETS if args.all else [args.planet]
    if not targets or targets == [None]:
        ap.error("pass --planet <name> or --all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ Per-planet training ]  device={device}  epochs={args.epochs}")

    summary = {}
    thresholds = {}
    for planet in targets:
        f = data_dir / f"{planet}.npz"
        if not f.exists():
            print(f"  {planet}: {f} missing — skipped")
            continue
        t0 = time.time()
        meta = train_planet(planet, data_dir, out_root / planet,
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, seed=args.seed, device=device,
                            mode_alpha=args.mode_alpha)
        meta["train_seconds"] = round(time.time() - t0, 1)
        summary[planet] = meta
        thresholds[planet] = meta["threshold"]
        print(f"     ({meta['train_seconds']:.0f}s)")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print(f"\n{'='*78}")
    print(f"  {'PLANET':10} {'AUC@40%':>8} {'F1fail':>8} {'MODE_ACC':>9} "
          f"{'THR':>6} {'RECALL':>7} {'PREC':>7}")
    print(f"{'='*78}")
    for p, m in summary.items():
        t = m["test"]["0.40"]; op = m["test_at_operating_point"]
        print(f"  {p:10} {t['auc']:>8.4f} {t['f1_fail']:>8.4f} "
              f"{t['mode_acc_on_failures']:>9.4f} {m['threshold']:>6.3f} "
              f"{op['recall']:>7.4f} {op['precision']:>7.4f}")
    print(f"{'='*78}\n  Saved → {out_root}\n")


if __name__ == "__main__":
    main()
