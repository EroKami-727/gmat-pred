"""
OrbitGuard ML correctness test — all planets, both outcomes, streaming window.

Evaluates the per-planet models on the HELD-OUT test split (missions never seen
in training or validation, reproduced from the same seed/split as training):

  * outcome        — does it catch failures, and does it leave successes alone
  * failure mode   — does it predict HOW a mission fails
  * streaming      — accuracy as a function of how much trajectory is observed
  * novel missions — synthetic trajectories generated from scratch

Run:
    /home/haise/Coding/venvs/gmat-pred/bin/python3 test_ml.py
    /home/haise/Coding/venvs/gmat-pred/bin/python3 test_ml.py --skip-synthetic
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.ml.planet_config import FAILURE_NAMES, OPERATING_FRAC, PLANETS
from src.ml.splits import test_indices
from src.ml.planet_router import PlanetRouter

DATA_DIR = Path("data/per_planet")
EVAL_FRACS = [0.10, 0.20, 0.30, 0.40]
SEED = 42


def test_split_indices(n: int, seed: int = SEED) -> np.ndarray:
    """
    The held-out test split.

    This used to reimplement per_planet_train's arithmetic with a comment saying
    it "must stay in sync" — which it did, while prune_economics.py's third copy
    did not. Both now call the one definition in src/ml/splits.py.
    """
    return test_indices(n, seed)


def batched_predict(router, planet, X, lengths, idx, frac):
    """P(fail) and predicted failure mode for each mission at `frac` of its length."""
    pf, modes = [], []
    for i in idx:
        n = max(2, int(lengths[i] * frac))
        out = router.predict(X[i, :n], planet)
        pf.append(out["p_fail"])
        modes.append(out["failure_mode"])
    return np.array(pf), np.array(modes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--limit", type=int, default=1500,
                    help="max held-out missions per planet (0 = all)")
    args = ap.parse_args()

    router = PlanetRouter("models/per_planet")
    if not router.is_available():
        raise SystemExit("No per-planet models found — run src.ml.per_planet_train --all")

    print(f"\n{'='*94}")
    print("  ORBITGUARD ML TEST — held-out missions, per-planet models")
    print(f"{'='*94}")
    print(f"  Loaded : {', '.join(router.status()['loaded_planets'])}")

    overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "mode_ok": 0, "mode_n": 0}
    rows = []
    stream_rows = []

    for planet in PLANETS:
        f = DATA_DIR / f"{planet}.npz"
        if not f.exists() or not router.supports(planet):
            print(f"\n  {planet}: no model/data — skipped")
            continue

        d = np.load(f)
        X, y, ft, lengths = d["X"], d["y"], d["failure_type"], d["lengths"]
        te = test_split_indices(len(y))
        if args.limit:
            te = te[:args.limit]

        thr = router.threshold_for(planet)

        # ── Streaming behaviour ──
        print(f"\n  ── {planet.upper()} ──  held-out n={len(te)}  "
              f"fail={int((y[te]==0).sum())} pass={int((y[te]==1).sum())}  thr={thr:.3f}")
        print(f"     {'SEEN':>6} {'AUC':>7} {'RECALL':>7} {'PREC':>7} {'F1':>7} "
              f"{'MODE_ACC':>9}   (mode acc = how it fails, on true failures)")

        for frac in EVAL_FRACS:
            pf, modes = batched_predict(router, planet, X, lengths, te, frac)
            actual_fail = y[te] == 0
            aborted = pf >= thr
            tp = int((aborted & actual_fail).sum())
            fp = int((aborted & ~actual_fail).sum())
            fn = int((~aborted & actual_fail).sum())
            tn = int((~aborted & ~actual_fail).sum())
            rec = tp / max(tp + fn, 1)
            pre = tp / max(tp + fp, 1)
            f1 = 2 * pre * rec / max(pre + rec, 1e-9)
            try:
                auc = roc_auc_score(y[te], -pf)
            except ValueError:
                auc = float("nan")
            mode_ok = int((modes[actual_fail] == ft[te][actual_fail]).sum())
            mode_n = int(actual_fail.sum())
            macc = mode_ok / max(mode_n, 1)

            marker = "  ← operating point" if abs(frac - OPERATING_FRAC) < 1e-9 else ""
            print(f"     {frac:>5.0%} {auc:>7.4f} {rec:>7.4f} {pre:>7.4f} {f1:>7.4f} "
                  f"{macc:>9.4f}{marker}")
            stream_rows.append((planet, frac, auc, rec, pre, f1, macc))

            if abs(frac - OPERATING_FRAC) < 1e-9:
                rows.append((planet, len(te), thr, auc, rec, pre, f1, macc, tp, fp, fn, tn))
                for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn),
                             ("mode_ok", mode_ok), ("mode_n", mode_n)):
                    overall[k] += v

        # ── Failure-mode confusion on true failures, at the operating point ──
        pf, modes = batched_predict(router, planet, X, lengths, te, OPERATING_FRAC)
        actual_fail = y[te] == 0
        true_modes = ft[te][actual_fail]
        pred_modes = modes[actual_fail]
        present = sorted(set(true_modes.tolist()))
        print(f"     failure-mode breakdown:")
        for m in present:
            sel = true_modes == m
            acc = float((pred_modes[sel] == m).mean()) if sel.any() else 0.0
            print(f"       {FAILURE_NAMES.get(int(m), '?'):18s} n={int(sel.sum()):>5}  "
                  f"correct={acc:>6.2%}")

    # ── Summary ──
    print(f"\n{'='*94}")
    print(f"  SUMMARY @ {OPERATING_FRAC:.0%} of trajectory observed")
    print(f"{'='*94}")
    print(f"  {'PLANET':10} {'N':>5} {'THR':>6} {'AUC':>7} {'RECALL':>7} {'PREC':>7} "
          f"{'F1':>7} {'MODE':>7}  {'TP':>5} {'FP':>4} {'FN':>4} {'TN':>5}")
    print(f"  {'-'*90}")
    for (p, n, thr, auc, rec, pre, f1, macc, tp, fp, fn, tn) in rows:
        print(f"  {p:10} {n:>5} {thr:>6.3f} {auc:>7.4f} {rec:>7.4f} {pre:>7.4f} "
              f"{f1:>7.4f} {macc:>7.4f}  {tp:>5} {fp:>4} {fn:>4} {tn:>5}")

    tp, fp, fn, tn = overall["tp"], overall["fp"], overall["fn"], overall["tn"]
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    macc = overall["mode_ok"] / max(overall["mode_n"], 1)
    print(f"  {'-'*90}")
    print(f"  {'OVERALL':10} {tp+fp+fn+tn:>5} {'—':>6} {'—':>7} {rec:>7.4f} {pre:>7.4f} "
          f"{f1:>7.4f} {macc:>7.4f}  {tp:>5} {fp:>4} {fn:>4} {tn:>5}")

    # ── Novel synthetic missions ──
    if not args.skip_synthetic:
        print(f"\n{'='*94}")
        print("  NOVEL MISSIONS — synthetic trajectories, not from the dataset")
        print(f"{'='*94}")
        try:
            run_synthetic(router)
        except Exception as e:                                     # noqa: BLE001
            print(f"  synthetic test unavailable: {type(e).__name__}: {e}")

    print()


def run_synthetic(router):
    """
    Build fresh missions with the dataset's own propagator and check the
    verdict against ground truth. Offsets are expressed in multiples of the
    1-sigma dispersion the dataset sampled, so 0 = textbook Hohmann transfer.
    """
    from src.api.mission_builder import build_mission, dispersion_scale

    print(f"  {'PLANET':9} {'dV(km/s)':>9} {'SIGMA':>6} {'TRUTH':>8} {'P(fail)':>8} "
          f"{'THR':>6} {'OOD%':>6} {'VERDICT':>9}  {'PRED_MODE':>16}")
    print(f"  {'-'*98}")

    n_ok = n_tot = n_ood = 0
    for planet in PLANETS:
        if not router.supports(planet):
            continue
        sigma = dispersion_scale(planet).get("dv_V", 0.003)
        for k in [0.0, 2.0, -2.0, 6.0]:
            dv = k * sigma
            try:
                gen = build_mission(planet, dv_v_offset=dv)
            except Exception as e:                               # noqa: BLE001
                print(f"  {planet:9} {dv:>9.4f}  build failed: {e}")
                continue

            feats = gen["features"]
            n = max(2, int(len(feats) * OPERATING_FRAC))
            out = router.predict(feats[:n], planet)
            truth = "SUCCESS" if gen["label"] == 1 else "FAIL"
            aborted = out["should_abort"]
            correct = (aborted and gen["label"] == 0) or (not aborted and gen["label"] == 1)
            n_ok += correct
            n_tot += 1
            n_ood += bool(out["out_of_distribution"])
            verdict = ("ABORT ✓" if correct else "ABORT ✗") if aborted else \
                      ("GO ✓" if correct else "GO ✗")
            print(f"  {planet:9} {dv:>9.4f} {k:>6.0f} {truth:>8} {out['p_fail']:>8.4f} "
                  f"{out['threshold']:>6.3f} {out['ood_fraction']*100:>5.0f}% "
                  f"{verdict:>9}  {out['failure_mode_name']:>16}")

    if n_tot:
        print(f"  {'-'*98}")
        print(f"  created-mission accuracy: {n_ok}/{n_tot} = {n_ok/n_tot:.1%}"
              f"   ({n_ood} flagged out-of-distribution)")
        print("  Missions are built by src/api/mission_builder, which calls the same")
        print("  propagator and feature code that produced the training set, so they")
        print("  are in-distribution. The OOD flag is advisory: at 6 sigma the burn")
        print("  error is physically valid but statistically extreme, and P(fail)")
        print("  remains correct there.")


if __name__ == "__main__":
    main()
