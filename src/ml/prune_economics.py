"""
Mission Pruning: how much GMAT compute can be skipped, and at what cost.

Two screening points are compared on identical splits:

  T0  — screen BEFORE running the simulation, from launch parameters only
        (TOI burn + parking-orbit orientation). A pruned mission costs zero
        propagation time.
  T40 — screen after propagating 40% of the trajectory, from telemetry
        (the per-planet Transformer). A pruned mission still costs 40%.

Compute is charged in RK4 propagation steps, which is what the simulator
actually spends: cost per mission is proportional to propagation duration, and
that spans ~100x between Mercury (175 d) and Neptune (~60,000 d).

Reported at a HIGH-RECALL operating point, because in pruning a false abort
throws away a good mission (expensive, silent) while a missed failure only
costs the compute you were already going to spend.

Usage:
    python -m src.ml.prune_economics
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_collection.gmat_runner import MissionConfig
from src.ml.planet_config import OPERATING_FRAC, PLANETS
from src.ml.planet_router import PlanetRouter

PARAMS = "/media/Data/Coding/gmat-pred/data/merged_all_v2/mission_params.parquet"
SUMMARY = "/media/Data/Coding/gmat-pred/data/merged_all_v2/summary.parquet"

# What the screener may look at before the simulation runs.
LAUNCH_COLS = ["dv_V_offset", "dv_N_offset", "dv_B_offset",
               "RAAN_offset", "AOP_offset", "INC_offset"]

TARGET_RECALL = 0.99      # of failures caught


def prop_days(planet: str) -> float:
    try:
        return float(MissionConfig("earth", planet).prop_days)
    except Exception:                                            # noqa: BLE001
        return 1.0


def load() -> pd.DataFrame:
    """
    Join launch parameters to outcomes ON THE KEY.

    params.sim_id is sorted ascending but summary.mission_id is NOT, so
    concatenating the two frames by row position pairs unrelated missions.
    An earlier version of this file did that and its numbers were wrong.
    """
    p = pq.read_table(PARAMS).to_pandas()
    s = pq.read_table(SUMMARY).to_pandas()
    df = p.merge(s[["mission_id", "label", "failure_type", "min_target_rmag"]],
                 left_on="sim_id", right_on="mission_id", how="inner",
                 validate="one_to_one")
    df["target"] = df["target"].str.lower()
    return df


def threshold_at_recall(p_fail: np.ndarray, y: np.ndarray, recall: float) -> float:
    """Lowest threshold catching `recall` of the failures."""
    fails = p_fail[y == 0]
    if len(fails) == 0:
        return 0.5
    return float(np.quantile(fails, 1.0 - recall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/prune_economics.json")
    args = ap.parse_args()

    df = load()
    router = PlanetRouter("models/per_planet")
    rows = []

    print(f"\n{'='*104}")
    print(f"  MISSION PRUNING ECONOMICS  (screen at T0 = launch params, "
          f"T40 = {OPERATING_FRAC:.0%} telemetry)")
    print(f"{'='*118}")
    print(f"  {'PLANET':9} {'PROP_d':>8} {'FAIL%':>6} | {'T0 AUC':>7} {'T0 LR':>7} | "
          f"{'T0 saved':>9} {'T0 FP%':>7} | {'T40 saved':>10} {'T40 FP%':>8} | "
          f"{'CASC sav':>9} {'CASC FP':>7}")
    print(f"  {'-'*118}")

    tot = {"t0_saved": 0.0, "t40_saved": 0.0, "total": 0.0,
           "t0_fp": 0, "t40_fp": 0, "n_pass": 0}

    for planet in PLANETS:
        z = np.load(f"data/per_planet/{planet}.npz")
        Xs, ys, Ls = z["X"], z["y"], z["lengths"]
        if "mission_ids" not in z:
            print(f"  {planet}: extract lacks mission_ids — run "
                  f"src.data_collection.recover_mission_ids")
            continue

        # Align launch params to the extract BY mission_id. The parquet is not
        # sorted by mission_id, so joining on row position pairs unrelated
        # missions — an earlier version of this script did exactly that.
        sub = df[df["target"] == planet]
        by_id = sub.set_index("sim_id" if "sim_id" in sub.columns else "mission_id")
        ids = z["mission_ids"]
        try:
            aligned = by_id.loc[ids]
        except KeyError:
            print(f"  {planet}: mission_ids not found in params table — skipped")
            continue
        assert (aligned["label"].values == ys).all(), f"{planet}: label mismatch after join"

        y = ys
        X = aligned[LAUNCH_COLS].values.astype(np.float64)

        rng = np.random.default_rng(42)
        perm = rng.permutation(len(y))
        n_tr = int(0.70 * len(y))
        tr, te = perm[:n_tr], perm[n_tr:]

        # ── T0 screen: gradient boosting + a linear model for reference ──
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                tree_method="hist", eval_metric="logloss",
                                n_jobs=6, verbosity=0)
        clf.fit(X[tr], y[tr])
        p_succ_t0 = clf.predict_proba(X[te])[:, 1]
        auc_t0 = roc_auc_score(y[te], p_succ_t0)

        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        auc_lr = roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1])

        pf_t0 = 1.0 - p_succ_t0
        thr_t0 = threshold_at_recall(pf_t0, y[te], TARGET_RECALL)
        prune_t0 = pf_t0 >= thr_t0

        # ── T40 screen: the per-planet telemetry model ──
        pf_t40 = np.empty(len(te))
        for j, i in enumerate(te):
            n = max(2, int(Ls[i] * OPERATING_FRAC))
            pf_t40[j] = router.predict(Xs[i, :n], planet)["p_fail"]
        thr_t40 = threshold_at_recall(pf_t40, ys[te], TARGET_RECALL)
        prune_t40 = pf_t40 >= thr_t40

        # ── Compute accounting ──
        d = prop_days(planet)
        n = len(te)
        total_cost = n * d                                  # run everything
        # T0: pruned missions cost nothing; the rest run in full.
        t0_cost = float((~prune_t0).sum()) * d
        # T40: pruned missions still cost the observed fraction.
        t40_cost = (float(prune_t40.sum()) * d * OPERATING_FRAC
                    + float((~prune_t40).sum()) * d)

        t0_saved = 1.0 - t0_cost / total_cost
        t40_saved = 1.0 - t40_cost / total_cost
        pass_mask = ys[te] == 1
        fp_t0 = float((prune_t0 & pass_mask).sum()) / max(pass_mask.sum(), 1)
        fp_t40 = float((prune_t40 & pass_mask).sum()) / max(pass_mask.sum(), 1)

        # ── Cascade: prune at T0 only where the cheap screen is confident,
        # let everything else run to 40% and decide with the telemetry model.
        # A false prune silently destroys a good mission, so the T0 stage is
        # tuned for precision and the T40 stage sweeps up the remainder.
        # The T0 confidence cut is fixed on TRAIN predictions — using test labels
        # to place it would leak. q is the quantile of P(fail) over training
        # successes, so q=0.999 means "prune early only above a level almost no
        # good training mission reached".
        pf_t0_tr = 1.0 - clf.predict_proba(X[tr])[:, 1]
        succ_tr = pf_t0_tr[y[tr] == 1]
        FP_BUDGET = 0.01

        cands = []
        for q in [0.99, 0.995, 0.999, 1.0]:
            t0_conf = float(np.quantile(succ_tr, q)) if len(succ_tr) else 1.0
            early = pf_t0 >= t0_conf
            late = (~early) & prune_t40
            cost = (float(late.sum()) * d * OPERATING_FRAC
                    + float((~early & ~late).sum()) * d)
            saved = 1.0 - cost / total_cost
            fp = float(((early | late) & pass_mask).sum()) / max(pass_mask.sum(), 1)
            rec = float(((early | late) & (ys[te] == 0)).sum()) / max((ys[te] == 0).sum(), 1)
            cands.append({"q": q, "saved": saved, "fp": fp, "recall": rec})

        feasible = [c for c in cands if c["fp"] <= FP_BUDGET]
        best = (max(feasible, key=lambda c: c["saved"]) if feasible
                else min(cands, key=lambda c: c["fp"]))

        tot.setdefault("casc_saved", 0.0)
        tot.setdefault("casc_fp", 0)
        tot["casc_saved"] += best["saved"] * total_cost
        tot["casc_fp"] += int(round(best["fp"] * pass_mask.sum()))

        tot["t0_saved"] += total_cost - t0_cost
        tot["t40_saved"] += total_cost - t40_cost
        tot["total"] += total_cost
        tot["t0_fp"] += int((prune_t0 & pass_mask).sum())
        tot["t40_fp"] += int((prune_t40 & pass_mask).sum())
        tot["n_pass"] += int(pass_mask.sum())

        print(f"  {planet:9} {d:>8.0f} {100*(y==0).mean():>5.1f}% | {auc_t0:>7.4f} "
              f"{auc_lr:>7.4f} | {t0_saved:>8.1%} {fp_t0:>6.1%} | "
              f"{t40_saved:>9.1%} {fp_t40:>7.1%} | {best['saved']:>8.1%} {best['fp']:>6.1%}")

        rows.append({"planet": planet, "prop_days": d, "auc_t0": round(auc_t0, 4),
                     "auc_logreg_t0": round(auc_lr, 4),
                     "compute_saved_t0": round(t0_saved, 4),
                     "compute_saved_t40": round(t40_saved, 4),
                     "false_prune_rate_t0": round(fp_t0, 4),
                     "false_prune_rate_t40": round(fp_t40, 4),
                     "cascade_saved": round(best["saved"], 4),
                     "cascade_false_prune": round(best["fp"], 4),
                     "cascade_recall": round(best["recall"], 4)})

    print(f"  {'-'*118}")
    print(f"  {'WEIGHTED':9} {'':>8} {'':>6} | {'':>7} {'':>7} | "
          f"{tot['t0_saved']/tot['total']:>8.1%} "
          f"{tot['t0_fp']/max(tot['n_pass'],1):>6.1%} | "
          f"{tot['t40_saved']/tot['total']:>9.1%} "
          f"{tot['t40_fp']/max(tot['n_pass'],1):>7.1%} | "
          f"{tot['casc_saved']/tot['total']:>8.1%} "
          f"{tot['casc_fp']/max(tot['n_pass'],1):>6.1%}")
    print(f"\n  Compute weighted by propagation duration, at {TARGET_RECALL:.0%} "
          f"failure recall.")
    print(f"  T0 dominates: a mission pruned before launch costs nothing, one pruned")
    print(f"  at {OPERATING_FRAC:.0%} has already burned {OPERATING_FRAC:.0%} of its budget.\n")

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target_recall": TARGET_RECALL,
                               "operating_frac": OPERATING_FRAC,
                               "per_planet": rows}, indent=2))
    print(f"  Saved → {out}\n")


if __name__ == "__main__":
    main()
