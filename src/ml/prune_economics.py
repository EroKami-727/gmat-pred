"""
Mission Pruning: how much simulation compute can be skipped, and at what cost.

Two screening points are compared on identical splits:

  T0  — screen BEFORE running the simulation, from launch parameters only
        (TOI burn + parking-orbit orientation). A pruned mission costs zero
        propagation time.
  T40 — screen after propagating 40% of the trajectory, from telemetry
        (the per-planet Transformer). A pruned mission still costs 40%.

Compute is charged in propagation-days, which is what the simulator actually
spends: cost per mission is proportional to propagation duration, and that spans
~100x between Mercury (127 d) and Neptune (13,419 d).

Reported at a HIGH-RECALL operating point, because in pruning a false abort
throws away a good mission (expensive, silent) while a missed failure only costs
the compute you were already going to spend.

Scope: the seven interplanetary targets. Moon is excluded by decision — see
EXCLUDED_TARGETS in src/ml/planet_config.py.


Evaluation protocol
-------------------
This script previously had two selection biases, both fixed here. They are
documented rather than quietly removed, because the corrected numbers are within
0.1 pp of the published ones and that fact is itself worth stating: the result
was not an artifact of the biases.

1. **Thresholds were fitted on the test labels.** `threshold_at_recall` was
   called with the test split's own labels, so the 99% recall operating point
   was chosen with knowledge of the answers — a deployed screen cannot do that.
   Thresholds are now fitted on the validation split and applied unchanged to
   test, so the reported failure recall is what the operating point actually
   achieves out of sample (99.06% weighted, not 99% by construction).

2. **The evaluation set contained the sequence model's validation split.** This
   script drew its own 70/30 partition with seed 42 while per_planet_train.py
   used 70/15/15 with seed 42 — same seed, same N, same permutation, so this
   script's "test" set was exactly the model's val + test. Half the missions the
   T40 screen was scored on were the ones its checkpoint and abort threshold had
   been selected on. Both now call src/ml/splits.py, which is the single
   definition of the partition.

Note the direction of bias (2): it flattered T40, the screen this analysis
concludes *against*. The negative result survives its own correction, which is
the strongest form it can take.

Usage:
    python -m src.ml.prune_economics
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_collection.gmat_runner import MissionConfig
from src.ml.planet_config import OPERATING_FRAC, PLANETS
from src.ml.planet_router import PlanetRouter
from src.ml.splits import train_val_test
from src.paths import params_parquet, require, summary_parquet

# What the screener may look at before the simulation runs: the six injection
# offsets. Nothing derived from the trajectory, because there isn't one yet.
LAUNCH_COLS = ["dv_V_offset", "dv_N_offset", "dv_B_offset",
               "RAAN_offset", "AOP_offset", "INC_offset"]

TARGET_RECALL = 0.99      # of failures caught, targeted on validation

# Cascade: the T0 stage prunes only where it is confident, everything else runs
# to 40% and is decided by telemetry. Budget on good missions destroyed.
FP_BUDGET = 0.01
CASCADE_QUANTILES = [0.99, 0.995, 0.999, 1.0]


def prop_days(planet: str) -> float:
    """Propagation duration for one mission to `planet`, in days."""
    try:
        return float(MissionConfig("earth", planet).prop_days)
    except Exception:                                            # noqa: BLE001
        return 1.0


def load() -> pd.DataFrame:
    """
    Join launch parameters to outcomes ON THE KEY.

    params.sim_id is sorted ascending but summary.mission_id is NOT, so
    concatenating the two frames by row position pairs unrelated missions.
    An earlier version of this file did that and its numbers were wrong
    (docs/RESEARCH_LEDGER.md, "Data-alignment defect").
    """
    p = pq.read_table(require(params_parquet())).to_pandas()
    s = pq.read_table(require(summary_parquet())).to_pandas()
    df = p.merge(s[["mission_id", "label", "failure_type", "min_target_rmag"]],
                 left_on="sim_id", right_on="mission_id", how="inner",
                 validate="one_to_one")
    df["target"] = df["target"].str.lower()
    return df


def threshold_at_recall(p_fail: np.ndarray, y: np.ndarray, recall: float) -> float:
    """
    Lowest P(fail) threshold catching `recall` of the failures in (p_fail, y).

    MUST be called with validation data. Passing the test split here is exactly
    the oracle-thresholding bug described in the module docstring.
    """
    fails = p_fail[y == 0]
    if len(fails) == 0:
        return 0.5
    return float(np.quantile(fails, 1.0 - recall))


def achieved(prune: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(failure recall, good-mission false-prune rate) for a prune decision."""
    fail, good = y == 0, y == 1
    rec = float((prune & fail).sum()) / max(int(fail.sum()), 1)
    fp = float((prune & good).sum()) / max(int(good.sum()), 1)
    return rec, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/prune_economics.json")
    ap.add_argument("--seed", type=int, default=42,
                    help="split seed; must match the seed the models were trained with")
    args = ap.parse_args()

    df = load()
    router = PlanetRouter("models/per_planet")
    rows = []

    print(f"\n{'='*118}")
    print(f"  MISSION PRUNING ECONOMICS  (screen at T0 = launch params, "
          f"T40 = {OPERATING_FRAC:.0%} telemetry)")
    print(f"  Thresholds fitted on validation, reported on the held-out test split.")
    print(f"{'='*118}")
    print(f"  {'PLANET':9} {'PROP_d':>8} {'FAIL%':>6} | {'T0 AUC':>7} {'T0 LR':>7} | "
          f"{'T0 saved':>9} {'T0 FP%':>7} {'T0 rec':>7} | {'T40 saved':>10} "
          f"{'T40 FP%':>8} {'T40 rec':>8} | {'CASC sav':>9} {'CASC FP':>8}")
    print(f"  {'-'*118}")

    tot = {"t0_saved": 0.0, "t40_saved": 0.0, "casc_saved": 0.0, "total": 0.0,
           "t0_fp": 0, "t40_fp": 0, "casc_fp": 0, "n_good": 0,
           "t0_caught": 0, "t40_caught": 0, "casc_caught": 0, "n_fail": 0}

    for planet in PLANETS:
        z = np.load(f"data/per_planet/{planet}.npz")
        Xs, ys, Ls = z["X"], z["y"], z["lengths"]
        if "mission_ids" not in z:
            print(f"  {planet:9} SKIPPED — extract lacks mission_ids; run "
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
            print(f"  {planet:9} SKIPPED — mission_ids not found in params table")
            continue
        assert (aligned["label"].values == ys).all(), f"{planet}: label mismatch after join"

        X = aligned[LAUNCH_COLS].values.astype(np.float64)

        # The SAME partition the per-planet model was trained under. Test is
        # touched once, below, for reporting only.
        tr, va, te = train_val_test(len(ys), args.seed)

        # ── T0 screen: gradient boosting on launch parameters ────────────────
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                tree_method="hist", eval_metric="logloss",
                                n_jobs=6, verbosity=0)
        clf.fit(X[tr], ys[tr])
        pf_va_t0 = 1.0 - clf.predict_proba(X[va])[:, 1]
        pf_te_t0 = 1.0 - clf.predict_proba(X[te])[:, 1]
        auc_t0 = roc_auc_score(ys[te], 1.0 - pf_te_t0)

        # Logistic regression on identical features, as a linearity check. It
        # scores ~0.49 (chance) everywhere: the parameter-to-outcome map is
        # strongly nonlinear, so this is a case for ML even though it turns out
        # not to be a case for sequence modelling.
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), ys[tr])
        auc_lr = roc_auc_score(ys[te], lr.predict_proba(sc.transform(X[te]))[:, 1])

        thr_t0 = threshold_at_recall(pf_va_t0, ys[va], TARGET_RECALL)
        prune_t0 = pf_te_t0 >= thr_t0

        # ── T40 screen: the per-planet telemetry model ───────────────────────
        def score(idx: np.ndarray) -> np.ndarray:
            out = np.empty(len(idx))
            for j, i in enumerate(idx):
                n = max(2, int(Ls[i] * OPERATING_FRAC))
                out[j] = router.predict(Xs[i, :n], planet)["p_fail"]
            return out

        pf_va_t40 = score(va)
        pf_te_t40 = score(te)
        thr_t40 = threshold_at_recall(pf_va_t40, ys[va], TARGET_RECALL)
        prune_t40 = pf_te_t40 >= thr_t40

        # ── Compute accounting ───────────────────────────────────────────────
        d = prop_days(planet)
        n = len(te)
        total_cost = n * d                                  # run everything
        # T0: pruned missions cost nothing; the rest run in full.
        t0_cost = float((~prune_t0).sum()) * d
        # T40: pruned missions have already burned the observed fraction.
        t40_cost = (float(prune_t40.sum()) * d * OPERATING_FRAC
                    + float((~prune_t40).sum()) * d)

        t0_saved = 1.0 - t0_cost / total_cost
        t40_saved = 1.0 - t40_cost / total_cost
        rec_t0, fp_t0 = achieved(prune_t0, ys[te])
        rec_t40, fp_t40 = achieved(prune_t40, ys[te])

        # ── Cascade ──────────────────────────────────────────────────────────
        # Prune at T0 only where the cheap screen is confident, let everything
        # else run to 40% and decide with the telemetry model. A false prune
        # silently destroys a good mission, so the T0 stage is tuned for
        # precision and the T40 stage sweeps up the remainder.
        #
        # The T0 confidence cut is a quantile of P(fail) over TRAINING
        # successes, and the quantile itself is chosen on VALIDATION. Test is
        # not consulted at any point in this selection.
        pf_tr_t0 = 1.0 - clf.predict_proba(X[tr])[:, 1]
        succ_tr = pf_tr_t0[ys[tr] == 1]

        cands = []
        for q in CASCADE_QUANTILES:
            t0_conf = float(np.quantile(succ_tr, q)) if len(succ_tr) else 1.0
            early_va = pf_va_t0 >= t0_conf
            late_va = (~early_va) & (pf_va_t40 >= thr_t40)
            rec_va, fp_va = achieved(early_va | late_va, ys[va])
            cost_va = (float(late_va.sum()) * d * OPERATING_FRAC
                       + float((~early_va & ~late_va).sum()) * d)
            cands.append({"q": q, "t0_conf": t0_conf,
                          "saved_va": 1.0 - cost_va / (len(va) * d),
                          "fp_va": fp_va, "rec_va": rec_va})

        feasible = [c for c in cands if c["fp_va"] <= FP_BUDGET]
        best = (max(feasible, key=lambda c: c["saved_va"]) if feasible
                else min(cands, key=lambda c: c["fp_va"]))

        # Apply the chosen configuration to test.
        early = pf_te_t0 >= best["t0_conf"]
        late = (~early) & prune_t40
        casc_cost = (float(late.sum()) * d * OPERATING_FRAC
                     + float((~early & ~late).sum()) * d)
        casc_saved = 1.0 - casc_cost / total_cost
        rec_casc, fp_casc = achieved(early | late, ys[te])

        # ── Aggregate ────────────────────────────────────────────────────────
        good, fail = ys[te] == 1, ys[te] == 0
        tot["t0_saved"] += total_cost - t0_cost
        tot["t40_saved"] += total_cost - t40_cost
        tot["casc_saved"] += total_cost - casc_cost
        tot["total"] += total_cost
        tot["t0_fp"] += int((prune_t0 & good).sum())
        tot["t40_fp"] += int((prune_t40 & good).sum())
        tot["casc_fp"] += int(((early | late) & good).sum())
        tot["n_good"] += int(good.sum())
        tot["t0_caught"] += int((prune_t0 & fail).sum())
        tot["t40_caught"] += int((prune_t40 & fail).sum())
        tot["casc_caught"] += int(((early | late) & fail).sum())
        tot["n_fail"] += int(fail.sum())

        print(f"  {planet:9} {d:>8.0f} {100*(ys==0).mean():>5.1f}% | {auc_t0:>7.4f} "
              f"{auc_lr:>7.4f} | {t0_saved:>8.1%} {fp_t0:>6.2%} {rec_t0:>6.2%} | "
              f"{t40_saved:>9.1%} {fp_t40:>7.2%} {rec_t40:>7.2%} | "
              f"{casc_saved:>8.1%} {fp_casc:>7.2%}")

        rows.append({"planet": planet, "prop_days": d, "n_test": int(n),
                     "auc_t0": round(auc_t0, 4), "auc_logreg_t0": round(auc_lr, 4),
                     "compute_saved_t0": round(t0_saved, 4),
                     "compute_saved_t40": round(t40_saved, 4),
                     "false_prune_rate_t0": round(fp_t0, 4),
                     "false_prune_rate_t40": round(fp_t40, 4),
                     "fail_recall_t0": round(rec_t0, 4),
                     "fail_recall_t40": round(rec_t40, 4),
                     "cascade_saved": round(casc_saved, 4),
                     "cascade_false_prune": round(fp_casc, 4),
                     "cascade_recall": round(rec_casc, 4)})

    # Weighted aggregate. This is the paper's headline row, so it is written to
    # the artifact rather than only printed — previously it existed solely in
    # stdout and the docs quoted it with nothing to check it against.
    w = {
        "compute_saved_t0": tot["t0_saved"] / tot["total"],
        "compute_saved_t40": tot["t40_saved"] / tot["total"],
        "cascade_saved": tot["casc_saved"] / tot["total"],
        "false_prune_rate_t0": tot["t0_fp"] / max(tot["n_good"], 1),
        "false_prune_rate_t40": tot["t40_fp"] / max(tot["n_good"], 1),
        "cascade_false_prune": tot["casc_fp"] / max(tot["n_good"], 1),
        "fail_recall_t0": tot["t0_caught"] / max(tot["n_fail"], 1),
        "fail_recall_t40": tot["t40_caught"] / max(tot["n_fail"], 1),
        "cascade_recall": tot["casc_caught"] / max(tot["n_fail"], 1),
        "n_good": tot["n_good"], "n_fail": tot["n_fail"],
    }

    print(f"  {'-'*118}")
    print(f"  {'WEIGHTED':9} {'':>8} {'':>6} | {'':>7} {'':>7} | "
          f"{w['compute_saved_t0']:>8.1%} {w['false_prune_rate_t0']:>6.2%} "
          f"{w['fail_recall_t0']:>6.2%} | "
          f"{w['compute_saved_t40']:>9.1%} {w['false_prune_rate_t40']:>7.2%} "
          f"{w['fail_recall_t40']:>7.2%} | "
          f"{w['cascade_saved']:>8.1%} {w['cascade_false_prune']:>7.2%}")
    print(f"\n  Compute weighted by propagation duration. Operating point targets "
          f"{TARGET_RECALL:.0%} failure recall on validation;")
    print(f"  the recall columns are what that threshold achieves on the held-out "
          f"test split.")
    print(f"  T0 dominates: a mission pruned before launch costs nothing, one pruned "
          f"at {OPERATING_FRAC:.0%} has already burned {OPERATING_FRAC:.0%}.\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "target_recall": TARGET_RECALL,
        "operating_frac": OPERATING_FRAC,
        "seed": args.seed,
        "protocol": ("thresholds fitted on the validation split, reported on the "
                     "held-out test split; partition from src/ml/splits.py"),
        "targets": PLANETS,
        "weighted": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in w.items()},
        "per_planet": rows,
    }, indent=2))
    print(f"  Saved → {out}\n")


if __name__ == "__main__":
    main()
