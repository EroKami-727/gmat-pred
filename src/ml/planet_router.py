"""
OrbitGuard Per-Planet Router
============================
Serving-side counterpart to `per_planet_train.py`. Loads one model per target
body and routes inference by planet name.

Replaces the old RegimeRouter, which grouped 3-4 planets behind a single model
and shared scaler. That sharing was the bug: a scaler fitted across planets has
an IQR spanning the cross-planet range, so within-planet mission-to-mission
variation shrank to ~1e-5 of the input range and the transformer could not
learn it (Venus emitted a constant P(fail)=0.020910 for every mission). Trees
were unaffected, which is why XGBoost baselines masked the problem.

Each planet directory holds:
    model.pt        — trained weights (dual head: outcome + failure mode)
    norm_stats.npz  — per-timestep mu/sd (L, F), fitted on that planet's train split
    meta.json       — threshold, downsample factor, metrics

Every prediction returns both the abort probability and the predicted failure
mode, so callers can report how a mission is expected to fail.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.ml.dataset import FEATURE_COLS
from src.ml.model import TrajectoryTransformer
from src.ml.planet_config import (
    FAILURE_NAMES, N_FAILURE_CLASSES, OPERATING_FRAC, PLANETS, downsample_for,
)

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5

# Per-timestep z-scores give excellent in-distribution discrimination precisely
# because the training spread at each timestep is tiny. The flip side is that
# genuinely novel inputs produce enormous z-scores (synthetic two-body missions
# hit |z| ~ 1e13), which saturates the model into a confident wrong answer.
# Clip so the network sees bounded input, and report how far out the input was
# so callers can distrust the number instead of acting on it.
Z_CLIP = 10.0
OOD_FRACTION_LIMIT = 0.25


class PlanetRouter:
    """Loads per-planet models and routes inference by target body."""

    def __init__(self, models_root: str | Path = "models/per_planet"):
        self.root = Path(models_root)
        self._caches: dict[str, dict] = {}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for planet in PLANETS:
            d = self.root / planet
            mp, sp, meta_p = d / "model.pt", d / "norm_stats.npz", d / "meta.json"
            if not (mp.exists() and sp.exists() and meta_p.exists()):
                continue
            try:
                meta = json.loads(meta_p.read_text())
                stats = np.load(sp)
                model = TrajectoryTransformer(
                    input_dim=len(FEATURE_COLS), output_dim=1,
                    task="binary", aux_dim=meta.get("aux_dim", N_FAILURE_CLASSES),
                )
                model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
                model.to(device).eval()
                entry = {
                    "model": model, "device": device,
                    "mu": stats["mu"].astype(np.float64),
                    "sd": stats["sd"].astype(np.float64),
                    "meta": meta,
                    "threshold": float(meta.get("threshold", DEFAULT_THRESHOLD)),
                    "downsample": int(meta.get("downsample", downsample_for(planet))),
                    "assist": None, "assist_window": None,
                }
                # Optional tree assist at the decision window. The Transformer
                # misses rare failure modes it has the information for (Uranus
                # surface_impact: 0.000 recall, vs 1.000 for a tree on the same
                # normalised input), so the two are fused once enough of the
                # trajectory is available.
                ap_, am_ = d / "assist.json", d / "assist_meta.json"
                if ap_.exists() and am_.exists():
                    try:
                        import xgboost as xgb
                        booster = xgb.XGBClassifier()
                        booster.load_model(ap_)
                        entry["assist"] = booster
                        entry["assist_window"] = int(json.loads(am_.read_text())["window"])
                    except Exception as e:                       # noqa: BLE001
                        log.warning("PlanetRouter: assist load failed for %s: %s", planet, e)
                self._caches[planet] = entry
                log.info("PlanetRouter: loaded %s (thr=%.3f)", planet, self._caches[planet]["threshold"])
            except Exception as e:                                  # noqa: BLE001
                log.warning("PlanetRouter: failed to load %s: %s", planet, e)

    def reload(self):
        self._caches.clear()
        self._load()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(name: str) -> str:
        return (name or "").strip().lower()

    def is_available(self) -> bool:
        return bool(self._caches)

    def supports(self, target: str) -> bool:
        return self._norm(target) in self._caches

    def threshold_for(self, target: str) -> float:
        c = self._caches.get(self._norm(target))
        return c["threshold"] if c else DEFAULT_THRESHOLD

    def downsample_for_target(self, target: str) -> int:
        c = self._caches.get(self._norm(target))
        return c["downsample"] if c else downsample_for(self._norm(target))

    def operating_frac(self, target: str) -> float:
        c = self._caches.get(self._norm(target))
        return float(c["meta"].get("operating_frac", OPERATING_FRAC)) if c else OPERATING_FRAC

    # ── Inference ─────────────────────────────────────────────────────────────

    def _prepare(self, cache: dict, features: np.ndarray) -> tuple[torch.Tensor, float]:
        """
        Per-timestep z-score using this planet's training statistics.
        Returns (tensor, ood_fraction) where ood_fraction is the share of
        values that fell outside the clip range before clipping.
        """
        X = np.asarray(features, dtype=np.float64)
        n, L = len(X), len(cache["mu"])
        if n <= L:
            mu, sd = cache["mu"][:n], cache["sd"][:n]
        else:
            # Longer than anything seen in training: hold the last known stats.
            pad = n - L
            mu = np.vstack([cache["mu"], np.repeat(cache["mu"][-1:], pad, axis=0)])
            sd = np.vstack([cache["sd"], np.repeat(cache["sd"][-1:], pad, axis=0)])

        with np.errstate(over="ignore", invalid="ignore"):
            Z = (X - mu) / sd
        Z = np.nan_to_num(Z, nan=0.0, posinf=Z_CLIP, neginf=-Z_CLIP)
        ood_fraction = float((np.abs(Z) > Z_CLIP).mean())
        Z = np.clip(Z, -Z_CLIP, Z_CLIP).astype(np.float32)
        return torch.from_numpy(Z).unsqueeze(0).to(cache["device"]), ood_fraction

    def predict(self, features_so_far: np.ndarray, target: str) -> dict:
        """
        Run the trajectory prefix through this planet's model.

        Returns {p_fail, failure_mode, failure_mode_name, mode_confidence,
                 threshold, should_abort, available}.
        """
        cache = self._caches.get(self._norm(target))
        if cache is None or len(features_so_far) == 0:
            return {"p_fail": 0.0, "failure_mode": 0, "failure_mode_name": "unknown",
                    "mode_confidence": 0.0, "threshold": DEFAULT_THRESHOLD,
                    "should_abort": False, "available": False,
                    "ood_fraction": 0.0, "out_of_distribution": False}

        x, ood_fraction = self._prepare(cache, features_so_far)
        mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=cache["device"])
        with torch.no_grad():
            logit, aux = cache["model"].forward_multitask(x, mask)
            p_success = float(torch.sigmoid(logit).item())
            if aux is not None:
                probs = torch.softmax(aux, dim=1)[0]
                # Report the most likely FAILURE mode, ignoring the success class,
                # so the UI can say how it would fail even before it is doomed.
                fail_probs = probs[1:]
                mode = int(fail_probs.argmax().item()) + 1
                conf = float(fail_probs[mode - 1].item())
            else:
                mode, conf = 7, 0.0

        p_fail_seq = 1.0 - p_success
        p_fail = p_fail_seq
        p_fail_assist = None

        # Fuse the tree assist once the prefix covers its window. Taking the max
        # keeps every failure either model is sure about; the threshold is
        # recalibrated on the fused score so precision is preserved.
        W = cache.get("assist_window")
        if cache.get("assist") is not None and W and x.shape[1] >= W:
            zw = x[0, :W].detach().cpu().numpy().reshape(1, -1)
            p_fail_assist = float(1.0 - cache["assist"].predict_proba(zw)[0, 1])
            p_fail = max(p_fail_seq, p_fail_assist)

        ood = ood_fraction > OOD_FRACTION_LIMIT
        return {
            "p_fail": p_fail,
            "p_fail_sequence": p_fail_seq,
            "p_fail_assist": p_fail_assist,
            "failure_mode": mode,
            "failure_mode_name": FAILURE_NAMES.get(mode, "unknown"),
            "mode_confidence": conf,
            "threshold": cache["threshold"],
            # Advisory, not a veto. An earlier version suppressed the abort when
            # ood was set; that was protecting against a generator that produced
            # wrong features. With missions built by the real propagator, an OOD
            # flag means "physically valid but statistically extreme" (e.g. a 17-
            # sigma burn error), where P(fail) is still correct — vetoing it threw
            # away right answers. Callers should surface the flag, not act on it.
            "should_abort": bool(p_fail >= cache["threshold"]),
            "available": True,
            "ood_fraction": round(ood_fraction, 4),
            "out_of_distribution": ood,
        }

    def infer_step(self, features_so_far: np.ndarray, target: str) -> float:
        """Backwards-compatible scalar P(fail) accessor."""
        return self.predict(features_so_far, target)["p_fail"]

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "loaded_planets": sorted(self._caches),
            "thresholds": {p: c["threshold"] for p, c in self._caches.items()},
            "downsample": {p: c["downsample"] for p, c in self._caches.items()},
            "metrics": {
                p: c["meta"].get("test", {}).get(f"{OPERATING_FRAC:.2f}", {})
                for p, c in self._caches.items()
            },
        }
