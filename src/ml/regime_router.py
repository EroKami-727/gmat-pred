"""
OrbitGuard Regime Router
========================
Routes trajectory inference to the correct regime model based on target body.

Regime split (excluding Moon — too short a transfer, genuinely different physics):
  Inner: Mercury, Venus, Mars
  Outer: Jupiter, Saturn, Uranus, Neptune

Falls back to the single full-dataset model if regime models haven't been trained yet,
so the API stays functional throughout the training pipeline.

Trained model directories expected at:
  models/inner_production/   ← trained on Mercury, Venus, Mars
  models/outer_production/   ← trained on Jupiter, Saturn, Uranus, Neptune
  models/thresholds.json     ← per-target P(fail) thresholds from calibration
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from src.ml.dataset import FEATURE_COLS
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer

log = logging.getLogger(__name__)

INNER_PLANETS = {"mercury", "venus", "mars"}
OUTER_PLANETS = {"jupiter", "saturn", "uranus", "neptune"}

# Default P(fail) threshold = 1 - calibrated P(success) 0.557
DEFAULT_THRESHOLD = 0.443


class RegimeRouter:
    """
    Loads inner/outer regime models and routes inference by target body name.
    Provides per-target calibrated thresholds after running per_target_calibration.py.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        # Keys: "inner" | "outer" | "fallback"
        self._caches: dict[str, dict | None] = {}
        # target_name (title-cased) -> P(fail) threshold
        self.thresholds: dict[str, float] = {}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_checkpoint(self, subdir: str) -> dict | None:
        """Load model + scaler from models_dir/subdir. Returns None if not present."""
        base = self.models_dir / subdir
        for arch in ["transformer", "lstm"]:
            mp = base / f"best_model_{arch}_binary.pt"
            sp = base / f"scaler_{arch}_binary.pkl"
            if mp.exists() and sp.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                with open(sp, "rb") as f:
                    scaler = pickle.load(f)
                input_dim = len(FEATURE_COLS)
                if arch == "transformer":
                    model = TrajectoryTransformer(input_dim=input_dim, output_dim=1, task="binary")
                else:
                    model = TrajectoryLSTM(input_dim=input_dim, output_dim=1, task="binary")
                model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
                model.to(device).eval()
                log.info("RegimeRouter: loaded %s from %s", arch, base)
                return {"model": model, "scaler": scaler, "arch": arch, "device": device}
        return None

    def _load(self):
        self._caches["inner"]    = self._load_checkpoint("inner_production")
        self._caches["outer"]    = self._load_checkpoint("outer_production")
        # Fallback: try known single-model dirs in priority order
        for fb_dir in ["transformer_production", "lstm_production", ""]:
            c = self._load_checkpoint(fb_dir)
            if c is not None:
                self._caches["fallback"] = c
                break
        else:
            self._caches["fallback"] = None

        thresholds_path = self.models_dir / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                self.thresholds = json.load(f)
            log.info("RegimeRouter: calibrated thresholds for %s", list(self.thresholds))

    # ── Routing ───────────────────────────────────────────────────────────────

    def _normalize(self, target_name: str) -> str:
        """Normalize target name to lowercase."""
        return target_name.strip().lower()

    def regime_for(self, target_name: str) -> str:
        """Return 'inner', 'outer', or 'fallback' for this target."""
        name = self._normalize(target_name)
        if name in INNER_PLANETS and self._caches.get("inner"):
            return "inner"
        if name in OUTER_PLANETS and self._caches.get("outer"):
            return "outer"
        return "fallback"

    def _cache_for(self, target_name: str) -> dict | None:
        regime = self.regime_for(target_name)
        return self._caches.get(regime) or self._caches.get("fallback")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run(self, cache: dict, features: np.ndarray) -> float:
        """Run model on features array, return P(success) in [0, 1]."""
        scaled = cache["scaler"].transform(features)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(cache["device"])
        with torch.no_grad():
            if cache["arch"] == "transformer":
                mask = torch.zeros(1, len(features), dtype=torch.bool, device=cache["device"])
                logit = cache["model"](x, mask)
            else:
                lengths = torch.tensor([len(features)], dtype=torch.long)
                logit = cache["model"](x, lengths)
        return float(torch.sigmoid(logit).item())

    def infer_step(self, features_so_far: np.ndarray, target_name: str) -> float:
        """
        Run inference on the trajectory seen so far.
        Returns P(fail) in [0, 1] — higher = more likely to fail.
        Uses the regime-appropriate model, falls back to the single model if needed.
        """
        cache = self._cache_for(target_name)
        if cache is None:
            return 0.0
        p_success = self._run(cache, features_so_far)
        return 1.0 - p_success

    # ── Thresholds ────────────────────────────────────────────────────────────

    def threshold_for(self, target_name: str) -> float:
        """Return calibrated P(fail) abort threshold for this target body."""
        name = self._normalize(target_name)
        return self.thresholds.get(name, DEFAULT_THRESHOLD)

    # ── Status ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return any(v is not None for v in self._caches.values())

    def status(self) -> dict:
        return {
            "inner_loaded":        self._caches.get("inner") is not None,
            "outer_loaded":        self._caches.get("outer") is not None,
            "fallback_loaded":     self._caches.get("fallback") is not None,
            "calibrated_targets":  list(self.thresholds.keys()),
            "thresholds":          self.thresholds,
        }

    def reload(self):
        """Hot-reload all models and thresholds (e.g. after training completes)."""
        self._caches.clear()
        self.thresholds.clear()
        self._load()
