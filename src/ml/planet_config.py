"""
Canonical per-planet sampling configuration.

Serving MUST feed the model telemetry at the same cadence it was trained on.
Both the extractor and the inference router import from here so the two can
never drift apart.

Native cadence in merged_all_v2 is 15 h per row for every target, so raw
mission length varies ~100x between Mercury (202 rows) and Neptune (21,471).
Downsampling each planet to ~TARGET_STEPS rows gives all planets comparable
sequence lengths.
"""

from __future__ import annotations

TARGET_STEPS = 100

# Raw cadence of the interplanetary targets. Moon differs (60 s) — use
# cadence_hours() rather than this constant directly.
NATIVE_CADENCE_HOURS = 15.0
CADENCE_HOURS = {"moon": 60.0 / 3600.0}

# Measured rows-per-mission in merged_all_v2 (484M rows total).
# Moon is a 6-day transfer sampled at 60 s (8,528 rows/mission); the
# interplanetary targets are sampled at 15 h. Downsampling to TARGET_STEPS
# normalises both onto comparable sequence lengths.
ROWS_PER_MISSION = {
    "moon": 8528,
    "mercury": 202, "venus": 280, "mars": 497, "jupiter": 1916,
    "saturn": 4241, "uranus": 11248, "neptune": 21471,
}

PLANETS = list(ROWS_PER_MISSION)

FAILURE_NAMES = {
    0: "success", 1: "surface_impact", 2: "orbit_too_high", 3: "missed_target",
    4: "source_impact", 5: "hyperbolic_flyby", 6: "degenerate_orbit", 7: "unknown",
}
N_FAILURE_CLASSES = 8

FAILURE_TYPE_MAP = {
    "success": 0, "surface_impact": 1, "orbit_too_high": 2,
    "missed_target": 3, "missed_moon": 3, "source_impact": 4,
    "earth_impact": 4, "hyperbolic_flyby": 5, "degenerate_orbit": 6,
    "unknown": 7,
}

# Fraction of a mission observed before the abort decision is made.
OPERATING_FRAC = 0.40


def downsample_for(planet: str, target_steps: int = TARGET_STEPS) -> int:
    """Rows to skip so `planet` yields ~target_steps rows per mission."""
    rpm = ROWS_PER_MISSION.get(planet.strip().lower(), 500)
    return max(1, round(rpm / target_steps))


def step_hours(planet: str, target_steps: int = TARGET_STEPS) -> float:
    """
    Wall-clock hours between consecutive model timesteps for `planet`.

    Synthetic missions are propagated on their own timestep, so they must be
    resampled to this spacing before inference or the positional structure
    will not match what the model was trained on.
    """
    base = CADENCE_HOURS.get(planet.strip().lower(), NATIVE_CADENCE_HOURS)
    return base * downsample_for(planet, target_steps)
