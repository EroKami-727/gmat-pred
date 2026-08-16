"""
Canonical per-planet sampling configuration.

Serving MUST feed the model telemetry at the same cadence it was trained on.
Both the extractor and the inference router import from here so the two can
never drift apart.

Native cadence in merged_all_v2 is 15 h per row for every target, so raw
mission length varies ~100x between Mercury (202 rows) and Neptune (21,471).
Downsampling each planet to ~TARGET_STEPS rows gives all planets comparable
sequence lengths.

Three target sets are defined here and they are not interchangeable:

    ALL_TARGETS      everything in the dataset, Moon included
    PLANETS          the seven-target study set — use for anything reported
    SERVING_TARGETS  what the router loads and the API offers, Moon included

Reported results use PLANETS. Training and calibration cover SERVING_TARGETS,
because the live simulator still offers Moon even though the paper does not
claim anything about it.
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

# Every target present in the dataset.
ALL_TARGETS = list(ROWS_PER_MISSION)

# Targets deliberately excluded from the study, and therefore from every
# reported table.
#
# Moon is out of scope. It is not an interplanetary transfer: a 6-day trajectory
# sampled at 60 s inside Earth's sphere of influence, against seven heliocentric
# transfers of 127-13,419 propagation-days sampled at 15 h. It shares neither
# the cost structure that the pruning economics are built on (its propagation is
# ~1/100th of the cheapest planet) nor the dynamical regime, and the
# leave-one-target-out audit already showed it as the single worst transfer
# case (AUC 0.296).
#
# It was previously dropped by accident rather than by decision — the Moon
# extract was regenerated after recover_mission_ids.py ran, so it lacks the
# mission_ids key, and prune_economics.py printed a skip line and carried on.
# The headline table has therefore always been seven targets. Making the
# exclusion explicit means the paper can state it as a scope decision, which is
# what it is, instead of leaving a reviewer to ask where the Moon went.
EXCLUDED_TARGETS = {"moon"}

#: The study set: seven interplanetary targets. This is what PLANETS means for
#: training, evaluation and every reported result.
PLANETS = [p for p in ALL_TARGETS if p not in EXCLUDED_TARGETS]

#: Serving keeps Moon. Its model is trained and calibrated (test F1 0.9888) and
#: the live simulator offers it as a target, so the router and API load the full
#: set. Excluded from the study != removed from the product.
SERVING_TARGETS = ALL_TARGETS

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
