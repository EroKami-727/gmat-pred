"""
Monte Carlo Mission Parameter Generator
========================================
Creates randomized sets of orbital parameters for GMAT simulations.
Deliberately uses wide variance so ~30-40% of generated missions fail
(crash into Earth, escape to infinity, or run out of fuel).
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0          # Mean Earth radius
EARTH_MU = 398600.4418            # GM of Earth  (km³/s²)
MOON_DISTANCE_KM = 384400.0      # Average Earth-Moon distance


@dataclass
class MissionParams:
    """Parameters for a single GMAT simulation run."""
    mission_id: int

    # Keplerian orbital elements
    sma: float       # Semi-major axis (km)
    ecc: float       # Eccentricity
    inc: float       # Inclination (deg)
    raan: float      # Right Ascension of Ascending Node (deg)
    aop: float       # Argument of Periapsis (deg)
    ta: float        # True Anomaly (deg)

    # Spacecraft properties
    dry_mass: float  # kg
    fuel_mass: float # kg

    # Simulation duration
    prop_days: float # Propagation time (days)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_inputs(
    num_missions: int = 200,
    seed: int | None = 42,
) -> List[MissionParams]:
    """
    Generate `num_missions` randomized parameter sets.

    The ranges are tuned so that roughly 30-40 % of missions end in failure:
    - Periapsis below Earth surface  → crash
    - Hyperbolic escape (ecc >= 1)    → lost in space
    - Very high orbits that never reach the Moon usefully

    Parameters
    ----------
    num_missions : int
        How many parameter sets to create.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    list[MissionParams]
        Each entry is a full set of inputs for one simulation.
    """
    rng = np.random.default_rng(seed)

    missions: List[MissionParams] = []

    for i in range(num_missions):
        # --- Orbital geometry -------------------------------------------
        # SMA: mix of LEO (6500-8000), MEO (8000-20000),
        #       high-orbit / trans-lunar (20000-420000)
        orbit_class = rng.choice(["LEO", "MEO", "HIGH"], p=[0.35, 0.30, 0.35])
        if orbit_class == "LEO":
            sma = rng.uniform(6400, 8000)        # some below surface → crash
        elif orbit_class == "MEO":
            sma = rng.uniform(8000, 20000)
        else:
            sma = rng.uniform(20000, 420000)

        # Eccentricity: mostly low, but sometimes dangerously high
        ecc = float(np.clip(rng.beta(1.5, 5.0) * 1.05, 0.0, 0.999))

        # Angular elements: uniform
        inc  = rng.uniform(0, 180)
        raan = rng.uniform(0, 360)
        aop  = rng.uniform(0, 360)
        ta   = rng.uniform(0, 360)

        # --- Spacecraft --------------------------------------------------
        dry_mass  = rng.uniform(500, 2000)
        fuel_mass = rng.uniform(50, 500)

        # --- Propagation -------------------------------------------------
        # 1–10 days (longer for high orbits)
        if orbit_class == "HIGH":
            prop_days = rng.uniform(3, 10)
        else:
            prop_days = rng.uniform(1, 5)

        missions.append(MissionParams(
            mission_id=i,
            sma=round(sma, 4),
            ecc=round(ecc, 6),
            inc=round(inc, 4),
            raan=round(raan, 4),
            aop=round(aop, 4),
            ta=round(ta, 4),
            dry_mass=round(dry_mass, 2),
            fuel_mass=round(fuel_mass, 2),
            prop_days=round(prop_days, 4),
        ))

    return missions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = generate_inputs(num_missions=5)
    for p in params:
        print(p)
