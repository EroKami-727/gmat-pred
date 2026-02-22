"""
Monte Carlo Dispersion Generator
==================================
Generates mission parameters by applying small random dispersions
around a FIXED nominal lunar transfer solution.

Nominal values come from a converged GMAT targeter run.
Dispersions represent realistic launch vehicle errors.

This is how NASA Monte Carlo dispersion analysis works:
  - Run targeter ONCE to find the nominal solution
  - Apply random perturbations to simulate real-world uncertainty
  - Some dispersions miss Moon naturally = physically real failures
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════
# NOMINAL SOLUTION — from converged GMAT targeter
# ═══════════════════════════════════════════════════════════════════════════

NOMINAL = {
    "TOI_V": 3.2337,           # km/s — prograde burn magnitude
    "TOI_N": 0.0,              # km/s — normal component
    "TOI_B": 0.0,              # km/s — binormal component
    "RAAN":  221.33,           # deg  — Right Ascension of Ascending Node
    "AOP":   358.309113604,    # deg  — Argument of Periapsis
    "INC":   28.7,             # deg  — Inclination
    "SMA":   6563.0,           # km   — Semi-major axis (LEO parking orbit)
    "ECC":   0.001,            # —    — Near-circular parking orbit
}

# ═══════════════════════════════════════════════════════════════════════════
# DISPERSION RANGES — (min, max) uniform bounds
# These simulate realistic launch vehicle errors and orbital uncertainties
# ═══════════════════════════════════════════════════════════════════════════

DISPERSION = {
    "dv_V":  (-0.006,  0.006),   # km/s — TOI burn magnitude error
    "dv_N":  (-0.003,  0.003),   # km/s — out-of-plane burn error
    "dv_B":  (-0.003,  0.003),   # km/s — binormal burn error
    "RAAN":  (-0.6,    0.6),     # deg  — orbital plane error
    "AOP":   (-0.6,    0.6),     # deg  — periapsis location error
    "INC":   (-0.1,    0.1),     # deg  — inclination error
}


@dataclass
class MissionParams:
    """Parameters for a single Monte Carlo simulation run."""
    sim_id: int

    # Absolute values (nominal + dispersion)
    TOI_V: float    # km/s
    TOI_N: float    # km/s
    TOI_B: float    # km/s
    RAAN:  float    # deg
    AOP:   float    # deg
    INC:   float    # deg
    SMA:   float    # km
    ECC:   float    # —

    # Offsets from nominal (for analysis)
    dv_V_offset:  float
    dv_N_offset:  float
    dv_B_offset:  float
    RAAN_offset:  float
    AOP_offset:   float
    INC_offset:   float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_inputs(
    num_missions: int = 3000,
    seed: int | None = 42,
) -> List[MissionParams]:
    """
    Generate `num_missions` dispersed parameter sets around the nominal solution.

    Each parameter = nominal_value + uniform_random(dispersion_min, dispersion_max)

    Parameters
    ----------
    num_missions : int
        Number of Monte Carlo runs to generate.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    list[MissionParams]
    """
    rng = np.random.default_rng(seed)
    missions: List[MissionParams] = []

    for i in range(num_missions):
        # Sample dispersions
        d_V    = rng.uniform(*DISPERSION["dv_V"])
        d_N    = rng.uniform(*DISPERSION["dv_N"])
        d_B    = rng.uniform(*DISPERSION["dv_B"])
        d_RAAN = rng.uniform(*DISPERSION["RAAN"])
        d_AOP  = rng.uniform(*DISPERSION["AOP"])
        d_INC  = rng.uniform(*DISPERSION["INC"])

        missions.append(MissionParams(
            sim_id=i,
            TOI_V=NOMINAL["TOI_V"] + d_V,
            TOI_N=NOMINAL["TOI_N"] + d_N,
            TOI_B=NOMINAL["TOI_B"] + d_B,
            RAAN=NOMINAL["RAAN"]   + d_RAAN,
            AOP=NOMINAL["AOP"]     + d_AOP,
            INC=NOMINAL["INC"]     + d_INC,
            SMA=NOMINAL["SMA"],
            ECC=NOMINAL["ECC"],
            dv_V_offset=round(d_V, 8),
            dv_N_offset=round(d_N, 8),
            dv_B_offset=round(d_B, 8),
            RAAN_offset=round(d_RAAN, 8),
            AOP_offset=round(d_AOP, 8),
            INC_offset=round(d_INC, 8),
        ))

    return missions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = generate_inputs(num_missions=5)
    for p in params:
        print(f"sim {p.sim_id}: TOI_V={p.TOI_V:.6f} (Δ{p.dv_V_offset:+.6f})  "
              f"RAAN={p.RAAN:.4f} (Δ{p.RAAN_offset:+.4f})  "
              f"AOP={p.AOP:.4f} (Δ{p.AOP_offset:+.4f})")
