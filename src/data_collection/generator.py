"""
Monte Carlo Dispersion Generator — Multi-Planet Mesh Topology
==============================================================
Generates mission parameters by applying small random dispersions
around a FIXED nominal transfer solution between any two solar
system bodies.

Nominal values are computed analytically via Hohmann transfer
approximation (for interplanetary) or from the converged GMAT
solution (for Earth-Moon).

Context features (mu_ratio, soi_ratio, dist_ratio) are attached
to every mission for the ML model to learn which "type" of transfer
it is seeing — enabling zero-shot generalization to unseen pairs.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════════════════
# PLANET REGISTRY — Physical constants for all major solar system bodies
# ═══════════════════════════════════════════════════════════════════════════

SUN_MU = 1.32712440018e11  # km³/s² — Sun gravitational parameter
AU_KM = 1.496e8            # km per AU

PLANET_REGISTRY: Dict[str, Dict[str, float]] = {
    "earth": {
        "mu":          398600.4418,          # km³/s²
        "radius":      6371.0,               # km
        "soi":         924000.0,             # km (Earth SOI relative to Sun)
        "orbit_radius": 1.0 * AU_KM,         # km — mean distance from Sun
        "orbit_period_days": 365.25,
    },
    "moon": {
        "mu":          4902.8,               # km³/s²
        "radius":      1737.4,               # km
        "soi":         66183.0,              # km (Moon SOI relative to Earth)
        "orbit_radius": 384400.0,            # km — distance from Earth
        "orbit_period_days": 27.32,
    },
    "mars": {
        "mu":          42828.4,              # km³/s²
        "radius":      3389.5,               # km
        "soi":         577000.0,             # km
        "orbit_radius": 1.524 * AU_KM,       # km
        "orbit_period_days": 687.0,
    },
    "jupiter": {
        "mu":          126686534.0,          # km³/s²
        "radius":      71492.0,              # km
        "soi":         48200000.0,           # km
        "orbit_radius": 5.203 * AU_KM,       # km
        "orbit_period_days": 4333.0,
    },
    "saturn": {
        "mu":          37931187.0,           # km³/s²
        "radius":      60268.0,              # km
        "soi":         54800000.0,           # km
        "orbit_radius": 9.537 * AU_KM,       # km
        "orbit_period_days": 10759.0,
    },
    "uranus": {
        "mu":          5793966.0,            # km³/s²
        "radius":      25559.0,              # km
        "soi":         51800000.0,           # km
        "orbit_radius": 19.19 * AU_KM,       # km
        "orbit_period_days": 30687.0,
    },
    "neptune": {
        "mu":          6836529.0,            # km³/s²
        "radius":      24764.0,              # km
        "soi":         86800000.0,           # km
        "orbit_radius": 30.07 * AU_KM,       # km
        "orbit_period_days": 60190.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# NOMINAL SOLUTIONS — Computed per mission type
# ═══════════════════════════════════════════════════════════════════════════

# Hard-coded converged GMAT solution for Earth-Moon (our best nominal)
EARTH_MOON_NOMINAL = {
    "TOI_V": 3.240,            # km/s — adjusted for RK4 parity
    "TOI_N": 0.0,              # km/s — normal component
    "TOI_B": 0.0,              # km/s — binormal component
    "RAAN":  221.33,           # deg
    "AOP":   358.309113604,    # deg
    "INC":   28.7,             # deg
    "SMA":   6563.0,           # km — LEO parking orbit
    "ECC":   0.001,
}

# Default dispersion bounds for Earth-Moon
EARTH_MOON_DISPERSION = {
    "dv_V":  (-0.006,  0.006),   # km/s
    "dv_N":  (-0.003,  0.003),   # km/s
    "dv_B":  (-0.003,  0.003),   # km/s
    "RAAN":  (-0.6,    0.6),     # deg
    "AOP":   (-0.6,    0.6),     # deg
    "INC":   (-0.1,    0.1),     # deg
}


def _hohmann_nominal(source: str, target: str) -> Dict[str, float]:
    """
    Compute approximate Hohmann transfer nominal parameters for any pair.

    For Earth-Moon: uses the hard-coded GMAT converged solution.
    For interplanetary: computes delta-V analytically from circular orbits.

    Returns a dict with the same keys as EARTH_MOON_NOMINAL.
    """
    if source == "earth" and target == "moon":
        return EARTH_MOON_NOMINAL.copy()

    src = PLANET_REGISTRY[source]
    tgt = PLANET_REGISTRY[target]

    # Determine the central body for the transfer
    # For interplanetary: central body = Sun
    # Source orbit radius and target orbit radius around the Sun
    r1 = src["orbit_radius"]
    r2 = tgt["orbit_radius"]

    # If target is closer to the sun, swap so r1 < r2 always
    # (the math is the same, we just flip the sign convention)
    inward = r2 < r1
    if inward:
        r1, r2 = r2, r1

    # Hohmann transfer ellipse semi-major axis
    a_transfer = (r1 + r2) / 2.0

    # delta-V at departure (from circular orbit at r1 to transfer ellipse)
    v_circular_1 = math.sqrt(SUN_MU / r1)
    v_transfer_1 = math.sqrt(SUN_MU * (2.0 / r1 - 1.0 / a_transfer))
    dv_departure = abs(v_transfer_1 - v_circular_1)

    # Convert heliocentric delta-V to hyperbolic excess velocity
    # then to the TOI burn from a low parking orbit around the source body
    v_park = math.sqrt(src["mu"] / (src["radius"] + 200.0))  # 200 km parking orbit
    v_inf = dv_departure  # hyperbolic excess speed
    v_burnout = math.sqrt(v_inf**2 + 2.0 * src["mu"] / (src["radius"] + 200.0))
    toi_v = v_burnout - v_park

    sma_park = src["radius"] + 200.0  # parking orbit radius = SMA for circular

    return {
        "TOI_V": toi_v,
        "TOI_N": 0.0,
        "TOI_B": 0.0,
        "RAAN":  0.0,       # arbitrary for synthetic propagator
        "AOP":   0.0,
        "INC":   0.0,       # ecliptic plane
        "SMA":   sma_park,
        "ECC":   0.001,
    }


def _hohmann_dispersions(source: str, target: str) -> Dict[str, tuple]:
    """
    Scale dispersion bounds based on transfer energy.
    Larger transfers get proportionally larger dispersions.
    """
    if source == "earth" and target == "moon":
        return EARTH_MOON_DISPERSION.copy()

    nominal = _hohmann_nominal(source, target)
    toi_v = nominal["TOI_V"]

    # Scale dispersions proportionally to burn magnitude
    # Earth-Moon TOI is ~3.23 km/s with ±0.006 km/s → ~0.19% relative
    scale = toi_v / 3.2337

    return {
        "dv_V":  (-0.006 * scale, 0.006 * scale),
        "dv_N":  (-0.003 * scale, 0.003 * scale),
        "dv_B":  (-0.003 * scale, 0.003 * scale),
        "RAAN":  (-0.6,   0.6),
        "AOP":   (-0.6,   0.6),
        "INC":   (-0.1,   0.1),
    }


def _compute_context_features(source: str, target: str) -> Dict[str, float]:
    """
    Compute the 3 dimensionless context features that fingerprint
    a mission type for the ML model.

    These are constant per mission (not per timestep), but will be
    broadcast to every row in the trajectory.
    """
    src = PLANET_REGISTRY[source]
    tgt = PLANET_REGISTRY[target]

    # Transfer distance (approximate)
    if source == "earth" and target == "moon":
        transfer_dist = tgt["orbit_radius"]  # ~384,400 km
    else:
        transfer_dist = abs(tgt["orbit_radius"] - src["orbit_radius"])

    mu_ratio = tgt["mu"] / SUN_MU
    soi_ratio = tgt["soi"] / max(transfer_dist, 1.0)
    dist_ratio = transfer_dist / AU_KM

    return {
        "mu_ratio":   mu_ratio,
        "soi_ratio":  soi_ratio,
        "dist_ratio": dist_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Mission Parameters Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MissionParams:
    """Parameters for a single Monte Carlo simulation run."""
    sim_id: int

    # Source and target body names
    source: str
    target: str

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

    # Context features (constant per mission, broadcast to every row)
    mu_ratio:   float
    soi_ratio:  float
    dist_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Input Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_inputs(
    num_missions: int = 3000,
    seed: int | None = 42,
    source: str = "earth",
    target: str = "moon",
    success_ratio: float = 0.0,
) -> List[MissionParams]:
    """
    Generate `num_missions` dispersed parameter sets around the nominal solution.

    Parameters
    ----------
    num_missions : int
        Number of Monte Carlo runs to generate.
    seed : int or None
        RNG seed for reproducibility.
    source : str
        Source body name (e.g., "earth", "jupiter").
    target : str
        Target body name (e.g., "moon", "mars").
    success_ratio : float
        If > 0, this fraction of missions will use tighter dispersions
        (±25% of normal) to bias toward success, improving class balance.
        Set to 0.0 for pure Monte Carlo (no bias).

    Returns
    -------
    list[MissionParams]
    """
    if source not in PLANET_REGISTRY:
        raise ValueError(f"Unknown source body: {source}. Available: {list(PLANET_REGISTRY.keys())}")
    if target not in PLANET_REGISTRY:
        raise ValueError(f"Unknown target body: {target}. Available: {list(PLANET_REGISTRY.keys())}")

    rng = np.random.default_rng(seed)
    nominal = _hohmann_nominal(source, target)
    dispersions = _hohmann_dispersions(source, target)
    context = _compute_context_features(source, target)

    # How many missions get biased toward success
    n_biased = int(num_missions * success_ratio) if success_ratio > 0 else 0

    missions: List[MissionParams] = []

    for i in range(num_missions):
        # If this mission should be biased toward success,
        # use a significantly tighter dispersion range
        if i < n_biased:
            # For Earth-Moon, the corridor is tiny. 10% of nominal dispersion
            # is more likely to hit it than 25%.
            scale = 0.1 if (source == "earth" and target == "moon") else 0.25
        else:
            scale = 1.0

        d_V    = rng.uniform(dispersions["dv_V"][0] * scale, dispersions["dv_V"][1] * scale)
        d_N    = rng.uniform(dispersions["dv_N"][0] * scale, dispersions["dv_N"][1] * scale)
        d_B    = rng.uniform(dispersions["dv_B"][0] * scale, dispersions["dv_B"][1] * scale)
        d_RAAN = rng.uniform(dispersions["RAAN"][0] * scale, dispersions["RAAN"][1] * scale)
        d_AOP  = rng.uniform(dispersions["AOP"][0]  * scale, dispersions["AOP"][1]  * scale)
        d_INC  = rng.uniform(dispersions["INC"][0]  * scale, dispersions["INC"][1]  * scale)

        missions.append(MissionParams(
            sim_id=i,
            source=source,
            target=target,
            TOI_V=nominal["TOI_V"] + d_V,
            TOI_N=nominal["TOI_N"] + d_N,
            TOI_B=nominal["TOI_B"] + d_B,
            RAAN=nominal["RAAN"]   + d_RAAN,
            AOP=nominal["AOP"]     + d_AOP,
            INC=nominal["INC"]     + d_INC,
            SMA=nominal["SMA"],
            ECC=nominal["ECC"],
            dv_V_offset=round(d_V, 8),
            dv_N_offset=round(d_N, 8),
            dv_B_offset=round(d_B, 8),
            RAAN_offset=round(d_RAAN, 8),
            AOP_offset=round(d_AOP, 8),
            INC_offset=round(d_INC, 8),
            mu_ratio=context["mu_ratio"],
            soi_ratio=context["soi_ratio"],
            dist_ratio=context["dist_ratio"],
        ))

    return missions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dispersed mission parameters")
    parser.add_argument("--source", type=str, default="earth")
    parser.add_argument("--target", type=str, default="moon")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--success-ratio", type=float, default=0.0)
    args = parser.parse_args()

    params = generate_inputs(
        num_missions=args.num,
        source=args.source,
        target=args.target,
        success_ratio=args.success_ratio,
    )
    for p in params:
        print(f"sim {p.sim_id}: TOI_V={p.TOI_V:.6f} (Δ{p.dv_V_offset:+.6f})  "
              f"RAAN={p.RAAN:.4f} (Δ{p.RAAN_offset:+.4f})  "
              f"ctx=[μ={p.mu_ratio:.2e}, SOI={p.soi_ratio:.4f}, d={p.dist_ratio:.4f}]")
