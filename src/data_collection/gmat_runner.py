"""
GMAT Mission Runner
====================
Two execution modes:

1. **Synthetic mode** (default) — Uses Keplerian two-body propagation with
   perturbations to generate time-series spacecraft data.  Works without
   GMAT installed.

2. **GMAT mode** — Generates a .script file from the base_mission template,
   invokes the GMAT binary, and parses the ReportFile output.  Requires
   GMAT to be installed.

Both modes produce identical DataFrame schemas so the rest of the pipeline
is agnostic to which was used.
"""

from __future__ import annotations

import math
import subprocess
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .generator import MissionParams

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
EARTH_MU = 398600.4418            # km³/s²
MOON_MU = 4902.8                  # km³/s²
MOON_DISTANCE_KM = 384400.0
MOON_ORBITAL_PERIOD_S = 27.32 * 86400.0   # ~27.32 days in seconds

# Output columns — matches the schema from docs/dataset_and_ml_guide.md
COLUMNS = [
    "mission_id",
    "elapsed_secs",
    "pos_x", "pos_y", "pos_z",         # Spacecraft position (km, EarthMJ2000Eq)
    "vel_x", "vel_y", "vel_z",         # Spacecraft velocity (km/s)
    "moon_x", "moon_y", "moon_z",      # Moon position      (km, EarthMJ2000Eq)
    "fuel_remaining",                   # kg
    "outcome",                          # 1 = success, 0 = failure
]


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic (two-body) runner
# ═══════════════════════════════════════════════════════════════════════════

def _keplerian_to_cartesian(sma, ecc, inc_deg, raan_deg, aop_deg, ta_deg, mu=EARTH_MU):
    """Convert Keplerian elements to Cartesian state [x,y,z,vx,vy,vz]."""
    inc  = math.radians(inc_deg)
    raan = math.radians(raan_deg)
    aop  = math.radians(aop_deg)
    ta   = math.radians(ta_deg)

    # Semi-latus rectum
    p = sma * (1 - ecc**2)
    r_mag = p / (1 + ecc * math.cos(ta))

    # Position & velocity in perifocal frame
    r_pf = np.array([r_mag * math.cos(ta),
                      r_mag * math.sin(ta),
                      0.0])
    v_pf = np.array([-math.sqrt(mu / p) * math.sin(ta),
                      math.sqrt(mu / p) * (ecc + math.cos(ta)),
                      0.0])

    # Rotation matrix: perifocal → ECI
    cos_O, sin_O = math.cos(raan), math.sin(raan)
    cos_w, sin_w = math.cos(aop),  math.sin(aop)
    cos_i, sin_i = math.cos(inc),  math.sin(inc)

    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i,  sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i,                      cos_w*sin_i,                       cos_i       ],
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return np.concatenate([r_eci, v_eci])


def _propagate_twobody(state: np.ndarray, dt: float, mu: float = EARTH_MU) -> np.ndarray:
    """
    Propagate a Cartesian state by dt seconds using velocity-Verlet
    (symplectic integrator — conserves energy better than RK4 for orbits).
    """
    r = state[:3].copy()
    v = state[3:].copy()

    r_mag = np.linalg.norm(r)
    a = -mu / r_mag**3 * r                   # gravitational acceleration

    # Half-step velocity
    v_half = v + 0.5 * dt * a
    # Full-step position
    r_new = r + dt * v_half
    # New acceleration
    r_new_mag = np.linalg.norm(r_new)
    a_new = -mu / r_new_mag**3 * r_new
    # Full-step velocity
    v_new = v_half + 0.5 * dt * a_new

    return np.concatenate([r_new, v_new])


def _moon_position(elapsed_secs: float) -> np.ndarray:
    """
    Approximate Moon position in EarthMJ2000Eq as a circular orbit.
    Good enough for synthetic training data — real GMAT will be more precise.
    """
    angle = 2 * math.pi * elapsed_secs / MOON_ORBITAL_PERIOD_S
    # Moon orbit inclined ~5.14° to ecliptic, ~23.4° ecliptic to equator
    # Simplify to ~18° effective inclination in J2000
    inc = math.radians(18.0)
    x = MOON_DISTANCE_KM * math.cos(angle)
    y = MOON_DISTANCE_KM * math.sin(angle) * math.cos(inc)
    z = MOON_DISTANCE_KM * math.sin(angle) * math.sin(inc)
    return np.array([x, y, z])


def _determine_outcome(trajectory: list[np.ndarray], sma: float, ecc: float) -> int:
    """
    Heuristic outcome classification:
        1 (success) — spacecraft stays in a stable orbit or reaches lunar vicinity
        0 (failure) — crashes (periapsis < Earth radius) or escapes
    """
    # Check periapsis
    periapsis = sma * (1 - ecc)
    if periapsis < EARTH_RADIUS_KM:
        return 0   # sub-surface orbit → crash

    # Check trajectory for Earth impact
    for state in trajectory:
        r_mag = np.linalg.norm(state[:3])
        if r_mag < EARTH_RADIUS_KM:
            return 0   # crashed

    # Check for escape (hyperbolic)
    if ecc >= 1.0:
        return 0

    # Check orbital energy (negative = bound, positive = escape)
    final_state = trajectory[-1]
    r_mag = np.linalg.norm(final_state[:3])
    v_mag = np.linalg.norm(final_state[3:])
    energy = 0.5 * v_mag**2 - EARTH_MU / r_mag
    if energy > 0:
        return 0   # escaping

    # If the orbit is very low and decaying, mark as failure
    if r_mag < EARTH_RADIUS_KM + 100:  # below 100 km altitude
        return 0

    return 1   # stable orbit — success


def run_synthetic(params: MissionParams, time_step: float = 60.0) -> pd.DataFrame:
    """
    Run a single mission using two-body synthetic propagation.

    Parameters
    ----------
    params : MissionParams
        Mission configuration.
    time_step : float
        Output interval in seconds (default 60s = 1 snapshot/minute).

    Returns
    -------
    pd.DataFrame
        Time-series data with columns matching COLUMNS.
    """
    # Convert Keplerian → Cartesian
    try:
        state = _keplerian_to_cartesian(
            params.sma, params.ecc, params.inc,
            params.raan, params.aop, params.ta,
        )
    except (ValueError, ZeroDivisionError):
        # Degenerate orbit — immediate failure
        return _empty_failure(params)

    total_secs = params.prop_days * 86400.0
    n_steps = int(total_secs / time_step)

    # Integration step (sub-step for accuracy)
    integration_dt = min(time_step, 30.0)  # max 30s integration step

    rows = []
    trajectory = []
    fuel = params.fuel_mass
    crashed = False

    for step_i in range(n_steps + 1):
        t = step_i * time_step
        moon = _moon_position(t)

        r_mag = np.linalg.norm(state[:3])

        # Check for crash
        if r_mag < EARTH_RADIUS_KM:
            crashed = True
            # Record the crash point and stop
            rows.append([
                params.mission_id, t,
                *state[:3], *state[3:],
                *moon,
                fuel, 0,  # outcome = 0
            ])
            break

        # Tiny fuel consumption model (drag-like depletion for realism)
        if fuel > 0:
            fuel_burn = 0.001 * time_step / 60.0  # ~0.001 kg/min baseline
            if r_mag < EARTH_RADIUS_KM + 400:     # more drag in LEO
                fuel_burn *= 3.0
            fuel = max(0.0, fuel - fuel_burn)

        rows.append([
            params.mission_id, t,
            *state[:3], *state[3:],
            *moon,
            round(fuel, 4), -1,  # outcome placeholder
        ])
        trajectory.append(state.copy())

        # Propagate to next time step using sub-steps
        if step_i < n_steps:
            subs = int(time_step / integration_dt)
            for _ in range(subs):
                state = _propagate_twobody(state, integration_dt)
                # Check crash during sub-step
                if np.linalg.norm(state[:3]) < EARTH_RADIUS_KM:
                    state[:3] = state[:3] / np.linalg.norm(state[:3]) * EARTH_RADIUS_KM
                    crashed = True
                    break
            if crashed:
                t_next = (step_i + 1) * time_step
                moon = _moon_position(t_next)
                rows.append([
                    params.mission_id, t_next,
                    *state[:3], *state[3:],
                    *moon,
                    fuel, 0,
                ])
                break

    if not rows:
        return _empty_failure(params)

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Set consistent outcome for ALL rows in this mission
    if crashed:
        df["outcome"] = 0
    else:
        outcome = _determine_outcome(trajectory, params.sma, params.ecc)
        df["outcome"] = outcome

    return df


def _empty_failure(params: MissionParams) -> pd.DataFrame:
    """Return a single-row failure DataFrame for degenerate orbits."""
    moon = _moon_position(0)
    row = [[
        params.mission_id, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        *moon,
        params.fuel_mass, 0,
    ]]
    return pd.DataFrame(row, columns=COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════
# GMAT binary runner (for when GMAT is installed)
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "gmat_scripts" / "base_mission.script"


def _generate_gmat_script(params: MissionParams, output_dir: Path) -> Path:
    """Generate a customised .script file from the template."""
    template = SCRIPT_TEMPLATE_PATH.read_text()

    replacements = {
        "GMAT Sat.SMA = 7000;":                f"GMAT Sat.SMA = {params.sma};",
        "GMAT Sat.ECC = 0.001;":               f"GMAT Sat.ECC = {params.ecc};",
        "GMAT Sat.INC = 28.5;":                f"GMAT Sat.INC = {params.inc};",
        "GMAT Sat.RAAN = 0;":                  f"GMAT Sat.RAAN = {params.raan};",
        "GMAT Sat.AOP = 0;":                   f"GMAT Sat.AOP = {params.aop};",
        "GMAT Sat.TA = 0;":                    f"GMAT Sat.TA = {params.ta};",
        "GMAT Sat.DryMass = 850;":             f"GMAT Sat.DryMass = {params.dry_mass};",
        "GMAT FuelTank.FuelMass = 300;":       f"GMAT FuelTank.FuelMass = {params.fuel_mass};",
        "Sat.ElapsedDays = 5":                 f"Sat.ElapsedDays = {params.prop_days}",
    }

    script = template
    for old, new in replacements.items():
        script = script.replace(old, new)

    report_path = output_dir / f"mission_{params.mission_id}_report.txt"
    script = script.replace(
        "GMAT MissionReport.Filename = 'output/mission_report.txt';",
        f"GMAT MissionReport.Filename = '{report_path}';",
    )

    script_path = output_dir / f"mission_{params.mission_id}.script"
    script_path.write_text(script)
    return script_path


def run_gmat(
    params: MissionParams,
    gmat_bin: str = "GMAT",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run a mission through the real GMAT binary.

    Parameters
    ----------
    params : MissionParams
    gmat_bin : str
        Path or command for the GMAT executable.
    output_dir : Path
        Directory for script and report files.

    Returns
    -------
    pd.DataFrame
    """
    if output_dir is None:
        output_dir = Path("data/gmat_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = _generate_gmat_script(params, output_dir)
    report_path = output_dir / f"mission_{params.mission_id}_report.txt"

    # Run GMAT
    result = subprocess.run(
        [gmat_bin, "--run", "--exit", str(script_path)],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"  ⚠ GMAT failed for mission {params.mission_id}: {result.stderr[:200]}")
        return _empty_failure(params)

    # Parse report file
    if not report_path.exists():
        return _empty_failure(params)

    df = pd.read_csv(
        report_path,
        sep=r"\s+",
        comment="%",
        header=0,
        names=[
            "elapsed_secs",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "moon_x", "moon_y", "moon_z",
            "fuel_remaining",
        ],
    )
    df.insert(0, "mission_id", params.mission_id)

    # Determine outcome from final state
    final = df.iloc[-1]
    r_mag = math.sqrt(final.pos_x**2 + final.pos_y**2 + final.pos_z**2)
    v_mag = math.sqrt(final.vel_x**2 + final.vel_y**2 + final.vel_z**2)
    energy = 0.5 * v_mag**2 - EARTH_MU / r_mag
    outcome = 0 if (energy > 0 or r_mag < EARTH_RADIUS_KM + 50) else 1
    df["outcome"] = outcome

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .generator import generate_inputs
    params = generate_inputs(num_missions=1)[0]
    print(f"Running synthetic simulation for: {params}")
    df = run_synthetic(params,  time_step=120.0)
    print(f"Result: {len(df)} rows, outcome={df['outcome'].iloc[0]}")
    print(df.head(10))
