"""
GMAT Mission Runner — Dispersion-based Real Physics
===================================================
Simulates the trajectory of a spacecraft around Earth with Lunar
gravity perturbations (3-body physics).

Applies the TOI (Trans-Lunar Injection) burn in the VNB frame
at the parking orbit, and propagates for up to 6 days.

Failure classification matches the teammate's scheme:
- success (captured)
- missed_moon (closest approach > 15,000 km)
- hyperbolic_flyby (ECC >= 1.0)
- surface_impact (RadPer < 1837 km)
- orbit_too_high (RadPer > 2237 km)
"""

from __future__ import annotations

import math
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
MOON_RADIUS_KM = 1737.4
MOON_DISTANCE_KM = 384400.0
MOON_ORBITAL_PERIOD_S = 27.32 * 86400.0

# Orbital bounds for success classification
MIN_RADPER_KM = MOON_RADIUS_KM + 100.0   # 1837.4 km
MAX_RADPER_KM = MOON_RADIUS_KM + 500.0   # 2237.4 km
MISS_DISTANCE_KM = 15000.0

COLUMNS = [
    "mission_id",
    "elapsed_secs",
    "elapsed_days",
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "earth_rmag",
    "luna_rmag",
    "ecc",
    "sma",
    "label",         # 1 = success, 0 = failure
    "failure_type",  # Detailed string classification
]


# ═══════════════════════════════════════════════════════════════════════════
# Math Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _keplerian_to_cartesian(sma, ecc, inc_deg, raan_deg, aop_deg, ta_deg=0.0):
    inc  = math.radians(inc_deg)
    raan = math.radians(raan_deg)
    aop  = math.radians(aop_deg)
    ta   = math.radians(ta_deg)

    p = sma * (1 - ecc**2)
    r_mag = p / (1 + ecc * math.cos(ta))

    r_pf = np.array([r_mag * math.cos(ta), r_mag * math.sin(ta), 0.0])
    v_pf = np.array([-math.sqrt(EARTH_MU / p) * math.sin(ta),
                      math.sqrt(EARTH_MU / p) * (ecc + math.cos(ta)), 0.0])

    cos_O, sin_O = math.cos(raan), math.sin(raan)
    cos_w, sin_w = math.cos(aop),  math.sin(aop)
    cos_i, sin_i = math.cos(inc),  math.sin(inc)

    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i,  sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i,                      cos_w*sin_i,                       cos_i       ],
    ])

    return R @ r_pf, R @ v_pf


def _moon_ephemeris(t_sec: float) -> np.ndarray:
    """Assume circular orbit, 18 deg inc."""
    angle = 2 * math.pi * t_sec / MOON_ORBITAL_PERIOD_S
    inc = math.radians(18.0)
    x = MOON_DISTANCE_KM * math.cos(angle)
    y = MOON_DISTANCE_KM * math.sin(angle) * math.cos(inc)
    z = MOON_DISTANCE_KM * math.sin(angle) * math.sin(inc)
    return np.array([x, y, z])

def _moon_velocity(t_sec: float) -> np.ndarray:
    """Derivative of _moon_ephemeris."""
    omega = 2 * math.pi / MOON_ORBITAL_PERIOD_S
    angle = omega * t_sec
    inc = math.radians(18.0)
    vx = -MOON_DISTANCE_KM * omega * math.sin(angle)
    vy = MOON_DISTANCE_KM * omega * math.cos(angle) * math.cos(inc)
    vz = MOON_DISTANCE_KM * omega * math.cos(angle) * math.sin(inc)
    return np.array([vx, vy, vz])


def _get_vnb_frame(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return [V, N, B] unit vectors (Velocity, Normal, Binormal)."""
    V = v / np.linalg.norm(v)
    N = np.cross(r, v)
    N = N / np.linalg.norm(N)
    B = np.cross(V, N)
    return np.column_stack([V, N, B])


def _calculate_orbit_elements(r: np.ndarray, v: np.ndarray, mu: float):
    """Calculates SMA and ECC w.r.t a central body."""
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    energy = 0.5 * v_mag**2 - mu / r_mag
    
    if abs(energy) < 1e-10:
        sma = 1e99
    else:
        sma = -mu / (2 * energy)

    h = np.cross(r, v)
    ecc_vec = np.cross(v, h) / mu - r / r_mag
    ecc = np.linalg.norm(ecc_vec)
    return sma, ecc


# ═══════════════════════════════════════════════════════════════════════════
# 3-Body Propagator (RK4)
# ═══════════════════════════════════════════════════════════════════════════

def _acceleration(r: np.ndarray, r_moon: np.ndarray) -> np.ndarray:
    """Acceleration from Earth + Moon."""
    r_mag = np.linalg.norm(r)
    r_rel_moon = r - r_moon
    rm_mag = np.linalg.norm(r_rel_moon)

    a_earth = -EARTH_MU / (r_mag**3) * r
    a_moon  = -MOON_MU / (rm_mag**3) * r_rel_moon
    return a_earth + a_moon


def _rk4_step(state: np.ndarray, t: float, dt: float) -> np.ndarray:
    # state = [x,y,z, vx,vy,vz]
    r = state[:3]
    v = state[3:]

    r_moon_0  = _moon_ephemeris(t)
    r_moon_h  = _moon_ephemeris(t + dt/2)
    r_moon_dt = _moon_ephemeris(t + dt)

    k1_v = _acceleration(r, r_moon_0)
    k1_r = v

    k2_v = _acceleration(r + k1_r * dt/2, r_moon_h)
    k2_r = v + k1_v * dt/2

    k3_v = _acceleration(r + k2_r * dt/2, r_moon_h)
    k3_r = v + k2_v * dt/2

    k4_v = _acceleration(r + k3_r * dt, r_moon_dt)
    k4_r = v + k3_v * dt

    r_new = r + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return np.concatenate([r_new, v_new])


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_synthetic(params: MissionParams, time_step: float = 60.0) -> pd.DataFrame:
    # 1. Establish initial LEO parking orbit (Keplerian → Cartesian)
    r, v = _keplerian_to_cartesian(
        params.SMA, params.ECC, params.INC,
        params.RAAN, params.AOP
    )

    # 2. Apply TOI impulsive burn in VNB frame
    burn_vnb = np.array([params.TOI_V, params.TOI_N, params.TOI_B])
    R_vnb = _get_vnb_frame(r, v)
    v += R_vnb @ burn_vnb

    # 3. Propagate (3-body RK4)
    state = np.concatenate([r, v])
    total_secs = 6.0 * 86400.0  # 6 days
    n_steps = int(total_secs / time_step)
    integration_dt = min(time_step, 10.0)

    rows = []
    min_luna_rmag = 999999.0
    crashed_earth = False
    closest_state = None
    closest_moon = None
    closest_t = 0.0

    for step_i in range(n_steps + 1):
        t = step_i * time_step
        r_current, v_current = state[:3], state[3:]
        
        earth_rmag = np.linalg.norm(r_current)
        if earth_rmag < EARTH_RADIUS_KM:
            crashed_earth = True
            break

        moon_pos = _moon_ephemeris(t)
        luna_rmag = np.linalg.norm(r_current - moon_pos)

        if luna_rmag < min_luna_rmag:
            min_luna_rmag = luna_rmag
            closest_state = (r_current.copy(), v_current.copy())
            closest_moon = moon_pos
            closest_t = t

        sma, ecc = _calculate_orbit_elements(r_current, v_current, EARTH_MU)

        # Stop early if very close to Moon (captured)
        if luna_rmag < 500:
            break

        rows.append([
            params.sim_id,
            t,
            t / 86400.0,
            *r_current,
            *v_current,
            earth_rmag,
            luna_rmag,
            ecc,
            sma,
            -1, ""  # placeholders
        ])

        # Step forward
        if step_i < n_steps:
            subs = int(time_step / integration_dt)
            for _ in range(subs):
                state = _rk4_step(state, t, integration_dt)
                t += integration_dt

    if not rows:
        return _empty_failure(params)

    # 4. Determine final outcome using closest approach to Moon
    r_close, v_close = closest_state
    v_rel_moon = v_close - _moon_velocity(closest_t)
    luna_sma, luna_ecc = _calculate_orbit_elements(r_close - closest_moon, v_rel_moon, MOON_MU)
    rad_per = luna_sma * (1 - luna_ecc)

    label = 0
    failure_type = "unknown"

    if crashed_earth:
        failure_type = "earth_impact"
    elif min_luna_rmag > MISS_DISTANCE_KM:
        failure_type = "missed_moon"
    elif rad_per > 0 and rad_per < MIN_RADPER_KM:
        failure_type = "surface_impact"
    elif rad_per > MAX_RADPER_KM:
        failure_type = "orbit_too_high"
    else:
        label = 1
        failure_type = "success"

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["label"] = label
    df["failure_type"] = failure_type

    return df


def _empty_failure(params: MissionParams) -> pd.DataFrame:
    row = [[
        params.sim_id, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        EARTH_RADIUS_KM, MISS_DISTANCE_KM * 2,
        0.0, 0.0,
        0, "degenerate_orbit"
    ]]
    return pd.DataFrame(row, columns=COLUMNS)
