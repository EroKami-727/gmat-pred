"""
GMAT Mission Runner — Multi-Planet Physics-Invariant 3-Body Propagator
======================================================================
Simulates the trajectory of a spacecraft around a source body with
target body gravity perturbations (3-body physics, RK4 integration).

Supports any source-target pair from the PLANET_REGISTRY by
parameterizing all physical constants, ephemeris, and classification
thresholds based on the mission parameters.

Phase 2 Enhancement: Computes physics-invariant features at every
timestep for ML generalization:
  - Target-Centric Synodic Frame coordinates (rel_x, rel_y, rel_z)
  - Specific Orbital Energy (spec_energy)
  - Flight Path Angle (fpa_deg)
  - Normalized Target Distance (norm_target_dist) — distance / SOI
  - Radial Velocity (radial_vel) — dr/dt toward target
  - Velocity Magnitude (vel_mag)

Phase 3 Enhancement: Context features per mission:
  - mu_ratio, soi_ratio, dist_ratio (from generator)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .generator import MissionParams, PLANET_REGISTRY, SUN_MU, AU_KM


# ---------------------------------------------------------------------------
# Column Schema
# ---------------------------------------------------------------------------
COLUMNS = [
    "mission_id",
    "elapsed_secs",
    "elapsed_days",
    # Raw Cartesian (kept for debugging / GMAT verification)
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    # Existing derived columns
    "earth_rmag",
    "luna_rmag",
    "ecc",
    "sma",
    # ── Physics-Invariant Features ──
    "rel_x", "rel_y", "rel_z",       # Target-Centric Synodic Frame
    "spec_energy",                     # Specific Orbital Energy (v²/2 - μ/r)
    "fpa_deg",                         # Flight Path Angle (degrees)
    "norm_target_dist",                # luna_rmag / SOI
    "radial_vel",                      # dr/dt toward target (km/s)
    "vel_mag",                         # |v| (km/s)
    # ── Context Features (constant per mission) ──
    "mu_ratio",
    "soi_ratio",
    "dist_ratio",
    # ── Source/Target identifiers ──
    "source_body",
    "target_body",
    # Targets
    "label",         # 1 = success, 0 = failure
    "failure_type",  # Detailed string classification
    "min_target_rmag",  # Regression target: closest approach (km)
]


# ═══════════════════════════════════════════════════════════════════════════
# Mission Configuration (derived from planet registry for any pair)
# ═══════════════════════════════════════════════════════════════════════════

class MissionConfig:
    """Physical constants and thresholds for a specific source-target pair."""

    def __init__(self, source: str, target: str):
        self.source_name = source
        self.target_name = target

        src = PLANET_REGISTRY[source]
        tgt = PLANET_REGISTRY[target]

        self.source_mu = src["mu"]
        self.source_radius = src["radius"]
        self.target_mu = tgt["mu"]
        self.target_radius = tgt["radius"]
        self.target_soi = tgt["soi"]

        # Transfer distance
        if source == "earth" and target == "moon":
            self.target_distance = tgt["orbit_radius"]  # 384,400 km
            self.target_orbital_period_s = tgt["orbit_period_days"] * 86400.0
        else:
            self.target_distance = abs(tgt["orbit_radius"] - src["orbit_radius"])
            self.target_orbital_period_s = tgt["orbit_period_days"] * 86400.0

        # Success corridor: periapsis altitude between 100 km and 500 km
        self.min_radper = self.target_radius + 100.0
        self.max_radper = self.target_radius + 500.0

        # Miss threshold: scaled by SOI
        # Earth-Moon miss was 15,000 km for SOI ~66,183 km (~22.7%)
        # Scale proportionally for other bodies
        self.miss_distance = self.target_soi * 0.227

        # Propagation time: scale by transfer time
        if source == "earth" and target == "moon":
            self.prop_days = 6.0
        else:
            # Hohmann transfer time ≈ π * sqrt(a_transfer³ / μ_sun)
            r1 = src["orbit_radius"]
            r2 = tgt["orbit_radius"]
            a_t = (r1 + r2) / 2.0
            t_hohmann_s = math.pi * math.sqrt(a_t**3 / SUN_MU)
            # Add 20% margin, convert to days
            self.prop_days = min(t_hohmann_s * 1.2 / 86400.0, 3650.0)  # cap at 10 years

        # Capture/impact radius (stop sim if we get this close)
        self.capture_radius = min(500.0, self.target_radius * 0.3)

        # Inclination for simplified ephemeris
        if source == "earth" and target == "moon":
            self.target_inc_deg = 18.0  # Moon's orbital inclination
        else:
            self.target_inc_deg = 0.0  # ecliptic plane approximation


# ═══════════════════════════════════════════════════════════════════════════
# Math Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _keplerian_to_cartesian(sma, ecc, inc_deg, raan_deg, aop_deg, mu, ta_deg=0.0):
    inc  = math.radians(inc_deg)
    raan = math.radians(raan_deg)
    aop  = math.radians(aop_deg)
    ta   = math.radians(ta_deg)

    p = sma * (1 - ecc**2)
    r_mag = p / (1 + ecc * math.cos(ta))

    r_pf = np.array([r_mag * math.cos(ta), r_mag * math.sin(ta), 0.0])
    v_pf = np.array([-math.sqrt(mu / p) * math.sin(ta),
                      math.sqrt(mu / p) * (ecc + math.cos(ta)), 0.0])

    cos_O, sin_O = math.cos(raan), math.sin(raan)
    cos_w, sin_w = math.cos(aop),  math.sin(aop)
    cos_i, sin_i = math.cos(inc),  math.sin(inc)

    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i,  sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i,                      cos_w*sin_i,                       cos_i       ],
    ])

    return R @ r_pf, R @ v_pf


def _target_ephemeris(t_sec: float, cfg: MissionConfig) -> np.ndarray:
    """
    Simplified circular-orbit ephemeris for the target body.

    For Earth-Moon: Moon orbits Earth (as before).
    For interplanetary: target orbits at its mean distance from the source.
    """
    omega = 2 * math.pi / cfg.target_orbital_period_s
    angle = omega * t_sec
    inc = math.radians(cfg.target_inc_deg)

    x = cfg.target_distance * math.cos(angle)
    y = cfg.target_distance * math.sin(angle) * math.cos(inc)
    z = cfg.target_distance * math.sin(angle) * math.sin(inc)
    return np.array([x, y, z])


def _target_velocity(t_sec: float, cfg: MissionConfig) -> np.ndarray:
    """Derivative of _target_ephemeris."""
    omega = 2 * math.pi / cfg.target_orbital_period_s
    angle = omega * t_sec
    inc = math.radians(cfg.target_inc_deg)

    vx = -cfg.target_distance * omega * math.sin(angle)
    vy =  cfg.target_distance * omega * math.cos(angle) * math.cos(inc)
    vz =  cfg.target_distance * omega * math.cos(angle) * math.sin(inc)
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
# Physics-Invariant Feature Computation
# ═══════════════════════════════════════════════════════════════════════════

def _compute_synodic_coords(r_sc: np.ndarray, r_source: np.ndarray, r_target: np.ndarray) -> np.ndarray:
    """
    Transform spacecraft position into a Target-Centric Synodic (Rotating) Frame.

    X-axis: Source -> Target direction
    Z-axis: Angular momentum of Source-Target system
    Y-axis: Completes right-hand system

    Origin: Target position
    """
    r_rel = r_sc - r_target

    e_hat = r_target - r_source
    e_norm = np.linalg.norm(e_hat)
    if e_norm < 1e-10:
        return r_rel  # fallback: no rotation
    e_x = e_hat / e_norm

    # Z-axis: approximate angular momentum direction
    e_z = np.array([0.0, 0.0, 1.0])
    e_z = e_z - np.dot(e_z, e_x) * e_x
    e_z_norm = np.linalg.norm(e_z)
    if e_z_norm > 1e-10:
        e_z = e_z / e_z_norm
    else:
        e_z = np.array([0.0, 0.0, 1.0])

    e_y = np.cross(e_z, e_x)

    syn_x = np.dot(r_rel, e_x)
    syn_y = np.dot(r_rel, e_y)
    syn_z = np.dot(r_rel, e_z)

    return np.array([syn_x, syn_y, syn_z])


def _compute_physics_features(r: np.ndarray, v: np.ndarray,
                               r_target: np.ndarray, cfg: MissionConfig):
    """
    Compute physics-invariant features at a single timestep.

    Returns: (rel_x, rel_y, rel_z, spec_energy, fpa_deg,
              norm_target_dist, radial_vel, vel_mag)
    """
    r_source = np.array([0.0, 0.0, 0.0])
    synodic = _compute_synodic_coords(r, r_source, r_target)

    # Specific Orbital Energy w.r.t source body
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    spec_energy = 0.5 * v_mag**2 - cfg.source_mu / r_mag

    # Flight Path Angle
    r_dot_v = np.dot(r, v)
    if r_mag > 0 and v_mag > 0:
        sin_gamma = r_dot_v / (r_mag * v_mag)
        sin_gamma = max(-1.0, min(1.0, sin_gamma))
        fpa_deg = math.degrees(math.asin(sin_gamma))
    else:
        fpa_deg = 0.0

    # Normalized Target Distance
    r_rel_target = r - r_target
    target_rmag = np.linalg.norm(r_rel_target)
    norm_target_dist = target_rmag / cfg.target_soi

    # Radial Velocity toward target
    if target_rmag > 1e-6:
        r_hat = r_rel_target / target_rmag
        radial_vel = np.dot(v, r_hat)
    else:
        radial_vel = 0.0

    return (synodic[0], synodic[1], synodic[2],
            spec_energy, fpa_deg, norm_target_dist, radial_vel, v_mag)


# ═══════════════════════════════════════════════════════════════════════════
# 3-Body Propagator (RK4)
# ═══════════════════════════════════════════════════════════════════════════

def _acceleration(r: np.ndarray, r_target: np.ndarray, cfg: MissionConfig) -> np.ndarray:
    """Acceleration from source + target body."""
    r_mag = np.linalg.norm(r)
    r_rel_target = r - r_target
    rt_mag = np.linalg.norm(r_rel_target)

    a_source = -cfg.source_mu / (r_mag**3) * r
    a_target = -cfg.target_mu / (rt_mag**3) * r_rel_target
    return a_source + a_target


def _rk4_step(state: np.ndarray, t: float, dt: float, cfg: MissionConfig) -> np.ndarray:
    r = state[:3]
    v = state[3:]

    r_t_0  = _target_ephemeris(t, cfg)
    r_t_h  = _target_ephemeris(t + dt/2, cfg)
    r_t_dt = _target_ephemeris(t + dt, cfg)

    k1_v = _acceleration(r, r_t_0, cfg)
    k1_r = v

    k2_v = _acceleration(r + k1_r * dt/2, r_t_h, cfg)
    k2_r = v + k1_v * dt/2

    k3_v = _acceleration(r + k2_r * dt/2, r_t_h, cfg)
    k3_r = v + k2_v * dt/2

    k4_v = _acceleration(r + k3_r * dt, r_t_dt, cfg)
    k4_r = v + k3_v * dt

    r_new = r + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return np.concatenate([r_new, v_new])


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_synthetic(params: MissionParams, time_step: float = 60.0) -> pd.DataFrame:
    """
    Run a single mission simulation using the 3-body RK4 propagator.

    Automatically configures all constants based on the source/target
    bodies specified in the MissionParams.
    """
    cfg = MissionConfig(params.source, params.target)

    # 1. Establish initial parking orbit (Keplerian → Cartesian)
    r, v = _keplerian_to_cartesian(
        params.SMA, params.ECC, params.INC,
        params.RAAN, params.AOP,
        mu=cfg.source_mu,
    )

    # 2. Apply TOI impulsive burn in VNB frame
    burn_vnb = np.array([params.TOI_V, params.TOI_N, params.TOI_B])
    R_vnb = _get_vnb_frame(r, v)
    v += R_vnb @ burn_vnb

    # 3. Propagate (3-body RK4)
    state = np.concatenate([r, v])
    total_secs = cfg.prop_days * 86400.0
    n_steps = int(total_secs / time_step)
    integration_dt = min(time_step, 10.0)

    rows = []
    min_target_rmag = 999999999.0
    crashed_source = False
    closest_state = None
    closest_target_pos = None
    closest_t = 0.0

    for step_i in range(n_steps + 1):
        t = step_i * time_step
        r_current, v_current = state[:3], state[3:]

        source_rmag = np.linalg.norm(r_current)
        if source_rmag < cfg.source_radius:
            crashed_source = True
            break

        target_pos = _target_ephemeris(t, cfg)
        target_rmag = np.linalg.norm(r_current - target_pos)

        if target_rmag < min_target_rmag:
            min_target_rmag = target_rmag
            closest_state = (r_current.copy(), v_current.copy())
            closest_target_pos = target_pos.copy()
            closest_t = t

        sma, ecc = _calculate_orbit_elements(r_current, v_current, cfg.source_mu)

        # Compute physics-invariant features
        (rel_x, rel_y, rel_z, spec_energy, fpa_deg,
         norm_target_dist, radial_vel, vel_mag) = _compute_physics_features(
            r_current, v_current, target_pos, cfg
        )

        # Stop early if inside capture radius
        if target_rmag < cfg.capture_radius:
            break

        rows.append([
            params.sim_id,
            t,
            t / 86400.0,
            *r_current,
            *v_current,
            source_rmag,
            target_rmag,
            ecc,
            sma,
            # Physics-invariant features
            rel_x, rel_y, rel_z,
            spec_energy, fpa_deg,
            norm_target_dist, radial_vel, vel_mag,
            # Context features (constant per mission)
            params.mu_ratio,
            params.soi_ratio,
            params.dist_ratio,
            # Body names
            params.source,
            params.target,
            # Placeholders (filled after propagation)
            -1, "", 0.0,
        ])

        # Step forward
        if step_i < n_steps:
            subs = int(time_step / integration_dt)
            for _ in range(subs):
                state = _rk4_step(state, t, integration_dt, cfg)
                t += integration_dt

    if not rows:
        return _empty_failure(params, cfg)

    # 4. Determine final outcome using closest approach to target
    r_close, v_close = closest_state
    v_rel_target = v_close - _target_velocity(closest_t, cfg)
    target_sma, target_ecc = _calculate_orbit_elements(
        r_close - closest_target_pos, v_rel_target, cfg.target_mu
    )
    rad_per = target_sma * (1 - target_ecc)

    label = 0
    failure_type = "unknown"

    if crashed_source:
        failure_type = "source_impact"
    elif min_target_rmag > cfg.miss_distance:
        failure_type = "missed_target"
    elif rad_per > 0 and rad_per < cfg.min_radper:
        failure_type = "surface_impact"
    elif rad_per > cfg.max_radper:
        failure_type = "orbit_too_high"
    else:
        label = 1
        failure_type = "success"

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["label"] = label
    df["failure_type"] = failure_type
    df["min_target_rmag"] = min_target_rmag

    return df


def _empty_failure(params: MissionParams, cfg: MissionConfig) -> pd.DataFrame:
    miss = cfg.miss_distance * 2
    row = [[
        params.sim_id, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        cfg.source_radius, miss,
        0.0, 0.0,
        # Physics-invariant placeholders
        0.0, 0.0, 0.0, 0.0, 0.0,
        miss / cfg.target_soi, 0.0, 0.0,
        # Context features
        params.mu_ratio, params.soi_ratio, params.dist_ratio,
        # Body names
        params.source, params.target,
        # Target
        0, "degenerate_orbit", miss,
    ]]
    return pd.DataFrame(row, columns=COLUMNS)
