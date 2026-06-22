"""
Numba JIT physics core — side experiment.
===========================================
Mirrors the hot-loop math in src/data_collection/gmat_runner.py
(_target_ephemeris, _target_velocity, _acceleration, _rk4_step,
_adaptive_integration_dt) exactly, but rewritten as plain-float/array
functions so Numba can compile them to native code.

This file intentionally duplicates rather than modifies the production
gmat_runner.py — kept isolated under experiments/ until validated.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def target_ephemeris_nb(t_sec, target_distance, omega, inc_rad):
    angle = omega * t_sec
    x = target_distance * np.cos(angle)
    y = target_distance * np.sin(angle) * np.cos(inc_rad)
    z = target_distance * np.sin(angle) * np.sin(inc_rad)
    out = np.empty(3)
    out[0], out[1], out[2] = x, y, z
    return out


@njit(cache=True, fastmath=False)
def acceleration_nb(r, r_target, source_mu, target_mu):
    r_mag = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    rx = r[0] - r_target[0]
    ry = r[1] - r_target[1]
    rz = r[2] - r_target[2]
    rt_mag = np.sqrt(rx * rx + ry * ry + rz * rz)

    sf = -source_mu / (r_mag ** 3)
    tf = -target_mu / (rt_mag ** 3)

    a = np.empty(3)
    a[0] = sf * r[0] + tf * rx
    a[1] = sf * r[1] + tf * ry
    a[2] = sf * r[2] + tf * rz
    return a


@njit(cache=True, fastmath=False)
def rk4_step_nb(state, t, dt, target_distance, omega, inc_rad, source_mu, target_mu):
    r = state[:3]
    v = state[3:]

    r_t_0 = target_ephemeris_nb(t, target_distance, omega, inc_rad)
    r_t_h = target_ephemeris_nb(t + dt / 2, target_distance, omega, inc_rad)
    r_t_dt = target_ephemeris_nb(t + dt, target_distance, omega, inc_rad)

    k1_v = acceleration_nb(r, r_t_0, source_mu, target_mu)
    k1_r = v

    k2_r_in = r + k1_r * dt / 2
    k2_v = acceleration_nb(k2_r_in, r_t_h, source_mu, target_mu)
    k2_r = v + k1_v * dt / 2

    k3_r_in = r + k2_r * dt / 2
    k3_v = acceleration_nb(k3_r_in, r_t_h, source_mu, target_mu)
    k3_r = v + k2_v * dt / 2

    k4_r_in = r + k3_r * dt
    k4_v = acceleration_nb(k4_r_in, r_t_dt, source_mu, target_mu)
    k4_r = v + k3_v * dt

    r_new = r + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_new = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    out = np.empty(6)
    out[:3] = r_new
    out[3:] = v_new
    return out


@njit(cache=True, fastmath=False)
def adaptive_dt_nb(r, t, target_distance, omega, inc_rad, target_soi, time_step, is_moon):
    if is_moon:
        return min(time_step, 10.0)

    r_from_source = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    target_pos = target_ephemeris_nb(t, target_distance, omega, inc_rad)
    dx = r[0] - target_pos[0]
    dy = r[1] - target_pos[1]
    dz = r[2] - target_pos[2]
    r_from_target = np.sqrt(dx * dx + dy * dy + dz * dz)

    if r_from_source < 100_000.0 or r_from_target < 250_000.0:
        return min(time_step, 30.0)
    if r_from_target < 5.0 * target_soi:
        return min(time_step, 300.0)
    return min(time_step, 900.0)


@njit(cache=True, fastmath=False)
def propagate_segment_nb(
    state, t_start, segment_end,
    target_distance, omega, inc_rad,
    source_mu, target_mu, source_radius, target_soi, capture_radius,
    time_step, is_moon,
):
    """
    Advance `state` from t_start to segment_end using adaptive RK4 substeps.
    Mirrors the inner `while t < segment_end:` loop in run_synthetic exactly,
    including per-substep closest-approach tracking.

    Returns
    -------
    state, t, crashed, captured, min_rmag, closest_r, closest_v, closest_target_pos, closest_t
    """
    t = t_start
    crashed = False
    captured = False
    min_rmag = 1.0e30
    closest_r = state[:3].copy()
    closest_v = state[3:].copy()
    closest_target_pos = target_ephemeris_nb(t, target_distance, omega, inc_rad)
    closest_t = t

    while t < segment_end:
        dt = adaptive_dt_nb(state[:3], t, target_distance, omega, inc_rad, target_soi, time_step, is_moon)
        dt = min(dt, segment_end - t)
        if dt <= 0:
            break

        state = rk4_step_nb(state, t, dt, target_distance, omega, inc_rad, source_mu, target_mu)
        t += dt

        r_step = state[:3]
        source_rmag_step = np.sqrt(r_step[0] ** 2 + r_step[1] ** 2 + r_step[2] ** 2)
        if source_rmag_step < source_radius:
            crashed = True
            break

        target_pos_after = target_ephemeris_nb(t, target_distance, omega, inc_rad)
        dx = r_step[0] - target_pos_after[0]
        dy = r_step[1] - target_pos_after[1]
        dz = r_step[2] - target_pos_after[2]
        target_rmag_after = np.sqrt(dx * dx + dy * dy + dz * dz)

        if target_rmag_after < min_rmag:
            min_rmag = target_rmag_after
            closest_r = r_step.copy()
            closest_v = state[3:].copy()
            closest_target_pos = target_pos_after.copy()
            closest_t = t

        if target_rmag_after < capture_radius:
            captured = True
            break

    return state, t, crashed, captured, min_rmag, closest_r, closest_v, closest_target_pos, closest_t


def warmup():
    """Trigger JIT compilation once, outside of timing measurements."""
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    propagate_segment_nb(
        state, 0.0, 900.0,
        384400.0, 2 * np.pi / (27.32 * 86400.0), 0.0,
        398600.4418, 4902.8, 6371.0, 66183.0, 500.0,
        900.0, True,
    )
