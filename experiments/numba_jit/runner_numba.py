"""
run_synthetic_numba — drop-in replacement for gmat_runner.run_synthetic
=========================================================================
Reuses the exact same outer orchestration, helper functions, and outcome
classification logic as the production src/data_collection/gmat_runner.py.
Only the substep RK4 propagation loop (the actual bottleneck — millions of
calls per mission for outer planets) is replaced with a Numba-JIT'd version.

This is a side experiment. It does not modify gmat_runner.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import MissionParams
from src.data_collection.gmat_runner import (
    COLUMNS,
    MissionConfig,
    _keplerian_to_cartesian,
    _get_vnb_frame,
    _calculate_orbit_elements,
    _compute_physics_features,
    _target_ephemeris,
    _target_velocity,
    _empty_failure,
)

from experiments.numba_jit.jit_physics import propagate_segment_nb


def run_synthetic_numba(params: MissionParams, time_step: float = 1800.0) -> pd.DataFrame:
    cfg = MissionConfig(params.source, params.target)

    r, v = _keplerian_to_cartesian(
        params.SMA, params.ECC, params.INC,
        params.RAAN, params.AOP,
        mu=cfg.source_mu,
    )
    burn_vnb = np.array([params.TOI_V, params.TOI_N, params.TOI_B])
    R_vnb = _get_vnb_frame(r, v)
    v += R_vnb @ burn_vnb

    state = np.concatenate([r, v])
    total_secs = cfg.prop_days * 86400.0
    n_steps = int(total_secs / time_step)

    is_moon = cfg.source_name == "earth" and cfg.target_name == "moon"
    omega = 2 * math.pi / cfg.target_orbital_period_s
    inc_rad = math.radians(cfg.target_inc_deg)

    rows = []
    min_target_rmag = math.inf
    crashed_source = False
    terminated = False
    closest_state = None
    closest_target_pos = None
    closest_t = 0.0

    for step_i in range(n_steps + 1):
        if terminated:
            break

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

        (rel_x, rel_y, rel_z, spec_energy, fpa_deg,
         norm_target_dist, radial_vel, vel_mag) = _compute_physics_features(
            r_current, v_current, target_pos, cfg
        )

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
            rel_x, rel_y, rel_z,
            spec_energy, fpa_deg,
            norm_target_dist, radial_vel, vel_mag,
            params.mu_ratio,
            params.soi_ratio,
            params.dist_ratio,
            params.source,
            params.target,
            -1, "", 0.0,
        ])

        if step_i < n_steps:
            segment_end = t + time_step
            (state, t_new, crashed, captured,
             seg_min_rmag, seg_closest_r, seg_closest_v,
             seg_closest_target_pos, seg_closest_t) = propagate_segment_nb(
                state, t, segment_end,
                cfg.target_distance, omega, inc_rad,
                cfg.source_mu, cfg.target_mu, cfg.source_radius,
                cfg.target_soi, cfg.capture_radius,
                time_step, is_moon,
            )

            if seg_min_rmag < min_target_rmag:
                min_target_rmag = seg_min_rmag
                closest_state = (seg_closest_r.copy(), seg_closest_v.copy())
                closest_target_pos = seg_closest_target_pos.copy()
                closest_t = seg_closest_t

            if crashed:
                crashed_source = True
                terminated = True
            elif captured:
                terminated = True

    if not rows:
        return _empty_failure(params, cfg)

    if closest_state is None or closest_target_pos is None or not math.isfinite(min_target_rmag):
        df = pd.DataFrame(rows, columns=COLUMNS)
        df["label"] = 0
        df["failure_type"] = "degenerate_orbit"
        df["min_target_rmag"] = min_target_rmag if math.isfinite(min_target_rmag) else cfg.miss_distance * 2
        return df

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
