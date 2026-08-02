"""
Simplified interplanetary trajectory generator for OrbitGuard mission creation.

Physics: two-body solar gravity, RK4 integrator, circular planetary orbits.
Features match the training schema (same FEATURE_COLS order and normalization).
"""

from __future__ import annotations
import numpy as np

MU_SUN = 132_712_440_018.0   # km³/s²
AU     = 149_597_870.7        # km

PLANET_DATA: dict[str, dict] = {
    "mercury": {"a_au": 0.387,  "mu": 22_030.0,       "soi_km":  112_000.0,    "period_s":  87.97 * 86400},
    "venus":   {"a_au": 0.723,  "mu": 324_859.0,       "soi_km":  617_000.0,    "period_s": 224.7  * 86400},
    "mars":    {"a_au": 1.524,  "mu": 42_828.0,        "soi_km":  577_000.0,    "period_s": 686.97 * 86400},
    "jupiter": {"a_au": 5.203,  "mu": 126_686_534.0,   "soi_km": 48_200_000.0,  "period_s": 4332.0 * 86400},
    "saturn":  {"a_au": 9.537,  "mu": 37_931_187.0,    "soi_km": 54_800_000.0,  "period_s": 10_759.0 * 86400},
    "uranus":  {"a_au": 19.191, "mu": 5_793_966.0,     "soi_km": 51_800_000.0,  "period_s": 30_687.0 * 86400},
    "neptune": {"a_au": 30.069, "mu": 6_836_529.0,     "soi_km": 86_800_000.0,  "period_s": 60_190.0 * 86400},
}
_EARTH = {"a_au": 1.0, "mu": 398_600.0, "soi_km": 924_000.0, "period_s": 365.25 * 86400}


def hohmann_c3(target: str) -> float:
    """Minimum launch C3 (km²/s²) for a Hohmann transfer to target."""
    p   = PLANET_DATA[target.lower()]
    r_e = _EARTH["a_au"] * AU
    r_t = p["a_au"] * AU
    a_tr = (r_e + r_t) / 2
    v_at_earth_tr = np.sqrt(MU_SUN * (2 / r_e - 1 / a_tr))
    v_earth_circ  = np.sqrt(MU_SUN / r_e)
    dv = abs(v_at_earth_tr - v_earth_circ)
    return round(dv ** 2, 2)


def optimal_phase_deg(target: str) -> float:
    """
    Optimal planet phase angle (degrees ahead of spacecraft) for Hohmann departure.
    The planet should be this many degrees ahead of Earth at departure so it arrives
    at the transfer orbit intercept at the right time.
    """
    p = PLANET_DATA[target.lower()]
    r_e  = _EARTH["a_au"] * AU
    r_t  = p["a_au"] * AU
    a_tr = (r_e + r_t) / 2
    t_transfer = np.pi * np.sqrt(a_tr ** 3 / MU_SUN)   # half-orbit time
    omega_t = 2 * np.pi / p["period_s"]                  # target angular velocity
    # For outer planets: target must be ahead by (π - omega_t × T)
    angle_target = np.pi - omega_t * t_transfer
    deg = float(np.degrees(angle_target)) % 360
    if deg > 180:
        deg -= 360
    return round(deg, 1)


def _planet_pos(name: str, t: float, phase0: float) -> np.ndarray:
    """Circular orbit position (km) at time t (s) with initial phase phase0 (rad)."""
    data  = PLANET_DATA[name] if name in PLANET_DATA else _EARTH
    omega = 2 * np.pi / data["period_s"]
    theta = phase0 + omega * t
    r     = data["a_au"] * AU
    return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])


def _earth_pos(t: float, earth_phase0: float) -> np.ndarray:
    omega = 2 * np.pi / _EARTH["period_s"]
    theta = earth_phase0 + omega * t
    r     = AU
    return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])


def _rk4_step(r: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    def acc(pos: np.ndarray) -> np.ndarray:
        r2 = float(np.dot(pos, pos))
        return -(MU_SUN / (r2 * np.sqrt(r2))) * pos

    k1r, k1v = v,                acc(r)
    k2r, k2v = v + .5*dt*k1v,   acc(r + .5*dt*k1r)
    k3r, k3v = v + .5*dt*k2v,   acc(r + .5*dt*k2r)
    k4r, k4v = v +    dt*k3v,   acc(r +    dt*k3r)

    r_new = r + (dt / 6) * (k1r + 2*k2r + 2*k3r + k4r)
    v_new = v + (dt / 6) * (k1v + 2*k2v + 2*k3v + k4v)
    return r_new, v_new


def _context_features(target: str) -> tuple[float, float, float]:
    """
    The three constant-per-mission context features, matching
    `src/data_collection/generator._compute_context_features` exactly.

    These are MISSION CONSTANTS, not per-timestep quantities. An earlier version
    of this module recomputed them from the instantaneous spacecraft-target
    distance, so they varied along the trajectory (Venus dist_ratio drifted
    0.820 -> 0.731 where the training data holds a constant 0.277).
    """
    p = PLANET_DATA[target.lower()]
    transfer_dist = abs(p["a_au"] - _EARTH["a_au"]) * AU     # km
    mu_ratio = p["mu"] / MU_SUN
    soi_ratio = p["soi_km"] / max(transfer_dist, 1.0)
    dist_ratio = transfer_dist / AU
    return mu_ratio, soi_ratio, dist_ratio


def _features(
    r_sc: np.ndarray, v_sc: np.ndarray,
    r_tgt: np.ndarray, r_earth: np.ndarray,
    mu_tgt: float, soi_km: float,
    context: tuple[float, float, float] | None = None,
) -> list[float]:
    """Return 13 features in FEATURE_COLS order."""
    rel = r_sc - r_tgt
    rel_x, rel_y, rel_z = float(rel[0]), float(rel[1]), float(rel[2])

    r_norm  = float(np.linalg.norm(r_sc))
    v_norm  = float(np.linalg.norm(v_sc))
    spec_energy = 0.5 * v_norm**2 - MU_SUN / r_norm

    r_hat      = r_sc / r_norm
    radial_vel = float(np.dot(v_sc, r_hat))
    v_tan      = max(0.0, v_norm**2 - radial_vel**2) ** 0.5
    fpa_deg    = float(np.degrees(np.arctan2(radial_vel, v_tan)))

    h       = np.cross(r_sc, v_sc)
    ecc_vec = np.cross(v_sc, h) / MU_SUN - r_sc / r_norm
    ecc     = float(np.linalg.norm(ecc_vec))

    dist_to_tgt     = float(np.linalg.norm(rel))
    norm_target_dist = dist_to_tgt / soi_km          # dist / SOI

    # Context features are constants of the mission, supplied by the caller.
    if context is None:
        mu_ratio = mu_tgt / MU_SUN
        soi_ratio = 0.0
        dist_ratio = 0.0
    else:
        mu_ratio, soi_ratio, dist_ratio = context

    earth_rmag = float(np.linalg.norm(r_sc - r_earth))
    vel_mag    = v_norm

    return [rel_x, rel_y, rel_z, spec_energy, fpa_deg, norm_target_dist,
            radial_vel, vel_mag, earth_rmag, ecc, mu_ratio, soi_ratio, dist_ratio]


def generate_mission(
    target: str,
    c3: float,
    phase_ahead_deg: float = None,
    earth_phase_deg: float = 0.0,
    dt_days: float = 1.0,
    n_feature_steps: int = 200,
) -> dict:
    """
    Generate a synthetic interplanetary trajectory from Earth to `target`.

    Args:
        target: planet name (lowercase)
        c3: launch energy in km²/s² (must be > 0)
        phase_ahead_deg: angular lead of target planet at departure. None = Hohmann optimal.
        earth_phase_deg: Earth's angular position at departure (sets geometry).
        dt_days: integrator timestep in days.
        n_feature_steps: number of feature vectors returned (downsampled from full propagation).

    Returns dict with keys: positions, telemetry, features (np.ndarray), label, target_body,
                            failure_type, total_steps, c3, phase_ahead_deg, metadata.
    """
    target = target.lower()
    if target not in PLANET_DATA:
        raise ValueError(f"Unknown target: {target}")

    pdata   = PLANET_DATA[target]
    a_tgt   = pdata["a_au"] * AU
    mu_tgt  = pdata["mu"]
    soi_km  = pdata["soi_km"]
    a_earth = AU

    if phase_ahead_deg is None:
        phase_ahead_deg = optimal_phase_deg(target)

    # Propagation duration: 2× Hohmann transfer time, min 180 days
    a_transfer = (a_earth + a_tgt) / 2
    t_hoh_s    = np.pi * np.sqrt(a_transfer**3 / MU_SUN)
    max_days   = max(180, int(t_hoh_s / 86400 * 2.2))
    dt_s       = dt_days * 86400.0

    # ── Initial conditions ─────────────────────────────────────────────────
    earth_phase0  = np.radians(earth_phase_deg)
    target_phase0 = earth_phase0 + np.radians(phase_ahead_deg)

    r_earth_0 = _earth_pos(0.0, earth_phase0)
    v_earth_0 = np.array([
        -np.sqrt(MU_SUN / a_earth) * np.sin(earth_phase0),
         np.sqrt(MU_SUN / a_earth) * np.cos(earth_phase0),
         0.0,
    ])

    # Prograde for outer planets, retrograde for inner
    v_earth_hat = v_earth_0 / np.linalg.norm(v_earth_0)
    dv_sign = -1.0 if a_tgt < a_earth else 1.0
    dv_mag  = np.sqrt(max(0.0, c3))

    r_sc = r_earth_0.copy()
    v_sc = v_earth_0 + dv_sign * dv_mag * v_earth_hat

    # ── Propagate ─────────────────────────────────────────────────────────
    all_r, all_v, all_t = [r_sc.copy()], [v_sc.copy()], [0.0]
    success = False
    min_dist = np.inf

    for step in range(max_days):
        t = step * dt_s
        r_sc, v_sc = _rk4_step(r_sc, v_sc, dt_s)
        r_tgt_now  = _planet_pos(target, t + dt_s, target_phase0)
        d = float(np.linalg.norm(r_sc - r_tgt_now))
        if d < min_dist:
            min_dist = d
        if d < soi_km * 1.5:
            success = True
        all_r.append(r_sc.copy())
        all_v.append(v_sc.copy())
        all_t.append(t + dt_s)
        if success and d > min_dist * 2:  # past closest approach
            break

    total_raw = len(all_r)
    # Aim for up to 1000 display points; for short missions keep all steps
    stride  = max(1, total_raw // min(n_feature_steps, 1000))
    indices = list(range(0, total_raw, stride))
    if (total_raw - 1) not in indices:
        indices.append(total_raw - 1)

    # ── Build outputs ──────────────────────────────────────────────────────
    positions, telemetry, feature_rows = [], [], []
    context = _context_features(target)

    for rank, idx in enumerate(indices):
        t    = all_t[idx]
        r    = all_r[idx]
        v    = all_v[idx]
        r_tgt   = _planet_pos(target, t, target_phase0)
        r_earth = _earth_pos(t, earth_phase0)

        feats = _features(r, v, r_tgt, r_earth, mu_tgt, soi_km, context)
        ep    = round((rank + 1) / len(indices), 5)

        rel = r - r_tgt
        positions.append({"step": rank, "elapsed_pct": ep,
                          "rel_x": round(float(rel[0]), 2),
                          "rel_y": round(float(rel[1]), 2)})
        telemetry.append({"step": rank, "elapsed_pct": ep,
                          "vel_mag":          round(feats[7], 3),
                          "ecc":              round(feats[9], 5),
                          "spec_energy":      round(feats[3], 2),
                          "earth_rmag":       round(feats[8], 1),
                          "fpa_deg":          round(feats[4], 4),
                          "norm_target_dist": round(feats[5], 5)})
        feature_rows.append(feats)

    features_arr = np.array(feature_rows, dtype=np.float32)
    label = 1 if success else 0

    return {
        "positions":     positions,
        "telemetry":     telemetry,
        "features":      features_arr,
        "label":         label,
        "target_body":   target,
        "failure_type":  "nominal" if success else "missed_target",
        "total_steps":   len(indices),
        "c3":            c3,
        "phase_ahead_deg": phase_ahead_deg,
        "min_dist_km":   round(min_dist, 0),
        "soi_km":        soi_km,
    }
