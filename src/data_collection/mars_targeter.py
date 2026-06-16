"""
Planet Targeting Diagnostics
============================
Searches Earth-target nominal parameters for the repository's synthetic
propagator before expensive dataset generation.

This is a diagnostic targeter for the current simplified dynamics. It does not
replace high-fidelity GMAT/SPICE validation, but it prevents generating large
all-failure interplanetary datasets from an unverified nominal.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import (
    MissionParams,
    PLANET_REGISTRY,
    _compute_context_features,
)
from src.data_collection.gmat_runner import (
    MissionConfig,
    _calculate_orbit_elements,
    _get_vnb_frame,
    _keplerian_to_cartesian,
    _rk4_step,
    _target_ephemeris,
    _target_velocity,
)


@dataclass(frozen=True)
class TargetingResult:
    toi_v: float
    aop: float
    min_target_rmag: float
    rad_per: float
    failure_type: str
    closest_day: float

    @property
    def label(self) -> int:
        return int(self.failure_type == "success")


def make_params(target: str, toi_v: float, aop: float, sim_id: int = 0) -> MissionParams:
    context = _compute_context_features("earth", target)
    return MissionParams(
        sim_id=sim_id,
        source="earth",
        target=target,
        TOI_V=float(toi_v),
        TOI_N=0.0,
        TOI_B=0.0,
        RAAN=0.0,
        AOP=float(aop),
        INC=0.0,
        SMA=6571.0,
        ECC=0.001,
        dv_V_offset=0.0,
        dv_N_offset=0.0,
        dv_B_offset=0.0,
        RAAN_offset=0.0,
        AOP_offset=0.0,
        INC_offset=0.0,
        mu_ratio=context["mu_ratio"],
        soi_ratio=context["soi_ratio"],
        dist_ratio=context["dist_ratio"],
    )


def make_mars_params(toi_v: float, aop: float, sim_id: int = 0) -> MissionParams:
    return make_params("mars", toi_v, aop, sim_id=sim_id)


def _adaptive_dt(r_sc: np.ndarray, target_pos: np.ndarray, cfg: MissionConfig,
                 far_dt: float, near_dt: float, fine_dt: float) -> float:
    source_rmag = np.linalg.norm(r_sc)
    target_rmag = np.linalg.norm(r_sc - target_pos)

    if source_rmag < 100_000.0 or target_rmag < 250_000.0:
        return fine_dt
    if target_rmag < 5.0 * cfg.target_soi:
        return near_dt
    return far_dt


def evaluate(params: MissionParams, far_dt: float = 900.0,
             near_dt: float = 300.0, fine_dt: float = 30.0) -> TargetingResult:
    cfg = MissionConfig(params.source, params.target)
    r, v = _keplerian_to_cartesian(
        params.SMA,
        params.ECC,
        params.INC,
        params.RAAN,
        params.AOP,
        mu=cfg.source_mu,
    )
    burn_vnb = np.array([params.TOI_V, params.TOI_N, params.TOI_B])
    v += _get_vnb_frame(r, v) @ burn_vnb
    state = np.concatenate([r, v])

    total_secs = cfg.prop_days * 86400.0
    t = 0.0
    min_target_rmag = math.inf
    closest_t = 0.0
    closest_state: tuple[np.ndarray, np.ndarray] | None = None
    closest_target_pos: np.ndarray | None = None
    crashed_source = False

    while t <= total_secs:
        r_current = state[:3]
        v_current = state[3:]
        source_rmag = np.linalg.norm(r_current)
        if source_rmag < cfg.source_radius:
            crashed_source = True
            break

        target_pos = _target_ephemeris(t, cfg)
        target_rmag = np.linalg.norm(r_current - target_pos)
        if target_rmag < min_target_rmag:
            min_target_rmag = target_rmag
            closest_t = t
            closest_state = (r_current.copy(), v_current.copy())
            closest_target_pos = target_pos.copy()

        if target_rmag < cfg.capture_radius:
            break

        dt = _adaptive_dt(r_current, target_pos, cfg, far_dt, near_dt, fine_dt)
        dt = min(dt, total_secs - t)
        if dt <= 0:
            break
        state = _rk4_step(state, t, dt, cfg)
        t += dt

    rad_per = math.inf
    if closest_state is not None and closest_target_pos is not None:
        r_close, v_close = closest_state
        v_rel_target = v_close - _target_velocity(closest_t, cfg)
        target_sma, target_ecc = _calculate_orbit_elements(
            r_close - closest_target_pos,
            v_rel_target,
            cfg.target_mu,
        )
        rad_per = target_sma * (1.0 - target_ecc)

    if crashed_source:
        failure_type = "source_impact"
    elif min_target_rmag > cfg.miss_distance:
        failure_type = "missed_target"
    elif rad_per > 0 and rad_per < cfg.min_radper:
        failure_type = "surface_impact"
    elif rad_per > cfg.max_radper:
        failure_type = "orbit_too_high"
    else:
        failure_type = "success"

    return TargetingResult(
        toi_v=float(params.TOI_V),
        aop=float(params.AOP),
        min_target_rmag=float(min_target_rmag),
        rad_per=float(rad_per),
        failure_type=failure_type,
        closest_day=float(closest_t / 86400.0),
    )


def sweep(target: str, v_min: float, v_max: float, v_count: int,
          aop_min: float, aop_max: float, aop_count: int,
          top: int, far_dt: float = 900.0,
          near_dt: float = 300.0, fine_dt: float = 30.0) -> list[TargetingResult]:
    results: list[TargetingResult] = []
    total = v_count * aop_count
    start = time.time()
    done = 0

    for toi_v in np.linspace(v_min, v_max, v_count):
        for aop in np.linspace(aop_min, aop_max, aop_count):
            result = evaluate(
                make_params(target, float(toi_v), float(aop)),
                far_dt=far_dt,
                near_dt=near_dt,
                fine_dt=fine_dt,
            )
            results.append(result)
            done += 1
            if done % max(1, total // 20) == 0:
                best = min(results, key=lambda r: r.min_target_rmag)
                print(
                    f"  {done:>5}/{total}  best: V={best.toi_v:.5f}, "
                    f"AOP={best.aop:.3f}, min={best.min_target_rmag:.1f} km, "
                    f"rad_per={best.rad_per:.1f}, {best.failure_type}",
                    flush=True,
                )

    target_radius = PLANET_REGISTRY[target]["radius"]
    target_radper = target_radius + 300.0
    results.sort(key=lambda r: (r.label == 0, abs(r.rad_per - target_radper), r.min_target_rmag))
    print(f"\nCompleted {total} evaluations in {time.time() - start:.1f}s")
    print("\nTop candidates:")
    for r in results[:top]:
        print(
            f"  label={r.label} {r.failure_type:<14} "
            f"V={r.toi_v:8.5f}  AOP={r.aop:8.3f}  "
            f"min={r.min_target_rmag:12.1f} km  "
            f"rad_per={r.rad_per:10.1f} km  day={r.closest_day:7.2f}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Earth-target nominal targeting parameters.")
    parser.add_argument("--target", type=str, default="mars", choices=sorted(PLANET_REGISTRY.keys()))
    parser.add_argument("--v-min", type=float, default=3.8)
    parser.add_argument("--v-max", type=float, default=4.2)
    parser.add_argument("--v-count", type=int, default=9)
    parser.add_argument("--aop-min", type=float, default=330.0)
    parser.add_argument("--aop-max", type=float, default=360.0)
    parser.add_argument("--aop-count", type=int, default=16)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--far-dt", type=float, default=900.0)
    parser.add_argument("--near-dt", type=float, default=300.0)
    parser.add_argument("--fine-dt", type=float, default=30.0)
    args = parser.parse_args()

    sweep(
        target=args.target,
        v_min=args.v_min,
        v_max=args.v_max,
        v_count=args.v_count,
        aop_min=args.aop_min,
        aop_max=args.aop_max,
        aop_count=args.aop_count,
        top=args.top,
        far_dt=args.far_dt,
        near_dt=args.near_dt,
        fine_dt=args.fine_dt,
    )


if __name__ == "__main__":
    main()
