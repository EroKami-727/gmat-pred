"""
In-Distribution Mission Builder
===============================
Builds user-defined missions with the SAME propagator and feature code that
produced the training dataset (`src/data_collection/gmat_runner.run_synthetic`),
so generated missions are scorable by the per-planet models.

Why this exists: `src/api/trajectory_gen.py` propagates simplified heliocentric
two-body motion with circular planetary orbits and computes Sun-referenced
orbital elements. The training data is source-body-centred — Venus step 0 has
spec_energy +9.10 (Earth-centric, 6564 km parking orbit) where the simplified
generator produced -514.9. Every mission it created scored |z| ~ 1e13 against
the per-timestep statistics and the router had to refuse a verdict.

Missions are parameterised the way the dataset is: a circular parking orbit plus
an impulsive Trans-Orbit-Insertion burn in the VNB (velocity / normal / binormal)
frame. Offsets are expressed relative to the Hohmann nominal, so 0 = textbook
transfer and non-zero = the kind of dispersion the dataset samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_collection.generator import (
    MissionParams, _compute_context_features, _hohmann_dispersions, _hohmann_nominal,
)
from src.data_collection.gmat_runner import MissionConfig, run_synthetic
from src.ml.dataset import FEATURE_COLS
from src.ml.planet_config import CADENCE_HOURS, NATIVE_CADENCE_HOURS, downsample_for


def step_seconds(target: str) -> float:
    """
    Raw propagation step for this target, matching the dataset's cadence.

    Not a constant: the interplanetary targets are sampled at 15 h but Moon is a
    6-day transfer sampled at 60 s. Using the interplanetary value for Moon
    produced a single-step mission.
    """
    return CADENCE_HOURS.get(target.strip().lower(), NATIVE_CADENCE_HOURS) * 3600.0

SOURCE = "earth"


def nominal_for(target: str) -> dict:
    """Hohmann nominal burn/orbit parameters for an Earth → target transfer."""
    return _hohmann_nominal(SOURCE, target.lower())


def dispersion_scale(target: str) -> dict:
    """1-sigma dispersions the dataset sampled, useful as UI slider ranges."""
    try:
        d = _hohmann_dispersions(SOURCE, target.lower())
    except Exception:                                            # noqa: BLE001
        return {}
    out = {}
    for k, v in d.items():
        # values may be (mean, sigma) tuples or plain sigmas
        out[k] = float(v[1]) if isinstance(v, (tuple, list)) and len(v) > 1 else float(v)
    return out


def build_mission(
    target: str,
    dv_v_offset: float = 0.0,
    dv_n_offset: float = 0.0,
    dv_b_offset: float = 0.0,
    raan_offset: float = 0.0,
    aop_offset: float = 0.0,
    inc_offset: float = 0.0,
    sim_id: int = 0,
) -> dict:
    """
    Propagate one mission and return positions, telemetry and model features.

    Offsets are added to the Hohmann nominal:
      dv_*_offset : km/s on the TOI burn (V = along-track, N/B = out-of-plane)
      *_offset    : degrees on the parking-orbit orientation
    """
    target = target.lower()
    nom = _hohmann_nominal(SOURCE, target)
    ctx = _compute_context_features(SOURCE, target)

    params = MissionParams(
        sim_id=sim_id, source=SOURCE, target=target,
        TOI_V=nom["TOI_V"] + dv_v_offset,
        TOI_N=nom.get("TOI_N", 0.0) + dv_n_offset,
        TOI_B=nom.get("TOI_B", 0.0) + dv_b_offset,
        RAAN=nom["RAAN"] + raan_offset,
        AOP=nom["AOP"] + aop_offset,
        INC=nom["INC"] + inc_offset,
        SMA=nom["SMA"], ECC=nom["ECC"],
        dv_V_offset=dv_v_offset, dv_N_offset=dv_n_offset, dv_B_offset=dv_b_offset,
        RAAN_offset=raan_offset, AOP_offset=aop_offset, INC_offset=inc_offset,
        **ctx,
    )

    df: pd.DataFrame = run_synthetic(params, time_step=step_seconds(target))
    if df is None or len(df) == 0:
        raise ValueError(f"propagation produced no telemetry for {target}")

    label = int(df["label"].iloc[0])
    failure_type = str(df["failure_type"].iloc[0])

    # Model input: same downsample the per-planet model was trained with.
    ds = downsample_for(target)
    feats = df[FEATURE_COLS].values.astype(np.float64)[::ds]

    # Visualisation payload — synodic-frame track, matching the dataset frame.
    positions, telemetry = [], []
    total = len(df)
    vis_stride = max(1, total // 400)
    for rank, i in enumerate(range(0, total, vis_stride)):
        row = df.iloc[i]
        ep = round((i + 1) / total, 5)
        # Key names must match the dataset trajectory endpoint (rel_x/rel_y/rel_z);
        # the orbital map reads those directly.
        positions.append({
            "step": rank, "elapsed_pct": ep,
            "rel_x": round(float(row["rel_x"]), 2),
            "rel_y": round(float(row["rel_y"]), 2),
            "rel_z": round(float(row["rel_z"]), 2),
        })
        telemetry.append({
            "step": rank, "elapsed_pct": ep,
            "elapsed_days": float(row.get("elapsed_days", row["elapsed_secs"] / 86400.0)),
            "spec_energy": float(row["spec_energy"]),
            "vel_mag": float(row["vel_mag"]),
            "earth_rmag": float(row["earth_rmag"]),
            "norm_target_dist": float(row["norm_target_dist"]),
            "fpa_deg": float(row["fpa_deg"]),
            "ecc": float(row["ecc"]),
        })

    min_rmag = (float(df["min_target_rmag"].iloc[0])
                if "min_target_rmag" in df.columns else None)

    return {
        "positions": positions,
        "telemetry": telemetry,
        "features": feats,
        "label": label,
        "failure_type": failure_type,
        "target_body": target,
        "total_steps": len(feats),
        "raw_steps": total,
        "downsample": ds,
        "min_target_rmag": min_rmag,
        "params": {
            "TOI_V": params.TOI_V, "TOI_N": params.TOI_N, "TOI_B": params.TOI_B,
            "RAAN": params.RAAN, "AOP": params.AOP, "INC": params.INC,
            "SMA": params.SMA, "ECC": params.ECC,
        },
        "nominal": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in nom.items()},
        "offsets": {
            "dv_v": dv_v_offset, "dv_n": dv_n_offset, "dv_b": dv_b_offset,
            "raan": raan_offset, "aop": aop_offset, "inc": inc_offset,
        },
    }
