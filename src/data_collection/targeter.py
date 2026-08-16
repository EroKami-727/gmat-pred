"""
Historical Earth-Moon nominal search (Nelder-Mead over TOI_V and RAAN).

Superseded by `find_nominal.py`, `exact_targeter.py` and `adaptive_targeter.py`,
and by the converged per-planet constants now living in `generator.py`. Kept for
provenance of how the Moon nominal was originally found; nothing imports it.

It had been dead in a stricter sense until now: `generator.NOMINAL` was split
into per-planet `EARTH_*_NOMINAL` dicts at some point and this import was never
updated, so the module raised ImportError on load. Repointed at
EARTH_MOON_NOMINAL — the objective minimises `luna_rmag`, so that is
unambiguously the dict it meant.
"""

import sys
from pathlib import Path
import scipy.optimize as opt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import MissionParams, EARTH_MOON_NOMINAL as NOMINAL
from src.data_collection.gmat_runner import run_synthetic

def objective(x):
    v, r = x
    params = MissionParams(
        sim_id=0,
        TOI_V=v, TOI_N=NOMINAL["TOI_N"], TOI_B=NOMINAL["TOI_B"],
        RAAN=r, AOP=NOMINAL["AOP"], INC=NOMINAL["INC"],
        SMA=NOMINAL["SMA"], ECC=NOMINAL["ECC"],
        dv_V_offset=0, dv_N_offset=0, dv_B_offset=0,
        RAAN_offset=0, AOP_offset=0, INC_offset=0
    )
    # Using 1800s for slightly finer resolution
    df = run_synthetic(params, time_step=1800.0)
    dist = df["luna_rmag"].min()
    print(f"Eval: V={v:.5f}, RAAN={r:.4f} -> {dist:.0f} km")
    return dist

if __name__ == "__main__":
    print("Running SciPy Nelder-Mead optimization to find synthetic nominal...")
    x0 = [3.197, 227.36]
    res = opt.minimize(
        objective, 
        x0, 
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 100, 'maxfev': 100}
    )
    print("\nOptimization Finished!")
    print(res)
