"""
RK4 Nominal Finder — Sweep TOI_V to find the success corridor
=============================================================
Quickly sweeps the burn magnitude to find the optimal value for the 
custom RK4 engine, as the GMAT-converged value might differ slightly.
"""

import numpy as np
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import MissionParams, EARTH_MOON_NOMINAL
from src.data_collection.gmat_runner import run_synthetic, PLANET_REGISTRY

def sweep():
    # Sweep around the GMAT value 3.2337
    toi_v_range = np.linspace(3.18, 3.28, 51)
    
    print(f"{'TOI_V':>10} | {'Outcome':>15} | {'Min R (km)':>12}")
    print("-" * 45)
    
    best_v = None
    best_dist = 999999
    
    for v in toi_v_range:
        params = MissionParams(
            sim_id=0,
            source="earth",
            target="moon",
            TOI_V=v,
            TOI_N=EARTH_MOON_NOMINAL["TOI_N"],
            TOI_B=EARTH_MOON_NOMINAL["TOI_B"],
            RAAN=EARTH_MOON_NOMINAL["RAAN"],
            AOP=EARTH_MOON_NOMINAL["AOP"],
            INC=EARTH_MOON_NOMINAL["INC"],
            SMA=EARTH_MOON_NOMINAL["SMA"],
            ECC=EARTH_MOON_NOMINAL["ECC"],
            dv_V_offset=0,
            dv_N_offset=0,
            dv_B_offset=0,
            RAAN_offset=0,
            AOP_offset=0,
            INC_offset=0,
            mu_ratio=0, # not used in runner math
            soi_ratio=0,
            dist_ratio=0
        )
        
        df = run_synthetic(params, time_step=300) # faster sweep
        last_row = df.iloc[-1]
        
        outcome = last_row["failure_type"]
        min_r = last_row["min_target_rmag"]
        
        indicator = "★ SUCCESS" if outcome == "success" else f"  {outcome}"
        print(f"{v:>10.5f} | {indicator:>15} | {min_r:>12.1f}")
        
        # Success is 100-500km altitude = 1837-2237km radius.
        # Center is 2037km.
        dist_to_center = abs(min_r - 2037.0)
        if outcome == "success" and dist_to_center < best_dist:
            best_dist = dist_to_center
            best_v = v

    print("-" * 45)
    if best_v:
        print(f"RECOMMENDED NOMINAL TOI_V: {best_v:.5f}")
    else:
        print("No success found in sweep. Check geometry.")

if __name__ == "__main__":
    sweep()
