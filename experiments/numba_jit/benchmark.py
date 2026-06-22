"""
Benchmark — production run_synthetic vs Numba JIT run_synthetic_numba.

Times both on identical mission sets across planets of increasing transfer
duration (Mars -> Saturn -> Uranus) to see how the speedup scales with
substep count, since that's what makes outer planets slow.

Usage
-----
  python experiments/numba_jit/benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.generator import generate_inputs
from src.data_collection.gmat_runner import run_synthetic
from experiments.numba_jit.runner_numba import run_synthetic_numba
from experiments.numba_jit import jit_physics


def bench_planet(planet: str, n_missions: int, time_step: float = 54000.0) -> dict:
    params_list = generate_inputs(
        source="earth", target=planet,
        num_missions=n_missions, seed=7, success_ratio=0.5,
    )

    t0 = time.perf_counter()
    for p in params_list:
        run_synthetic(p, time_step=time_step)
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in params_list:
        run_synthetic_numba(p, time_step=time_step)
    t_jit = time.perf_counter() - t0

    return {
        "planet": planet,
        "n_missions": n_missions,
        "t_orig": t_orig,
        "t_jit": t_jit,
        "speedup": t_orig / t_jit if t_jit > 0 else float("inf"),
    }


def main():
    print("=" * 70)
    print("  NUMBA JIT BENCHMARK")
    print("=" * 70)

    print("\n▸ Warming up JIT compilation (excluded from timing)...")
    jit_physics.warmup()
    # Also warm up via one real call through the wrapper, since rk4_step_nb
    # etc. get specialized per actual call-site argument types on first use.
    warm_params = generate_inputs(source="earth", target="mars", num_missions=1, seed=1)[0]
    run_synthetic_numba(warm_params, time_step=54000.0)
    print("  Done.")

    # Increasing transfer duration -> increasing substep count
    plan = [
        ("mars", 5),
        ("saturn", 3),
        ("uranus", 2),
    ]

    results = []
    for planet, n in plan:
        print(f"\n▸ Benchmarking {n} missions: earth → {planet}")
        r = bench_planet(planet, n)
        results.append(r)
        print(
            f"  Original : {r['t_orig']:.2f}s ({r['t_orig']/r['n_missions']:.3f}s/mission)"
        )
        print(
            f"  Numba JIT: {r['t_jit']:.2f}s ({r['t_jit']/r['n_missions']:.3f}s/mission)"
        )
        print(f"  Speedup  : {r['speedup']:.2f}x")

    print(f"\n{'=' * 70}")
    print(f"  {'Planet':<10} {'Orig s/mission':>16} {'JIT s/mission':>16} {'Speedup':>10}")
    print(f"  {'-'*54}")
    for r in results:
        print(
            f"  {r['planet']:<10} "
            f"{r['t_orig']/r['n_missions']:>16.3f} "
            f"{r['t_jit']/r['n_missions']:>16.3f} "
            f"{r['speedup']:>9.2f}x"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
