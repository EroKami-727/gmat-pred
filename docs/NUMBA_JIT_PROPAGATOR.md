# Numba JIT Propagator — `experiments/numba_jit/`

Status: **validated, used in production for the Neptune dataset (10,000
missions, generated 2026-06-19).** Production `gmat_runner.py` is
unmodified — this is a parallel, opt-in fast path.

## Why this exists

Outer-planet mission generation was prohibitively slow. The compute cost
of `run_synthetic()` is driven by the number of RK4 substeps, which scales
with total flight duration (Hohmann transfer time), not with the recorded
telemetry cadence:

| Planet | Hohmann transfer time | Substeps/mission (≈900s deep-space step) |
|---|---:|---:|
| Mars | ~311 days | ~30,000 |
| Uranus | ~19 years | ~675,000 |
| Neptune | ~37 years | ~1,300,000 |

At 14 parallel workers, Uranus (10,000 missions) took **~12 hours**.
Neptune was projected at **15-24 hours** — impractical to leave a laptop
running for. The bottleneck is `_acceleration()` / `_rk4_step()` in
`src/data_collection/gmat_runner.py`, which calls `np.linalg.norm` and
numpy array ops on 3-element arrays millions of times per mission. For
arrays that small, numpy's per-call dispatch/allocation overhead dwarfs
the actual floating-point work.

## What it does

[Numba](https://numba.pydata.org/) JIT-compiles the hot substep loop to
native machine code, bypassing the Python interpreter and numpy's generic
array machinery entirely. Measured speedup at production cadence
(54,000s recording interval):

| Planet | Original | JIT | Speedup |
|---|---:|---:|---:|
| Mars | 2.50 s/mission | 0.07 s/mission | **37x** |
| Saturn | 16.8 s/mission | 0.60 s/mission | **28x** |
| Uranus | 37.3 s/mission | 1.42 s/mission | **26x** |

Neptune (10,000 missions, 14 workers) finished in **under an hour** with
the JIT engine, versus a projected 15-24 hours with the original.

## How it works

Only the actual bottleneck is touched. Everything else — parameter
generation (`generate_inputs`), outer recording loop, outcome
classification (success/failure_type), DataFrame schema, batching, and
parquet writing — is **byte-for-byte the same code path** as production.

```
experiments/numba_jit/
├── jit_physics.py                       Numba-compiled hot loop
├── runner_numba.py                      run_synthetic_numba() — drop-in
│                                         replacement for run_synthetic(),
│                                         reuses all production helpers
├── build_database_jit.py                CLI generation script, mirrors
│                                         src/data_collection/build_database.py
│                                         with run_func swapped to the JIT runner
├── validate.py                          Synthetic test missions vs production
├── validate_against_real_jupiter.py     Real production data vs JIT replay
└── benchmark.py                         Timing comparison across planets
```

### `jit_physics.py`

Pure-numeric, `@njit`-decorated mirrors of the production functions, using
plain floats/fixed-size arrays instead of `MissionConfig` attribute access
(Numba's `nopython` mode can't operate on arbitrary Python objects):

- `target_ephemeris_nb` ← mirrors `_target_ephemeris`
- `acceleration_nb` ← mirrors `_acceleration`
- `rk4_step_nb` ← mirrors `_rk4_step`
- `adaptive_dt_nb` ← mirrors `_adaptive_integration_dt`
- `propagate_segment_nb` ← the actual hot loop (mirrors the `while t <
  segment_end:` substep loop inside `run_synthetic`), including per-substep
  closest-approach tracking (`min_target_rmag`) so outcome classification
  stays identical to production.

### `runner_numba.py`

`run_synthetic_numba(params, time_step)` — same signature as production
`run_synthetic`. Imports the real `MissionConfig`, `_keplerian_to_cartesian`,
`_get_vnb_frame`, `_calculate_orbit_elements`, `_compute_physics_features`,
`_target_ephemeris`, `_target_velocity`, `_empty_failure`, and `COLUMNS`
directly from `src/data_collection/gmat_runner.py`. The only swap is the
substep propagation call: `propagate_segment_nb(...)` instead of the
production inline `while` loop.

### `build_database_jit.py`

Near line-for-line copy of `src/data_collection/build_database.py`, with
`run_func = partial(run_synthetic_numba, time_step=time_step)` instead of
`run_synthetic`. Same CLI flags, same multiprocessing/batching/parquet
logic. Does not support `--append` mode (wasn't needed for this use case;
add it by porting that block from `build_database.py` if needed later).

Usage:

```bash
python -m experiments.numba_jit.build_database_jit \
  --source earth --target neptune \
  --num-missions 10000 --output-dir data/neptune \
  --seed 42 --success-ratio 0.35 --batch-size 10 \
  --workers 14 --time-step 54000
```

## Validation

Two layers of evidence, in increasing order of strength:

### 1. Synthetic test missions (`validate.py`)

32 freshly generated test missions across 4 planets (Moon, Mars, Venus,
Mercury), covering both success and multiple failure_type outcomes.
**32/32 matched exactly** on row count, label, and failure_type. Max
floating-point position drift: 2.6e-4 km (sub-millimeter) — expected
IEEE754 reordering noise from JIT compilation over million-substep
integrations, not a defect.

### 2. Real production Jupiter data (`validate_against_real_jupiter.py`)

The strongest check: loads the **exact `MissionParams`** that produced the
already-generated, on-disk Jupiter dataset (10,000 missions, original
non-JIT pipeline), replays 50 of them through `run_synthetic_numba`, and
diffs against the real recorded telemetry for those same `sim_id`s.

- Sample spanned success, `orbit_too_high`, and `surface_impact` outcomes.
- **50/50 matched exactly** on row count, label, and failure_type.
- Max absolute position error: 0.036 km (36 m) over trajectories spanning
  **hundreds of millions of km** — relative error ≈ 5×10⁻¹¹. This is far
  below the systematic approximation error already inherent in the
  simplified 3-body/circular-ephemeris physics model itself.

**Why not bit-identical?** Numba compiles via LLVM with different
instruction selection/ordering than CPython + numpy's BLAS-backed
`np.linalg.norm`. IEEE754 floating-point arithmetic is not strictly
associative, so reordering operations over ~10⁵-10⁶ accumulated RK4 steps
produces small drift. This is universal to any from-scratch reimplementation
of a long iterative numeric integration — it does not indicate a logic bug,
which is why the *outcome* (label/failure_type), not bit-exact telemetry,
is the validation criterion that matters for dataset integrity.

### Re-running validation

```bash
# Synthetic missions across 4 planets
python -m experiments.numba_jit.validate

# Against real on-disk Jupiter data
python -m experiments.numba_jit.validate_against_real_jupiter \
  --real-dir /media/Data/Coding/gmat-pred/data/jupiter --n-sample 50
```

## Known gotcha: Numba cache and module path

Numba's `@njit(cache=True)` pickles a reference to the function's
*defining module name* into its on-disk cache (`__pycache__/*.nbi`/`.nbc`).
If `jit_physics.py` is ever imported under a different qualified name
(e.g. bare `jit_physics` vs `experiments.numba_jit.jit_physics`), stale
cache entries raise `ModuleNotFoundError` inside Numba's loader. Fix: wipe
`find experiments -name __pycache__ -exec rm -rf {} +` and let it
recompile. This is why all entry points import `jit_physics` via the
`experiments.numba_jit.jit_physics` package path, never a bare relative
import — multiprocessing workers must resolve the same qualified name as
the main process.

## What was NOT changed

- `src/data_collection/gmat_runner.py` and `build_database.py` are
  untouched.
- Mercury, Venus, Mars, Jupiter, Saturn, Uranus, and Moon datasets were
  already generated with the original (non-JIT) pipeline before this
  experiment existed — they were not regenerated.
- Only Neptune was generated with the JIT engine (`build_database_jit.py`),
  because validation against real Jupiter data passed first.
- `--append` mode is not implemented in `build_database_jit.py`.

## If you want to use this for future planets/re-runs

It's safe to use `build_database_jit.py` going forward for any new
interplanetary generation — the validation evidence above covers the
exact production cadence (54,000s) and a representative range of outcome
types. For a new planet not yet covered by validation (anything beyond
Mercury–Neptune, e.g. a non-default source body), re-run
`validate.py` with that planet added to the `planets` list before trusting
it for paper data.
