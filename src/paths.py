"""
Where the bulk data lives.

The mission tables are far too large to sit in the repository (the merged
mission table alone is ~13 GB), so they live on a separate volume. Until this
module existed, that volume's absolute path was pasted into nine different
files — as real module-level constants in `src/ml/prune_economics.py` and
`experiments/numba_jit/validate_against_real_jupiter.py`, and as copy-paste
usage examples in the docstrings of everything else.

That is fine on the machine the results were produced on and useless anywhere
else, which matters as soon as a reviewer or a collaborator tries to run the
pipeline. Resolution order here:

    1. $ORBITGUARD_DATA, if set
    2. DEFAULT_DATA_ROOT, the reference machine's path

so the reference machine keeps working untouched, and anyone else exports one
variable:

    export ORBITGUARD_DATA=/path/to/merged_all_v2
"""

from __future__ import annotations

import os
from pathlib import Path

# The reference machine. Kept as the fallback so existing invocations — and
# run_pipeline.sh — behave exactly as before when the variable is unset.
DEFAULT_DATA_ROOT = Path("/media/Data/Coding/gmat-pred/data/merged_all_v2")

#: Repository root, derived from this file's location rather than the process
#: working directory, so scripts work when invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Directory holding missions/params/summary parquet for the merged dataset."""
    env = os.environ.get("ORBITGUARD_DATA")
    return Path(env).expanduser() if env else DEFAULT_DATA_ROOT


def missions_parquet() -> Path:
    """Full per-timestep telemetry table (~13 GB, streamed, never loaded whole)."""
    return data_root() / "missions.parquet"


def params_parquet() -> Path:
    """One row per mission: the six launch-parameter offsets, keyed by sim_id."""
    return data_root() / "mission_params.parquet"


def summary_parquet() -> Path:
    """One row per mission: outcome label and failure type, keyed by mission_id."""
    return data_root() / "summary.parquet"


def per_planet_dir() -> Path:
    """Compact per-planet .npz extracts. Small enough to live beside the code."""
    return REPO_ROOT / "data" / "per_planet"


def require(path: Path, hint: str = "") -> Path:
    """
    Fail loudly and usefully when data is missing.

    A bare FileNotFoundError deep inside pyarrow does not tell someone that they
    need to set an environment variable, so say so.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Set ORBITGUARD_DATA to the directory holding the merged dataset "
            f"(currently resolving to {data_root()}).\n"
            + (f"{hint}\n" if hint else "")
        )
    return path
