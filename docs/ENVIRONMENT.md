# Environment

## The interpreter

Everything in this repository runs under **one** interpreter:

```
/home/haise/Coding/venvs/gmat-pred/bin/python3     # CPython 3.12.13, created by uv
```

`run_pipeline.sh` hardcodes it as `$VENV`, and `test_ml.py`'s docstring names it.
Nothing runs against the system Python.

Verify a working environment with:

```bash
/home/haise/Coding/venvs/gmat-pred/bin/python3 - <<'PY'
import numpy, pandas, torch, xgboost, sklearn, numba
print("numpy ", numpy.__version__)
print("torch ", torch.__version__, "cuda:", torch.cuda.is_available())
print("numba ", numba.__version__)
PY
```

Expected on the reference machine: numpy 2.5.2, torch 2.13.0+cu130 with CUDA
available, numba 0.67.0.

## Trap: do not create a repo-local `.venv/`

A `.venv/` directory used to sit at the repo root: a Python 3.14 stub holding
only `fastapi`, `uvicorn`, `click`, `pip` and their transitive deps — no numpy,
no torch, no xgboost. **It has been deleted.** Do not recreate one.

It was dangerous precisely because it looked right. Editors and language servers
auto-select a repo-local `.venv/` over anything else, so the Python extension
reported every scientific import as missing and any command run through the
editor's selected interpreter died with `ModuleNotFoundError: No module named
'numpy'` — while the same command run against the real interpreter worked fine.
It is gitignored, so it never appeared in a diff to explain itself.

`.venv/` stays in `.gitignore` so an accidental recreation is not committed. If
your editor reports numpy/torch/pandas as uninstalled, it has selected a stub
like that one: point the interpreter at the path at the top of this file.

## Rebuilding from scratch

```bash
uv venv --python 3.12 ~/Coding/venvs/gmat-pred
VIRTUAL_ENV=~/Coding/venvs/gmat-pred uv pip install -r requirements.txt
```

`requirements.txt` is pinned to the versions the reported results were produced
under. Installing `numba` does **not** force a numpy downgrade at these pins
(numba 0.67 supports numpy 2.5) — verified with `uv pip install --dry-run`.

## Dataset location

Bulk data lives outside the repo and is resolved through
`src/paths.py`, which reads `$ORBITGUARD_DATA` and falls back to the reference
machine's path. See that module, and set the variable if your data lives
elsewhere:

```bash
export ORBITGUARD_DATA=/media/Data/Coding/gmat-pred/data/merged_all_v2
```

## Verifying a working install

```bash
PYTHONPATH=. python test_api.py --quick   # serving layer, app mounted in-process
PYTHONPATH=. python test_ml.py --limit 300 --skip-synthetic   # models on held-out data
```

`test_api.py` needs no server running and no GPU; it will report the dataset as
unmounted rather than failing obscurely if `$ORBITGUARD_DATA` is wrong, which
makes it the fastest check that an environment is set up correctly.

## Optional / legacy

- `seaborn` is used only by `src/data_collection/eda_report.py`.
- `src/frontend/app.py` is a **legacy Streamlit prototype**, superseded by the
  React app in `frontend/`. `streamlit` is deliberately not in
  `requirements.txt`; that file is kept for provenance only.
