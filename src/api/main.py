"""
OrbitGuard FastAPI Backend
==========================
Serves the React dashboard with real ML data.

Endpoints
---------
GET  /api/health                → system/GPU status
GET  /api/status                → best model info + AUC
GET  /api/ablation              → ablation_results.json (or 404)
GET  /api/metrics               → per-epoch training metrics
POST /api/train/start           → spawn training subprocess
GET  /api/train/stream/{job_id} → SSE stream of training stdout
GET  /api/train/stop/{job_id}   → terminate training job
GET  /api/simulator/missions    → sample mission IDs from dataset
GET  /api/simulator/stream      → SSE realtime prediction per mission

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import pickle
import re
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.ml.dataset import TrajectoryDataset, FEATURE_COLS
from src.ml.model import TrajectoryLSTM, TrajectoryTransformer
from src.ml.planet_router import PlanetRouter
# The API surfaces every target it can serve, Moon included; the
# seven-target study set is a reporting concept, not a serving one.
from src.ml.planet_config import SERVING_TARGETS as ALL_PLANETS, OPERATING_FRAC
from src.api.trajectory_gen import hohmann_c3, optimal_phase_deg, PLANET_DATA
from src.paths import missions_parquet
from src.api.mission_builder import build_mission, nominal_for, dispersion_scale


# ═══════════════════════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="OrbitGuard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
PLANET_MODELS_DIR = Path("models/per_planet")
REPORTS_DIR = Path("reports/ablation")

# Global state for training jobs
_processes: dict[str, subprocess.Popen] = {}
_log_queues: dict[str, list[str]] = {}

# Thread pool for blocking inference
_executor = ThreadPoolExecutor(max_workers=4)

# Per-planet router (lazy-loaded, one instance)
_router: PlanetRouter | None = None

# In-memory store for user-generated missions (mission_id → trajectory/features dict)
_gen_cache: dict[str, dict] = {}
_gen_counter = 0


def _load_router() -> PlanetRouter:
    global _router
    if _router is None:
        _router = PlanetRouter(PLANET_MODELS_DIR)
    return _router

# Regex to parse train.py epoch line:
# Epoch 01 | Train Loss: 0.6234 (45.23%) | Val Loss: 0.5891 (51.12%) | F1=0.312 AUC=0.623 | lr=...
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|.*?Train Loss:\s*([\d.]+).*?Val Loss:\s*([\d.]+)"
    r".*?(?:F1=([\d.]+)\s+)?AUC=([\d.]+)"
)


# ═══════════════════════════════════════════════════════════════════════════
# Model Cache (load once for simulator)
# ═══════════════════════════════════════════════════════════════════════════

_model_cache: dict = {}  # {"model": ..., "scaler": ..., "arch": ...}


def _find_best_model() -> tuple[Optional[Path], Optional[Path], str]:
    """Return (model_path, scaler_path, arch) for the best available model."""
    for arch in ["transformer", "lstm"]:
        for subdir in ["", "transformer_production/", "lstm_production/"]:
            mp = MODELS_DIR / subdir / f"best_model_{arch}_binary.pt"
            sp = MODELS_DIR / subdir / f"scaler_{arch}_binary.pkl"
            if mp.exists() and sp.exists():
                return mp, sp, arch
    return None, None, "transformer"


def _load_model_cached() -> dict:
    if _model_cache:
        return _model_cache

    mp, sp, arch = _find_best_model()
    if mp is None:
        return {}

    with open(sp, "rb") as f:
        scaler = pickle.load(f)

    input_dim = len(FEATURE_COLS)
    if arch == "transformer":
        model = TrajectoryTransformer(input_dim=input_dim, output_dim=1, task="binary")
    else:
        model = TrajectoryLSTM(input_dim=input_dim, output_dim=1, task="binary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    _model_cache.update({"model": model, "scaler": scaler, "arch": arch,
                          "device": device, "model_path": str(mp)})
    return _model_cache


# ═══════════════════════════════════════════════════════════════════════════
# System Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info: dict = {"status": "ok", "device": device}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


@app.get("/api/system")
async def system_snapshot():
    """Realtime system snapshot — polled by Overview terminal every 5s."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snap: dict = {"device": device, "timestamp": __import__("time").strftime("%H:%M:%S")}
    if device == "cuda":
        snap["gpu"] = torch.cuda.get_device_name(0)
        snap["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        snap["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        snap["gpu_util_pct"] = round(torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, 1)

    # Active training jobs
    active_jobs = [jid for jid, proc in _processes.items() if proc.poll() is None]
    snap["active_training_jobs"] = active_jobs

    # Latest ablation result
    abl_path = REPORTS_DIR / "ablation_results.json"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        snap["ablation_fractions_done"] = len(abl)
        snap["ablation_best_auc"] = round(max(r["test_auc"] for r in abl), 4) if abl else None
    else:
        snap["ablation_fractions_done"] = 0
        snap["ablation_best_auc"] = None

    # Model info
    mp, sp, arch = _find_best_model()
    snap["model_exists"] = mp is not None
    snap["arch"] = arch if mp else None

    cache = _load_model_cached()
    if cache:
        snap["model_params"] = sum(p.numel() for p in cache["model"].parameters())

    return snap


@app.get("/api/status")
async def status():
    mp, sp, arch = _find_best_model()
    if mp is None:
        return {"model_exists": False, "arch": None, "best_auc": None}

    best_auc = None
    for subdir in ["", "transformer_production/", "lstm_production/"]:
        metrics_path = MODELS_DIR / subdir / f"metrics_{arch}_binary.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            if metrics:
                best_auc = round(max(m.get("auc", 0) for m in metrics), 4)
            break

    total_params = None
    try:
        cache = _load_model_cached()
        if cache:
            total_params = sum(p.numel() for p in cache["model"].parameters())
    except Exception:
        pass

    return {
        "model_exists": True,
        "arch": arch,
        "model_path": str(mp),
        "best_auc": best_auc,
        "total_params": total_params,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ML Data Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/ablation")
async def get_ablation():
    path = REPORTS_DIR / "ablation_results.json"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No ablation results yet.",
                     "hint": "Run: python3 -m src.ml.ablation --data data/merged/missions.parquet --epochs 30"}
        )
    with open(path) as f:
        return json.load(f)


@app.get("/api/metrics")
async def get_metrics(arch: str = Query("transformer"), task: str = Query("binary")):
    for a in [arch, "transformer", "lstm"]:
        for subdir in ["", f"{a}_production/"]:
            path = MODELS_DIR / subdir / f"metrics_{a}_{task}.json"
            if path.exists():
                with open(path) as f:
                    return {"arch": a, "metrics": json.load(f)}
    return JSONResponse(status_code=404, content={"error": "No metrics found. Train a model first."})


# ═══════════════════════════════════════════════════════════════════════════
# Generic Job Helper
# ═══════════════════════════════════════════════════════════════════════════

def _start_job(cmd: list[str]) -> str:
    """Launch a subprocess, stream its stdout into the shared log queue, return job_id."""
    job_id = str(uuid.uuid4())[:8]
    _log_queues[job_id] = []

    def run():
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(Path.cwd()),
        )
        _processes[job_id] = proc
        for line in proc.stdout:
            _log_queues[job_id].append(line.rstrip())
        proc.wait()
        _log_queues[job_id].append(f"__DONE__:{proc.returncode}")

    threading.Thread(target=run, daemon=True).start()
    return job_id


# Generic stream / stop aliases (any job_id, not just training)
@app.get("/api/jobs/stream/{job_id}")
async def stream_job(job_id: str):
    return await stream_training(job_id)


@app.get("/api/jobs/stop/{job_id}")
async def stop_job(job_id: str):
    return await stop_training(job_id)


# ═══════════════════════════════════════════════════════════════════════════
# Training Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/train/start")
async def start_training(config: dict):
    job_id = str(uuid.uuid4())[:8]

    arch_label = config.get("arch", "TrajectoryTransformer")
    model_key = "transformer" if "Transformer" in arch_label else "lstm"
    lr = config.get("lr", "1e-3")
    batch = config.get("batch", 32)
    epochs = config.get("epochs", 30)
    early_exit = config.get("earlyExit", "1.0")
    hidden = config.get("hidden", 128)
    data = config.get("data", "data/merged/missions.parquet")
    output_dir = config.get("outputDir", f"models/{model_key}_production")

    cmd = [
        "python3", "-m", "src.ml.train",
        "--data", str(data),
        "--model", model_key,
        "--lr", str(lr),
        "--batch-size", str(batch),
        "--epochs", str(epochs),
        "--early-exit", str(early_exit),
        "--hidden-dim", str(hidden),
        "--output-dir", str(output_dir),
    ]

    _log_queues[job_id] = []

    def run():
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(Path.cwd()),
        )
        _processes[job_id] = proc
        for line in proc.stdout:
            _log_queues[job_id].append(line.rstrip())
        proc.wait()
        _log_queues[job_id].append(f"__DONE__:{proc.returncode}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return {"job_id": job_id, "status": "started", "command": " ".join(cmd)}


@app.get("/api/train/stream/{job_id}")
async def stream_training(job_id: str):
    if job_id not in _log_queues:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    async def generate():
        idx = 0
        while True:
            queue = _log_queues.get(job_id, [])
            while idx < len(queue):
                line = queue[idx]
                idx += 1

                if line.startswith("__DONE__:"):
                    code = int(line.split(":")[1])
                    yield f"data: {json.dumps({'type': 'done', 'exit_code': code})}\n\n"
                    return

                # Try to parse structured epoch data
                m = _EPOCH_RE.search(line)
                if m:
                    epoch_data = {
                        "type": "epoch",
                        "epoch": int(m.group(1)),
                        "train_loss": float(m.group(2)),
                        "val_loss": float(m.group(3)),
                        "f1": float(m.group(4)) if m.group(4) else None,
                        "auc": float(m.group(5)),
                    }
                    yield f"data: {json.dumps(epoch_data)}\n\n"

                yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"

            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/train/stop/{job_id}")
async def stop_training(job_id: str):
    proc = _processes.get(job_id)
    if proc and proc.poll() is None:
        proc.terminate()
        _log_queues.get(job_id, []).append("__DONE__:-1")
        return {"status": "stopped"}
    return {"status": "not_running"}


# ═══════════════════════════════════════════════════════════════════════════
# Simulator Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/simulator/router/status")
async def router_status():
    """Return which regime models are loaded and calibrated thresholds."""
    return _load_router().status()


@app.post("/api/simulator/router/reload")
async def router_reload():
    """Hot-reload regime models and thresholds after training completes."""
    global _router
    _router = None  # Force re-init on next request
    status = _load_router().status()
    return {"reloaded": True, **status}


@app.get("/api/simulator/thresholds")
async def get_thresholds():
    """Return per-target calibrated P(fail) thresholds for the simulator UI."""
    router = _load_router()
    return {target: router.threshold_for(target) for target in sorted(ALL_PLANETS)}


def _dataset_path(data: str | None) -> Path:
    """
    Resolve the mission table a simulator request should read.

    `data` used to default to the literal "data/merged/missions.parquet" — a
    March generation the per-planet models were never trained on — while the
    frontend papered over it by sending an absolute path from its own machine on
    every request. Both halves were wrong: the server default pointed at the
    wrong dataset, and the browser had no business knowing a filesystem path at
    all (it breaks the moment the dashboard is served from anywhere but the box
    holding the data).

    The server now resolves its own default from $ORBITGUARD_DATA. An explicit
    `data` parameter still overrides, for comparing against an older generation.
    """
    return Path(data) if data else missions_parquet()


@app.get("/api/simulator/missions")
async def sample_missions(
    data: str | None = Query(None, description="Override the dataset path; defaults to the configured root ($ORBITGUARD_DATA)."),
    n: int = Query(12),
    seed: int = Query(42),
    target: str = Query("ALL"),
):
    """
    Return N random mission IDs and their true labels from the dataset.
    Streams in small batches and stops early — never reads the whole file into RAM.
    """
    data_path = _dataset_path(data)
    if not data_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Dataset not found: {data_path}"})

    def _load():
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pyarrow as pa
        pf = pq.ParquetFile(data_path)
        cols = ["mission_id", "label", "failure_type", "target_body"]
        available = [c for c in cols if c in pf.schema_arrow.names]

        want_target = target.lower() if target != "ALL" else None
        seen: dict[int, dict] = {}
        stop_at = n * 20

        # Read row groups in RANDOM order, not file order. Missions are written
        # in mission_id order and the first block of each planet is the
        # targeter's near-nominal seeds: reading from the start returned a pool
        # that was mostly successes (Jupiter/Saturn/Neptune had zero failures in
        # the first 240) and, for Uranus, only the grazing failures the model is
        # known to miss. Random row groups give a pool that reflects the dataset.
        rng = np.random.default_rng(seed)
        groups = rng.permutation(pf.num_row_groups)
        # Cap per group so the pool spans many regions of the file: a single
        # row group holds a contiguous mission_id range, and taking a whole one
        # would just move the bias rather than remove it.
        per_group_cap = max(2, stop_at // 8)

        for gi in groups:
            tbl = pf.read_row_group(int(gi), columns=available)
            if want_target:
                mask = pc.equal(pc.utf8_lower(tbl.column("target_body")),
                                pa.scalar(want_target))
                tbl = tbl.filter(mask)
            if tbl.num_rows == 0:
                continue
            df = tbl.to_pandas()
            taken = 0
            for mid, grp in df.groupby("mission_id", sort=False):
                mid = int(mid)
                if mid not in seen:
                    row = grp.iloc[0]
                    seen[mid] = {
                        "mission_id":   mid,
                        "label":        int(row["label"])         if "label"        in row.index else None,
                        "failure_type": str(row["failure_type"])  if "failure_type" in row.index else "unknown",
                        "target":       str(row["target_body"])   if "target_body"  in row.index else "unknown",
                    }
                    taken += 1
                    if taken >= per_group_cap:
                        break
            if len(seen) >= stop_at:
                break

        missions = list(seen.values())
        if not missions:
            return []
        idx = rng.choice(len(missions), min(n, len(missions)), replace=False)
        return [missions[i] for i in idx]

    loop = asyncio.get_event_loop()
    missions = await loop.run_in_executor(_executor, _load)
    return {"missions": missions}


@app.get("/api/simulator/model_report")
async def model_report():
    """
    Live per-planet production metrics, read from each model's meta.json.

    Serving this rather than hardcoding numbers in the dashboard keeps the
    report honest after every retrain — an earlier hardcoded LOTO table kept
    displaying results for a model architecture that no longer exists.
    """
    router = _load_router()
    out = []
    for planet in ALL_PLANETS:
        c = router._caches.get(planet)
        if not c:
            out.append({"planet": planet, "trained": False})
            continue
        meta = c["meta"]
        at_op = meta.get("test", {}).get(f"{OPERATING_FRAC:.2f}", {})
        recal = meta.get("recalibrated_test") or meta.get("test_at_operating_point", {})
        out.append({
            "planet": planet, "trained": True,
            "auc": recal.get("auc", at_op.get("auc")),
            "f1": recal.get("f1", at_op.get("f1")),
            "recall": recal.get("recall"),
            "precision": recal.get("precision"),
            "mode_acc": at_op.get("mode_acc_on_failures"),
            "threshold": meta.get("threshold"),
            "n_missions": meta.get("n_missions"),
            "downsample": meta.get("downsample"),
            "has_assist": c.get("assist") is not None,
        })
    return {"operating_frac": OPERATING_FRAC, "planets": out}


@app.get("/api/simulator/planet_info")
async def planet_info():
    """
    Planet parameters for the mission creator UI.

    `nominal` is the Hohmann reference the creator's offsets are relative to,
    and `sigma` is the 1-sigma dispersion the dataset sampled — useful as slider
    scale. Dispersions are small (Venus dv_V sigma = 0.003 km/s), so an offset of
    a few tenths of a km/s is tens of sigma out and will be flagged
    out-of-distribution even though it propagates correctly.
    """
    out = {}
    for name, data in PLANET_DATA.items():
        entry = {
            "a_au":              data["a_au"],
            "hohmann_c3":        hohmann_c3(name),
            "optimal_phase_deg": optimal_phase_deg(name),
            "soi_km":            data["soi_km"],
            "model_trained":     _load_router().supports(name),
        }
        try:
            entry["nominal"] = nominal_for(name)
            entry["sigma"] = dispersion_scale(name)
        except Exception:                                        # noqa: BLE001
            pass
        out[name] = entry
    return out


@app.post("/api/simulator/generate")
async def generate_mission_endpoint(body: dict):
    """
    Build a mission from user parameters and return a "gen_" mission_id that the
    trajectory/stream endpoints accept.

    Uses the same propagator and feature code as the training dataset
    (src/api/mission_builder), so generated missions are in-distribution and the
    per-planet models can score them. Parameters are offsets from the Hohmann
    nominal; see /api/simulator/planet_info for nominal values and the 1-sigma
    dispersions the dataset sampled.
    """
    global _gen_counter
    target = str(body.get("target", "mars")).lower()

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(body.get(key, default))
        except (TypeError, ValueError):
            return default

    def _gen():
        return build_mission(
            target,
            dv_v_offset=_f("dv_v_offset"), dv_n_offset=_f("dv_n_offset"),
            dv_b_offset=_f("dv_b_offset"), raan_offset=_f("raan_offset"),
            aop_offset=_f("aop_offset"),   inc_offset=_f("inc_offset"),
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _gen)
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=400,
                            content={"error": f"{type(e).__name__}: {e}"})

    _gen_counter += 1
    mission_id = f"gen_{_gen_counter}"
    _gen_cache[mission_id] = result
    result_out = {k: v for k, v in result.items() if k != "features"}
    result_out["mission_id"] = mission_id
    return result_out


@app.get("/api/simulator/trajectory")
async def get_trajectory(
    mission_id: str = Query(...),
    data: str | None = Query(None, description="Override the dataset path; defaults to the configured root ($ORBITGUARD_DATA)."),
    downsample: int = Query(2),
):
    """
    Return pre-loaded trajectory positions and telemetry for orbital map visualization.
    Positions are in the synodic frame (rel_x, rel_y relative to target at origin).
    """
    # Serve from in-memory cache for generated missions
    if isinstance(mission_id, str) and mission_id.startswith("gen_"):
        cached = _gen_cache.get(mission_id)
        if cached is None:
            return JSONResponse(status_code=404, content={"error": f"Generated mission {mission_id} not found"})
        return {
            "mission_id":   mission_id,
            "total_steps":  cached["total_steps"],
            "label":        cached["label"],
            "failure_type": cached["failure_type"],
            "target_body":  cached["target_body"],
            "positions":    cached["positions"],
            "telemetry":    cached["telemetry"],
        }

    mid_int   = int(mission_id)
    data_path = _dataset_path(data)
    if not data_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Dataset not found: {data_path}"})

    def _load():
        import pyarrow.dataset as ds_arrow
        import pandas as pd
        wanted = [
            "mission_id", "elapsed_secs", "label", "failure_type", "target_body",
            "rel_x", "rel_y", "rel_z",
            "vel_mag", "ecc", "spec_energy", "earth_rmag", "fpa_deg", "norm_target_dist",
        ]
        dataset   = ds_arrow.dataset(str(data_path), format="parquet")
        available = [c for c in wanted if c in dataset.schema.names]
        tbl = dataset.to_table(
            filter=ds_arrow.field("mission_id") == mid_int,
            columns=available,
        )
        if tbl.num_rows == 0:
            return None
        mdf = tbl.to_pandas().sort_values("elapsed_secs").reset_index(drop=True)
        mdf = mdf.iloc[::downsample].reset_index(drop=True)
        total = len(mdf)

        def safe(row, col, dec=4):
            v = row.get(col, 0.0)
            try:
                return round(float(v), dec)
            except Exception:
                return 0.0

        positions, telemetry = [], []
        for i in range(total):
            row = mdf.iloc[i]
            ep = round((i + 1) / total, 5)
            positions.append({
                "step": i,
                "elapsed_pct": ep,
                "rel_x": safe(row, "rel_x", 2),
                "rel_y": safe(row, "rel_y", 2),
            })
            telemetry.append({
                "step": i,
                "elapsed_pct": ep,
                "vel_mag":         safe(row, "vel_mag",         3),
                "ecc":             safe(row, "ecc",             5),
                "spec_energy":     safe(row, "spec_energy",     2),
                "earth_rmag":      safe(row, "earth_rmag",      1),
                "fpa_deg":         safe(row, "fpa_deg",         4),
                "norm_target_dist":safe(row, "norm_target_dist",5),
            })

        label = int(mdf["label"].iloc[0]) if "label" in mdf.columns else None
        failure_type = str(mdf["failure_type"].iloc[0]) if "failure_type" in mdf.columns else "unknown"
        target_body = str(mdf["target_body"].iloc[0]) if "target_body" in mdf.columns else "unknown"

        return {
            "mission_id": mission_id,
            "total_steps": total,
            "label": label,
            "failure_type": failure_type,
            "target_body": target_body,
            "positions": positions,
            "telemetry": telemetry,
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _load)
    if result is None:
        return JSONResponse(status_code=404, content={"error": f"Mission {mission_id} not found"})
    return result


@app.get("/api/simulator/stream")
async def simulator_stream(
    mission_id: str = Query(...),
    data: str | None = Query(None, description="Override the dataset path; defaults to the configured root ($ORBITGUARD_DATA)."),
    step_delay_ms: int = Query(80),
    threshold: float = Query(0.5),
    min_elapsed_pct: float = Query(0.4, description="Minimum trajectory fraction before cancellation is allowed (match training early_exit_frac)"),
):
    """
    SSE stream that replays a mission timestep-by-timestep through the model.
    Emits:
      {"type": "step",   "timestep": N, "elapsed_pct": 0.12, "probability": 0.34}
      {"type": "cancel", "timestep": N, "elapsed_pct": 0.40, "probability": 0.87}
      {"type": "done",   "true_label": 0, "final_prob": 0.91, "was_correct": true}
    """
    router = _load_router()
    if not router.is_available():
        async def _no_model():
            yield f"data: {json.dumps({'type': 'error', 'text': 'No trained model found. Train a model first.'})}\n\n"
        return StreamingResponse(_no_model(), media_type="text/event-stream")

    # loop must be in scope for the generate() closure regardless of which branch we take
    loop = asyncio.get_event_loop()

    # ── Load features: generated mission (from cache) or parquet ──────────
    if mission_id.startswith("gen_"):
        cached = _gen_cache.get(mission_id)
        if cached is None:
            async def _no_gen():
                yield f"data: {json.dumps({'type': 'error', 'text': f'Generated mission {mission_id} not found'})}\n\n"
            return StreamingResponse(_no_gen(), media_type="text/event-stream")
        features     = cached["features"]
        true_label   = cached["label"]
        failure_type = cached["failure_type"]
        target_body  = cached["target_body"]
    else:
        mid_int   = int(mission_id)
        data_path = _dataset_path(data)
        if not data_path.exists():
            async def _no_data():
                yield f"data: {json.dumps({'type': 'error', 'text': f'Dataset not found: {data_path}'})}\n\n"
            return StreamingResponse(_no_data(), media_type="text/event-stream")

        def _load_mission():
            import pyarrow.dataset as ds_arrow
            import pandas as pd
            needed = list(set(FEATURE_COLS + ["mission_id", "label", "failure_type", "elapsed_secs", "target_body"]))
            dataset = ds_arrow.dataset(str(data_path), format="parquet")
            schema_names = dataset.schema.names
            needed = [c for c in needed if c in schema_names]
            tbl = dataset.to_table(
                filter=ds_arrow.field("mission_id") == mid_int,
                columns=needed,
            )
            if tbl.num_rows == 0:
                return None, None, None, None
            mission_df   = tbl.to_pandas().sort_values("elapsed_secs").reset_index(drop=True)
            target_body  = str(mission_df["target_body"].iloc[0]).lower() if "target_body" in mission_df.columns else "unknown"
            # Cadence must match training exactly; the model records its own factor.
            ds = router.downsample_for_target(target_body)
            mission_df   = mission_df.iloc[::ds].reset_index(drop=True)
            feats        = mission_df[FEATURE_COLS].values.astype(np.float64)
            label        = int(mission_df["label"].iloc[0])
            failure_type = str(mission_df["failure_type"].iloc[0]) if "failure_type" in mission_df.columns else "unknown"
            return feats, label, failure_type, target_body

        features, true_label, failure_type, target_body = await loop.run_in_executor(_executor, _load_mission)

    if features is None:
        async def _not_found():
            yield f"data: {json.dumps({'type': 'error', 'text': f'Mission {mission_id} not found'})}\n\n"
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    delay_s = step_delay_ms / 1000.0
    total_steps = len(features)

    # Use calibrated per-target threshold; fall back to user-supplied value
    cal_threshold = router.threshold_for(target_body) if target_body else threshold
    effective_threshold = cal_threshold

    # Emit info header
    header = {"type": "info", "total_steps": total_steps, "true_label": true_label,
               "failure_type": failure_type, "mission_id": mission_id, "target_body": target_body,
               "model": "per_planet", "model_available": router.supports(target_body),
               "calibrated_threshold": round(effective_threshold, 4)}

    async def generate():
        yield f"data: {json.dumps(header)}\n\n"

        canceled = False
        final_prob = 0.0
        pred_mode, pred_mode_name, mode_conf = 0, "unknown", 0.0
        is_ood, ood_frac = False, 0.0
        stride = max(1, total_steps // 150)
        # Models are trained on prefixes up to OPERATING_FRAC of mission length.
        # Past that the input is out-of-distribution, so freeze the last valid
        # prediction rather than feeding longer sequences (which previously drove
        # every outer-planet mission to a false abort near 70%).
        infer_cutoff = int(total_steps * OPERATING_FRAC)

        for i in range(0, total_steps):
            if i % stride != 0 and i != total_steps - 1:
                continue

            elapsed_pct = round((i + 1) / total_steps, 4)

            if i <= infer_cutoff:
                res = await loop.run_in_executor(
                    _executor, router.predict, features[:i + 1], target_body)
                final_prob = res["p_fail"]
                pred_mode = res["failure_mode"]
                pred_mode_name = res["failure_mode_name"]
                mode_conf = res["mode_confidence"]
                is_ood = res["out_of_distribution"]
                ood_frac = res["ood_fraction"]

            event = {"type": "step", "timestep": i + 1, "elapsed_pct": elapsed_pct,
                     "probability": round(final_prob, 4),
                     "predicted_failure_mode": pred_mode_name,
                     "mode_confidence": round(mode_conf, 4),
                     "out_of_distribution": is_ood, "ood_fraction": ood_frac}
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(delay_s)

            # Abort check uses the last in-distribution probability (held constant
            # past the operating point). `is_ood` rides along as an advisory flag
            # for the UI rather than blocking the abort.
            if (final_prob >= effective_threshold and not canceled
                    and elapsed_pct >= min_elapsed_pct):
                canceled = True
                cancel_event = {"type": "cancel", "timestep": i + 1,
                                "elapsed_pct": elapsed_pct, "probability": round(final_prob, 4),
                                "threshold_used": round(effective_threshold, 4),
                                "predicted_failure_mode": pred_mode_name,
                                "mode_confidence": round(mode_conf, 4)}
                yield f"data: {json.dumps(cancel_event)}\n\n"
                break

        was_correct = (canceled and true_label == 0) or (not canceled and true_label == 1)
        mode_correct = (pred_mode_name == str(failure_type).lower()) if canceled else None
        done_event = {"type": "done", "true_label": true_label, "final_prob": round(final_prob, 4),
                      "was_correct": was_correct, "canceled": canceled,
                      "predicted_failure_mode": pred_mode_name,
                      "actual_failure_type": failure_type,
                      "mode_correct": mode_correct,
                      "out_of_distribution": is_ood, "ood_fraction": ood_frac}
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Endpoints — Baselines, Multi-Seed, Arch Ablation, Tables
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/experiments/baselines/start")
async def start_baselines(config: dict):
    fracs = str(config.get("exitFracs", "0.1 0.2 0.3 0.4 0.6 1.0")).split()
    cmd = [
        "python3", "-m", "src.ml.baselines",
        "--data",              str(config.get("data", "data/merged/missions.parquet")),
        "--exit-fracs",        *fracs,
        "--seed",              str(config.get("seed", 42)),
        "--downsample-factor", str(config.get("downsampleFactor", 15)),
    ]
    job_id = _start_job(cmd)
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/experiments/baselines/results")
async def get_baselines_results():
    path = Path("reports/baselines/baseline_results.json")
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "No baseline results yet. Run the baselines experiment first."})
    return json.loads(path.read_text())


@app.post("/api/experiments/multi_seed/start")
async def start_multi_seed(config: dict):
    seeds = str(config.get("seeds", "42 123 456 789 1024")).split()
    cmd = [
        "python3", "-m", "src.ml.multi_seed",
        "--data",        str(config.get("data", "data/merged/missions.parquet")),
        "--seeds",       *seeds,
        "--early-exit",  str(config.get("earlyExit", "0.4")),
        "--epochs",      str(config.get("epochs", 30)),
        "--batch-size",  str(config.get("batchSize", 32)),
        "--lr",          str(config.get("lr", "1e-3")),
        "--output-dir",  "reports/multi_seed",
    ]
    job_id = _start_job(cmd)
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/experiments/multi_seed/results")
async def get_multi_seed_results():
    path = Path("reports/multi_seed/summary.json")
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "No multi-seed results yet. Run the multi-seed experiment first."})
    return json.loads(path.read_text())


@app.post("/api/experiments/arch_ablation/start")
async def start_arch_ablation(config: dict):
    variants = config.get("variants", ["full_transformer", "no_cls_token", "no_pos_encoding", "no_context_features", "lstm"])
    cmd = [
        "python3", "-m", "src.ml.arch_ablation",
        "--data",        str(config.get("data", "data/merged/missions.parquet")),
        "--early-exit",  str(config.get("earlyExit", "0.4")),
        "--epochs",      str(config.get("epochs", 30)),
        "--seed",        str(config.get("seed", 42)),
        "--variants",    *variants,
        "--output-dir",  "reports/arch_ablation",
    ]
    job_id = _start_job(cmd)
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/experiments/arch_ablation/results")
async def get_arch_ablation_results():
    path = Path("reports/arch_ablation/arch_ablation_results.json")
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "No arch ablation results yet. Run the arch ablation experiment first."})
    return json.loads(path.read_text())


@app.post("/api/experiments/tables/start")
async def start_tables(config: dict):
    cmd = [
        "python3", "-m", "src.ml.results_table",
        "--ablation",   "reports/ablation/ablation_results.json",
        "--baselines",  "reports/baselines/baseline_results.json",
        "--arch",       "reports/arch_ablation/arch_ablation_results.json",
        "--multi-seed", "reports/multi_seed/summary.json",
        "--output-dir", "reports/tables",
    ]
    job_id = _start_job(cmd)
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/experiments/tables")
async def get_tables():
    """Return the content of all generated table files (.md and .tex)."""
    tables_dir = Path("reports/tables")
    result = {}
    for name in ["main_comparison", "arch_ablation", "multi_seed"]:
        md_path  = tables_dir / f"{name}.md"
        tex_path = tables_dir / f"{name}.tex"
        result[name] = {
            "md":  md_path.read_text()  if md_path.exists()  else None,
            "tex": tex_path.read_text() if tex_path.exists() else None,
        }
    return result


@app.post("/api/experiments/interpretability/start")
async def start_interpretability(config: dict):
    cmd = [
        "python3", "-m", "src.ml.interpretability",
        "--data",               str(config.get("data",             "data/merged/missions.parquet")),
        "--model-path",         str(config.get("modelPath",        "models/transformer_production/best_model_transformer_binary.pt")),
        "--scaler-path",        str(config.get("scalerPath",       "models/transformer_production/scaler_transformer_binary.pkl")),
        "--n-sample-missions",  str(config.get("nSampleMissions",  10)),
        "--n-attr-missions",    str(config.get("nAttrMissions",    200)),
        "--n-failure-missions", str(config.get("nFailureMissions", 500)),
        "--early-exit",         str(config.get("earlyExit",        0.4)),
        "--seed",               str(config.get("seed",             42)),
        "--output-dir",         "reports/interpretability",
    ]
    job_id = _start_job(cmd)
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/experiments/interpretability/results")
async def get_interpretability_results():
    path = Path("reports/interpretability/summary.json")
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "No interpretability results yet. Run the analysis first."})
    return json.loads(path.read_text())
