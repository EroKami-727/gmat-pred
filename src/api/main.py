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
REPORTS_DIR = Path("reports/ablation")

# Global state for training jobs
_processes: dict[str, subprocess.Popen] = {}
_log_queues: dict[str, list[str]] = {}

# Thread pool for blocking inference
_executor = ThreadPoolExecutor(max_workers=4)

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

@app.get("/api/simulator/missions")
async def sample_missions(
    data: str = Query("data/merged/missions.parquet"),
    n: int = Query(12),
    seed: int = Query(42),
):
    """Return N random mission IDs and their true labels from the dataset."""
    data_path = Path(data)
    if not data_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Dataset not found: {data}"})

    def _load():
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(data_path)
        cols = ["mission_id", "label", "failure_type"]
        available = [c for c in cols if c in pf.schema_arrow.names]
        tbl = pf.read(columns=available)
        df = tbl.to_pandas()
        summary = df.groupby("mission_id").first().reset_index()
        rng = np.random.default_rng(seed)
        sample = summary.sample(min(n, len(summary)), random_state=int(rng.integers(0, 9999)))
        return [
            {
                "mission_id": int(row["mission_id"]),
                "label": int(row["label"]) if "label" in row else None,
                "failure_type": str(row["failure_type"]) if "failure_type" in row else "unknown",
            }
            for _, row in sample.iterrows()
        ]

    loop = asyncio.get_event_loop()
    missions = await loop.run_in_executor(_executor, _load)
    return {"missions": missions}


@app.get("/api/simulator/stream")
async def simulator_stream(
    mission_id: int = Query(...),
    data: str = Query("data/merged/missions.parquet"),
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
    cache = _load_model_cached()
    if not cache:
        async def _no_model():
            yield f"data: {json.dumps({'type': 'error', 'text': 'No trained model found. Train a model first.'})}\n\n"
        return StreamingResponse(_no_model(), media_type="text/event-stream")

    data_path = Path(data)
    if not data_path.exists():
        async def _no_data():
            yield f"data: {json.dumps({'type': 'error', 'text': f'Dataset not found: {data}'})}\n\n"
        return StreamingResponse(_no_data(), media_type="text/event-stream")

    def _load_mission():
        import pyarrow.parquet as pq
        import pandas as pd
        pf = pq.ParquetFile(data_path)
        needed = list(set(FEATURE_COLS + ["mission_id", "label", "failure_type", "elapsed_secs"]))
        needed = [c for c in needed if c in pf.schema_arrow.names]
        frames = []
        for batch in pf.iter_batches(batch_size=500_000, columns=needed):
            df = batch.to_pandas()
            chunk = df[df["mission_id"] == mission_id]
            if len(chunk):
                frames.append(chunk)
        if not frames:
            return None, None, None
        mission_df = pd.concat(frames).sort_values("elapsed_secs").reset_index(drop=True)
        # Downsample (factor 15)
        mission_df = mission_df.iloc[::15].reset_index(drop=True)
        features = mission_df[FEATURE_COLS].values.astype(np.float32)
        label = int(mission_df["label"].iloc[0])
        failure_type = str(mission_df.get("failure_type", pd.Series(["unknown"])).iloc[0])
        return features, label, failure_type

    model = cache["model"]
    scaler = cache["scaler"]
    device = cache["device"]
    arch = cache["arch"]

    def _infer_step(features_so_far: np.ndarray) -> float:
        scaled = scaler.transform(features_so_far)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features_so_far)], dtype=torch.long)
        with torch.no_grad():
            if arch == "transformer":
                mask = torch.zeros(1, len(features_so_far), dtype=torch.bool, device=device)
                logit = model(x, mask)
            else:
                logit = model(x, lengths)
        return float(torch.sigmoid(logit).item())

    loop = asyncio.get_event_loop()
    features, true_label, failure_type = await loop.run_in_executor(_executor, _load_mission)

    if features is None:
        async def _not_found():
            yield f"data: {json.dumps({'type': 'error', 'text': f'Mission {mission_id} not found'})}\n\n"
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    delay_s = step_delay_ms / 1000.0
    total_steps = len(features)

    # Emit info header
    header = {"type": "info", "total_steps": total_steps, "true_label": true_label,
               "failure_type": failure_type, "mission_id": mission_id}

    async def generate():
        yield f"data: {json.dumps(header)}\n\n"

        canceled = False
        final_prob = 0.0
        # Stream every 5th step for speed, always include last
        stride = max(1, total_steps // 60)

        for i in range(0, total_steps):
            if i % stride != 0 and i != total_steps - 1:
                continue

            prob = 1.0 - await loop.run_in_executor(_executor, _infer_step, features[:i + 1])  # model outputs P(success); invert to P(fail)
            elapsed_pct = round((i + 1) / total_steps, 4)
            final_prob = prob

            event = {"type": "step", "timestep": i + 1, "elapsed_pct": elapsed_pct,
                     "probability": round(prob, 4)}
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(delay_s)

            # Early-exit cancellation — only after model has seen enough data to be reliable
            if prob >= threshold and not canceled and elapsed_pct >= min_elapsed_pct:
                canceled = True
                cancel_event = {"type": "cancel", "timestep": i + 1,
                                "elapsed_pct": elapsed_pct, "probability": round(prob, 4)}
                yield f"data: {json.dumps(cancel_event)}\n\n"
                break

        was_correct = (canceled and true_label == 0) or (not canceled and true_label == 1)
        done_event = {"type": "done", "true_label": true_label, "final_prob": round(final_prob, 4),
                      "was_correct": was_correct, "canceled": canceled}
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
