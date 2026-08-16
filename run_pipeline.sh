#!/usr/bin/env bash
# OrbitGuard per-planet rebuild — runs fully detached from the editor.
#
#   setsid nohup ./run_pipeline.sh > /dev/null 2>&1 &
#
# Survives VS Code restarts/crashes. Progress: tail -f .runlogs/pipeline.log
set -u

cd /home/haise/Coding/Projects/gmat-pred || exit 1

VENV=/home/haise/Coding/venvs/gmat-pred/bin/python3
DATA="${ORBITGUARD_DATA:-/media/Data/Coding/gmat-pred/data/merged_all_v2}/missions.parquet"
LOGDIR=.runlogs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/pipeline.log"

# Keep CPU pressure modest so the desktop stays responsive.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== pipeline start $(date -Is) pid=$$ ===" > "$LOG"

# ── Stage 1: extract any missing planet ───────────────────────────────────────
MISSING=()
for p in moon mercury venus mars jupiter saturn uranus neptune; do
    [ -f "data/per_planet/$p.npz" ] || MISSING+=("$p")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ">>> extracting: ${MISSING[*]}" >> "$LOG"
    nice -n 10 "$VENV" -m src.data_collection.extract_per_planet \
        --data "$DATA" --out-dir data/per_planet \
        --planets "${MISSING[@]}" >> "$LOG" 2>&1
    rc=$?
    echo ">>> extract exit=$rc" >> "$LOG"
    [ $rc -ne 0 ] && { echo "=== ABORT: extraction failed ===" >> "$LOG"; exit 1; }
else
    echo ">>> all planet extracts present" >> "$LOG"
fi

# ── Stage 2: train one model per planet ───────────────────────────────────────
echo ">>> training all planets $(date -Is)" >> "$LOG"
nice -n 10 "$VENV" -m src.ml.per_planet_train \
    --all --epochs 60 --data-dir data/per_planet \
    --out-root models/per_planet >> "$LOG" 2>&1
rc=$?
echo ">>> train exit=$rc" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== ABORT: training failed ===" >> "$LOG"; exit 1; }

# ── Stage 3: tree assist + threshold calibration ──────────────────────────────
echo ">>> assist + recalibrate $(date -Is)" >> "$LOG"
nice -n 10 "$VENV" -m src.ml.train_assist --all >> "$LOG" 2>&1
nice -n 10 "$VENV" -m src.ml.recalibrate --limit 1200 >> "$LOG" 2>&1
echo ">>> assist/recal exit=$?" >> "$LOG"

echo "=== pipeline done $(date -Is) ===" >> "$LOG"
