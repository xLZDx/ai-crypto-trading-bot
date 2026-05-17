#!/usr/bin/env bash
# runpod_train_cpu.sh -- Train all CPU models on RunPod, then upload artifacts.
#
# Models trained: base, trend, futures, scalping, regime, meta
# (TFT and OFT are GPU-only -- skipped automatically, no crash)
#
# Cost target: AMD EPYC 7402P @ $0.318/hr, ~6h run = ~$1.90 total.
#
# Usage:
#   bash scripts/runpod_train_cpu.sh
#
# Dry run:
#   DRY_RUN=1 bash scripts/runpod_train_cpu.sh
#
# Skip upload (debug):
#   SKIP_UPLOAD=1 bash scripts/runpod_train_cpu.sh
#
# Force retrain even if models are fresh:
#   FORCE_RETRAIN=1 bash scripts/runpod_train_cpu.sh

set -euo pipefail

PROJECT_DIR="/workspace"
LOG_DIR="${PROJECT_DIR}/logs"
ARTIFACTS_DIR="${AI_TRADER_ARTIFACTS_DIR:-${PROJECT_DIR}/data/artifacts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_FOLDER="${GDRIVE_ARTIFACTS_FOLDER:-AI-Trader-Backup/artifacts}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
AUTO_STOP="${AUTO_STOP:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

# Use all available CPUs (24 for EPYC 7402P, 52 for 2x Gold 6230R, etc.)
NCPUS=$(nproc)
# Leave 2 cores for OS/SSH overhead
TRAIN_THREADS=$(( NCPUS > 4 ? NCPUS - 2 : NCPUS ))

mkdir -p "${LOG_DIR}" "${ARTIFACTS_DIR}"

log() { echo "[$(date -u '+%H:%M:%S')] $*"; }
ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
RUN_LOG="${LOG_DIR}/runpod_cpu_train_$(date -u +%Y%m%d_%H%M%S).log"
SCRIPT_START=$(date +%s)

log "============================================================"
log " RunPod CPU Training -- base/trend/futures/scalping/regime/meta"
log " $(ts)"
log " CPUs: ${NCPUS}  TRAIN_THREADS: ${TRAIN_THREADS}"
log "============================================================"
log " DRY_RUN=${DRY_RUN}  SKIP_UPLOAD=${SKIP_UPLOAD}  FORCE_RETRAIN=${FORCE_RETRAIN}"
log " Logging to: ${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

cd "${PROJECT_DIR}"

# ── 1. Pre-flight ─────────────────────────────────────────────────────────────
log "[check] Python + sklearn check..."
python3 - <<'PYCHECK'
import sklearn, pandas, joblib, numpy, sys, os
print(f"  sklearn   : {sklearn.__version__}")
print(f"  pandas    : {pandas.__version__}")
print(f"  numpy     : {numpy.__version__}")
print(f"  CPUs      : {os.cpu_count()}")
PYCHECK
log "[check] OK."

# ── 2. Set SKIP_IF_FRESH_S ────────────────────────────────────────────────────
# FORCE_RETRAIN=1 sets skip window to 0 (retrain everything).
# Default: 48h (skip models trained in last 48h -- resume-mode for crashes).
if [ "${FORCE_RETRAIN}" = "1" ]; then
    export AI_TRADER_TRAIN_SKIP_IF_FRESH_S=0
    log "[train] FORCE_RETRAIN=1 -- all models will be retrained."
else
    export AI_TRADER_TRAIN_SKIP_IF_FRESH_S=172800
    log "[train] Resume mode: skipping models trained within last 48h."
fi

# ── 3. Pull parquet data from Google Drive ────────────────────────────────────
log "------------------------------------------------------------"
log "[data] Syncing parquet from ${RCLONE_REMOTE}:AI-Trader-Backup/test2/..."
log "[data] Only downloads new/changed files (incremental)."
log "------------------------------------------------------------"

PARQUET_SRC="${RCLONE_REMOTE}:AI-Trader-Backup/test2/AI trading assistance/data/parquet"
PARQUET_DST="${PROJECT_DIR}/data/parquet"
mkdir -p "${PARQUET_DST}"

if [ "${DRY_RUN}" = "1" ]; then
    log "[data][DRY_RUN] Would rclone sync ${PARQUET_SRC}/ ${PARQUET_DST}/"
elif command -v rclone &>/dev/null; then
    DATA_START=$(date +%s)
    rclone sync "${PARQUET_SRC}/" "${PARQUET_DST}/" \
        --fast-list \
        --transfers 8 \
        --checkers 16 \
        --progress \
        --log-level INFO \
        2>&1 | tail -10 || log "[data][WARN] parquet sync failed -- training with existing data"
    DATA_ELAPSED=$(( $(date +%s) - DATA_START ))
    log "[data] Sync done in ${DATA_ELAPSED}s ($(( DATA_ELAPSED / 60 ))m)"
else
    log "[data][WARN] rclone not found -- skipping data sync. Run: curl https://rclone.org/install.sh | bash"
fi

# ── 4. Train all CPU models ───────────────────────────────────────────────────
log "------------------------------------------------------------"
log "[train] Starting full CPU training pipeline..."
log "[train] Expected: base(5TF) + trend(5TF) + futures(5TF) + scalping(2TF)"
log "[train]         + regime(1TF) + meta(4TF) = 22 (model x TF) combos"
log "[train] x 20 symbols = ~320 runs @ ${TRAIN_THREADS} threads each"
log "[train] Estimated wall time: ~5-7h on ${NCPUS}-core machine"
log "------------------------------------------------------------"

export AI_TRADER_TRAIN_CPU_THREADS="${TRAIN_THREADS}"
# OMP/MKL threads are set inside train_all_models.py from AI_TRADER_TRAIN_CPU_THREADS.
# Set them here too so any subprocess that bypasses train_all_models.py also respects the cap.
export OMP_NUM_THREADS="${TRAIN_THREADS}"
export MKL_NUM_THREADS="${TRAIN_THREADS}"
export OPENBLAS_NUM_THREADS="${TRAIN_THREADS}"

TRAIN_START=$(date +%s)

if [ "${DRY_RUN}" = "1" ]; then
    log "[train][DRY_RUN] Would run: python3 -m src.engine.train_all_models"
    sleep 3
else
    python3 -m src.engine.train_all_models 2>&1 \
        || log "[train][WARN] train_all_models exited non-zero -- check log for details"
fi

TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
log "[train] Training done in $(( TRAIN_ELAPSED / 60 ))m (${TRAIN_ELAPSED}s)"

# ── 4. Export artifacts ───────────────────────────────────────────────────────
log "------------------------------------------------------------"
log "[export] Running artifact_exporter..."
log "------------------------------------------------------------"
if [ "${DRY_RUN}" = "1" ]; then
    log "[export][DRY_RUN] Would export artifacts to ${ARTIFACTS_DIR}"
else
    python3 - <<PYEXPORT
import sys; sys.path.insert(0, '${PROJECT_DIR}')
from src.engine.artifact_exporter import export_artifacts
result = export_artifacts()
exported = result.get('exported', [])
print(f"  Exported {len(exported)} artifacts -> ${ARTIFACTS_DIR}")
for f in exported[:15]:
    print(f"    {f}")
PYEXPORT
fi

# ── 5. Upload to Google Drive ─────────────────────────────────────────────────
if [ "${SKIP_UPLOAD}" = "0" ]; then
    log "------------------------------------------------------------"
    log "[upload] Syncing artifacts to ${RCLONE_REMOTE}:${GDRIVE_FOLDER}..."
    log "------------------------------------------------------------"

    DATESTAMP=$(date -u '+%Y-%m-%dT%H%M%SZ')
    DEST="${RCLONE_REMOTE}:${GDRIVE_FOLDER}/${DATESTAMP}"

    if [ "${DRY_RUN}" = "1" ]; then
        log "[upload][DRY_RUN] Would rclone copy ${ARTIFACTS_DIR}/ ${DEST}/"
    else
        if command -v rclone &>/dev/null; then
            rclone copy "${ARTIFACTS_DIR}/" "${DEST}/" \
                --include "*.joblib" \
                --include "*.json" \
                --include "*.pt" \
                --include "*.db" \
                --progress \
                --log-level INFO \
                2>&1 | tail -20 || log "[upload][WARN] rclone upload failed"

            echo "${DATESTAMP}" | rclone rcat "${RCLONE_REMOTE}:${GDRIVE_FOLDER}/current" 2>/dev/null || true
            log "[upload] Done. latest: ${DATESTAMP}"
        else
            log "[upload][WARN] rclone not found -- run: curl https://rclone.org/install.sh | bash"
        fi
    fi
else
    log "[upload] SKIP_UPLOAD=1 -- skipped."
fi

# ── 6. Cost estimate ──────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
COST=$(python3 -c "print(f'\${${TOTAL_ELAPSED}/3600 * 0.318:.2f}')" 2>/dev/null || echo "?")
log "============================================================"
log " Training complete."
log " Wall time  : ${TOTAL_MIN} min (${TOTAL_ELAPSED}s)"
log " Cost est.  : ~\$${COST}  (EPYC 7402P @ \$0.318/hr)"
log " Log file   : ${RUN_LOG}"
log " Artifacts  : ${ARTIFACTS_DIR}"
log "============================================================"

# ── 7. Auto-stop ──────────────────────────────────────────────────────────────
if [ "${AUTO_STOP}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
    log "[stop] AUTO_STOP=1 -- halting pod in 60s."
    log "       Set AUTO_STOP=0 to keep the pod alive."
    sleep 60
    kill -9 1 2>/dev/null || true
fi
