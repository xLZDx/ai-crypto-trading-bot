#!/usr/bin/env bash
# cherry_train_cpu.sh -- Train all CPU models on CherryServers, then upload artifacts.
#
# Server: AMD EPYC 7402P, 24 cores, 64 GB RAM, ~0.272 EUR/hr
# Models trained: base, trend, futures, scalping, regime, meta
# (TFT and OFT are GPU-only -- skipped automatically)
#
# Prerequisites (run once via cherry_setup_cpu.sh or remote_setup.sh):
#   ~/ai-trading/  -- project cloned here
#   ~/ai-trading/venv  -- virtualenv with requirements installed
#
# Usage:
#   bash ~/ai-trading/scripts/cherry_train_cpu.sh
#
# Dry run:
#   DRY_RUN=1 bash ~/ai-trading/scripts/cherry_train_cpu.sh
#
# Skip upload (debug):
#   SKIP_UPLOAD=1 bash ~/ai-trading/scripts/cherry_train_cpu.sh
#
# Force retrain all models (ignore freshness):
#   FORCE_RETRAIN=1 bash ~/ai-trading/scripts/cherry_train_cpu.sh
#
# Keep server alive after training (don't poweroff):
#   AUTO_STOP=0 bash ~/ai-trading/scripts/cherry_train_cpu.sh

set -euo pipefail

PROJECT_DIR="${HOME}/ai-trading"
VENV="${PROJECT_DIR}/venv"
LOG_DIR="${PROJECT_DIR}/logs"
ARTIFACTS_DIR="${AI_TRADER_ARTIFACTS_DIR:-${PROJECT_DIR}/data/artifacts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_FOLDER="${GDRIVE_ARTIFACTS_FOLDER:-AI-Trader-Backup/artifacts}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
AUTO_STOP="${AUTO_STOP:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
HOURLY_RATE="${HOURLY_RATE:-0.272}"  # EUR/hr for EPYC 7402P on CherryServers

# Use all available CPUs (24 for EPYC 7402P), leave 2 for OS/SSH overhead
NCPUS=$(nproc)
TRAIN_THREADS=$(( NCPUS > 4 ? NCPUS - 2 : NCPUS ))

mkdir -p "${LOG_DIR}" "${ARTIFACTS_DIR}"

log() { echo "[$(date -u '+%H:%M:%S')] $*"; }
ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
RUN_LOG="${LOG_DIR}/cherry_cpu_train_$(date -u +%Y%m%d_%H%M%S).log"
SCRIPT_START=$(date +%s)

log "============================================================"
log " CherryServers CPU Training -- base/trend/futures/scalping/regime/meta"
log " $(ts)"
log " CPUs: ${NCPUS}  TRAIN_THREADS: ${TRAIN_THREADS}"
log " Project: ${PROJECT_DIR}"
log "============================================================"
log " DRY_RUN=${DRY_RUN}  SKIP_UPLOAD=${SKIP_UPLOAD}  FORCE_RETRAIN=${FORCE_RETRAIN}"
log " Logging to: ${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

cd "${PROJECT_DIR}"

# ── 1. Activate virtualenv ────────────────────────────────────────────────────
log "[env] Activating venv: ${VENV}/bin/activate"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

log "[check] Python + sklearn check..."
python3 - <<'PYCHECK'
import sklearn, pandas, joblib, numpy, sys, os
print(f"  python    : {sys.version.split()[0]}")
print(f"  sklearn   : {sklearn.__version__}")
print(f"  pandas    : {pandas.__version__}")
print(f"  numpy     : {numpy.__version__}")
print(f"  CPUs      : {os.cpu_count()}")
PYCHECK
log "[check] OK."

# ── 2. Set freshness skip window ──────────────────────────────────────────────
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
    log "[data][WARN] rclone not found. Install: curl https://rclone.org/install.sh | sudo bash"
    log "[data][WARN] Training will proceed with any data already present."
fi

# ── 4. Train all CPU models ───────────────────────────────────────────────────
log "------------------------------------------------------------"
log "[train] Starting full CPU training pipeline..."
log "[train] Expected: base(4TF) + trend(4TF) + futures(4TF) + scalping(2TF)"
log "[train]         + regime(1TF) + meta(4TF) = 19 (model x TF) combos"
log "[train] x 20 symbols = ~380 runs @ ${TRAIN_THREADS} threads each"
log "[train] Estimated wall time: ~5-7h on ${NCPUS}-core machine"
log "------------------------------------------------------------"

export AI_TRADER_TRAIN_CPU_THREADS="${TRAIN_THREADS}"
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

# ── 5. Export artifacts ───────────────────────────────────────────────────────
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

# ── 6. Upload to Google Drive ─────────────────────────────────────────────────
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
            log "[upload][WARN] rclone not found -- artifacts saved locally only"
        fi
    fi
else
    log "[upload] SKIP_UPLOAD=1 -- skipped."
fi

# ── 7. Cost estimate ──────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
COST=$(python3 -c "print(f'{${TOTAL_ELAPSED}/3600 * ${HOURLY_RATE:.3f}:.2f}')" 2>/dev/null || echo "?")
log "============================================================"
log " Training complete."
log " Wall time  : ${TOTAL_MIN} min (${TOTAL_ELAPSED}s)"
log " Cost est.  : ~EUR ${COST}  (EPYC 7402P @ ${HOURLY_RATE} EUR/hr)"
log " Log file   : ${RUN_LOG}"
log " Artifacts  : ${ARTIFACTS_DIR}"
log "============================================================"

# ── 8. Auto-stop ──────────────────────────────────────────────────────────────
# CherryServers: sudo poweroff stops the server and billing.
# Set AUTO_STOP=0 to keep server alive for debugging.
if [ "${AUTO_STOP}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
    log "[stop] AUTO_STOP=1 -- powering off server in 60s."
    log "       Ctrl+C or set AUTO_STOP=0 to keep alive."
    sleep 60
    sudo poweroff 2>/dev/null || true
fi
