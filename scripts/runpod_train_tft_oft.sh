#!/usr/bin/env bash
# runpod_train_tft_oft.sh -- Train TFT + OFT on RunPod GPU, then upload artifacts.
#
# Cost target: RTX 4090 @ $0.69/hr → ~$0.70–1.40 total per weekly run.
#
# What this script does:
#   1. Verify CUDA is live
#   2. Train TFT for each applicable timeframe (1h, 4h)
#   3. Train OFT + SAC (joint_oft_rl.py) for 1m
#   4. Export artifacts via artifact_exporter.py
#   5. Upload artifacts to Google Drive (or GCS) via rclone
#   6. Print cost estimate
#   7. STOP the pod automatically (saves money — no idle billing)
#
# Usage:
#   bash scripts/runpod_train_tft_oft.sh
#
# Dry run (skip actual training, just test the pipeline):
#   DRY_RUN=1 bash scripts/runpod_train_tft_oft.sh
#
# Skip specific stages:
#   SKIP_TFT=1 bash scripts/runpod_train_tft_oft.sh   # only OFT + upload
#   SKIP_OFT=1 bash scripts/runpod_train_tft_oft.sh   # only TFT + upload
#   SKIP_UPLOAD=1 bash scripts/runpod_train_tft_oft.sh # no upload (debug)

set -euo pipefail

PROJECT_DIR="/workspace"
LOG_DIR="${PROJECT_DIR}/logs"
ARTIFACTS_DIR="${AI_TRADER_ARTIFACTS_DIR:-${PROJECT_DIR}/data/artifacts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_FOLDER="${GDRIVE_ARTIFACTS_FOLDER:-AI-Trader-Backup/artifacts}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TFT="${SKIP_TFT:-0}"
SKIP_OFT="${SKIP_OFT:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
AUTO_STOP="${AUTO_STOP:-1}"   # set AUTO_STOP=0 to keep pod alive after training

mkdir -p "${LOG_DIR}" "${ARTIFACTS_DIR}"

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date -u '+%H:%M:%S')] $*"; }
ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
RUN_LOG="${LOG_DIR}/runpod_train_$(date -u +%Y%m%d_%H%M%S).log"
SCRIPT_START=$(date +%s)

log "============================================================"
log " RunPod GPU Training — TFT + OFT"
log " $(ts)"
log "============================================================"
log " DRY_RUN=${DRY_RUN}  SKIP_TFT=${SKIP_TFT}  SKIP_OFT=${SKIP_OFT}"
log " Logging to: ${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

cd "${PROJECT_DIR}"

# ── 1. CUDA check ─────────────────────────────────────────────────────────────
log "[check] Verifying CUDA..."
python3 - <<'PYCHECK'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available! Check RunPod GPU allocation.", file=sys.stderr)
    sys.exit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  GPU: {name}  VRAM: {vram:.1f} GB")
print(f"  torch: {torch.__version__}  CUDA: {torch.version.cuda}")
PYCHECK
log "[check] CUDA OK."

# ── 2. TFT training ───────────────────────────────────────────────────────────
if [ "${SKIP_TFT}" = "0" ]; then
    log "------------------------------------------------------------"
    log "[TFT] Starting Temporal Fusion Transformer training..."
    log "------------------------------------------------------------"

    TFT_TIMEFRAMES=("1h" "4h")
    for TF in "${TFT_TIMEFRAMES[@]}"; do
        log "[TFT] timeframe=${TF} ..."
        TF_START=$(date +%s)
        if [ "${DRY_RUN}" = "1" ]; then
            log "[TFT][DRY_RUN] Would run: python3 -m src.engine.train_tft_model --timeframe ${TF}"
            sleep 2
        else
            python3 -m src.engine.train_tft_model \
                --timeframe "${TF}" \
                --epochs 25 \
                --min-epochs 3 \
                --patience 6 \
                2>&1 || log "[TFT][WARN] train_tft_model failed for ${TF} — continuing"
        fi
        TF_ELAPSED=$(( $(date +%s) - TF_START ))
        log "[TFT] ${TF} done in ${TF_ELAPSED}s"
    done
    log "[TFT] All timeframes complete."
else
    log "[TFT] SKIP_TFT=1 — skipped."
fi

# ── 3. OFT + SAC joint training ───────────────────────────────────────────────
if [ "${SKIP_OFT}" = "0" ]; then
    log "------------------------------------------------------------"
    log "[OFT] Starting Order Flow Transformer + SAC training..."
    log "------------------------------------------------------------"

    OFT_SYMBOLS=("BTC/USDT" "ETH/USDT" "SOL/USDT")
    for SYM in "${OFT_SYMBOLS[@]}"; do
        SYM_SAFE="${SYM//\//_}"
        log "[OFT] symbol=${SYM} tf=1m ..."
        OFT_START=$(date +%s)
        if [ "${DRY_RUN}" = "1" ]; then
            log "[OFT][DRY_RUN] Would run: python3 -m src.training.joint_oft_rl --symbol ${SYM} --tf 1m"
            sleep 2
        else
            python3 -m src.training.joint_oft_rl \
                --symbol "${SYM}" \
                --tf 1m \
                --epochs 5 \
                --episodes 50 \
                2>&1 || log "[OFT][WARN] joint_oft_rl failed for ${SYM} — continuing"
        fi
        OFT_ELAPSED=$(( $(date +%s) - OFT_START ))
        log "[OFT] ${SYM} done in ${OFT_ELAPSED}s"
    done
    log "[OFT] All symbols complete."
else
    log "[OFT] SKIP_OFT=1 — skipped."
fi

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
for f in exported[:10]:
    print(f"    {f}")
PYEXPORT
fi

# ── 5. Upload to Google Drive ─────────────────────────────────────────────────
if [ "${SKIP_UPLOAD}" = "0" ]; then
    log "------------------------------------------------------------"
    log "[upload] Syncing artifacts to ${RCLONE_REMOTE}:${GDRIVE_FOLDER}..."
    log "------------------------------------------------------------"

    # Dated subfolder for rollback
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

            # Update 'current' pointer so install_artifacts.sh knows what to pull
            echo "${DATESTAMP}" | rclone rcat "${RCLONE_REMOTE}:${GDRIVE_FOLDER}/current" 2>/dev/null || true
            log "[upload] Done. latest: ${DATESTAMP}"
        else
            log "[upload][WARN] rclone not found — skipping upload. Run: curl https://rclone.org/install.sh | bash"
        fi
    fi
else
    log "[upload] SKIP_UPLOAD=1 — skipped."
fi

# ── 6. Cost estimate ──────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
COST=$(python3 -c "print(f'\${${TOTAL_ELAPSED}/3600 * 0.69:.2f}')" 2>/dev/null || echo "?")
log "============================================================"
log " Training complete."
log " Wall time  : ${TOTAL_MIN} min (${TOTAL_ELAPSED}s)"
log " Cost est.  : ~${COST}  (RTX 4090 @ \$0.69/hr)"
log " Log file   : ${RUN_LOG}"
log " Artifacts  : ${ARTIFACTS_DIR}"
log "============================================================"

# ── 7. Auto-stop pod to avoid idle billing ────────────────────────────────────
if [ "${AUTO_STOP}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
    log "[stop] AUTO_STOP=1 — halting pod in 60s to stop billing."
    log "       Set AUTO_STOP=0 to keep the pod alive."
    log "       Or: kill this script within 60s."
    sleep 60
    # RunPod: kill -9 1 causes the container to exit and billing stops
    kill -9 1 2>/dev/null || true
fi
