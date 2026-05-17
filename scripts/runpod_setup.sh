#!/usr/bin/env bash
# runpod_setup.sh -- one-time bootstrap for RunPod GPU container.
#
# RunPod-specific assumptions:
#   - PyTorch 2.8.0 + CUDA 12.1 already installed in the base image
#   - /workspace is the persistent volume (50 GB)
#   - Running as root, no sudo needed
#   - Python binary: python3 (already on PATH)
#
# Usage (paste into RunPod SSH terminal):
#   cd /workspace
#   git clone https://github.com/xLZDx/ai-crypto-trading-bot . && bash scripts/runpod_setup.sh
#
# After setup, run training:
#   bash scripts/runpod_train_tft_oft.sh

set -euo pipefail

GITHUB_REPO="https://github.com/xLZDx/ai-crypto-trading-bot"
PROJECT_DIR="/workspace"
DATA_DIR="${PROJECT_DIR}/data"
PARQUET_DIR="${DATA_DIR}/parquet"
ARTIFACTS_DIR="${DATA_DIR}/artifacts"
LOG_DIR="${PROJECT_DIR}/logs"

echo ""
echo "============================================================"
echo " RunPod GPU setup — AI Crypto Trading Bot"
echo " $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

# ── 1. Pull latest code (repo may already be cloned) ─────────────────────────
if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "[setup] Repo exists — pulling latest..."
    git -C "${PROJECT_DIR}" fetch origin
    git -C "${PROJECT_DIR}" reset --hard origin/main || git -C "${PROJECT_DIR}" reset --hard origin/master
else
    echo "[setup] Cloning repo..."
    git clone "${GITHUB_REPO}" "${PROJECT_DIR}"
fi

# ── 2. Create required directories ───────────────────────────────────────────
echo "[setup] Creating directories..."
mkdir -p "${PARQUET_DIR}" "${ARTIFACTS_DIR}" "${LOG_DIR}" \
         "${DATA_DIR}/models" "${DATA_DIR}/risk/drift_baselines" \
         "${DATA_DIR}/training_runs" "${DATA_DIR}/retrain_regressions" \
         "${PROJECT_DIR}/models"

# ── 3. Install Python dependencies ───────────────────────────────────────────
# Skip torch/torchvision/torchaudio — already installed in RunPod image.
# darts will auto-detect and use the existing torch installation.
echo "[setup] Installing dependencies (torch already in image, skipping reinstall)..."
pip install --upgrade pip --quiet

# Install everything except torch variants to avoid overwriting RunPod's CUDA build
pip install --quiet \
    --ignore-installed \
    --constraint <(echo "torch==$(python3 -c 'import torch; print(torch.__version__)')" 2>/dev/null || echo "") \
    -r requirements.txt 2>&1 | grep -E "^(Collecting|Installing|Successfully|ERROR|WARNING)" || true

# Sanity checks
python3 - <<'PYCHECK'
import torch, darts, pandas, sklearn, joblib, pyarrow
print(f"  torch     : {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  darts     : {darts.__version__}")
print(f"  pandas    : {pandas.__version__}")
print(f"  GPU count : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU 0     : {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: CUDA not available!")
PYCHECK

# ── 4. .env configuration ─────────────────────────────────────────────────────
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
    cat >> "${PROJECT_DIR}/.env" << EOF

# ---- RunPod overrides (auto-added by runpod_setup.sh) ----
AI_TRADER_ARTIFACTS_DIR=${ARTIFACTS_DIR}
AI_TRADER_PIPELINE_LOCAL=1
# Add your MODEL_MANIFEST_KEY, GCS_BUCKET, RCLONE_REMOTE here:
# MODEL_MANIFEST_KEY=
# GCS_BUCKET=your-bucket-name
# RCLONE_REMOTE=gdrive
EOF
    echo ""
    echo "  [!] .env created. Edit it NOW to add MODEL_MANIFEST_KEY:"
    echo "      nano ${PROJECT_DIR}/.env"
    echo ""
fi

# ── 5. rclone (for uploading artifacts to Google Drive) ───────────────────────
if ! command -v rclone &>/dev/null; then
    echo "[setup] Installing rclone..."
    curl -fsSL https://rclone.org/install.sh | bash -s -- --quiet 2>&1 | tail -3
else
    echo "[setup] rclone: $(rclone version 2>/dev/null | head -1)"
fi

# ── 6. Data sync instructions ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete."
echo "============================================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "A) Set your MODEL_MANIFEST_KEY in .env:"
echo "   nano ${PROJECT_DIR}/.env"
echo ""
echo "B) Sync parquet data from your local machine (run on RAZER, not here):"
echo "   On Razer PowerShell:"
cat << 'RSYNC_HELP'
   # Get RunPod SSH details from https://www.runpod.io/console/pods
   # Format: ssh root@<pod-ip> -p <port>
   $POD_IP="<your-pod-ip>"
   $POD_PORT="<your-pod-port>"
   rsync -avz --progress --partial `
     "D:\test 2\AI trading assistance\data\parquet\" `
     "root@${POD_IP}:/workspace/data/parquet/" -e "ssh -p ${POD_PORT}"
RSYNC_HELP
echo ""
echo "C) Or configure rclone for GCS/GDrive and pull data:"
echo "   rclone config"
echo "   rclone copy gdrive:AI-Trader-Backup/data/parquet /workspace/data/parquet"
echo ""
echo "D) Run training:"
echo "   bash scripts/runpod_train_tft_oft.sh"
echo ""
