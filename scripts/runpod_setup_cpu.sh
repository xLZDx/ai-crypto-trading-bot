#!/usr/bin/env bash
# runpod_setup_cpu.sh -- one-time bootstrap for RunPod CPU pod (Ubuntu 22.04).
#
# RunPod CPU pod assumptions:
#   - Ubuntu 22.04, Python 3.10+ already on PATH
#   - /workspace is the persistent volume
#   - Running as root, no sudo needed
#   - NO GPU -- TFT/OFT will be skipped automatically
#
# Usage (paste into RunPod SSH terminal):
#   cd /workspace
#   git clone https://github.com/xLZDx/ai-crypto-trading-bot . && bash scripts/runpod_setup_cpu.sh
#
# After setup, run training:
#   bash scripts/runpod_train_cpu.sh

set -euo pipefail

GITHUB_REPO="https://github.com/xLZDx/ai-crypto-trading-bot"
PROJECT_DIR="/workspace"
DATA_DIR="${PROJECT_DIR}/data"
PARQUET_DIR="${DATA_DIR}/parquet"
ARTIFACTS_DIR="${DATA_DIR}/artifacts"
LOG_DIR="${PROJECT_DIR}/logs"

echo ""
echo "============================================================"
echo " RunPod CPU setup -- AI Crypto Trading Bot"
echo " $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "[setup] Repo exists -- pulling latest..."
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

# ── 3. System packages ────────────────────────────────────────────────────────
echo "[setup] Checking system packages..."
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev build-essential libgomp1 \
        rsync curl git tmux htop 2>&1 | tail -5
fi

# ── 4. Install Python dependencies ───────────────────────────────────────────
# CPU-only torch: tiny install (~250MB vs 2GB CUDA build).
# Required because train_all_models.py imports train_tft_model at module level;
# TFT will fail gracefully at runtime (no GPU) and training continues.
echo "[setup] Installing torch CPU-only..."
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo "[setup] Installing project dependencies..."
pip install --quiet \
    --constraint <(echo "torch==$(python3 -c 'import torch; print(torch.__version__)')" 2>/dev/null || echo "") \
    -r "${PROJECT_DIR}/requirements.txt" 2>&1 | grep -E "^(Collecting|Installing|Successfully|ERROR|WARNING)" || true

# ── 5. Sanity check ───────────────────────────────────────────────────────────
python3 - <<'PYCHECK'
import pandas, sklearn, joblib, numpy
print(f"  pandas    : {pandas.__version__}")
print(f"  sklearn   : {sklearn.__version__}")
print(f"  numpy     : {numpy.__version__}")
try:
    import torch
    print(f"  torch     : {torch.__version__}  (CUDA: {torch.cuda.is_available()} -- expected False)")
except ImportError:
    print("  torch     : not installed (ok for CPU-only)")
import os
print(f"  CPUs      : {os.cpu_count()}")
PYCHECK

# ── 6. .env configuration ─────────────────────────────────────────────────────
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
    cat >> "${PROJECT_DIR}/.env" << EOF

# ---- RunPod CPU overrides (auto-added by runpod_setup_cpu.sh) ----
AI_TRADER_ARTIFACTS_DIR=${ARTIFACTS_DIR}
AI_TRADER_PIPELINE_LOCAL=1
AI_TRADER_TRAIN_CPU_THREADS=$(nproc)
# Add your MODEL_MANIFEST_KEY here:
# MODEL_MANIFEST_KEY=
# RCLONE_REMOTE=gdrive
EOF
    echo ""
    echo "  [!] .env created. Edit it NOW to add MODEL_MANIFEST_KEY:"
    echo "      nano ${PROJECT_DIR}/.env"
    echo ""
fi

# ── 7. rclone ─────────────────────────────────────────────────────────────────
if ! command -v rclone &>/dev/null; then
    echo "[setup] Installing rclone..."
    curl -fsSL https://rclone.org/install.sh | bash -s -- --quiet 2>&1 | tail -3
else
    echo "[setup] rclone: $(rclone version 2>/dev/null | head -1)"
fi

# ── 8. Print next steps ───────────────────────────────────────────────────────
NCPUS=$(nproc)
echo ""
echo "============================================================"
echo " Setup complete.  CPUs available: ${NCPUS}"
echo "============================================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "A) Set MODEL_MANIFEST_KEY in .env:"
echo "   nano ${PROJECT_DIR}/.env"
echo ""
echo "B) Sync parquet data from Razer (run this on RAZER, not here):"
echo "   \$POD_IP=\"<your-pod-ip>\""
echo "   \$POD_PORT=\"<your-pod-port>\""
cat << 'RSYNC_HELP'
   rsync -avz --progress --partial `
     "D:\test 2\AI trading assistance\data\parquet\" `
     "root@${POD_IP}:/workspace/data/parquet/" -e "ssh -p ${POD_PORT}"
RSYNC_HELP
echo ""
echo "C) Or pull from Google Drive (if already backed up there):"
echo "   rclone config   # set up gdrive remote once"
echo "   rclone copy gdrive:AI-Trader-Backup/test2/\"AI trading assistance\"/data/parquet \\"
echo "                  /workspace/data/parquet"
echo ""
echo "D) Run training (~6h on ${NCPUS} cores):"
echo "   bash scripts/runpod_train_cpu.sh"
echo ""
