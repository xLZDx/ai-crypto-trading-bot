#!/usr/bin/env bash
# remote_setup.sh -- one-time bootstrap for the dedicated training server.
#
# Run this script immediately after SSH-ing into a fresh Ubuntu 22.04 box.
# Assumes: Python 3.10+, NVMe RAID 0 mounted at /data, sudo access.
#
# Usage:
#   ssh user@<training-server-ip>
#   bash remote_setup.sh
#
# After setup:
#   1. Edit ~/ai-trading/.env  (API keys + AI_TRADER_ARTIFACTS_DIR)
#   2. Run 'rclone config' to connect the 'gcs' remote
#   3. Use the tmux commands printed at the end to start training

set -euo pipefail

GITHUB_REPO="https://github.com/xLZDx/ai-crypto-trading-bot"
PROJECT_DIR="${HOME}/ai-trading"
DATA_ROOT="/data"
ARTIFACTS_DIR="${DATA_ROOT}/artifacts"
PARQUET_DIR="${DATA_ROOT}/parquet_db"
CACHE_DIR="${DATA_ROOT}/cache/duckdb_temp"

# ── 1. Directory layout ───────────────────────────────────────────────────────
echo "[setup] Creating data directories..."
sudo mkdir -p "${ARTIFACTS_DIR}" "${PARQUET_DIR}" "${CACHE_DIR}"
sudo chown -R "$(whoami)" "${DATA_ROOT}"

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
echo "[setup] Cloning repository..."
if [ ! -d "${PROJECT_DIR}/.git" ]; then
  git clone "${GITHUB_REPO}" "${PROJECT_DIR}"
else
  echo "[setup] Repo already exists at ${PROJECT_DIR} -- pulling latest..."
  git -C "${PROJECT_DIR}" pull --ff-only
fi

# ── 3. Python virtual environment ─────────────────────────────────────────────
echo "[setup] Creating virtual environment..."
cd "${PROJECT_DIR}"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
echo "[setup] Python environment ready."

# ── 4. .env configuration ─────────────────────────────────────────────────────
if [ ! -f "${PROJECT_DIR}/.env" ]; then
  cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
  # Inject remote-specific overrides automatically
  cat >> "${PROJECT_DIR}/.env" << EOF

# ---- Remote server overrides (auto-added by remote_setup.sh) ----
AI_TRADER_ARTIFACTS_DIR=${ARTIFACTS_DIR}
AI_TRADER_PIPELINE_LOCAL=1
EOF
  echo "[setup] Created .env -- EDIT IT NOW to add your API keys."
  echo "        Required: API_KEY, API_SECRET, MODEL_MANIFEST_KEY, ZMQ_BUS_KEY"
else
  echo "[setup] .env already exists -- skipping creation."
fi

# ── 5. rclone ─────────────────────────────────────────────────────────────────
if ! command -v rclone &> /dev/null; then
  echo "[setup] Installing rclone..."
  curl https://rclone.org/install.sh | sudo bash
  echo "[setup] Run 'rclone config' to configure your GCS remote (name it 'gcs')."
else
  echo "[setup] rclone already installed: $(rclone version | head -1)"
fi

# ── 6. tmux instructions ──────────────────────────────────────────────────────
ACTIVATE="source ${PROJECT_DIR}/venv/bin/activate"
RUN_CMD="cd ${PROJECT_DIR} && ${ACTIVATE} && python -m src.engine.pipeline_orchestrator"
SYNC_CMD="bash ${PROJECT_DIR}/scripts/rclone_sync.sh"

echo ""
echo "============================================================"
echo " Setup complete. Next steps:"
echo "============================================================"
echo ""
echo "1. Edit your credentials:"
echo "   nano ${PROJECT_DIR}/.env"
echo ""
echo "2. Configure rclone GCS remote (if not done yet):"
echo "   rclone config"
echo ""
echo "3. Start a persistent training session:"
echo ""
echo "   # Create a tmux session"
echo "   tmux new-session -d -s training -n orchestrator"
echo ""
echo "   # Start the pipeline (runs train + backtest + artifact export)"
echo "   tmux send-keys -t training:orchestrator '${RUN_CMD}' Enter"
echo ""
echo "   # Attach to watch progress"
echo "   tmux attach -t training"
echo "   # Detach without killing:  Ctrl-B then D"
echo ""
echo "4. After training completes, sync artifacts to GCS:"
echo "   ${SYNC_CMD}"
echo ""
echo "5. On your trading VPS, pull the fresh artifacts:"
echo "   rclone copy gcs:your-trading-artifacts/artifacts/ models/"
echo "   # The bot detects the new files within 60 s and hot-reloads."
echo "============================================================"
