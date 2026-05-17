#!/usr/bin/env bash
# sync_parquet.sh -- one-time rsync of historical Parquet files to the remote
# training server.  Run this from your local machine (WSL, Git Bash, or any
# POSIX shell) ONCE before the first remote training run.
#
# The Parquet store (~48 GB) only needs to be transferred once.  Subsequent
# training sweeps pull incremental OHLCV from the exchange (ccxt/binance) and
# append to the remote store, so this script is not needed again unless you
# provision a new server.
#
# Usage:
#   bash scripts/sync_parquet.sh user@<training-server-ip>
#
# Examples:
#   bash scripts/sync_parquet.sh ubuntu@10.0.0.42
#   bash scripts/sync_parquet.sh ubuntu@10.0.0.42 --dry-run
#
# Env vars:
#   LOCAL_PARQUET   -- source path on this machine (default below)
#   REMOTE_PARQUET  -- destination path on the server (default /data/parquet_db)

set -euo pipefail

REMOTE="${1:-}"
if [ -z "${REMOTE}" ]; then
  echo "Usage: bash scripts/sync_parquet.sh user@server-ip [--dry-run]"
  exit 1
fi

DRY_RUN=""
if [ "${2:-}" = "--dry-run" ]; then
  DRY_RUN="--dry-run"
  echo "[sync_parquet] DRY RUN -- no files will be transferred"
fi

# Adjust LOCAL_PARQUET if your project lives elsewhere.
# On Windows with WSL/Git Bash: /mnt/d/test\ 2/...
LOCAL_PARQUET="${LOCAL_PARQUET:-/mnt/d/test 2/AI trading assistance/data/parquet/}"
REMOTE_PARQUET="${REMOTE_PARQUET:-/data/parquet_db/}"

echo "[sync_parquet] Source : ${LOCAL_PARQUET}"
echo "[sync_parquet] Dest   : ${REMOTE}:${REMOTE_PARQUET}"
echo "[sync_parquet] Size estimate: up to 48 GB -- may take 30-90 min on 1 Gbps"

rsync -avz \
  --progress    \
  --partial     \
  --checksum    \
  ${DRY_RUN}    \
  --include "*.parquet" \
  --exclude "*" \
  "${LOCAL_PARQUET}" \
  "${REMOTE}:${REMOTE_PARQUET}"

echo "[sync_parquet] done -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
