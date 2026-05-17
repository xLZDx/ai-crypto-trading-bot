#!/usr/bin/env bash
# rclone_sync.sh -- push lightweight training artifacts to Google Cloud Storage.
#
# Run on the remote training server after pipeline_orchestrator.py finishes.
# The script syncs only the small files (.joblib, .json, .db) -- never the
# 48 GB Parquet store or raw logs.
#
# Prerequisites:
#   1. rclone installed: curl https://rclone.org/install.sh | sudo bash
#   2. GCS remote configured: rclone config  (name the remote 'gcs')
#      For a service-account key:
#        rclone config create gcs google cloud storage
#          service_account_file /path/to/sa-key.json
#
# Usage:
#   bash scripts/rclone_sync.sh
#
# Env vars (override defaults):
#   AI_TRADER_ARTIFACTS_DIR  -- source dir  (default /data/artifacts)
#   GCS_BUCKET               -- destination  (default your-trading-artifacts)
#   RCLONE_REMOTE            -- rclone remote name (default gcs)

set -euo pipefail

ARTIFACTS_DIR="${AI_TRADER_ARTIFACTS_DIR:-/data/artifacts}"
GCS_BUCKET="${GCS_BUCKET:-your-trading-artifacts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gcs}"
LOG_FILE="/tmp/rclone_sync_$(date +%Y%m%d_%H%M%S).log"

echo "[rclone_sync] source : ${ARTIFACTS_DIR}"
echo "[rclone_sync] dest   : ${RCLONE_REMOTE}:${GCS_BUCKET}/artifacts/"
echo "[rclone_sync] log    : ${LOG_FILE}"

rclone sync "${ARTIFACTS_DIR}/" "${RCLONE_REMOTE}:${GCS_BUCKET}/artifacts/" \
  --include "*.joblib" \
  --include "*.json"   \
  --include "*.db"     \
  --progress           \
  --log-level INFO     \
  --log-file "${LOG_FILE}"

echo "[rclone_sync] done -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
