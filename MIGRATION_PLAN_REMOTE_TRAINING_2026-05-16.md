# ML Training Migration Plan — Remote Dedicated Server
**Date:** 2026-05-16  
**Status:** Approved — pending execution  
**Reviewed by:** architect agent (27 tool calls, 204 s)

---

## Architecture Overview

```
Local Windows (development)
  └── git push ──────────────────────┐
                                     │
Remote GPU Server (training)         │
  ├── git clone ◄────────────────────┘
  ├── /data/parquet_db/  (rsync once, ~48 GB)
  ├── /data/db/          (rsync once, DuckDB telemetry)
  ├── pipeline_orchestrator.py  (runs in tmux)
  ├── artifact_exporter.py  ──► /data/artifacts/
  └── rclone_sync.sh  ──────────► GCS bucket (dated subfolder)

Trading VPS (inference)
  ├── install_artifacts.sh  ◄── GCS bucket
  │     └── pulls *.joblib + meta + manifest → models/
  ├── bot (main.py) hot-reloads when manifest.json mtime changes
  └── models/manifest.json  ← re-signed at install time
```

---

## Deliverables (14 items)

### A — Infrastructure (no code, one-time)

| # | What | Why |
|---|------|-----|
| A1 | Push repo to private GitHub; add SSH key to remote server | Code deployment without manual file copy |
| A2 | rsync `data/parquet/` → `/data/parquet_db/` on GPU server (~48 GB) | Historical OHLCV needed for training |
| A3 | rsync `data/db/` → `/data/db/` on GPU server | DuckDB telemetry tables (`backtest_results`, `training_runs`, `model_wf_folds`) are **not rebuildable** from Parquet — if absent, backtest phase has no baseline and `overall_ok` goes false, blocking artifact export |
| A4 | Enable **GCS object versioning** on the artifacts bucket | Rollback path: if a new model is worse, previous version survives |
| A5 | VS Code Remote-SSH connection | IDE operates locally; all execution on remote |

### B — New file: `src/engine/artifact_exporter.py`

Selects the best trained variant per key (highest `walk_forward_mean_acc` from meta JSON, fallback to `accuracy`). Exports to `ARTIFACTS_DIR` (env var `AI_TRADER_ARTIFACTS_DIR`, default `data/artifacts/`).

**What it exports:**
- `{key}_best.joblib` + `{key}_best_meta.json` — for every trained joblib key (base, trend, futures, scalping, meta, regime)
- ALL per-TF variants (e.g. `base_4h_model.joblib`, `base_15m_model.joblib`) — because `MultiTFPredictor` auto-discovers every per-TF file in `models/`; shipping only the best-per-key alias leaves stale variants on the trading VPS
- `best_model.joblib` + `best_model_meta.json` — alias pointing to best base model
- `optuna.db` — Optuna SQLite study, with `PRAGMA wal_checkpoint(TRUNCATE)` run before copy to flush WAL

### C — Modify `src/engine/pipeline_orchestrator.py`

At end of `run_pipeline()` (after `overall_ok`, before `_write_status`):
- If `overall_ok`: call `export_artifacts()` in a non-fatal try/except
- Embed result dict in `pipeline_status.json` under key `"artifacts"`

Change size: ~10 lines inside existing `run_pipeline()`.

### D — Modify `src/analysis/multi_tf_predictor.py`

Hot-reload with corrected locking design (architect found original over-locked):

- **`_mtimes: dict[str, float]`** — tracks mtime of each loaded file
- **Watch target: `models/manifest.json` mtime, NOT individual `.joblib` mtimes.**  
  Reason: trainer writes joblib first, then signs and updates manifest last. Watching manifest mtime guarantees binary is fully signed before reload triggers. Watching joblib mtime risks loading an unsigned file during the write→sign gap.
- **`reload()`** — builds fresh `_predictors` dict entirely outside any lock, then does a single `self._predictors = new_dict` assignment (GIL-atomic). No lock held during `joblib.load`.
- **No lock on predict path** — dict-reference reads are GIL-atomic; a swap mid-predict returns a result from either old or new model, both valid.
- **`start_watcher(interval_s=60)`** — daemon thread, polls manifest mtime, calls `reload()` on change.

### E — Modify `src/main.py`

After the 4 `MultiTFPredictor` constructions at lines 110-113:  
Call `pred.start_watcher(interval_s=60)` on each. ~4 lines added.

### F — New file: `scripts/rclone_sync.sh` (runs on GPU server)

Syncs `ARTIFACTS_DIR` to `gcs:{GCS_BUCKET}/artifacts/{DATESTAMP}/` (dated subdirectory for rollback). Updates a `current` pointer file so the trading VPS knows which run to pull.

```
gcs:bucket/artifacts/
  2026-05-16T14:00Z/
    best_model.joblib
    best_model_meta.json
    base_best.joblib
    base_4h_model.joblib      ← all per-TF variants
    ...
    optuna.db
    manifest.json             ← re-keyed manifest (see G)
  current                     ← contains the latest datestamp
```

### G — New file: `scripts/install_artifacts.sh` (runs on trading VPS)

Critical missing seam (architect Risk 1 + Risk 2):

1. Read `current` pointer from GCS → get latest datestamp
2. Pull all `*.joblib`, `*.json`, `*.db` from `gcs:bucket/artifacts/{datestamp}/` → `models/.staging/`
3. Rename per-TF files to legacy names: `base_best.joblib` → `btc_rf_model.joblib` etc. (using `LEGACY_MODEL_NAME` mapping from `src/utils/model_paths.py`)
4. **Re-sign the manifest**: run `sign_model()` for each installed file — writes fresh HMAC entries keyed to `models/<filename>` rel-paths that `model_integrity.py` expects. Requires `MODEL_MANIFEST_KEY` in `.env`.
5. Atomic move: `mv models/.staging/* models/`
6. Watcher detects `models/manifest.json` mtime changed → `reload()` → bot picks up new weights, no restart

**Why re-signing is mandatory:** `model_integrity.py:184-192` keys HMAC by path relative to project root. A file copied from `data/artifacts/` to `models/` changes its rel-key; old manifest entry no longer matches → `ModelIntegrityError` on load.

### H — New file: `scripts/remote_setup.sh` (GPU server, one-time)

Creates directories, git clones, sets up venv, scaffolds `.env`, installs rclone, prints tmux startup commands.

### I — New file: `scripts/sync_data.sh` (runs from local machine, one-time)

Rsyncs `data/parquet/` AND `data/db/` with `--checksum --partial`.  
**Do NOT use `--delete`** on `data/db/` — local Windows bot may still be writing to it.

### J — Config updates

- `.env.example`: add `AI_TRADER_ARTIFACTS_DIR`, `GCS_BUCKET`, `RCLONE_REMOTE`
- `.gitignore`: add `data/artifacts/`
- `models/manifest.json`: already not git-ignored — ships with clone as baseline, overwritten at install time on VPS

### K — `MODEL_MANIFEST_KEY` propagation (operational, no code)

Same key must be in `.env` on all three machines: local Windows, GPU server, trading VPS.  
Self-test: sign a dummy file, verify it — must pass before declaring the remote ready.

---

## Execution Order

```
Phase 1 — Prep (local, no server needed)
  J  Config files (.env.example, .gitignore)
  B  artifact_exporter.py
  C  pipeline_orchestrator.py
  D  multi_tf_predictor.py  (hot-reload rewrite)
  E  main.py                (start_watcher calls)
  F  rclone_sync.sh
  G  install_artifacts.sh
  H  remote_setup.sh
  I  sync_data.sh
  → git push

Phase 2 — Server bootstrap (one-time)
  bash remote_setup.sh
  bash sync_data.sh user@server    ← ~48 GB + db, 30-90 min
  Set MODEL_MANIFEST_KEY in remote .env
  Run HMAC self-test

Phase 3 — First remote training run
  tmux: python -m src.engine.pipeline_orchestrator
  bash rclone_sync.sh

Phase 4 — Trading VPS wiring
  Set MODEL_MANIFEST_KEY in VPS .env
  bash install_artifacts.sh        ← pull from GCS, re-sign, install
  Confirm bot logs show hot-reload within 60 s
```

---

## Risk Register

| Severity | Risk | Mitigation |
|----------|------|------------|
| CRITICAL | Bot reads only from `models/`, not `data/artifacts/` | `install_artifacts.sh` (G) renames and installs to `models/` |
| CRITICAL | HMAC rel-key mismatch after copy | Re-sign at install time in G |
| CRITICAL | `data/db/` absent on remote → backtest baseline missing, `overall_ok=false` | `sync_data.sh` (I) transfers both stores |
| HIGH | Over-locked hot-reload blocks predict path | Copy-on-write swap, manifest.json as watch target (D) |
| HIGH | Exporter ships only one model per key, stale per-TF variants remain | Export all per-TF variants (B) |
| MEDIUM | No rollback if new model underperforms | GCS dated subdirs + object versioning (A4, F) |
| MEDIUM | `MODEL_MANIFEST_KEY` not synced across machines | Explicit step K + self-test |
| MEDIUM | SQLite WAL not checkpointed before optuna.db copy | `PRAGMA wal_checkpoint` in exporter (B) |
| MEDIUM | `pipeline_status.json` split-brain if local box also runs pipeline | Sync `pipeline_status.json` alongside artifacts, or add remote-status endpoint to dashboard |
| LOW | Symlink swap rejected by `model_integrity.py` | `install_artifacts.sh` uses atomic `mv`, not symlinks |
| LOW | `training_rules.json` drifts if edited live on remote | Operational rule: edits go through git only, never live on remote |

---

## Files Affected

### Modified (4 files)
- `src/engine/pipeline_orchestrator.py` — ~10 lines added in `run_pipeline()`
- `src/analysis/multi_tf_predictor.py` — hot-reload rewrite (~80 lines added)
- `src/main.py` — ~4 lines added in `MultiAssetTrader.__init__`
- `.env.example` — 3 new env vars

### 1 line added
- `.gitignore` — `data/artifacts/`

### New files (5)
- `src/engine/artifact_exporter.py`
- `scripts/rclone_sync.sh`
- `scripts/install_artifacts.sh`
- `scripts/remote_setup.sh`
- `scripts/sync_data.sh`

### NOT changed
- `src/utils/model_integrity.py`
- `src/utils/model_paths.py`
- `src/analysis/ml_predictor.py`
- `src/engine/train_all_models.py`
- `data/training_rules.json`
- All dashboard code
