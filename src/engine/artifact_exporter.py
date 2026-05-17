"""
artifact_exporter -- post-training artifact packaging for remote-server migration.

After each successful training sweep the pipeline orchestrator calls
export_artifacts().  It selects the highest-accuracy (key, tf) variant
for every trained model family, writes a clean set of lightweight files
to ARTIFACTS_DIR, then rclone_sync.sh can push them to GCS without
touching the 48 GB Parquet/DuckDB store.

Exported files
--------------
best_model.joblib       -- best base model binary (primary trading signal)
best_model_meta.json    -- its metadata: features, thresholds, walk_forward stats,
                           HMAC signature field (added by sign_model at training time)
{key}_best.joblib       -- best model binary for each trained key
{key}_best_meta.json    -- corresponding meta per key
optuna.db               -- Optuna SQLite study (all trial history + hyperparameters)

ARTIFACTS_DIR is set via AI_TRADER_ARTIFACTS_DIR env var.
  Local default : <project_root>/data/artifacts/
  Remote server : /data/artifacts/   (on NVMe RAID 0 mount)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv(
    'AI_TRADER_ARTIFACTS_DIR',
    str(PROJECT_ROOT / 'data' / 'artifacts'),
))
OPTUNA_DB_SRC = PROJECT_ROOT / 'data' / 'optuna_orchestrator.db'

# Model families serialised as joblib (sklearn).  TFT/OFT use .pt (torch)
# and are handled separately if needed.
_JOBLIB_KEYS = frozenset({'base', 'trend', 'futures', 'scalping', 'meta', 'regime'})


def _best_artifact_for_key(key: str) -> tuple[Path, Path] | None:
    """Return (model_path, meta_path) for the highest-scoring trained variant
    of *key*, or None if nothing is on disk yet.

    Scoring priority: walk_forward_mean_acc > accuracy > 0.0.
    """
    from src.utils.model_paths import (
        list_per_tf_artifacts,
        MODELS_DIR,
        LEGACY_MODEL_NAME,
        LEGACY_META_NAME,
    )

    candidates: list[tuple[float, Path, Path]] = []

    # Per-TF variants written by the multi-TF trainer
    for _tf, model_path, meta_path in list_per_tf_artifacts(key):
        if not model_path.exists() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        score = float(meta.get('walk_forward_mean_acc') or meta.get('accuracy') or 0.0)
        candidates.append((score, model_path, meta_path))

    # Canonical / legacy file (backwards-compat name written by trainer)
    legacy_model = MODELS_DIR / LEGACY_MODEL_NAME.get(key, '')
    legacy_meta  = MODELS_DIR / LEGACY_META_NAME.get(key, '')
    if legacy_model.exists() and legacy_meta.exists():
        try:
            meta  = json.loads(legacy_meta.read_text(encoding='utf-8'))
            score = float(meta.get('walk_forward_mean_acc') or meta.get('accuracy') or 0.0)
            candidates.append((score, legacy_model, legacy_meta))
        except (OSError, json.JSONDecodeError):
            pass

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    _, best_model, best_meta = candidates[0]
    return best_model, best_meta


def export_artifacts() -> dict:
    """Package the lightweight training outputs into ARTIFACTS_DIR.

    Safe to call after a partial sweep -- keys with no on-disk model are
    skipped with a warning rather than raising.  Always returns a summary
    dict so the caller can embed it in the pipeline status file.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    errors:   list[str] = []
    best_base: tuple[Path, Path] | None = None

    # 1. Per-key best model
    for key in sorted(_JOBLIB_KEYS):
        result = _best_artifact_for_key(key)
        if result is None:
            logger.warning("artifact_exporter: no trained model found for key=%s -- skipping", key)
            continue

        model_src, meta_src = result
        dst_model = ARTIFACTS_DIR / f'{key}_best.joblib'
        dst_meta  = ARTIFACTS_DIR / f'{key}_best_meta.json'
        try:
            shutil.copy2(model_src, dst_model)
            shutil.copy2(meta_src,  dst_meta)
            exported.extend([str(dst_model), str(dst_meta)])
            logger.info("artifact_exporter: exported %s -> %s (src=%s)",
                        key, dst_model.name, model_src.name)
            if key == 'base':
                best_base = (model_src, meta_src)
        except OSError as exc:
            errors.append(f'{key}: {exc}')
            logger.error("artifact_exporter: copy failed for key=%s: %s", key, exc)

    # 2. best_model.joblib -- generic alias for the primary trading model (base)
    if best_base:
        bm_dst   = ARTIFACTS_DIR / 'best_model.joblib'
        bm_meta  = ARTIFACTS_DIR / 'best_model_meta.json'
        try:
            shutil.copy2(best_base[0], bm_dst)
            shutil.copy2(best_base[1], bm_meta)
            exported.extend([str(bm_dst), str(bm_meta)])
            logger.info("artifact_exporter: exported best_model.joblib (src=%s)", best_base[0].name)
        except OSError as exc:
            errors.append(f'best_model: {exc}')
            logger.error("artifact_exporter: copy failed for best_model: %s", exc)

    # 3. Optuna SQLite study
    if OPTUNA_DB_SRC.exists():
        dst_db = ARTIFACTS_DIR / 'optuna.db'
        try:
            shutil.copy2(OPTUNA_DB_SRC, dst_db)
            exported.append(str(dst_db))
            logger.info("artifact_exporter: exported optuna.db")
        except OSError as exc:
            errors.append(f'optuna.db: {exc}')
            logger.error("artifact_exporter: copy failed for optuna.db: %s", exc)
    else:
        logger.warning("artifact_exporter: optuna_orchestrator.db not found -- skipping")

    summary = {
        'artifacts_dir':   str(ARTIFACTS_DIR),
        'exported_count':  len(exported),
        'exported':        exported,
        'errors':          errors,
    }
    if errors:
        logger.warning("artifact_exporter: %d export(s) failed: %s", len(errors), errors)
    else:
        logger.info("artifact_exporter: %d artifacts written to %s",
                    len(exported), ARTIFACTS_DIR)
    return summary
