"""Audit + clean unused model artifacts in `models/`.

Strategy:
  1. List every file in `models/`.
  2. Cross-reference against what `inference_engine.py` and `ml_predictor.py`
     actually load.
  3. Move anything not loaded to `models/_archived/` (don't hard-delete) so
     the user can recover if needed.
  4. Print a summary.

Usage:
    python scripts/cleanup_models.py             # dry-run (default)
    python scripts/cleanup_models.py --apply     # actually move files
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"
ARCHIVE_DIR  = MODELS_DIR / "_archived"

# Filenames of models actively loaded by current code paths.
ACTIVE_LOADED = {
    # Phase 2
    "regime_classifier.joblib",
    "regime_classifier_meta.json",
    "oft_model.pt",
    # Legacy that's still wired to inference_engine / ml_predictor
    "tft_model.pt",
    "btc_rf_model.joblib",
    "btc_rf_model_meta.json",
    "scalping_model.joblib",
    "scalping_model_meta.json",
    "futures_short_model.joblib",
    "futures_short_model_meta.json",
    "trend_model.joblib",
    "trend_model_meta.json",
    "meta_labeler.joblib",
    "meta_labeler_meta.json",
}

logger = logging.getLogger("cleanup_models")


def discover_referenced(project_root: Path) -> set[str]:
    """Grep through src/ for any string ending in .joblib or .pt
    so we don't accidentally archive a checkpoint someone references."""
    refs = set(ACTIVE_LOADED)
    pat = re.compile(r"['\"]([\w./\\-]+\.(?:joblib|pt|json))['\"]")
    for p in (project_root / "src").rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in pat.finditer(text):
            refs.add(Path(m.group(1)).name)
    return refs


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually move files. Default = dry-run.")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        logger.warning("models/ dir doesn't exist -- nothing to do")
        return 0

    referenced = discover_referenced(PROJECT_ROOT)
    logger.info("References found: %d filenames", len(referenced))

    files = [p for p in MODELS_DIR.iterdir() if p.is_file()]
    archive_targets = [p for p in files if p.name not in referenced]

    logger.info("Total model files: %d", len(files))
    logger.info("Active / referenced: %d", len(files) - len(archive_targets))
    logger.info("Archive candidates: %d", len(archive_targets))
    for p in archive_targets:
        logger.info("  - %s (%.1f MB)", p.name, p.stat().st_size / 1e6)

    if args.apply and archive_targets:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p in archive_targets:
            dst = ARCHIVE_DIR / p.name
            try:
                shutil.move(str(p), str(dst))
                moved += 1
            except Exception as exc:
                logger.warning("could not move %s: %s", p.name, exc)
        logger.info("Moved %d files to %s", moved, ARCHIVE_DIR)
    elif archive_targets:
        logger.info("DRY-RUN -- re-run with --apply to actually move them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
