"""
Upload 1s CSV.gz files to Google Drive (G:) using chunked I/O.

Google Drive's virtual filesystem (GVFS) rejects Copy-Item / shutil.copy2
for large binary files (WinError 1117). Chunked open()+write() works fine.

Usage:
    python scripts/upload_1s_to_gdrive.py
    python scripts/upload_1s_to_gdrive.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
GDRIVE_DST   = Path(r"G:\AI BOT\AI trading assistance\data\archive_1s")
CHUNK_SIZE   = 4 * 1024 * 1024  # 4 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _chunk_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        for chunk in iter(lambda: f_in.read(CHUNK_SIZE), b""):
            f_out.write(chunk)


def upload(dry_run: bool = False) -> None:
    files = sorted(RAW_DIR.glob("*_1s.csv.gz")) + sorted(RAW_DIR.glob("*_spot_1s.csv.gz"))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    log.info("Files to upload: %d %s", len(unique), "(DRY RUN)" if dry_run else "")

    ok = 0
    fail = 0
    skip = 0

    for src in unique:
        dst = GDRIVE_DST / src.name
        size_mb = round(src.stat().st_size / 1024**2, 1)

        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            log.info("  SKIP %s (%s MB) -- already on GDrive", src.name, size_mb)
            skip += 1
            continue

        log.info("  %s (%s MB) -> GDrive...", src.name, size_mb)
        if dry_run:
            ok += 1
            continue

        try:
            _chunk_copy(src, dst)
            written_mb = round(dst.stat().st_size / 1024**2, 1)
            log.info("    Done: %s MB written", written_mb)
            ok += 1
        except Exception as e:
            log.error("    FAILED: %s", e)
            fail += 1

    log.info(
        "Upload %s: %d uploaded, %d skipped, %d failed",
        "simulated" if dry_run else "complete",
        ok, skip, fail,
    )
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload 1s CSV.gz files to Google Drive")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload(dry_run=args.dry_run)
