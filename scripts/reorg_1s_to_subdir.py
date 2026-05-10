"""
One-off reorganization: move legacy 1-sec parquet partitions into the new
{SYMBOL}/1s/ subdirectory layout.

Background: the original `migrate_1sec_to_parquet.py` wrote to
`data/parquet/{SYMBOL}/yyyymm=YYYY-MM/`. After multi-timeframe support
landed, the canonical layout became
`data/parquet/{SYMBOL}/{TIMEFRAME}/yyyymm=YYYY-MM/`.

This script moves the legacy 1-sec partitions into `{SYMBOL}/1s/` so the
ParquetStore queries with `timeframe='1s'` find them.

Idempotent — symbols whose partitions are already under `1s/` are skipped.

Usage:
    python scripts/reorg_1s_to_subdir.py
    python scripts/reorg_1s_to_subdir.py --dry-run
    python scripts/reorg_1s_to_subdir.py --symbol BTC/USDT
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.parquet_store import DEFAULT_BASE_DIR, SUPPORTED_TIMEFRAMES

logger = logging.getLogger("reorg_1s")


def reorg_symbol(sym_dir: Path, *, dry_run: bool = False) -> int:
    """Move yyyymm=* dirs under sym_dir into sym_dir/1s/.

    Returns the number of partitions moved. If sym_dir/1s/ already exists
    AND there are no top-level legacy partitions, returns 0 (no-op).
    """
    legacy_partitions = [
        d for d in sym_dir.iterdir()
        if d.is_dir() and d.name.startswith("yyyymm=")
    ]
    if not legacy_partitions:
        return 0

    target_root = sym_dir / "1s"
    target_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in legacy_partitions:
        dst = target_root / src.name
        if dst.exists():
            logger.warning("[reorg] target exists, skipping: %s", dst)
            continue
        if dry_run:
            logger.info("[reorg] DRY-RUN move %s -> %s", src, dst)
        else:
            shutil.move(str(src), str(dst))
            logger.info("[reorg] moved %s -> %s", src.name, dst)
        moved += 1
    return moved


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    p.add_argument("--symbol", default="", help="Restrict to one symbol (e.g. BTC/USDT)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        logger.error("base dir does not exist: %s", base)
        return 1

    if args.symbol:
        target = args.symbol.replace("/", "_").upper()
        sym_dirs = [base / target] if (base / target).exists() else []
    else:
        sym_dirs = sorted(d for d in base.iterdir() if d.is_dir())

    grand_moved = 0
    for sym_dir in sym_dirs:
        # Skip if this is already a "timeframe" wrapper (shouldn't be), or
        # if there are no legacy yyyymm partitions at the top level.
        if sym_dir.name in SUPPORTED_TIMEFRAMES:
            continue
        moved = reorg_symbol(sym_dir, dry_run=args.dry_run)
        if moved:
            logger.info("[reorg] %s: %d partitions moved", sym_dir.name, moved)
        grand_moved += moved

    logger.info("=" * 60)
    logger.info("Total partitions moved: %d", grand_moved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
