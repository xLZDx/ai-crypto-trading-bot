"""
One-time migration: convert *_spot_1s.csv.gz files in data/raw/historical/
into the Parquet store at data/parquet/{symbol}/{YYYY-MM}/data.parquet.

Idempotent — already-converted months are skipped. Safe to re-run.

Usage:
    python scripts/migrate_1sec_to_parquet.py
    python scripts/migrate_1sec_to_parquet.py --symbol BTC/USDT
    python scripts/migrate_1sec_to_parquet.py --raw-dir D:/other/path
    python scripts/migrate_1sec_to_parquet.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.parquet_store import ParquetStore, DEFAULT_BASE_DIR

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "historical"

logger = logging.getLogger("migrate_1sec")


def discover_csv_files(raw_dir: Path, suffix: str = "_spot_1s.csv.gz") -> list[tuple[str, Path]]:
    """Return list of (symbol, path) pairs for every matching CSV."""
    if not raw_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for path in sorted(raw_dir.glob(f"*{suffix}")):
        # File name: BTC_USDT_spot_1s.csv.gz → symbol BTC/USDT
        stem = path.name.removesuffix(suffix)
        if "_" not in stem:
            continue
        base, quote = stem.rsplit("_", 1)
        symbol = f"{base}/{quote}"
        out.append((symbol, path))
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser(description="Migrate 1-sec CSV history into the Parquet store")
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing *_spot_1s.csv.gz")
    p.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR), help="Parquet store base directory")
    p.add_argument("--symbol",  default="", help="Restrict to one symbol (e.g. BTC/USDT)")
    p.add_argument("--suffix",  default="_spot_1s.csv.gz", help="CSV file suffix to match")
    p.add_argument("--dry-run", action="store_true", help="List work without ingesting")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    files   = discover_csv_files(raw_dir, args.suffix)

    if args.symbol:
        files = [(s, p) for (s, p) in files if s == args.symbol]

    if not files:
        logger.warning("No CSVs matched in %s", raw_dir)
        return 1

    total_size = sum(f.stat().st_size for (_, f) in files)
    logger.info("Found %d CSV file(s), total compressed size %.2f GB", len(files), total_size / 1e9)

    if args.dry_run:
        for sym, path in files:
            logger.info("  • %s → %s", sym, path.name)
        return 0

    store = ParquetStore(out_dir)
    grand_rows = 0
    grand_months = 0
    grand_skipped = 0
    t0 = time.time()

    for i, (sym, path) in enumerate(files, 1):
        logger.info("[%d/%d] Ingesting %s (%s, %.0f MB)…",
                    i, len(files), sym, path.name, path.stat().st_size / 1e6)
        try:
            res = store.ingest_csv(path, sym)
            grand_rows    += res["rows_total"]
            grand_months  += res["months_written"]
            grand_skipped += res["skipped_months"]
            logger.info(
                "    ↳ %d months written, %d skipped, %d rows total",
                res["months_written"], res["skipped_months"], res["rows_total"],
            )
        except Exception as exc:
            logger.exception("    ↳ FAILED: %s", exc)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done in %.1f s — %d months written, %d skipped, %d rows total",
                elapsed, grand_months, grand_skipped, grand_rows)
    logger.info("Parquet store: %s", out_dir)

    # Summary
    summary = store.status()
    logger.info("Symbols on disk: %d, total %.2f GB", summary["symbols"], summary["size_gb"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
