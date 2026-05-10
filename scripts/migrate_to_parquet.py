"""
General-purpose CSV → Parquet migration for any timeframe.

Layout written:
    data/parquet/{SYMBOL}/{TIMEFRAME}/yyyymm=YYYY-MM/data_*.parquet

Discovers files by suffix in `data/raw/`:
    {SYMBOL}_{TIMEFRAME}.csv.gz       # e.g. BTC_USDT_1m.csv.gz, BTC_USDT_1d.csv.gz
    {SYMBOL}_funding.csv.gz           # funding rate data

Idempotent — already-converted months are auto-skipped.
Single-pass (no ORDER BY) — relies on the input being chronological.

Usage:
    python scripts/migrate_to_parquet.py --timeframe 1m
    python scripts/migrate_to_parquet.py --timeframe 1d
    python scripts/migrate_to_parquet.py --timeframe funding
    python scripts/migrate_to_parquet.py --timeframe 1m --symbol BTC/USDT
    python scripts/migrate_to_parquet.py --timeframe 1m --dry-run
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

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_TIMEFRAMES = ("1s", "1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M", "funding")

logger = logging.getLogger("migrate_to_parquet")


def discover_csv_files(raw_dir: Path, timeframe: str) -> list[tuple[str, Path]]:
    """Return list of (symbol, csv_path) for every {SYMBOL}_{TIMEFRAME}.csv.gz."""
    if not raw_dir.exists():
        return []
    suffix = f"_{timeframe}.csv.gz"
    out: list[tuple[str, Path]] = []
    for path in sorted(raw_dir.glob(f"*{suffix}")):
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
    p = argparse.ArgumentParser(description="Migrate {tf} CSVs to Parquet")
    p.add_argument("--timeframe", required=True, choices=SUPPORTED_TIMEFRAMES,
                   help="Timeframe identifier (e.g. 1m, 1d, funding)")
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                   help="Directory containing CSV.gz files")
    p.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR),
                   help="Parquet store base directory")
    p.add_argument("--symbol", default="",
                   help="Restrict to one symbol (e.g. BTC/USDT)")
    p.add_argument("--dry-run", action="store_true", help="List work without ingesting")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    files   = discover_csv_files(raw_dir, args.timeframe)

    if args.symbol:
        files = [(s, p) for (s, p) in files if s == args.symbol]

    if not files:
        logger.warning("No %s CSVs matched in %s", args.timeframe, raw_dir)
        return 1

    total_size = sum(f.stat().st_size for (_, f) in files)
    logger.info("Found %d %s file(s), total compressed size %.2f GB",
                len(files), args.timeframe, total_size / 1e9)

    if args.dry_run:
        for sym, path in files:
            logger.info("  - %s -> %s", sym, path.name)
        return 0

    store = ParquetStore(out_dir)
    grand_rows = 0
    grand_months = 0
    grand_skipped = 0
    t0 = time.time()

    for i, (sym, path) in enumerate(files, 1):
        logger.info("[%d/%d] Ingesting %s %s (%s, %.0f MB)...",
                    i, len(files), sym, args.timeframe, path.name,
                    path.stat().st_size / 1e6)
        try:
            res = store.ingest_csv(path, sym, timeframe=args.timeframe)
            grand_rows    += res["rows_total"]
            grand_months  += res["months_written"]
            grand_skipped += res["skipped_months"]
            logger.info(
                "    %d months written, %d skipped, %d rows now in store",
                res["months_written"], res["skipped_months"], res["rows_total"],
            )
        except Exception as exc:
            logger.exception("    FAILED: %s", exc)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done in %.1f s -- %d months written, %d skipped, %d rows total",
                elapsed, grand_months, grand_skipped, grand_rows)
    logger.info("Parquet store: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
