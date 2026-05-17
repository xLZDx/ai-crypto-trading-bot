"""One-shot migration of `data/raw/cryptocompare_news.csv` into the Parquet store.

Schema written:
    timestamp (DATETIME)  — published_at
    title     (VARCHAR)
    summary   (VARCHAR)
    source    (VARCHAR)
    score     (DOUBLE)    — placeholder, computed by FinBERT downstream

Layout:
    data/parquet/_news/yyyymm=YYYY-MM/data.parquet     (Hive-partitioned)

Idempotent: re-runs skip months already on disk.

Usage:
    python scripts/migrate_news_to_parquet.py
    python scripts/migrate_news_to_parquet.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.parquet_store import ParquetStore, DEFAULT_BASE_DIR

NEWS_CSV    = PROJECT_ROOT / "data" / "raw" / "cryptocompare_news.csv"
NEWS_SYMBOL = "_NEWS"          # pseudo-symbol so it gets its own dir
NEWS_TF     = "news"           # timeframe slot

logger = logging.getLogger("migrate_news")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(NEWS_CSV))
    p.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("News CSV not found at %s", csv_path)
        return 1

    size_mb = csv_path.stat().st_size / 1e6
    logger.info("Migrating %s (%.1f MB) -> Parquet", csv_path.name, size_mb)
    if args.dry_run:
        logger.info("DRY RUN -- no files written.")
        return 0

    store = ParquetStore(Path(args.out_dir))
    # The news CSV uses 'published_at' as its time column (ISO strings).
    # ParquetStore.ingest_csv defaults `timestamp_col="timestamp"` — we pass
    # `published_at` explicitly so the partition key is computed correctly.
    res = store.ingest_csv(
        csv_path, NEWS_SYMBOL,
        timestamp_col="published_at",
        timeframe=NEWS_TF,
    )
    logger.info("Done: %d months written, %d skipped, %d rows total",
                res["months_written"], res["skipped_months"], res["rows_total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
