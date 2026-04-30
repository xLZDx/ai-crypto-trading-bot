"""
Historical data ingestion pipeline — imports CSV.gz archives into QuestDB.

Reads existing data/raw/historical/*_spot_1s.csv.gz files and
data/raw/*_1m.csv.gz / *_1h.csv.gz / *_1d.csv.gz files.

Deduplication is handled by QuestDB's DEDUP UPSERT KEYS on (ts, symbol, timeframe).

Usage:
    python -m src.database.ingest_pipeline                      # all symbols/timeframes
    python -m src.database.ingest_pipeline --symbol BTC/USDT   # single symbol
    python -m src.database.ingest_pipeline --timeframe 1m       # single timeframe only
    python -m src.database.ingest_pipeline --since 2025-01-01  # only after date
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR  = PROJECT_ROOT / "data" / "raw"
HIST_DIR = RAW_DIR / "historical"
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"

BATCH_SIZE = 10_000   # rows per ILP write call
LOG_EVERY  = 100_000  # log progress every N rows


def _watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        with WATCHLIST_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _iter_gz(path: Path, since: datetime | None = None) -> Iterator[dict]:
    """Yield row dicts from a OHLCV csv.gz file."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "").strip()
            if not ts_str or ts_str == "timestamp":
                continue
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if since is not None and dt <= since:
                continue
            yield {
                "timestamp": dt,
                "open":   float(row.get("open", 0) or 0),
                "high":   float(row.get("high", 0) or 0),
                "low":    float(row.get("low", 0) or 0),
                "close":  float(row.get("close", 0) or 0),
                "volume": float(row.get("volume", 0) or 0),
                "funding_rate": float(row.get("funding_rate", 0) or 0),
            }


def ingest_file(path: Path, symbol: str, timeframe: str, client, since: datetime | None = None) -> int:
    """Ingest one csv.gz file into QuestDB. Returns rows written."""
    if not path.exists():
        return 0

    from src.database.questdb_client import _to_ns, _tag

    # Check last stored timestamp — skip if already up to date
    if since is None:
        since = client.get_latest_candle_ts(symbol, timeframe)

    size_mb = round(path.stat().st_size / 1e6, 1)
    logger.info("[%s/%s] Ingesting %s (%.0f MB) | since=%s",
                symbol, timeframe, path.name, size_mb,
                since.strftime("%Y-%m-%d") if since else "beginning")

    sym_tag  = _tag(symbol)
    tf_tag   = _tag(timeframe)
    lines    = []
    written  = 0
    skipped  = 0

    for bar in _iter_gz(path, since):
        ts_ns = _to_ns(bar["timestamp"])
        if ts_ns is None:
            skipped += 1
            continue
        lines.append(
            f"market_data,symbol={sym_tag},timeframe={tf_tag} "
            f"open={bar['open']},"
            f"high={bar['high']},"
            f"low={bar['low']},"
            f"close={bar['close']},"
            f"volume={bar['volume']},"
            f"funding_rate={bar['funding_rate']} "
            f"{ts_ns}"
        )
        if len(lines) >= BATCH_SIZE:
            if client.write_ilp(lines):
                written += len(lines)
            else:
                logger.warning("ILP write failed at row %d — retrying once", written)
                time.sleep(1)
                client.write_ilp(lines)   # one retry
            lines = []
            if written % LOG_EVERY < BATCH_SIZE:
                logger.info("  … %d rows written", written)

    if lines:
        if client.write_ilp(lines):
            written += len(lines)

    logger.info("[%s/%s] Done — %d rows written, %d skipped", symbol, timeframe, written, skipped)
    return written


def ingest_symbol(symbol: str, timeframes: list[str] | None, client, since: datetime | None) -> dict:
    """Ingest all timeframe files for one symbol. Returns summary dict."""
    safe = symbol.replace("/", "_")
    summary: dict[str, int] = {}

    tf_map = {
        "1s": HIST_DIR / f"{safe}_spot_1s.csv.gz",
        "1m": RAW_DIR  / f"{safe}_1m.csv.gz",
        "1h": RAW_DIR  / f"{safe}_1h.csv.gz",
        "1d": RAW_DIR  / f"{safe}_1d.csv.gz",
    }

    for tf, path in tf_map.items():
        if timeframes and tf not in timeframes:
            continue
        if not path.exists():
            logger.debug("[%s/%s] File not found: %s", symbol, tf, path.name)
            continue
        rows = ingest_file(path, symbol, tf, client, since=since)
        summary[tf] = rows

    return summary


def run(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    since: datetime | None = None,
) -> None:
    from src.database.questdb_client import get_client
    from src.database.schema import create_all

    client = get_client()
    if not client.is_available():
        logger.error("QuestDB is not running. Start it with: docker-compose up -d questdb")
        return

    logger.info("Ensuring tables exist…")
    create_all(client)

    if symbols is None:
        symbols = _watchlist()

    logger.info("=" * 60)
    logger.info("Starting ingestion: %d symbols, timeframes=%s", len(symbols), timeframes or "all")
    logger.info("=" * 60)

    total = 0
    for sym in symbols:
        summary = ingest_symbol(sym, timeframes, client, since)
        rows = sum(summary.values())
        total += rows
        logger.info("✓ %s — %s rows total", sym, rows)

    logger.info("=" * 60)
    logger.info("DONE — %d total rows ingested", total)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest historical CSV.gz data into QuestDB")
    parser.add_argument("--symbol",    nargs="+", metavar="SYM",  help="e.g. BTC/USDT ETH/USDT")
    parser.add_argument("--timeframe", nargs="+", metavar="TF",   help="e.g. 1m 1h 1d (default: all)")
    parser.add_argument("--since",     metavar="DATE",             help="e.g. 2025-01-01 (skip older)")
    args = parser.parse_args()

    since_dt = None
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    run(
        symbols=args.symbol,
        timeframes=args.timeframe,
        since=since_dt,
    )
