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


def _log_ingestion(client, path: Path, symbol: str, timeframe: str,
                   rows_written: int, first_ts: datetime | None, last_ts: datetime | None) -> None:
    """Record a completed file ingestion to csv_ingestion_log."""
    try:
        from src.database.questdb_client import _to_ns, _tag, _now_ns
        file_size = path.stat().st_size if path.exists() else 0
        now_ns    = _now_ns()
        fts = _to_ns(first_ts) or now_ns
        lts = _to_ns(last_ts)  or now_ns
        line = (
            f"csv_ingestion_log,symbol={_tag(symbol)},timeframe={_tag(timeframe)} "
            f'filename="{path.name}",'
            f'source_path="{str(path).replace(chr(92), "/")}",'
            f"rows_written={rows_written}i,"
            f"file_size_bytes={file_size}i,"
            f"first_bar_ts={fts}i,"
            f"last_bar_ts={lts}i "
            f"{now_ns}"
        )
        client.write_ilp([line])
    except Exception as exc:
        logger.debug("Could not write to csv_ingestion_log: %s", exc)


def _get_ingestion_log(client, filename: str) -> dict | None:
    """Return the most recent ingestion record for a filename, or None."""
    try:
        rows = client.query(
            f"SELECT filename, rows_written, file_size_bytes, last_bar_ts "
            f"FROM csv_ingestion_log "
            f"WHERE filename='{filename}' "
            f"ORDER BY ts DESC LIMIT 1"
        )
        return rows[0] if rows else None
    except Exception:
        return None


def ingest_file(path: Path, symbol: str, timeframe: str, client,
                since: datetime | None = None, force: bool = False) -> int:
    """
    Ingest one csv.gz file into QuestDB. Returns rows written.

    Skips files that are already fully ingested (same filename + same file size).
    Records completion to csv_ingestion_log so the index stays up to date.
    Pass force=True to re-ingest even if the log says it's done.
    """
    if not path.exists():
        return 0

    from src.database.questdb_client import _to_ns, _tag

    file_size = path.stat().st_size

    # ── Skip check: already ingested at same file size ───────────────────────
    if not force and since is None:
        log_entry = _get_ingestion_log(client, path.name)
        if log_entry and log_entry.get("file_size_bytes") == file_size:
            logger.info("[%s/%s] %s already ingested (%d rows, %d MB) — skipping.",
                        symbol, timeframe, path.name,
                        log_entry["rows_written"], round(file_size / 1e6))
            return 0
        # File grew since last ingest → append only new rows
        if log_entry and log_entry.get("last_bar_ts"):
            try:
                last_ts_str = str(log_entry["last_bar_ts"])
                since = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                logger.info("[%s/%s] File grew since last ingest — appending from %s",
                            symbol, timeframe, since.strftime("%Y-%m-%d %H:%M"))
            except Exception:
                pass

    # Fallback: use last stored timestamp to avoid duplicates
    if since is None:
        since = client.get_latest_candle_ts(symbol, timeframe)

    size_mb = round(file_size / 1e6, 1)
    logger.info("[%s/%s] Ingesting %s (%.0f MB) | since=%s",
                symbol, timeframe, path.name, size_mb,
                since.strftime("%Y-%m-%d") if since else "beginning")

    sym_tag  = _tag(symbol)
    tf_tag   = _tag(timeframe)
    lines    = []
    written  = 0
    skipped  = 0
    first_ts: datetime | None = None
    last_ts:  datetime | None = None

    for bar in _iter_gz(path, since):
        ts_ns = _to_ns(bar["timestamp"])
        if ts_ns is None:
            skipped += 1
            continue
        if first_ts is None:
            first_ts = bar["timestamp"]
        last_ts = bar["timestamp"]
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
                client.write_ilp(lines)
            lines = []
            if written % LOG_EVERY < BATCH_SIZE:
                logger.info("  … %d rows written", written)

    if lines:
        if client.write_ilp(lines):
            written += len(lines)

    logger.info("[%s/%s] Done — %d rows written, %d skipped", symbol, timeframe, written, skipped)

    # ── Record to index ───────────────────────────────────────────────────────
    if written > 0:
        _log_ingestion(client, path, symbol, timeframe, written, first_ts, last_ts)

    return written


def ingest_symbol(symbol: str, timeframes: list[str] | None, client,
                  since: datetime | None, force: bool = False) -> dict:
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
        rows = ingest_file(path, symbol, tf, client, since=since, force=force)
        summary[tf] = rows

    return summary


def check_ingestion_status(symbols: list[str] | None = None,
                           timeframes: list[str] | None = None) -> list[dict]:
    """
    Return per-file ingestion status without writing anything.
    Each row: {filename, symbol, timeframe, exists, file_size_mb, ingested, rows_written, up_to_date}
    """
    from src.database.questdb_client import get_client
    client = get_client()
    available = client.is_available()

    if symbols is None:
        symbols = _watchlist()

    rows_out = []
    for sym in symbols:
        safe = sym.replace("/", "_")
        tf_map = {
            "1s": HIST_DIR / f"{safe}_spot_1s.csv.gz",
            "1m": RAW_DIR  / f"{safe}_1m.csv.gz",
            "1h": RAW_DIR  / f"{safe}_1h.csv.gz",
            "1d": RAW_DIR  / f"{safe}_1d.csv.gz",
        }
        for tf, path in tf_map.items():
            if timeframes and tf not in timeframes:
                continue
            exists = path.exists()
            file_size = path.stat().st_size if exists else 0
            log_entry = _get_ingestion_log(client, path.name) if available else None
            ingested = log_entry is not None
            logged_size = log_entry.get("file_size_bytes", 0) if log_entry else 0
            rows_written = log_entry.get("rows_written", 0) if log_entry else 0
            up_to_date = ingested and (logged_size == file_size)
            rows_out.append({
                "symbol":        sym,
                "timeframe":     tf,
                "filename":      path.name,
                "exists":        exists,
                "file_size_mb":  round(file_size / 1e6, 1) if exists else 0,
                "ingested":      ingested,
                "rows_written":  rows_written,
                "up_to_date":    up_to_date,
            })
    return rows_out


def run(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    since: datetime | None = None,
    force: bool = False,
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
    logger.info("Starting ingestion: %d symbols, timeframes=%s%s",
                len(symbols), timeframes or "all", " [FORCE]" if force else "")
    logger.info("=" * 60)

    total = 0
    for sym in symbols:
        summary = ingest_symbol(sym, timeframes, client, since, force=force)
        rows = sum(summary.values())
        total += rows
        logger.info("✓ %s — %d rows written", sym, rows)

    logger.info("=" * 60)
    logger.info("DONE — %d total rows written", total)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest historical CSV.gz data into QuestDB")
    parser.add_argument("--symbol",     nargs="+", metavar="SYM",  help="e.g. BTC/USDT ETH/USDT")
    parser.add_argument("--timeframe",  nargs="+", metavar="TF",   help="e.g. 1m 1h 1d (default: all)")
    parser.add_argument("--since",      metavar="DATE",             help="e.g. 2025-01-01 (skip older)")
    parser.add_argument("--force",      action="store_true",        help="Re-ingest even if already logged")
    parser.add_argument("--check-only", action="store_true",        help="Print pending files without writing")
    args = parser.parse_args()

    if args.check_only:
        status = check_ingestion_status(args.symbol, args.timeframe)
        done   = [r for r in status if r["up_to_date"]]
        pending = [r for r in status if r["exists"] and not r["up_to_date"]]
        missing = [r for r in status if not r["exists"]]
        print(f"\n{'File':<45} {'TF':<4} {'Size MB':>8}  {'Status'}")
        print("-" * 75)
        for r in status:
            if not r["exists"]:
                status_str = "  — no file"
            elif r["up_to_date"]:
                status_str = f"  ✓ done ({r['rows_written']:,} rows)"
            else:
                status_str = f"  ⏳ PENDING ({r['rows_written']:,} rows so far)"
            print(f"  {r['filename']:<43} {r['timeframe']:<4} {r['file_size_mb']:>8.1f}  {status_str}")
        print(f"\nSummary: {len(done)} done, {len(pending)} pending, {len(missing)} no file")
        sys.exit(0)

    since_dt = None
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    run(
        symbols=args.symbol,
        timeframes=args.timeframe,
        since=since_dt,
        force=args.force,
    )
