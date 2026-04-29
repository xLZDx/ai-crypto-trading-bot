"""
Binance Archive Downloader — downloads full historical 1-second OHLCV data
from https://data.binance.vision for all watchlist coins.

The standard Binance klines API only serves the last 24-48 h of 1s data.
The data.binance.vision archive has complete monthly zip files going back
to each coin's listing date.

Strategy
--------
- For each symbol, inspect the existing _spot_1s.csv.gz to find the last
  stored timestamp, then download only months that are missing.
- Data is appended in chronological order; nothing is re-downloaded.
- Multiple symbols are processed concurrently (MAX_WORKERS threads).

Output files
------------
  data/raw/{SYMBOL}_spot_1s.csv.gz   (historical archive, appended here)

Usage
-----
  python -m src.data_ingestion.binance_archive_downloader
  python -m src.data_ingestion.binance_archive_downloader --symbols BTC/USDT ETH/USDT
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("archive_dl")

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RAW_DIR        = PROJECT_ROOT / "data" / "raw" / "historical"   # pre-2026 archive lives here
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"

MAX_WORKERS = 3
RETRY_LIMIT = 4
BASE_URL    = "https://data.binance.vision/data/spot/monthly/klines"
CSV_HEADER  = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote",
]


def _watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        with WATCHLIST_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]


def _archive_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def _gz_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol.replace('/', '_')}_spot_1s.csv.gz"


def _last_timestamp(gz: Path) -> datetime | None:
    if not gz.exists():
        return None
    try:
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            last = deque(f, maxlen=1)
        if not last:
            return None
        ts_str = last[0].split(",")[0].strip()
        if ts_str == "timestamp":
            return None
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.warning("Could not read last timestamp from %s: %s", gz.name, exc)
        logger.warning("Corrupted archive detected. Deleting %s to force redownload...", gz.name)
        try:
            gz.unlink()
        except OSError as del_exc:
            logger.error("Failed to delete corrupted file: %s", del_exc)
        return None


def _first_timestamp(gz: Path) -> datetime | None:
    if not gz.exists():
        return None
    try:
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            next(f)
            first = next(f, None)
        if not first:
            return None
        return datetime.strptime(first.split(",")[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            gz.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _months_to_download(last_ts: datetime | None) -> list[tuple[int, int]]:
    """Return (year, month) list of months to download, starting after last_ts."""
    if last_ts is None:
        start = datetime(2017, 8, 1, tzinfo=timezone.utc)
    else:
        nxt = last_ts.replace(day=1) + timedelta(days=32)
        start = nxt.replace(day=1)

    # Stop at last *completed* month (never download current month mid-way)
    now = datetime.now(timezone.utc)
    end_month = (now.replace(day=1) - timedelta(days=1)).replace(day=1)

    months = []
    cur = start
    while cur <= end_month:
        months.append((cur.year, cur.month))
        nxt = cur.replace(day=1) + timedelta(days=32)
        cur = nxt.replace(day=1)
    return months


def _download_month_zip(sym_bin: str, year: int, month: int) -> bytes | None:
    fname = f"{sym_bin}-1s-{year:04d}-{month:02d}.zip"
    url   = f"{BASE_URL}/{sym_bin}/1s/{fname}"
    delay = 2.0
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            logger.warning("[%s] %04d-%02d attempt %d: %s", sym_bin, year, month, attempt + 1, exc)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def _convert_row(row: list) -> list | None:
    try:
        ts_ms = int(row[0])
        dt    = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return [dt,
                float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                float(row[5]), float(row[7]), int(row[8]),
                float(row[9]), float(row[10])]
    except (IndexError, ValueError, OSError, TypeError):
        return None


def _append_zip_to_gz(zip_bytes: bytes, gz_path: Path, after_ts: datetime | None) -> int:
    """Extract zip → convert → append to gz. Returns rows written."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    mode    = "at" if gz_path.exists() else "wt"
    written = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
        if csv_name is None:
            return 0
        with zf.open(csv_name) as raw_csv:
            text_io = io.TextIOWrapper(raw_csv, encoding="utf-8")
            reader  = csv.reader(text_io)
            with gzip.open(gz_path, mode, newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                if mode == "wt":
                    writer.writerow(CSV_HEADER)
                for row in reader:
                    if not row or not row[0].isdigit():
                        continue
                    converted = _convert_row(row)
                    if converted is None:
                        continue
                    if after_ts is not None:
                        try:
                            row_dt = datetime.strptime(converted[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                            if row_dt <= after_ts:
                                continue
                        except ValueError:
                            continue
                    writer.writerow(converted)
                    written += 1
    return written


def download_symbol(symbol: str) -> dict:
    gz       = _gz_path(symbol)
    sym_bin  = _archive_symbol(symbol)
    last_ts  = _last_timestamp(gz)
    months   = _months_to_download(last_ts)
    first_ts = _first_timestamp(gz)

    logger.info("[%s] File: %s | Stored: %s → %s | Months to fetch: %d",
                symbol, gz.name,
                first_ts.strftime("%Y-%m") if first_ts else "none",
                last_ts.strftime("%Y-%m-%d") if last_ts else "none",
                len(months))

    if not months:
        logger.info("[%s] Already up-to-date.", symbol)
        return {"symbol": symbol, "months_downloaded": 0, "rows_written": 0}

    total_rows = 0
    downloaded = 0

    for year, month in months:
        logger.info("[%s] Downloading %04d-%02d …", symbol, year, month)
        zip_bytes = _download_month_zip(sym_bin, year, month)
        if zip_bytes is None:
            logger.info("[%s] %04d-%02d not in archive — skipping.", symbol, year, month)
            continue
        rows = _append_zip_to_gz(zip_bytes, gz, after_ts=last_ts)
        total_rows += rows
        downloaded += 1
        # Advance cutoff so next month doesn't re-write rows
        last_ts = datetime(year, month % 12 + 1 if month < 12 else 1,
                           1, tzinfo=timezone.utc) - timedelta(seconds=1)
        size_mb = round(gz.stat().st_size / 1024**2, 0)
        logger.info("[%s] %04d-%02d +%d rows | file %.0f MB", symbol, year, month, rows, size_mb)
        time.sleep(0.3)

    size_gb = round(gz.stat().st_size / 1024**3, 2) if gz.exists() else 0
    logger.info("[%s] DONE — %d months, %d rows, %.2f GB", symbol, downloaded, total_rows, size_gb)
    return {"symbol": symbol, "months_downloaded": downloaded, "rows_written": total_rows}


def download_all(symbols: list[str] | None = None) -> None:
    if symbols is None:
        symbols = _watchlist()

    logger.info("=" * 60)
    logger.info("Binance Archive 1s Downloader  —  %d symbols", len(symbols))
    logger.info("Archive: %s", BASE_URL)
    logger.info("=" * 60)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_symbol, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r = fut.result()
                logger.info("✅ [%s] %d months  %d rows", sym, r["months_downloaded"], r["rows_written"])
            except Exception as exc:
                logger.error("❌ [%s] %s", sym, exc)

    logger.info("=" * 60)
    logger.info("ALL SYMBOLS COMPLETE")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance 1s archive data from data.binance.vision")
    parser.add_argument("--symbols", nargs="+", metavar="SYM",
                        help="e.g. BTC/USDT ETH/USDT  (default: all watchlist coins)")
    args = parser.parse_args()
    download_all(symbols=args.symbols)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
