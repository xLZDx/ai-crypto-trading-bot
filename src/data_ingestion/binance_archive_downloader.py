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
RAW_DIR        = PROJECT_ROOT / "data" / "raw"                  # 1m / 1h / 1d / 1mo live here
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"   # 1s lives here
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"
LISTING_CACHE  = PROJECT_ROOT / "data" / "binance_listing_dates.json"

# Parallelism. Network-bound, so 8 is safe; user can go higher with env var.
MAX_WORKERS = int(os.getenv("ARCHIVE_MAX_WORKERS", "8"))
RETRY_LIMIT = 4
BASE_URL    = "https://data.binance.vision/data/spot/monthly/klines"
CSV_HEADER  = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote",
]

# Binance archive supports these kline intervals.
SUPPORTED_TF = ("1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                "6h", "8h", "12h", "1d", "3d", "1w", "1mo")


def _output_dir_for(timeframe: str) -> Path:
    """1s lives in data/raw/historical/; everything else in data/raw/."""
    return HISTORICAL_DIR if timeframe == "1s" else RAW_DIR


def _output_filename(symbol: str, timeframe: str) -> str:
    """1s preserves legacy `_spot_1s.csv.gz`; others use `_{tf}.csv.gz`."""
    safe = symbol.replace("/", "_")
    if timeframe == "1s":
        return f"{safe}_spot_1s.csv.gz"
    return f"{safe}_{timeframe}.csv.gz"


def _watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        with WATCHLIST_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]


def _archive_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def _gz_path(symbol: str, timeframe: str = "1s") -> Path:
    return _output_dir_for(timeframe) / _output_filename(symbol, timeframe)


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


def _zip_url(sym_bin: str, year: int, month: int, timeframe: str) -> str:
    fname = f"{sym_bin}-{timeframe}-{year:04d}-{month:02d}.zip"
    return f"{BASE_URL}/{sym_bin}/{timeframe}/{fname}"


def _zip_exists(sym_bin: str, year: int, month: int, timeframe: str = "1s") -> bool:
    """Cheap existence probe via HEAD — avoids a full GET → 404 round-trip.

    data.binance.vision answers HEAD with the same status code and
    Content-Length headers it would for GET, so this is ~10× cheaper for
    "not in archive" months (the dominant case for delisted coins).
    """
    url = _zip_url(sym_bin, year, month, timeframe)
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False  # treat network errors as "doesn't exist" for this run


def _download_month_zip(sym_bin: str, year: int, month: int, timeframe: str = "1s",
                        skip_if_missing: bool = True) -> bytes | None:
    """GET the zip if it exists. Returns None on 404 / persistent errors.

    With `skip_if_missing=True` (default), we HEAD first to avoid a
    body-transfer for missing months.
    """
    if skip_if_missing and not _zip_exists(sym_bin, year, month, timeframe):
        return None

    url = _zip_url(sym_bin, year, month, timeframe)
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


# ─── Listing-date cache (skip months before a coin existed) ─────────────────

def _load_listing_cache() -> dict[str, str]:
    """Map symbol → first-known yyyy-mm. Updated lazily as we successfully download."""
    if not LISTING_CACHE.exists():
        return {}
    try:
        return json.loads(LISTING_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_listing_cache(cache: dict[str, str]) -> None:
    try:
        LISTING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LISTING_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("could not save listing cache: %s", exc)


def _filter_months_by_listing(symbol: str, months: list[tuple[int, int]],
                              cache: dict[str, str]) -> list[tuple[int, int]]:
    """Drop months strictly before the cached listing date for this symbol."""
    listed = cache.get(symbol)
    if not listed:
        return months
    try:
        ly, lm = (int(p) for p in listed.split("-"))
    except Exception:
        return months
    return [(y, m) for (y, m) in months if (y, m) >= (ly, lm)]


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
    gz_path.parent.mkdir(parents=True, exist_ok=True)
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


def download_symbol(symbol: str, timeframe: str = "1s",
                    listing_cache: dict[str, str] | None = None) -> dict:
    gz       = _gz_path(symbol, timeframe)
    sym_bin  = _archive_symbol(symbol)
    last_ts  = _last_timestamp(gz)
    months   = _months_to_download(last_ts)
    first_ts = _first_timestamp(gz)

    if listing_cache is None:
        listing_cache = _load_listing_cache()
    pre_skip = len(months)
    months = _filter_months_by_listing(symbol, months, listing_cache)
    skipped_pre = pre_skip - len(months)

    logger.info("[%s/%s] File: %s | Stored: %s → %s | Months to fetch: %d (skipped %d pre-listing)",
                symbol, timeframe, gz.name,
                first_ts.strftime("%Y-%m") if first_ts else "none",
                last_ts.strftime("%Y-%m-%d") if last_ts else "none",
                len(months), skipped_pre)

    if not months:
        logger.info("[%s/%s] Already up-to-date.", symbol, timeframe)
        return {"symbol": symbol, "timeframe": timeframe,
                "months_downloaded": 0, "rows_written": 0,
                "months_skipped_404": 0}

    total_rows = 0
    downloaded = 0
    skipped_404 = 0
    first_success: tuple[int, int] | None = None

    for year, month in months:
        zip_bytes = _download_month_zip(sym_bin, year, month, timeframe,
                                        skip_if_missing=True)
        if zip_bytes is None:
            skipped_404 += 1
            continue
        rows = _append_zip_to_gz(zip_bytes, gz, after_ts=last_ts)
        total_rows += rows
        downloaded += 1
        if first_success is None:
            first_success = (year, month)
        last_ts = datetime(year, month % 12 + 1 if month < 12 else 1,
                           1, tzinfo=timezone.utc) - timedelta(seconds=1)
        size_mb = round(gz.stat().st_size / 1024**2, 0)
        logger.info("[%s/%s] %04d-%02d +%d rows | file %.0f MB",
                    symbol, timeframe, year, month, rows, size_mb)
        time.sleep(0.1)  # gentle on the archive CDN

    # Cache first successful month so future runs skip pre-listing months.
    if first_success and symbol not in listing_cache:
        listing_cache[symbol] = f"{first_success[0]:04d}-{first_success[1]:02d}"
        _save_listing_cache(listing_cache)

    size_gb = round(gz.stat().st_size / 1024**3, 2) if gz.exists() else 0
    logger.info("[%s/%s] DONE — %d months, %d rows, %d 404s, %.2f GB",
                symbol, timeframe, downloaded, total_rows, skipped_404, size_gb)
    return {"symbol": symbol, "timeframe": timeframe,
            "months_downloaded": downloaded, "rows_written": total_rows,
            "months_skipped_404": skipped_404}


def download_all(symbols: list[str] | None = None, timeframe: str = "1s") -> None:
    if timeframe not in SUPPORTED_TF:
        raise ValueError(f"timeframe {timeframe!r} not in {SUPPORTED_TF}")
    if symbols is None:
        symbols = _watchlist()

    logger.info("=" * 60)
    logger.info("Binance Archive Downloader  —  %d symbols  tf=%s  workers=%d",
                len(symbols), timeframe, MAX_WORKERS)
    logger.info("Archive: %s/{SYM}/%s/", BASE_URL, timeframe)
    logger.info("Output: %s", _output_dir_for(timeframe))
    logger.info("=" * 60)

    listing_cache = _load_listing_cache()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_symbol, s, timeframe, listing_cache): s
                   for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r = fut.result()
                logger.info("DONE [%s/%s] %d months  %d rows  %d 404s",
                            sym, timeframe,
                            r["months_downloaded"], r["rows_written"],
                            r.get("months_skipped_404", 0))
            except Exception as exc:
                logger.error("FAIL [%s/%s] %s", sym, timeframe, exc)

    logger.info("=" * 60)
    logger.info("ALL SYMBOLS COMPLETE  (timeframe=%s)", timeframe)
    logger.info("=" * 60)


def download_all_timeframes_parallel(symbols: list[str] | None = None,
                                     timeframes: list[str] | None = None) -> None:
    """Cross-timeframe parallelism — schedules every (sym, tf) into one big pool.

    Faster than sequentially calling download_all for each tf, because
    the worker pool stays saturated even when one tf finishes faster than
    others (e.g. 1mo has only ~80 months vs 1m has thousands).
    """
    if symbols is None:
        symbols = _watchlist()
    if timeframes is None:
        timeframes = ["1m", "1d", "1mo"]
    invalid = [tf for tf in timeframes if tf not in SUPPORTED_TF]
    if invalid:
        raise ValueError(f"timeframes {invalid} not in {SUPPORTED_TF}")

    logger.info("=" * 60)
    logger.info("Binance Archive Downloader (cross-TF) — %d sym × %d tf × workers=%d",
                len(symbols), len(timeframes), MAX_WORKERS)
    logger.info("=" * 60)

    listing_cache = _load_listing_cache()
    jobs = [(s, tf) for s in symbols for tf in timeframes]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_symbol, s, tf, listing_cache): (s, tf)
                   for s, tf in jobs}
        for fut in as_completed(futures):
            s, tf = futures[fut]
            try:
                r = fut.result()
                logger.info("DONE [%s/%s] %d months  %d rows  %d 404s",
                            s, tf, r["months_downloaded"], r["rows_written"],
                            r.get("months_skipped_404", 0))
            except Exception as exc:
                logger.error("FAIL [%s/%s] %s", s, tf, exc)

    logger.info("=" * 60)
    logger.info("ALL (symbol, tf) JOBS COMPLETE")
    logger.info("=" * 60)


def main() -> None:
    global MAX_WORKERS
    parser = argparse.ArgumentParser(
        description="Download Binance archive klines from data.binance.vision"
    )
    parser.add_argument("--symbols", nargs="+", metavar="SYM",
                        help="e.g. BTC/USDT ETH/USDT  (default: all watchlist coins)")
    parser.add_argument("--timeframe", default="1s", choices=SUPPORTED_TF,
                        help="Kline interval (default 1s)")
    parser.add_argument("--all-timeframes", nargs="+",
                        help="Download multiple timeframes IN PARALLEL, e.g. "
                             "--all-timeframes 1m 1d 1mo")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override MAX_WORKERS (default 8)")
    args = parser.parse_args()

    if args.workers:
        MAX_WORKERS = max(1, int(args.workers))

    if args.all_timeframes:
        download_all_timeframes_parallel(symbols=args.symbols,
                                         timeframes=args.all_timeframes)
    else:
        download_all(symbols=args.symbols, timeframe=args.timeframe)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
