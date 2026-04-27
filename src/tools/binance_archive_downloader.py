"""
Bulk historical 1-second kline downloader from data.binance.vision.

Usage (defaults to WL coins, 1s interval, 2019-present, spot+futures):
    python src/tools/binance_archive_downloader.py

Override options:
    python src/tools/binance_archive_downloader.py --interval 1m --start-year 2017
    python src/tools/binance_archive_downloader.py --market spot
    python src/tools/binance_archive_downloader.py --start-year 2022
"""
import os
import sys
import json
import gzip
import zipfile
import requests
import io
import csv
import time
import argparse
import collections
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── helpers ──────────────────────────────────────────────────────────────────

def get_last_timestamp(filename):
    """Reads the last timestamp from a gzipped CSV to allow resuming."""
    try:
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            last_line = collections.deque(f, maxlen=1)[0]
        if last_line and not last_line.startswith('timestamp'):
            ts_str = last_line.split(',')[0]
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def download_monthly_archive(symbol, interval, year, month, market="spot", retries=3):
    """Fetches the official monthly ZIP from data.binance.vision with retry."""
    month_str = f"{month:02d}"
    if market == "futures":
        url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip")
    else:
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip")

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return None  # No data for this month — normal for early dates
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    [!] Network error (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [!] Failed after {retries} attempts: {e}")
                return None


def process_and_append(zip_content, output_filepath, is_first_write, last_dt=None):
    """Extracts CSV from ZIP in memory, formats it, and appends to the master file."""
    with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            text_stream = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.reader(text_stream)
            mode = 'wt' if is_first_write else 'at'
            with gzip.open(output_filepath, mode=mode, newline='', encoding='utf-8') as out_f:
                writer = csv.writer(out_f)
                if is_first_write:
                    writer.writerow([
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'quote_volume', 'trades_count',
                        'taker_buy_base', 'taker_buy_quote'
                    ])
                rows_written = 0
                for row in reader:
                    try:
                        if not row or len(row) < 11:
                            continue
                        ts_ms = int(row[0])
                        dt_obj = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                        if last_dt and dt_obj <= last_dt:
                            continue
                        dt = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([dt, row[1], row[2], row[3], row[4],
                                         row[5], row[7], row[8], row[9], row[10]])
                        rows_written += 1
                    except (ValueError, OSError, TypeError, IndexError):
                        continue
                return rows_written


def bulk_download(symbol_raw, interval, start_year, market="spot"):
    """Iterates through every month from start_year to present and builds a master dataset."""
    symbol = symbol_raw.replace('/', '')
    end_year = datetime.now(timezone.utc).year

    raw_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    output_filepath = os.path.join(raw_dir, f"{symbol_raw.replace('/', '_')}_{market}_{interval}.csv.gz")

    last_dt = get_last_timestamp(output_filepath) if os.path.exists(output_filepath) else None
    resume_info = f" (resuming from {last_dt.strftime('%Y-%m')})" if last_dt else ""

    print(f"\n  [{market.upper()}] {symbol_raw} [{interval}]{resume_info}")

    is_first_write = not os.path.exists(output_filepath) or last_dt is None
    total_months = 0
    total_rows = 0
    now = datetime.now(timezone.utc)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == now.year and month >= now.month:
                continue  # Skip current / future months
            if last_dt and (year < last_dt.year or (year == last_dt.year and month < last_dt.month)):
                is_first_write = False
                continue  # Already downloaded

            print(f"    {year}-{month:02d} ... ", end="", flush=True)
            zip_content = download_monthly_archive(symbol, interval, year, month, market)

            if zip_content:
                rows = process_and_append(zip_content, output_filepath, is_first_write, last_dt)
                is_first_write = False
                total_months += 1
                total_rows += rows
                size_mb = os.path.getsize(output_filepath) / 1_048_576
                print(f"OK  +{rows:,} rows  (file: {size_mb:.1f} MB)")
            else:
                print("not found")

    print(f"  ✅ Done {symbol_raw} [{market}]: {total_months} months, {total_rows:,} new rows → {output_filepath}")


# ── entry point ───────────────────────────────────────────────────────────────

def load_watchlist(default_symbols):
    wl_path = os.path.join(project_root, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r', encoding='utf-8') as f:
            symbols = json.load(f)
        print(f"Loaded {len(symbols)} coins from watchlist.json")
        return symbols
    print(f"watchlist.json not found — using defaults: {default_symbols}")
    return default_symbols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk download Binance kline archives")
    parser.add_argument('--interval',   default='1s',  help='Kline interval (default: 1s)')
    parser.add_argument('--start-year', default=2019,  type=int,
                        help='First year to download (default: 2019 — when 1s data became available)')
    parser.add_argument('--market',     default='both', choices=['spot', 'futures', 'both'],
                        help='Market(s) to download (default: both)')
    args = parser.parse_args()

    DEFAULTS = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT', 'ETH/USDT']
    symbols  = load_watchlist(DEFAULTS)
    markets  = ['spot', 'futures'] if args.market == 'both' else [args.market]

    print(f"\n{'='*60}")
    print(f"  Binance Archive Downloader")
    print(f"  Interval  : {args.interval}")
    print(f"  Start year: {args.start_year}")
    print(f"  Markets   : {', '.join(markets)}")
    print(f"  Coins     : {len(symbols)}")
    print(f"  ⚠️  1s data can be very large (GBs per coin). Be patient.")
    print(f"{'='*60}")

    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        for mkt in markets:
            bulk_download(sym, args.interval, args.start_year, mkt)

    print(f"\n{'='*60}")
    print(f"  All {len(symbols)} coins downloaded.")
    print(f"{'='*60}")
