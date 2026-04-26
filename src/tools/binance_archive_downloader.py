import os
import sys
import json
import gzip
import zipfile
import requests
import io
import csv
from datetime import datetime, timezone
import collections

# Ensure Python sees the root folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_last_timestamp(filename):
    """Efficiently finds the last timestamp in the gzipped CSV to allow resuming."""
    try:
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            last_line = collections.deque(f, maxlen=1)[0]
        if last_line and not last_line.startswith('timestamp'):
            ts_str = last_line.split(',')[0]
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None

def download_monthly_archive(symbol, interval, year, month, market="spot"):
    """Fetches the official monthly ZIP archive from Binance's data server."""
    month_str = f"{month:02d}"
    if market == "futures":
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip"
    else:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 404:
            return None # Data doesn't exist for this month (e.g. coin not listed yet)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"    [!] Error downloading {url}: {e}")
        return None

def process_and_append_data(zip_content, output_filepath, is_first_write):
    """Extracts the CSV from the ZIP in memory, formats it, and appends to our master dataset."""
    with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
        # There is exactly one CSV inside these archives
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            decoded_file = f.read().decode('utf-8').splitlines()
            reader = csv.reader(decoded_file)
            
            mode = 'wt' if is_first_write else 'at'
            with gzip.open(output_filepath, mode=mode, newline='', encoding='utf-8') as out_f:
                writer = csv.writer(out_f)
                if is_first_write:
                    writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count', 'taker_buy_base', 'taker_buy_quote'])
                
                for row in reader:
                    try:
                        if not row or len(row) < 11: continue
                        # Skip headers or corrupted timestamp values
                        ts_ms = int(row[0])
                        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        # Extract and write the 10 specific columns our ML models require
                        writer.writerow([dt, row[1], row[2], row[3], row[4], row[5], row[7], row[8], row[9], row[10]])
                    except (ValueError, OSError, TypeError, IndexError):
                        # Silently skip any invalid rows (like CSV headers)
                        continue

def bulk_download_for_symbol(symbol_raw, interval, start_year=2014, end_year=None, market="spot"):
    """Iterates through every month of every year and compiles a massive historical dataset."""
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
        
    symbol = symbol_raw.replace('/', '')
    print(f"\n======================================================")
    print(f"Starting MASSIVE bulk download for {symbol_raw} [{interval}] ({market.upper()})")
    print(f"Years: {start_year} to {end_year}")
    print(f"Source: data.binance.vision")
    print(f"======================================================")

    raw_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    output_filepath = os.path.join(raw_dir, f"{symbol_raw.replace('/', '_')}_{market}_{interval}.csv.gz")
    
    last_dt = get_last_timestamp(output_filepath) if os.path.exists(output_filepath) else None

    is_first_write = True
    total_months = 0
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Don't try to download future months
            if year == datetime.now(timezone.utc).year and month >= datetime.now(timezone.utc).month:
                continue
                
            # Skip months we already fully downloaded in a previous run
            if last_dt and (year < last_dt.year or (year == last_dt.year and month < last_dt.month)):
                is_first_write = False
                continue
                
            print(f"[*] Fetching {year}-{month:02d}...", end=" ", flush=True)
            zip_content = download_monthly_archive(symbol, interval, year, month, market)
            
            if zip_content:
                process_and_append_data(zip_content, output_filepath, is_first_write)
                is_first_write = False
                total_months += 1
                print("SUCCESS")
            else:
                print("NOT FOUND (Skipping)")
                
    print(f"\n✅ Finished {symbol_raw} [{interval}]. Downloaded {total_months} months of data.")
    print(f"Saved to: {output_filepath}")

if __name__ == "__main__":
    # Target explicitly requested symbols for 1s data
    symbols = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT', 'ETH/USDT']
    current_year = datetime.now(timezone.utc).year
    start_year = current_year - 10
        
    print(f"Targeting {len(symbols)} coins: {symbols}")
    
    for sym in symbols:
        # Download 1s data for past 10 years for both Spot and Futures
        bulk_download_for_symbol(sym, '1s', start_year=start_year, market='spot')
        bulk_download_for_symbol(sym, '1s', start_year=start_year, market='futures')
