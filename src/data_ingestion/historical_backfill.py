import time
import os
import csv
from datetime import datetime, timezone, timedelta
import gzip
import requests
import logging

logger = logging.getLogger(__name__)

def backfill_history(symbol='BTC/USDT', timeframe='1m', days=365):
    """
    Downloads long-term historical data from Binance in chunks.
    Allows fetching 1 year of 1m data (approx 525,600 candles).
    """
    logger.info(f"Backfilling data for {symbol} [{timeframe}] for {days} days...")

    # Calculate timestamps
    now = datetime.now(timezone.utc)
    since_date = now - timedelta(days=days)
    since_ms = int(since_date.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    filename = os.path.join(raw_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv.gz")
    
    # Check if file exists, and skip if it does to prevent redownloading
    if os.path.exists(filename):
        logger.info(f"[{symbol} - {timeframe}] History file already exists ({filename}). Skipping.")
        return
        
    total_downloaded = 0
    limit = 1000

    try:
        with gzip.open(filename, mode='wt', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count', 'taker_buy_base', 'taker_buy_quote'])

            while since_ms < now_ms:
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol.replace('/', '')}&interval={timeframe}&startTime={since_ms}&limit={limit}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                ohlcv = response.json()
                
                if not ohlcv:
                    break
                
                for row in ohlcv:
                    dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([dt, row[1], row[2], row[3], row[4], row[5], row[7], row[8], row[9], row[10]])
                
                total_downloaded += len(ohlcv)
                since_ms = ohlcv[-1][0] + 1
                
                time.sleep(0.1) # Be nice to Binance API limits

        logger.info(f"Successfully backfilled {total_downloaded} candles for {symbol}. Saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error while backfilling data for {symbol}: {e}")

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT']
    for sym in symbols:
        # Backfill 6 years of 1-minute data (WARNING: This will take gigabytes and a lot of time!)
        backfill_history(symbol=sym, timeframe='1m', days=6*365)
        # Backfill 6 years of 1-hour data (matches the main bot timeframe)
        backfill_history(symbol=sym, timeframe='1h', days=6*365)
        # Backfill 6 years of 1-day data (for Random Forest ML model training)
        backfill_history(symbol=sym, timeframe='1d', days=6*365)
