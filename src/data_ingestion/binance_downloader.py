import os
import csv
from datetime import datetime, timezone
import gzip
import requests
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def download_history(symbol='BTC/USDT', timeframe='1h', limit=1000):
    """
    Updates existing raw historical data with the latest candles from Binance.
    Merges new data with old data to prevent overwriting the ML history.
    """
    logger.info(f"Updating latest {limit} candles for {symbol} [{timeframe}]...")

    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.replace('/', '')}&interval={timeframe}&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        ohlcv = response.json()
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(project_root, 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        filename = os.path.join(raw_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv.gz")
        
        new_data = []
        for row in ohlcv:
            dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            new_data.append([dt, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[7]), float(row[8]), float(row[9]), float(row[10])])
            
        new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count', 'taker_buy_base', 'taker_buy_quote'])
        
        if os.path.exists(filename):
            old_df = pd.read_csv(filename)
            combined_df = pd.concat([old_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
        else:
            combined_df = new_df
            
        combined_df.to_csv(filename, index=False, compression='gzip')
                
        logger.info(f"Successfully merged new candles. Total history size: {len(combined_df)} rows.")
        
    except Exception as e:
        logger.error(f"Error while downloading data for {symbol}: {e}")

if __name__ == "__main__":
    download_history(symbol='BTC/USDT', timeframe='1h', limit=1000)
