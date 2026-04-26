import os
import csv
import time
from datetime import datetime, timezone
import gzip
import requests
import logging
import collections
import ccxt

logger = logging.getLogger(__name__)

_RETRY_CODES = {429, 418}
_MAX_RETRIES = 5


def fetch_funding_rates(symbol='BTC/USDT', limit=1000):
    """
    Fetches historical funding rates using ccxt. 
    Crucial for pairs trading and futures backtesting unit economics.
    """
    logger.info(f"Fetching funding rates for {symbol}...")
    try:
        # For Binance USDT-M futures, symbols typically look like BTC/USDT:USDT or BTC/USDT
        exchange = ccxt.binance({'enableRateLimit': True})
        # Try fetching funding rate history
        funding_history = exchange.fetch_funding_rate_history(symbol=symbol, limit=limit)
        logger.info(f"Successfully fetched {len(funding_history)} funding rate records for {symbol}.")
        return funding_history
    except Exception as e:
        logger.error(f"Error fetching funding rates for {symbol}: {e}")
        return []

def _get_with_retry(url: str, timeout: int = 10) -> requests.Response:
    """GET with exponential backoff on Binance rate-limit (429) or IP-ban (418) responses."""
    delay = 2.0
    for attempt in range(_MAX_RETRIES):
        response = requests.get(url, timeout=timeout)
        if response.status_code in _RETRY_CODES:
            retry_after = int(response.headers.get("Retry-After", delay))
            logger.warning(f"Binance rate limit ({response.status_code}). Waiting {retry_after}s (attempt {attempt + 1}/{_MAX_RETRIES})...")
            time.sleep(retry_after)
            delay = min(delay * 2, 60)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"Binance API blocked after {_MAX_RETRIES} retries: {url}")


def get_last_timestamp(filename):
    """Efficiently finds the last timestamp in the gzipped CSV without loading into memory."""
    try:
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            last_line = collections.deque(f, maxlen=1)[0]
        if last_line and not last_line.startswith('timestamp'):
            ts_str = last_line.split(',')[0]
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
    except Exception as e:
        logger.debug(f"Could not read last timestamp from {filename}: {e}")
    return None


def _validate_ohlcv_row(row) -> bool:
    """Basic sanity check: high >= open/close >= low, all positive."""
    try:
        o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        return h >= max(o, c) and l <= min(o, c) and l > 0
    except (IndexError, ValueError):
        return False


def download_history(symbol='BTC/USDT', timeframe='1h', limit=1000):
    """
    Smartly appends only the newest candles from Binance to the historical archive.
    Prevents loading massive datasets into memory and completely avoids file overwriting.
    """
    logger.info(f"Updating latest {limit} candles for {symbol} [{timeframe}]...")

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(project_root, 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        filename = os.path.join(raw_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv.gz")

        last_ts = get_last_timestamp(filename) if os.path.exists(filename) else None

        safe_symbol = symbol.replace('/', '')
        url = f"https://api.binance.com/api/v3/klines?symbol={safe_symbol}&interval={timeframe}&limit={limit}"
        if last_ts:
            url += f"&startTime={last_ts + 1}"

        response = _get_with_retry(url)
        ohlcv = response.json()

        if not isinstance(ohlcv, list):
            logger.error(f"[{symbol}] Unexpected response format: {type(ohlcv)}")
            return

        if not ohlcv:
            logger.info(f"[{symbol}] Data is already up to date.")
            return

        mode = 'at' if os.path.exists(filename) else 'wt'
        valid_rows = 0
        with gzip.open(filename, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if mode == 'wt':
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                  'quote_volume', 'trades_count', 'taker_buy_base', 'taker_buy_quote'])
            for row in ohlcv:
                if not _validate_ohlcv_row(row):
                    logger.warning(f"[{symbol}] Skipping invalid OHLCV row: {row[:5]}")
                    continue
                dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([dt, float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                                  float(row[5]), float(row[7]), float(row[8]), float(row[9]), float(row[10])])
                valid_rows += 1

        logger.info(f"Appended {valid_rows}/{len(ohlcv)} valid candles to {filename}.")

    except Exception as e:
        logger.error(f"Error while downloading data for {symbol}: {e}")


if __name__ == "__main__":
    download_history(symbol='BTC/USDT', timeframe='1h', limit=1000)
