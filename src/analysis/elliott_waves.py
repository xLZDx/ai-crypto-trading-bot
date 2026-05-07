import csv
import logging
from typing import List, Dict
import gzip

logging.basicConfig(level=logging.INFO, format='%(message)s')

class ElliottWaveAnalyzer:
    """
    Basic market analyzer for finding Elliott Waves.
    Uses the ZigZag algorithm (finding peaks and troughs) in pure Python without heavy libraries.
    """
    def __init__(self, deviation_percent=2.0):
        self.deviation_percent = deviation_percent / 100.0
        
    def load_data(self, filepath: str, tail_n: int = 0) -> List[Dict]:
        """Load CSV data.  tail_n > 0 returns only the last tail_n rows (memory-safe).

        Per-row try/except so a single malformed row (e.g. a resampler gap
        emitted as ',,,,,') doesn't abort the whole load. Skipped rows are
        counted and logged once at WARN, not 60 times per cycle.
        """
        from collections import deque
        data_buf: deque = deque(maxlen=tail_n if tail_n > 0 else None)
        skipped = 0
        try:
            open_func = gzip.open if filepath.endswith('.gz') else open
            mode = 'rt' if filepath.endswith('.gz') else 'r'
            with open_func(filepath, mode, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        data_buf.append({
                            'timestamp': row['timestamp'],
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'quote_volume': float(row.get('quote_volume', 0)),
                            'trades_count': float(row.get('trades_count', 0)),
                            'taker_buy_base': float(row.get('taker_buy_base', 0)),
                            'taker_buy_quote': float(row.get('taker_buy_quote', 0))
                        })
                    except (ValueError, KeyError, TypeError):
                        skipped += 1
                        continue
            data = list(data_buf)
            if skipped:
                logging.warning(f"Loaded {len(data)} candles from {filepath} (skipped {skipped} malformed rows)")
            else:
                logging.info(f"Loaded {len(data)} candles from {filepath}")
            return data
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {e}")
            return []

    def calculate_zigzag(self, data: List[Dict]) -> List[Dict]:
        """
        Determines extremes (Pivot High / Pivot Low).
        If the price bounces from the max/min by more than deviation_percent,
        we set a new point. This is the basis for finding a 5-wave Elliott structure.
        """
        if not data:
            return []

        pivots = []
        last_pivot = data[0]
        last_pivot['type'] = 'unknown' # 'high' or 'low'
        
        is_uptrend = True
        
        for i in range(1, len(data)):
            current = data[i]
            
            if is_uptrend:
                # If we are in an uptrend, look for a new maximum
                if current['high'] > last_pivot['high']:
                    last_pivot = current
                    last_pivot['type'] = 'high'
                # If price dropped by X% from the last maximum, trend changed
                elif current['low'] < last_pivot['high'] * (1 - self.deviation_percent):
                    pivots.append(last_pivot)
                    last_pivot = current
                    last_pivot['type'] = 'low'
                    is_uptrend = False
            else:
                # If we are in a downtrend, look for a new minimum
                if current['low'] < last_pivot['low']:
                    last_pivot = current
                    last_pivot['type'] = 'low'
                # If price rose by X% from the last minimum, trend changed
                elif current['high'] > last_pivot['low'] * (1 + self.deviation_percent):
                    pivots.append(last_pivot)
                    last_pivot = current
                    last_pivot['type'] = 'high'
                    is_uptrend = True
                    
        pivots.append(last_pivot) # Add the last point
        return pivots

if __name__ == "__main__":
    analyzer = ElliottWaveAnalyzer(deviation_percent=3.0) # Look for movements greater than 3%
    filepath = 'data/raw/BTC_USDT_1h.csv'
    
    data = analyzer.load_data(filepath)
    if data:
        pivots = analyzer.calculate_zigzag(data)
        logging.info(f"Found {len(pivots)} pivot points on the chart:")
        
        # Print the last 10 found waves
        for p in pivots[-10:]:
            trend = "📈 PEAK (Wave Top)" if p['type'] == 'high' else "📉 TROUGH (Wave Bottom)"
            logging.info(f"{p['timestamp']} | {trend} | Price: {p['close']:.2f}")
