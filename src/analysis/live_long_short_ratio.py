"""
Live Binance long/short ratio fetcher — Phase X2 feature stream.

Crowded-trade contrarian signal:
  - long_short_account_ratio > 3.5: extreme long crowding → mean-reversion short
  - long_short_account_ratio < 0.3: extreme short crowding → squeeze risk

Binance Futures endpoint:
    GET /futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=1h

ccxt doesn't have a unified method for this (it's a data endpoint, not
trading), so we hit the raw HTTP API. Same TTL cache + lock pattern as
live_funding.py / live_open_interest.py.

Cf. updated_architecture_plan_en.md §11.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SEC: float = 300.0   # 5-min cache (endpoint updates every 5 min)

_lock: threading.Lock = threading.Lock()
_cache: dict[str, tuple[dict, float]] = {}   # {symbol_period: (data, expires)}

_BINANCE_BASE = "https://fapi.binance.com"
_ENDPOINT = "/futures/data/topLongShortAccountRatio"


def _to_binance_symbol(symbol: str) -> str:
    """'BTC_USDT' or 'BTC/USDT:USDT' or 'BTC/USDT' → 'BTCUSDT'."""
    if not symbol:
        return ''
    s = symbol.replace('/', '').replace('_', '').replace(':USDT', '').replace(':', '')
    return s.upper()


def fetch_long_short_ratio(
    symbol: str,
    *,
    period: str = '1h',
    ttl_sec: float = _DEFAULT_TTL_SEC,
) -> Optional[dict]:
    """Return latest long/short ratio dict for *symbol* + *period*, or None.

    Output:
        {
          'long_account': float,   # share of accounts net-long
          'short_account': float,  # share of accounts net-short
          'long_short_ratio': float,   # = long_account / short_account
          'timestamp': int (ms),
        }

    Period values accepted by Binance: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d.
    """
    if period not in {'5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'}:
        logger.warning("[live_lsr] invalid period %r; defaulting to 1h", period)
        period = '1h'

    cache_key = f'{symbol}@{period}'
    now = time.monotonic()
    with _lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            value, expires_at = cached
            if now < expires_at:
                return value
        try:
            import urllib.request
            import json as _json
            bs = _to_binance_symbol(symbol)
            url = f"{_BINANCE_BASE}{_ENDPOINT}?symbol={bs}&period={period}&limit=1"
            with urllib.request.urlopen(url, timeout=5) as resp:
                raw = _json.loads(resp.read().decode('utf-8'))
            if not raw or not isinstance(raw, list):
                logger.warning("[live_lsr] empty payload for %s", symbol)
                return None
            row = raw[0]
            out = {
                'long_account':      float(row.get('longAccount') or 0.0),
                'short_account':     float(row.get('shortAccount') or 0.0),
                'long_short_ratio':  float(row.get('longShortRatio') or 0.0),
                'timestamp':         int(row.get('timestamp') or 0),
            }
            _cache[cache_key] = (out, now + ttl_sec)
            return out
        except Exception as exc:
            logger.warning(
                "[live_lsr] fetch_long_short_ratio(%s) failed: %s", symbol, exc,
            )
            return None


def clear_cache() -> None:
    """For test teardown."""
    with _lock:
        _cache.clear()
