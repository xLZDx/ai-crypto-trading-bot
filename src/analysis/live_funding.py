"""
Live funding-rate fetcher with TTL cache.

Design principles:
- Single shared ccxt exchange instance (lazy-init inside the lock).
- threading.Lock wraps both the cache-read AND cache-write in one acquisition,
  preventing a TOCTOU race where two threads both see a miss and double-fetch.
- enableRateLimit=True on the ccxt instance honours Binance rate limits.
- Fail-closed: any exchange error returns None so callers can gate on it.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# TTL for cached funding rate in seconds (Binance updates every 8 hours;
# 5-minute cache is conservative enough to avoid stale data in live trading).
_DEFAULT_TTL_SEC: float = 300.0

_lock: threading.Lock = threading.Lock()
_exchange = None  # ccxt.binanceusdm, lazy-initialised
_cache: dict[str, tuple[float, float]] = {}  # {symbol: (rate, expires_monotonic)}


def _to_ccxt_perpetual(symbol: str) -> str:
    """Normalize the bot's internal '<BASE>_USDT' format to ccxt's
    '<BASE>/USDT:USDT' perpetual symbol expected by binanceusdm.

    Examples:
      'DOGE_USDT'     -> 'DOGE/USDT:USDT'   (internal -> ccxt perpetual)
      'BTC/USDT:USDT' -> 'BTC/USDT:USDT'    (already ccxt format, no change)
      'BTC/USDT'      -> 'BTC/USDT'         (ccxt spot — pass through; caller
                                             is responsible for the type)

    Quote-asset support: only '_USDT' is translated. '_BUSD' / '_USDC' / coin-
    margined ('BTC_BTC') tails pass through unchanged — ccxt will reject them
    and FuturesAgent's fail-closed funding gate (futures_agent.py:113) will
    block the trade, which is the correct behavior for an unsupported pair.

    Why this exists: bot agents pass '<BASE>_USDT' (FuturesAgent.symbols comes
    from main.py as '<BASE>_USDT' strings). ccxt.binanceusdm.fetch_funding_rate
    expects perpetual notation. Without translation, every futures-funding
    call raises `binanceusdm does not have market symbol DOGE_USDT` and the
    funding gate fail-closes — blocking every futures trade indefinitely.
    Caused the 2026-05-13 `[live_funding] failed × 1989` banner warning.
    """
    if not symbol:
        # Empty / None — defensive guard. Callers should pass strings, but a
        # malformed signal can produce '' from payload.get('symbol', '').
        return symbol or ''
    if '/' in symbol or ':' in symbol:
        return symbol
    if symbol.endswith('_USDT'):
        base = symbol[:-len('_USDT')]
        return f"{base}/USDT:USDT"
    return symbol


def _get_exchange():
    """Return the shared ccxt binanceusdm instance, creating it if needed.

    Must be called with _lock already held.
    """
    global _exchange
    if _exchange is None:
        import ccxt  # deferred import so the module loads without ccxt in tests
        _exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "timeout": 5000,          # 5s hard cap; bounds worst-case lock-hold on outage
            "options": {"defaultType": "future"},
        })
    return _exchange


def fetch_funding_rate(
    symbol: str,
    *,
    ttl_sec: float = _DEFAULT_TTL_SEC,
) -> Optional[float]:
    """Return the current funding rate for *symbol*, using a TTL cache.

    The cache check and write both happen inside the same lock acquisition,
    eliminating the TOCTOU window that would exist if they used separate locks.

    Args:
        symbol:  ccxt-style symbol, e.g. "BTC/USDT:USDT".
        ttl_sec: How long (seconds) a cached value is considered fresh.

    Returns:
        Funding rate as a float (e.g. 0.0001 = 0.01%), or None on any error.
    """
    now = time.monotonic()

    with _lock:
        cached = _cache.get(symbol)
        if cached is not None:
            rate, expires_at = cached
            if now < expires_at:
                logger.debug("[live_funding] cache hit %s rate=%.6f", symbol, rate)
                return rate

        # Cache miss or expired — fetch while still holding the lock.
        # This prevents a second thread from also seeing a miss and stampeding.
        try:
            exchange = _get_exchange()
            ccxt_symbol = _to_ccxt_perpetual(symbol)
            data = exchange.fetch_funding_rate(ccxt_symbol)
            rate = float(data["fundingRate"])
            _cache[symbol] = (rate, now + ttl_sec)
            logger.debug("[live_funding] fetched %s (ccxt=%s) rate=%.6f",
                         symbol, ccxt_symbol, rate)
            return rate
        except Exception as exc:
            logger.warning("[live_funding] fetch_funding_rate(%s) failed: %s", symbol, exc)
            return None


def clear_cache() -> None:
    """Clear all cached entries (primarily for test teardown)."""
    with _lock:
        _cache.clear()
        logger.debug("[live_funding] cache cleared")
