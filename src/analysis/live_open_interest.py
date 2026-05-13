"""
Live open-interest fetcher with TTL cache — Phase X2 feature stream.

Mirror of `src/analysis/live_funding.py`:
  - Single shared ccxt.binanceusdm instance (lazy + locked).
  - 5-minute TTL cache (OI updates ~once per minute on Binance).
  - Symbol normalization via `_to_ccxt_perpetual` from live_funding.

Open interest is a leading indicator of forced-deleveraging cascades.
Spikes / drops of >5% OI in 1h often precede liquidation cluster moves.
Cf. updated_architecture_plan_en.md §11 (Microstructure & flow).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SEC: float = 300.0   # 5-min, matches funding-rate cache

_lock: threading.Lock = threading.Lock()
_exchange = None  # ccxt.binanceusdm, lazy-init
_cache: dict[str, tuple[float, float]] = {}  # {symbol: (oi_usdt, expires)}


def _get_exchange():
    """Lazy + locked ccxt instance. Caller must hold _lock."""
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "timeout": 5000,
            "options": {"defaultType": "future"},
        })
    return _exchange


def fetch_open_interest(
    symbol: str,
    *,
    ttl_sec: float = _DEFAULT_TTL_SEC,
) -> Optional[float]:
    """Return current open interest for *symbol* in USDT notional, or None.

    Same symbol-format normalization as funding fetcher — accepts bot
    internal '<BASE>_USDT' and converts to ccxt's '<BASE>/USDT:USDT'.

    Returns None on any error (fail-closed contract — callers like
    FuturesAgent can fall back to the candle-level OI feature if the live
    fetch fails).
    """
    from src.analysis.live_funding import _to_ccxt_perpetual
    now = time.monotonic()
    with _lock:
        cached = _cache.get(symbol)
        if cached is not None:
            value, expires_at = cached
            if now < expires_at:
                return value
        try:
            exchange = _get_exchange()
            ccxt_symbol = _to_ccxt_perpetual(symbol)
            data = exchange.fetch_open_interest(ccxt_symbol)
            # ccxt unified format: {symbol, baseVolume, openInterestAmount,
            # openInterestValue, timestamp, datetime, info}
            # Prefer openInterestValue (already in quote currency = USDT).
            oi_usdt = data.get('openInterestValue')
            if oi_usdt is None:
                # Fallback: some venues return openInterestAmount only; multiply
                # by last price.
                amount = float(data.get('openInterestAmount') or 0.0)
                ticker = exchange.fetch_ticker(ccxt_symbol)
                last = float((ticker or {}).get('last') or 0.0)
                oi_usdt = amount * last
            oi_usdt = float(oi_usdt or 0.0)
            _cache[symbol] = (oi_usdt, now + ttl_sec)
            return oi_usdt
        except Exception as exc:
            logger.warning(
                "[live_oi] fetch_open_interest(%s) failed: %s", symbol, exc,
            )
            return None


def clear_cache() -> None:
    """For test teardown."""
    with _lock:
        _cache.clear()
