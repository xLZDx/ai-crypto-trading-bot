"""
Tests for the ccxt-perpetual symbol normalizer in src/analysis/live_funding.py.

The 2026-05-13 incident showed FuturesAgent flooding the dashboard with
`fetch_funding_rate(DOGE_USDT) failed: binanceusdm does not have market
symbol DOGE_USDT` because the agent passes the bot's internal '<BASE>_USDT'
format but ccxt.binanceusdm expects the perpetual format '<BASE>/USDT:USDT'.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_internal_format_translated():
    from src.analysis.live_funding import _to_ccxt_perpetual
    assert _to_ccxt_perpetual('DOGE_USDT')  == 'DOGE/USDT:USDT'
    assert _to_ccxt_perpetual('AVAX_USDT')  == 'AVAX/USDT:USDT'
    assert _to_ccxt_perpetual('BTC_USDT')   == 'BTC/USDT:USDT'
    assert _to_ccxt_perpetual('SHIB_USDT')  == 'SHIB/USDT:USDT'


def test_already_ccxt_perpetual_passthrough():
    from src.analysis.live_funding import _to_ccxt_perpetual
    # If a caller already uses ccxt format, we don't double-translate.
    assert _to_ccxt_perpetual('DOGE/USDT:USDT') == 'DOGE/USDT:USDT'


def test_ccxt_spot_format_passthrough():
    from src.analysis.live_funding import _to_ccxt_perpetual
    # Spot format ('BASE/QUOTE' with no ':USDT') is left alone — the caller
    # explicitly asked for spot. binanceusdm will reject this if used on the
    # futures endpoint, but that's the caller's responsibility.
    assert _to_ccxt_perpetual('BTC/USDT') == 'BTC/USDT'


def test_unknown_format_passthrough():
    from src.analysis.live_funding import _to_ccxt_perpetual
    # We don't try to be clever about formats we don't recognize.
    assert _to_ccxt_perpetual('BTCUSDT') == 'BTCUSDT'  # no '_' separator
    assert _to_ccxt_perpetual('') == ''


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
