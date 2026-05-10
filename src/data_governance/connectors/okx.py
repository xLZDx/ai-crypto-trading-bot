"""OKX public klines connector. Free, no auth."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class OKXConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="okx", host="www.okx.com",
        priority=0, requires_auth=False,
        default_poll_interval_sec=300,
        category="market",
        description="OKX V5 spot/perps public candles.",
    )
    BASE = "https://www.okx.com/api/v5/market/history-candles"

    _TF = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

    def is_available(self) -> bool:
        return True

    def pull_history(self, *, symbols=None, timeframes=None,
                     since=None, until=None, lookback_days=30, **kw) -> int:
        symbols = symbols or ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        timeframes = timeframes or ["1m", "1h", "1d"]
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        until = until or datetime.now(timezone.utc)
        qdb = self._qdb()
        rows_total = 0
        for sym in symbols:
            for tf in timeframes:
                bar = self._TF.get(tf)
                if not bar:
                    continue
                start_ms = int(since.timestamp() * 1000)
                end_ms   = int(until.timestamp() * 1000)
                cursor = end_ms
                while cursor > start_ms:
                    url = (f"{self.BASE}?instId={sym}&bar={bar}"
                           f"&before={cursor}&limit=100")
                    r = self._http_get(url)
                    if r is None or r.status_code != 200:
                        break
                    data = (r.json() or {}).get("data") or []
                    if not data:
                        break
                    bars = []
                    for row in data:
                        bars.append({
                            "timestamp": int(row[0]),
                            "open": float(row[1]), "high": float(row[2]),
                            "low":  float(row[3]), "close": float(row[4]),
                            "volume": float(row[5]),
                        })
                    out_sym = sym.replace("-", "/")
                    n = qdb.write_market_candles_bulk(out_sym, f"okx_{tf}", bars)
                    rows_total += n
                    cursor = int(data[-1][0]) - 1
                    if len(data) < 100:
                        break
        logger.info("[okx] pull_history wrote %d bars", rows_total)
        return rows_total
