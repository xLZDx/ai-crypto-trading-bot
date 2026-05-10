"""Coinbase Exchange (formerly Pro) public candles."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class CoinbaseConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="coinbase", host="api.exchange.coinbase.com",
        priority=0, requires_auth=False,
        default_poll_interval_sec=300,
        category="market",
        description="Coinbase Exchange spot candles.",
    )
    BASE = "https://api.exchange.coinbase.com/products"

    _TF_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900,
                       "1h": 3600, "6h": 21600, "1d": 86400}

    def is_available(self) -> bool:
        return True

    def pull_history(self, *, symbols=None, timeframes=None,
                     since=None, until=None, lookback_days=30, **kw) -> int:
        symbols = symbols or ["BTC-USD", "ETH-USD", "SOL-USD"]
        timeframes = timeframes or ["1m", "1h", "1d"]
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        until = until or datetime.now(timezone.utc)
        qdb = self._qdb()
        rows_total = 0
        for sym in symbols:
            for tf in timeframes:
                gran = self._TF_GRANULARITY.get(tf)
                if not gran:
                    continue
                # Coinbase returns max 300 bars; chunk the time range.
                cur_start = since
                while cur_start < until:
                    cur_end = min(cur_start + timedelta(seconds=300 * gran), until)
                    url = (f"{self.BASE}/{sym}/candles?granularity={gran}"
                           f"&start={cur_start.isoformat()}&end={cur_end.isoformat()}")
                    r = self._http_get(url)
                    if r is None or r.status_code != 200:
                        break
                    data = r.json() or []
                    if not data:
                        cur_start = cur_end
                        continue
                    # Coinbase rows: [time, low, high, open, close, volume]
                    bars = []
                    for row in data:
                        bars.append({
                            "timestamp": int(row[0]) * 1000,
                            "low":  float(row[1]), "high":  float(row[2]),
                            "open": float(row[3]), "close": float(row[4]),
                            "volume": float(row[5]),
                        })
                    out_sym = sym.replace("-", "/")
                    n = qdb.write_market_candles_bulk(out_sym, f"cb_{tf}", bars)
                    rows_total += n
                    cur_start = cur_end
        logger.info("[coinbase] pull_history wrote %d bars", rows_total)
        return rows_total
