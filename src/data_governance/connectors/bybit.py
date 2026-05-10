"""Bybit V5 public klines connector. Free, no auth."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class BybitConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="bybit", host="api.bybit.com",
        priority=0, requires_auth=False,
        default_poll_interval_sec=300,
        category="market",
        description="Bybit V5 spot/perps public klines.",
    )
    BASE = "https://api.bybit.com/v5/market/kline"

    _TF = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}

    def is_available(self) -> bool:
        return True

    def pull_history(self, *, symbols=None, timeframes=None,
                     since=None, until=None, lookback_days=30, **kw) -> int:
        symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        timeframes = timeframes or ["1m", "1h", "1d"]
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        until = until or datetime.now(timezone.utc)
        qdb = self._qdb()
        rows_total = 0
        for sym in symbols:
            for tf in timeframes:
                interval = self._TF.get(tf)
                if not interval:
                    continue
                start_ms = int(since.timestamp() * 1000)
                end_ms   = int(until.timestamp() * 1000)
                cursor = end_ms
                while cursor > start_ms:
                    url = (f"{self.BASE}?category=spot&symbol={sym}"
                           f"&interval={interval}&start={start_ms}&end={cursor}&limit=1000")
                    r = self._http_get(url)
                    if r is None or r.status_code != 200:
                        break
                    data = (r.json() or {}).get("result", {}).get("list") or []
                    if not data:
                        break
                    bars = []
                    for row in data:
                        ts = int(row[0])
                        bars.append({
                            "timestamp": ts,
                            "open": float(row[1]), "high": float(row[2]),
                            "low": float(row[3]), "close": float(row[4]),
                            "volume": float(row[5]),
                        })
                    n = qdb.write_market_candles_bulk(
                        f"{sym[:-4]}/{sym[-4:]}" if sym.endswith("USDT") else sym,
                        f"bybit_{tf}", bars,
                    )
                    rows_total += n
                    # Bybit returns descending; advance cursor older
                    cursor = int(data[-1][0]) - 1
                    if len(data) < 1000:
                        break
        logger.info("[bybit] pull_history wrote %d bars", rows_total)
        return rows_total
