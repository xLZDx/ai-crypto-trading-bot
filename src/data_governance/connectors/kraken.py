"""Kraken public OHLC."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class KrakenConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="kraken", host="api.kraken.com",
        priority=0, requires_auth=False,
        default_poll_interval_sec=300,
        category="market",
        description="Kraken public OHLC (spot, EUR + USD pairs).",
    )
    BASE = "https://api.kraken.com/0/public/OHLC"

    _TF = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

    def is_available(self) -> bool:
        return True

    def pull_history(self, *, symbols=None, timeframes=None,
                     since=None, **kw) -> int:
        symbols = symbols or ["XBTUSD", "ETHUSD", "SOLUSD"]
        timeframes = timeframes or ["1h", "1d"]
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=30)
        qdb = self._qdb()
        rows_total = 0
        for sym in symbols:
            for tf in timeframes:
                interval = self._TF.get(tf)
                if not interval:
                    continue
                since_s = int(since.timestamp())
                url = f"{self.BASE}?pair={sym}&interval={interval}&since={since_s}"
                r = self._http_get(url)
                if r is None or r.status_code != 200:
                    continue
                resp = r.json() or {}
                if resp.get("error"):
                    logger.debug("[kraken] %s err: %s", sym, resp.get("error"))
                    continue
                # Kraken nests data under the resolved pair key (varies)
                payload = resp.get("result", {})
                key = next((k for k in payload if k != "last"), None)
                if not key:
                    continue
                bars = []
                for row in payload[key]:
                    bars.append({
                        "timestamp": int(row[0]) * 1000,
                        "open":  float(row[1]), "high":  float(row[2]),
                        "low":   float(row[3]), "close": float(row[4]),
                        "volume": float(row[6]),
                    })
                out_sym = key  # already a Kraken canonical pair string
                n = qdb.write_market_candles_bulk(out_sym, f"krk_{tf}", bars)
                rows_total += n
        logger.info("[kraken] pull_history wrote %d bars", rows_total)
        return rows_total
