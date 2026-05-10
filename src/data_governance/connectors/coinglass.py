"""CoinGlass — aggregated funding rates across all 8 major venues.

Free tier: limited symbols + lower granularity.
Paid tier: all symbols + 1m granularity. Pass `COINGLASS_API_KEY` env var.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class CoinGlassConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="coinglass", host="open-api.coinglass.com",
        priority=2, requires_auth=True,
        default_poll_interval_sec=600,
        category="derivatives",
        description="Aggregated funding rates across 8 major venues. "
                    "Set COINGLASS_API_KEY env var.",
    )
    BASE = "https://open-api.coinglass.com/public/v2"

    def is_available(self) -> bool:
        return bool(os.getenv("COINGLASS_API_KEY"))

    def pull_history(self, *, symbols=None, **kw) -> int:
        api_key = os.getenv("COINGLASS_API_KEY")
        if not api_key:
            return 0
        symbols = symbols or ["BTC", "ETH", "SOL"]
        qdb = self._qdb()
        n = 0
        for sym in symbols:
            url = f"{self.BASE}/funding?symbol={sym}"
            r = self._http_get(url, headers={"coinglassSecret": api_key})
            if r is None or r.status_code != 200:
                continue
            data = (r.json() or {}).get("data") or []
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            for venue in data:
                rate = venue.get("rate")
                if rate is None:
                    continue
                qdb.write_signal(
                    f"funding_{sym.lower()}_{venue.get('exchange', 'unknown').lower()}",
                    {"rate": float(rate)}, ts_val=ts_ms,
                )
                n += 1
        logger.info("[coinglass] %d funding points written", n)
        return n
