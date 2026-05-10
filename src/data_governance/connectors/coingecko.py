"""CoinGecko market caps + dominance + Fear & Greed alternative."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class CoinGeckoConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="coingecko", host="api.coingecko.com",
        priority=0, requires_auth=False,
        default_poll_interval_sec=600,
        category="macro",
        description="CoinGecko global market cap, BTC dominance, top 250 daily.",
    )
    BASE = "https://api.coingecko.com/api/v3"

    def is_available(self) -> bool:
        return True

    def pull_history(self, **kw) -> int:
        qdb = self._qdb()
        rows_total = 0

        # Global stats — single point, written as a "global_market" pseudo-bar
        r = self._http_get(f"{self.BASE}/global")
        if r is not None and r.status_code == 200:
            d = (r.json() or {}).get("data", {})
            total_mcap = float(d.get("total_market_cap", {}).get("usd", 0))
            total_vol  = float(d.get("total_volume", {}).get("usd", 0))
            btc_dom    = float(d.get("market_cap_percentage", {}).get("btc", 0))
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            qdb.write_signal("global", {
                "total_mcap_usd": total_mcap,
                "total_volume_usd": total_vol,
                "btc_dominance":  btc_dom,
            }, ts_val=ts_ms)
            rows_total += 1

        return rows_total
