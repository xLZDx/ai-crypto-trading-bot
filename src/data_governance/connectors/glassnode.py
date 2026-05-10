"""Glassnode on-chain metrics — free tier."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class GlassnodeConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="glassnode", host="api.glassnode.com",
        priority=2, requires_auth=True,
        default_poll_interval_sec=14400,
        category="onchain",
        description="Glassnode on-chain (active addresses, exchange flows). "
                    "Set GLASSNODE_API_KEY env var (free tier limited).",
    )
    BASE = "https://api.glassnode.com/v1/metrics"

    METRICS = {
        "addresses/active_count":    "active_addresses",
        "transactions/transfers_volume_exchanges_net": "exchange_net_flow",
        "indicators/sopr":           "sopr",
        "supply/profit_relative":    "supply_profit_pct",
    }

    def is_available(self) -> bool:
        return bool(os.getenv("GLASSNODE_API_KEY"))

    def pull_history(self, *, asset: str = "BTC", **kw) -> int:
        api_key = os.getenv("GLASSNODE_API_KEY")
        if not api_key:
            return 0
        qdb = self._qdb()
        n = 0
        for path, alias in self.METRICS.items():
            url = (f"{self.BASE}/{path}?a={asset}&api_key={api_key}"
                   f"&i=24h&f=JSON")
            r = self._http_get(url)
            if r is None or r.status_code != 200:
                continue
            for point in r.json() or []:
                ts_ms = int(point["t"]) * 1000
                v = point.get("v")
                if v is None:
                    continue
                qdb.write_signal(f"glassnode_{asset.lower()}_{alias}",
                                 {"value": float(v)}, ts_val=ts_ms)
                n += 1
        logger.info("[glassnode] %d points written for %s", n, asset)
        return n
