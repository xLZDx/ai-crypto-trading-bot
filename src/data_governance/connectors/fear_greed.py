"""Crypto Fear & Greed Index (alternative.me) — free, no auth."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class FearGreedConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="fear_greed", host="api.alternative.me",
        priority=0, requires_auth=False,
        default_poll_interval_sec=3600,
        category="sentiment",
        description="Crypto Fear & Greed Index (0-100).",
    )
    BASE = "https://api.alternative.me/fng/"

    def is_available(self) -> bool:
        return True

    def pull_history(self, *, limit: int = 365, **kw) -> int:
        qdb = self._qdb()
        r = self._http_get(f"{self.BASE}?limit={int(limit)}")
        if r is None or r.status_code != 200:
            return 0
        data = (r.json() or {}).get("data") or []
        rows = 0
        for item in data:
            ts_ms = int(item["timestamp"]) * 1000
            qdb.write_signal("fear_greed", {
                "value":  float(item["value"]),
            }, ts_val=ts_ms)
            rows += 1
        logger.info("[fear_greed] %d points written", rows)
        return rows
