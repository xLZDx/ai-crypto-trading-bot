"""Santiment social/dev/whale metrics — free tier requires API key."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class SantimentConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="santiment", host="api.santiment.net",
        priority=2, requires_auth=True,
        default_poll_interval_sec=14400,
        category="sentiment",
        description="Santiment social volume + dev activity + whale moves. "
                    "Set SANTIMENT_API_KEY env var (free tier limited).",
    )
    GRAPHQL = "https://api.santiment.net/graphiql"

    SLUGS = ("bitcoin", "ethereum", "solana")

    def is_available(self) -> bool:
        return bool(os.getenv("SANTIMENT_API_KEY"))

    def pull_history(self, *, days: int = 30, **kw) -> int:
        api_key = os.getenv("SANTIMENT_API_KEY")
        if not api_key:
            return 0
        qdb = self._qdb()
        n = 0
        # Santiment uses a GraphQL POST. We use a simple REST-ish wrapper
        # via their /api/v2/timeseries endpoint when available.
        for slug in self.SLUGS:
            url = (f"https://api.santiment.net/v1/social_volume_total"
                   f"?slug={slug}&from={days}d&apikey={api_key}")
            r = self._http_get(url)
            if r is None or r.status_code != 200:
                continue
            data = r.json() or []
            for point in data:
                ts = point.get("datetime") or point.get("date")
                v  = point.get("value")
                if not ts or v is None:
                    continue
                ts_ms = int(datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                            .timestamp() * 1000)
                qdb.write_signal(f"santiment_social_{slug}",
                                 {"value": float(v)}, ts_val=ts_ms)
                n += 1
        logger.info("[santiment] %d social points written", n)
        return n
