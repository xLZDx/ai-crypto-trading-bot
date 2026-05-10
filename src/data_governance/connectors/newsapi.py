"""NewsAPI.org connector — broad news (100 requests/day free)."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class NewsAPIConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="newsapi", host="newsapi.org",
        priority=2, requires_auth=True,
        default_poll_interval_sec=3600,
        category="news",
        description="NewsAPI.org broad news, crypto category. "
                    "Set NEWSAPI_KEY env var (100 req/day free).",
    )
    BASE = "https://newsapi.org/v2/everything"

    QUERIES = ("crypto OR bitcoin OR ethereum",)

    def is_available(self) -> bool:
        return bool(os.getenv("NEWSAPI_KEY"))

    def pull_history(self, *, hours: int = 24, **kw) -> int:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return 0
        qdb = self._qdb()
        n = 0
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        for q in self.QUERIES:
            url = (f"{self.BASE}?q={q}&from={since.isoformat()}"
                   f"&sortBy=publishedAt&pageSize=50&apiKey={api_key}")
            r = self._http_get(url)
            if r is None or r.status_code != 200:
                continue
            for art in (r.json() or {}).get("articles") or []:
                ts = art.get("publishedAt")
                if not ts:
                    continue
                ts_ms = int(datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            .timestamp() * 1000)
                qdb.write_news_sentiment({
                    "ts":     ts_ms,
                    "symbol": "",
                    "source": f"newsapi/{(art.get('source') or {}).get('name', '?')}",
                    "title":  (art.get("title") or "")[:512],
                    "url":    art.get("url", ""),
                    "score":  0.0,
                })
                n += 1
        logger.info("[newsapi] %d items written", n)
        return n
