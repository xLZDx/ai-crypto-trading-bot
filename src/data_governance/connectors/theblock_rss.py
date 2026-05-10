"""The Block RSS connector — curated institutional news. Free.

Uses `feedparser` (already in requirements.txt).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import mktime

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class TheBlockRSSConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="theblock_rss", host="www.theblock.co",
        priority=1, requires_auth=False,
        default_poll_interval_sec=3600,
        category="news",
        description="The Block (institutional crypto news). RSS, no auth.",
    )
    FEED = "https://www.theblock.co/rss.xml"

    def is_available(self) -> bool:
        try:
            import feedparser  # noqa: F401
            return True
        except ImportError:
            return False

    def pull_history(self, **kw) -> int:
        try:
            import feedparser
        except ImportError:
            return 0
        try:
            d = feedparser.parse(self.FEED)
        except Exception as exc:
            logger.debug("[theblock] feed err: %s", exc)
            return 0
        qdb = self._qdb()
        n = 0
        for entry in (d.entries or [])[:50]:
            t = entry.get("published_parsed") or entry.get("updated_parsed")
            ts_ms = int(mktime(t) * 1000) if t \
                else int(datetime.now(timezone.utc).timestamp() * 1000)
            qdb.write_news_sentiment({
                "ts":     ts_ms,
                "symbol": "",
                "source": "theblock",
                "title":  (entry.get("title") or "")[:512],
                "url":    entry.get("link", ""),
                "score":  0.0,
            })
            n += 1
        logger.info("[theblock] %d items written", n)
        return n
