"""CryptoCompare News API — free 100k requests/month.

Reuses the existing `cryptocompare_news.csv` schema (timestamp, title,
summary, source) and writes new headlines to QuestDB `news_sentiment`
table. FinBERT scoring (already in feature_engineering.add_finbert_sentiment)
runs downstream.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class CryptoCompareNewsConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="cryptocompare_news", host="min-api.cryptocompare.com",
        priority=1, requires_auth=False,    # API key optional but recommended
        default_poll_interval_sec=600,
        category="news",
        description="Aggregated crypto news from ~20 sources. "
                    "Set CRYPTOCOMPARE_API_KEY env var for a higher quota.",
    )
    BASE = "https://min-api.cryptocompare.com/data/v2/news/"

    def is_available(self) -> bool:
        return True

    def pull_history(self, *, lang: str = "EN", limit: int = 50, **kw) -> int:
        url = f"{self.BASE}?lang={lang}"
        api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
        if api_key:
            url += f"&api_key={api_key}"
        r = self._http_get(url)
        if r is None or r.status_code != 200:
            return 0
        items = (r.json() or {}).get("Data") or []
        qdb = self._qdb()
        n = 0
        for item in items[:int(limit)]:
            try:
                qdb.write_news_sentiment({
                    "ts":      int(item["published_on"]) * 1000,
                    "symbol":  ",".join(item.get("categories", "").split("|")),
                    "source":  item.get("source", "cc"),
                    "title":   item.get("title", "")[:512],
                    "url":     item.get("url", ""),
                    "score":   0.0,    # filled by FinBERT downstream
                })
                n += 1
            except Exception as exc:
                logger.debug("[ccnews] write failed: %s", exc)
        logger.info("[cryptocompare_news] %d items written", n)
        return n
