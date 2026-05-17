"""Reddit sentiment via PRAW. Requires REDDIT_CLIENT_ID/SECRET/USER_AGENT env vars.

Free, no rate-limit concerns at our volume.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class RedditConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="reddit", host="www.reddit.com",
        priority=2, requires_auth=True,
        default_poll_interval_sec=1800,
        category="sentiment",
        description="Reddit r/cryptocurrency + r/bitcoin top posts (PRAW). "
                    "Requires REDDIT_CLIENT_ID/SECRET/USER_AGENT env vars.",
    )

    SUBREDDITS = ("cryptocurrency", "bitcoin", "ethereum", "altcoin")

    def is_available(self) -> bool:
        return all(os.getenv(k) for k in
                   ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"))

    def pull_history(self, *, limit: int = 25, **kw) -> int:
        if not self.is_available():
            return 0
        try:
            import praw
        except ImportError:
            logger.info("[reddit] praw not installed -- skipping. "
                        "Run: pip install --no-cache-dir praw")
            return 0
        try:
            reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT", "ai-trading/0.1"),
            )
        except Exception as exc:
            logger.warning("[reddit] auth failed: %s", exc)
            return 0
        qdb = self._qdb()
        n = 0
        for sub in self.SUBREDDITS:
            try:
                for post in reddit.subreddit(sub).hot(limit=int(limit)):
                    qdb.write_news_sentiment({
                        "ts":     int(post.created_utc) * 1000,
                        "symbol": "",
                        "source": f"reddit/{sub}",
                        "title":  (post.title or "")[:512],
                        "url":    f"https://reddit.com{post.permalink}",
                        "score":  float(post.score) / 1000.0,
                    })
                    n += 1
            except Exception as exc:
                logger.debug("[reddit] %s err: %s", sub, exc)
        logger.info("[reddit] %d posts written", n)
        return n
