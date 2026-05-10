"""YouTube transcript connector — pulls captions from a watchlist of channels.

`youtube-transcript-api` is already in requirements.txt. The connector is
configured by `data/youtube_watchlist.json` listing channel ids; missing
file → no-op. Failure-soft.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WATCHLIST    = PROJECT_ROOT / "data" / "youtube_watchlist.json"


@register
class YouTubeTranscriptsConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="youtube", host="www.youtube.com",
        priority=2, requires_auth=False,
        default_poll_interval_sec=14400,
        category="news",
        description="YouTube transcripts for influencer commentary "
                    "(crypto channel watchlist at data/youtube_watchlist.json).",
    )

    def is_available(self) -> bool:
        if not WATCHLIST.exists():
            return False
        try:
            import youtube_transcript_api  # noqa: F401
            return True
        except ImportError:
            return False

    def _video_ids(self) -> list[str]:
        if not WATCHLIST.exists():
            return []
        try:
            cfg = json.loads(WATCHLIST.read_text(encoding="utf-8"))
            return list(cfg.get("video_ids") or [])
        except Exception:
            return []

    def pull_history(self, **kw) -> int:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return 0
        qdb = self._qdb()
        n = 0
        for vid in self._video_ids():
            try:
                trs = YouTubeTranscriptApi.get_transcript(vid)
            except Exception as exc:
                logger.debug("[youtube] %s err: %s", vid, exc)
                continue
            full = " ".join(p.get("text", "") for p in trs)[:512]
            qdb.write_news_sentiment({
                "ts":     int(datetime.now(timezone.utc).timestamp() * 1000),
                "symbol": "",
                "source": f"youtube/{vid}",
                "title":  full,
                "url":    f"https://www.youtube.com/watch?v={vid}",
                "score":  0.0,
            })
            n += 1
        logger.info("[youtube] %d transcripts written", n)
        return n
