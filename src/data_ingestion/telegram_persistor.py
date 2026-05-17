"""Telegram → QuestDB persistor.

Wraps the existing `src/analysis/telegram_monitor.TelegramMonitor` so every
inbound message is written to QuestDB's `news_sentiment` table (same schema
as the CryptoCompare news connector). The bot's in-memory cache is kept
intact for low-latency reactions; the DB write adds durability for training.

Designed to fail-soft: missing Telegram creds → log + return.

Run:
    python -m src.data_ingestion.telegram_persistor
    python -m src.data_ingestion.telegram_persistor --channels VilarsoPro vilarsofree
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("telegram_persistor")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--channels", nargs="+",
                   default=["VilarsoPro", "vilarsofree", "mr_mozart"])
    p.add_argument("--poll-sec", type=int, default=60)
    args = p.parse_args()

    if not (os.getenv("TELEGRAM_API_ID") and os.getenv("TELEGRAM_API_HASH")):
        logger.warning("TELEGRAM_API_ID / TELEGRAM_API_HASH not set -- exiting.")
        return 1

    try:
        from src.analysis.telegram_monitor import TelegramMonitor
    except Exception as exc:
        logger.error("Could not import TelegramMonitor: %s", exc)
        return 1

    try:
        from src.database.parquet_client import get_client as get_questdb
    except Exception as exc:
        logger.error("Could not import ParquetClient: %s", exc)
        return 1

    qdb = get_questdb()
    monitor = TelegramMonitor(channels=args.channels)
    logger.info("Persisting Telegram messages from %s -> QuestDB news_sentiment",
                args.channels)

    seen: set[str] = set()
    while True:
        try:
            # TelegramMonitor exposes `latest_signals()` — a list of dicts with
            # source, text, ts, parsed sentiment. (Existing API.)
            messages = []
            if hasattr(monitor, "latest_signals"):
                messages = monitor.latest_signals()
            elif hasattr(monitor, "fetch"):
                messages = monitor.fetch()
            for m in messages:
                key = f"{m.get('source', '?')}::{m.get('id', m.get('ts', ''))}"
                if key in seen:
                    continue
                seen.add(key)
                ts_ms = int(m.get("ts") or
                            datetime.now(timezone.utc).timestamp() * 1000)
                qdb.write_news_sentiment({
                    "ts":     ts_ms,
                    "symbol": m.get("symbol") or "",
                    "source": f"telegram/{m.get('source', '?')}",
                    "title":  (m.get("text") or "")[:512],
                    "url":    "",
                    "score":  float(m.get("score") or 0.0),
                })
            if messages:
                logger.info("[telegram] %d new messages persisted", len(messages))
        except Exception as exc:
            logger.warning("[telegram] poll error: %s", exc)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    sys.exit(main())
