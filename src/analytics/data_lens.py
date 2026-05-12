"""
DataLens — single time-aligned query API across all data sources.

For training and backtesting we need, for any (symbol, time-window):
  • OHLCV from the primary venue (Binance) — Parquet cold path
  • OHLCV from secondary venues (Bybit, OKX, Coinbase, Kraken) — QuestDB hot
  • Funding rates — Parquet
  • News headlines + sentiment — Parquet (`_news`)
  • Macro context (FRED, F&G, CoinGecko global) — QuestDB `model_signals`
  • Regime label — `regime_classifier`

DataLens stitches these together into one DataFrame indexed by timestamp.
Missing sources are silently dropped — the bot stays usable even if e.g.
FRED isn't configured.

Usage:
    lens = DataLens()
    df = lens.training_frame(symbol="BTC/USDT", timeframe="1h",
                             start="2024-01-01", end="2024-12-31")
    # Columns: open, high, low, close, volume, funding_rate, fear_greed,
    #          dxy, vix, news_sentiment_24h, regime, ...
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLens:
    """Read-only, time-aligned join across hot+cold stores."""

    def __init__(self):
        self._qdb = None
        self._parquet = None

    def _q(self):
        if self._qdb is None:
            from src.database.parquet_client import get_client
            self._qdb = get_client()
        return self._qdb

    def _p(self):
        if self._parquet is None:
            from src.database.parquet_store import get_store
            self._parquet = get_store()
        return self._parquet

    # ── Single source readers ──────────────────────────────────────────────

    def ohlcv(self, symbol: str, timeframe: str, start, end):
        """Primary OHLCV from Parquet (preferred — full history)."""
        return self._p().query(symbol, start=start, end=end, timeframe=timeframe)

    def funding(self, symbol: str, start, end):
        return self._p().query(symbol, start=start, end=end, timeframe="funding")

    def news(self, since=None):
        """Recent news with sentiment from the news Parquet partition."""
        return self._p().query("_NEWS", start=since, end=None, timeframe="news")

    # ── Combined training frame ────────────────────────────────────────────

    def training_frame(self, *, symbol: str, timeframe: str,
                       start, end,
                       include_funding: bool = True,
                       include_news_24h: bool = True,
                       include_macro: bool = False):
        """Time-aligned join. Returns a pandas DataFrame.

        Missing optional sources are silently skipped. The base index is
        the OHLCV timestamps for `(symbol, timeframe)`.
        """
        import pandas as pd
        df = self.ohlcv(symbol, timeframe, start, end)
        if df is None or df.empty:
            logger.warning("[DataLens] no OHLCV for %s/%s", symbol, timeframe)
            return df

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if include_funding:
            try:
                fnd = self.funding(symbol, start, end)
                if fnd is not None and not fnd.empty:
                    fnd = fnd[["timestamp", "fundingRate"]] if "fundingRate" in fnd.columns \
                          else fnd[["timestamp", "funding_rate"]]
                    fnd.columns = ["timestamp", "funding_rate"]
                    fnd["timestamp"] = pd.to_datetime(fnd["timestamp"])
                    df = pd.merge_asof(df, fnd.sort_values("timestamp"),
                                       on="timestamp", direction="backward")
            except Exception as exc:
                logger.debug("[DataLens] funding skipped: %s", exc)

        if include_news_24h:
            try:
                news_df = self.news(since=start)
                if news_df is not None and not news_df.empty:
                    news_df = news_df.copy()
                    if "published_at" in news_df.columns:
                        news_df["timestamp"] = pd.to_datetime(news_df["published_at"])
                    elif "ts" in news_df.columns:
                        news_df["timestamp"] = pd.to_datetime(news_df["ts"], unit="ms")
                    if "score" not in news_df.columns:
                        news_df["score"] = 0.0
                    # 24-hour rolling mean sentiment
                    news_df = news_df.sort_values("timestamp").set_index("timestamp")
                    rolling = news_df["score"].rolling("24h").mean()
                    rolling_df = rolling.reset_index().rename(
                        columns={"score": "news_sentiment_24h"})
                    df = pd.merge_asof(df, rolling_df.sort_values("timestamp"),
                                       on="timestamp", direction="backward")
            except Exception as exc:
                logger.debug("[DataLens] news skipped: %s", exc)

        if include_macro:
            # Macro indicators are point-in-time signals stored in QuestDB
            # `model_signals`. Fetch them for the same time range.
            try:
                macro_keys = ["fred_dxy", "fred_vix", "fred_us_10y",
                              "fear_greed", "tvl_ethereum"]
                start_iso = (datetime.fromisoformat(str(start).replace("Z", "+00:00"))
                             if not isinstance(start, datetime) else start).isoformat()
                end_iso = (datetime.fromisoformat(str(end).replace("Z", "+00:00"))
                           if not isinstance(end, datetime) else end).isoformat()
                for key in macro_keys:
                    # Phase A5 (2026-05-12): parameterized. All three
                    # values come from internal sources (hardcoded
                    # macro_keys list + datetime → ISO conversion)
                    # but parameterizing eliminates the risk class
                    # entirely.
                    sql = ("SELECT timestamp, value FROM model_signals "
                           "WHERE symbol = ? "
                           "AND timestamp >= ? AND timestamp < ?")
                    rows = self._q().query(sql, params=[key, start_iso, end_iso])
                    if not rows:
                        continue
                    macro_df = __import__("pandas").DataFrame(rows)
                    macro_df["timestamp"] = __import__("pandas").to_datetime(macro_df["timestamp"])
                    macro_df = macro_df.rename(columns={"value": key}).sort_values("timestamp")
                    df = __import__("pandas").merge_asof(
                        df, macro_df, on="timestamp", direction="backward",
                    )
            except Exception as exc:
                logger.debug("[DataLens] macro skipped: %s", exc)

        return df


__all__ = ["DataLens"]
