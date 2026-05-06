"""
Parquet-first feature reader — Phase 10.

Replaces `analyzers[symbol].load_data(csv_path)` calls in `main.py` with
queries against the Parquet store. Falls back to the legacy CSV.gz path
if Parquet isn't populated yet — so the live bot keeps working even
during the cut-over.

Read order:
  1. Parquet store (`data/parquet/{SYM}/{tf}/yyyymm=*/`) — preferred
  2. Legacy CSV.gz (`data/raw/{SYM}_{tf}.csv.gz`) — fallback

Returned shape matches what `ElliottWaveAnalyzer.load_data` produces:
a list[dict] with keys timestamp / open / high / low / close / volume.
"""
from __future__ import annotations

import gzip
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

_TF_BAR_SECONDS = {
    "1s":  1,        "1m":  60,       "5m":  300,
    "15m": 900,      "1h":  3600,     "4h":  14400,
    "1d":  86400,    "1w":  604800,   "1mo": 2592000,
}


def _parquet_load(symbol: str, timeframe: str, tail_n: int):
    """Read the most recent `tail_n` bars from the Parquet store."""
    try:
        from src.database.parquet_store import get_store
    except Exception:
        return None
    store = get_store()
    secs = _TF_BAR_SECONDS.get(timeframe, 3600)
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=secs * (tail_n + 50))
    try:
        df = store.query(symbol, start=start, end=end, timeframe=timeframe)
    except Exception as exc:
        logger.debug("[feature_reader] parquet query failed: %s", exc)
        return None
    if df is None or df.empty:
        return None
    df = df.tail(tail_n).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


def _csv_load(symbol: str, timeframe: str, tail_n: int):
    """Fallback to the legacy CSV.gz path."""
    safe = symbol.replace("/", "_")
    candidates = [
        RAW_DIR / f"{safe}_{timeframe}.csv.gz",
        RAW_DIR / "historical" / f"{safe}_{timeframe}.csv.gz",
        RAW_DIR / "historical" / f"{safe}_spot_{timeframe}.csv.gz",
    ]
    for p in candidates:
        if p.exists():
            try:
                import csv
                rows: list[dict] = []
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            rows.append({
                                "timestamp": row["timestamp"],
                                "open":  float(row.get("open", 0)),
                                "high":  float(row.get("high", 0)),
                                "low":   float(row.get("low", 0)),
                                "close": float(row.get("close", 0)),
                                "volume": float(row.get("volume", 0)),
                            })
                        except (ValueError, KeyError):
                            continue
                return rows[-tail_n:] if rows else None
            except Exception as exc:
                logger.debug("[feature_reader] csv read failed: %s", exc)
    return None


def load_recent_bars(symbol: str, timeframe: str, tail_n: int = 1000):
    """Public API. Tries Parquet first, falls back to CSV.gz.

    Returns: list[dict] or None.
    """
    rows = _parquet_load(symbol, timeframe, tail_n)
    if rows:
        logger.debug("[feature_reader] %s/%s -> parquet (%d rows)",
                     symbol, timeframe, len(rows))
        return rows
    rows = _csv_load(symbol, timeframe, tail_n)
    if rows:
        logger.debug("[feature_reader] %s/%s -> csv (%d rows)",
                     symbol, timeframe, len(rows))
    return rows


def load_news_recent(hours: int = 48):
    """Recent news headlines + sentiment from the news Parquet partition.

    Reads `data/parquet/_NEWS/news/yyyymm=*/*.parquet` directly via DuckDB.
    The news schema's time column is `ts` (or `published_at` on legacy
    partitions) — different from the OHLCV `timestamp` column the generic
    parquet_store.query expects, so we query inline rather than through
    that helper.

    Returns a list of dicts; empty list if no partitions / no data."""
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    glob = str(project_root / "data" / "parquet" / "_NEWS" / "news" /
               "yyyymm=*" / "*.parquet").replace("\\", "/")
    if not list(Path(project_root / "data" / "parquet" / "_NEWS" / "news").glob(
            "yyyymm=*/*.parquet")):
        return []
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=int(hours))
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        # union_by_name lets us read mixed schemas (legacy `published_at`
        # files alongside the new `ts`-based GDELT/Reddit scrapers). We
        # discover which timestamp columns actually exist before referencing
        # them, because DuckDB's binder is strict — referencing a missing
        # column even inside COALESCE/TRY_CAST raises BinderException.
        union_cols = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{glob}', union_by_name=True) LIMIT 0"
        ).fetchall()]
        ts_candidates = [c for c in ("ts", "published_at", "timestamp")
                         if c in union_cols]
        if not ts_candidates:
            con.close()
            return []
        ts_expr = "COALESCE(" + ", ".join(
            f"TRY_CAST({c} AS TIMESTAMP)" for c in ts_candidates
        ) + ")"
        sql = f"""
        WITH src AS (
            SELECT * FROM read_parquet('{glob}', union_by_name=True)
        )
        SELECT *
        FROM src
        WHERE {ts_expr} BETWEEN ? AND ?
        ORDER BY {ts_expr}
        """
        df = con.execute(sql, [start, end]).df()
        con.close()
    except Exception:
        return []
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


__all__ = ["load_recent_bars", "load_news_recent"]
