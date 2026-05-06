"""
gdelt_news_backfill — backfill historical crypto news from GDELT 2.0 for
every coin in the watchlist, going back to each coin's listing date (or
2015-02 = GDELT 2.0 start, whichever is later).

GDELT (https://api.gdeltproject.org/api/v2/doc/doc) is free, requires no
API key, and returns articles with a tone score (-100..+100) we use as a
zero-cost sentiment proxy. Limitations:

  - One query returns up to 250 articles. We page by month-window so each
    coin × month bucket fits.
  - Tone is GDELT's coarse aggregate, not a finance-tuned sentiment score.
    Better than nothing, much cheaper than running FinBERT over millions
    of articles. We can swap it for a model later.
  - GDELT has false positives: "Cardano" matches an Italian coach,
    "UNI" matches universities, "SOL" can mean Spanish "sun". Per-coin
    queries use coin-name + ticker context (`(Bitcoin OR BTC) AND
    cryptocurrency`) to suppress unrelated hits.

Output layout matches the existing news ingest in data/parquet/_NEWS/news/:
   yyyymm=YYYY-MM/<COIN>.parquet
each Parquet has: ts (datetime), title, url, source, language, tone (float),
                  coin (str), domain (str)

Run via:  python -m src.data_ingestion.gdelt_news_backfill [--coin BTC]
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
NEWS_OUT_ROOT = PROJECT_ROOT / "data" / "parquet" / "_NEWS" / "news"

GDELT_API     = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_T0      = datetime(2015, 2, 18, tzinfo=timezone.utc)  # GDELT 2.0 start
PER_QUERY_MAX = 250
# GDELT explicitly asks for ≥5s between requests in their 429 message.
# We honour that + give 0.5s slack. Total runtime: 108 month-windows ×
# 20 coins × 5.5s ≈ 3.3 hours per full pass.
SLEEP_BETWEEN = 5.5
MAX_RETRIES   = 3
RETRY_BACKOFF = 30   # 429 → wait 30s before retry


# Per-coin queries. Each maps to (keyword_query, listing_date).
# keyword_query is the GDELT search expression — combines name + ticker
# with crypto context to suppress homonyms (UNI: universities, SOL: sun).
COIN_QUERIES: dict[str, tuple[str, datetime]] = {
    "BTC":   ('(Bitcoin OR "BTC")',                                   datetime(2015, 2, 18, tzinfo=timezone.utc)),
    "ETH":   ('(Ethereum OR "ETH")',                                  datetime(2015, 7, 30, tzinfo=timezone.utc)),
    "SOL":   ('(Solana OR "SOL/USD" OR "SOLUSDT") cryptocurrency',    datetime(2020, 8, 11, tzinfo=timezone.utc)),
    "ADA":   ('(Cardano OR "ADA/USD" OR "ADAUSDT") cryptocurrency',   datetime(2018, 4, 17, tzinfo=timezone.utc)),
    "BNB":   ('(Binance Coin OR "BNB") cryptocurrency',               datetime(2017, 7, 25, tzinfo=timezone.utc)),
    "XRP":   ('(XRP OR Ripple) cryptocurrency',                       datetime(2018, 5, 4,  tzinfo=timezone.utc)),
    "DOGE":  ('(Dogecoin OR "DOGE") cryptocurrency',                  datetime(2019, 7, 5,  tzinfo=timezone.utc)),
    "TRX":   ('(Tron OR "TRX") cryptocurrency',                       datetime(2018, 8, 30, tzinfo=timezone.utc)),
    "AVAX":  ('(Avalanche OR "AVAX") cryptocurrency',                 datetime(2020, 9, 22, tzinfo=timezone.utc)),
    "SHIB":  ('("Shiba Inu" OR "SHIB") cryptocurrency',               datetime(2021, 5, 11, tzinfo=timezone.utc)),
    "DOT":   ('(Polkadot OR "DOT") cryptocurrency',                   datetime(2020, 8, 19, tzinfo=timezone.utc)),
    "LINK":  ('(Chainlink OR "LINK") cryptocurrency',                 datetime(2019, 1, 16, tzinfo=timezone.utc)),
    "NEAR":  ('("NEAR Protocol" OR "NEARUSDT") cryptocurrency',       datetime(2020, 10, 14, tzinfo=timezone.utc)),
    "UNI":   ('(Uniswap OR "UNI") cryptocurrency',                    datetime(2020, 9, 17, tzinfo=timezone.utc)),
    "LTC":   ('(Litecoin OR "LTC") cryptocurrency',                   datetime(2017, 12, 13, tzinfo=timezone.utc)),
    "APT":   ('(Aptos OR "APT") cryptocurrency',                      datetime(2022, 10, 19, tzinfo=timezone.utc)),
    "ATOM":  ('(Cosmos OR "ATOM") cryptocurrency',                    datetime(2019, 4, 29, tzinfo=timezone.utc)),
    "HBAR":  ('(Hedera OR "HBAR") cryptocurrency',                    datetime(2019, 9, 17, tzinfo=timezone.utc)),
    "ICP":   ('("Internet Computer" OR "ICP") cryptocurrency',        datetime(2021, 5, 10, tzinfo=timezone.utc)),
    "SUI":   ('("Sui" OR "SUIUSDT") cryptocurrency',                  datetime(2023, 5, 3,  tzinfo=timezone.utc)),
}


def _month_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Generate (month_start, month_end) pairs covering [start, end)."""
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    out: list[tuple[datetime, datetime]] = []
    while cur < end:
        # Next month start
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        out.append((max(cur, start), min(nxt, end)))
        cur = nxt
    return out


def _gdelt_fmt(dt: datetime) -> str:
    """GDELT timestamp format: YYYYMMDDHHMMSS, all UTC."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def fetch_window(query: str, start: datetime, end: datetime,
                 max_records: int = PER_QUERY_MAX,
                 timeout: int = 30) -> list[dict]:
    """Fetch up to max_records articles in [start, end) for a query.
    Retries on 429 with exponential backoff. Returns empty list on
    persistent failure so the caller can keep iterating."""
    params = {
        "query":         query,
        "mode":          "ArtList",
        "format":        "JSON",
        "startdatetime": _gdelt_fmt(start),
        "enddatetime":   _gdelt_fmt(end),
        "maxrecords":    str(max_records),
        "sort":          "DateAsc",
    }
    backoff = RETRY_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(GDELT_API, params=params, timeout=timeout)
            if r.status_code == 429:
                logger.info("GDELT 429 on %s — sleeping %ds (attempt %d/%d)",
                            query[:30], backoff, attempt + 1, MAX_RETRIES)
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code != 200:
                logger.warning("GDELT %s: HTTP %d (%s)",
                               query[:40], r.status_code, r.text[:80])
                return []
            if not r.text.strip():
                return []
            try:
                data = r.json()
            except json.JSONDecodeError:
                return []
            return data.get("articles") or []
        except Exception as exc:
            logger.warning("GDELT fetch %s [%s..%s] try %d: %s",
                           query[:30], start.date(), end.date(), attempt + 1, exc)
            time.sleep(backoff)
            backoff *= 2
    return []


def _to_records(articles: list[dict], coin: str) -> list[dict]:
    """Normalise GDELT article dicts into our schema."""
    out: list[dict] = []
    for a in articles:
        url = a.get("url", "")
        title = (a.get("title") or "").strip()[:500]
        if not url or not title:
            continue
        # GDELT timestamp format: YYYYMMDDTHHMMSSZ
        ts_raw = a.get("seendate") or ""
        try:
            ts = datetime.strptime(ts_raw, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        try:
            tone = float(a.get("tone") or 0)
        except (TypeError, ValueError):
            tone = 0.0
        out.append({
            "ts":       ts,
            "title":    title,
            "url":      url[:500],
            "source":   "gdelt",
            "language": (a.get("language") or "")[:8],
            "tone":     tone,
            "coin":     coin,
            "domain":   (a.get("domain") or "")[:120],
        })
    return out


def backfill_coin(
    coin: str,
    *,
    end: datetime | None = None,
    out_root: Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Download monthly news for one coin from its listing date → end.
    Writes Parquet files keyed by yyyymm. Skips months whose file already
    exists (idempotent — safe to re-run)."""
    if coin not in COIN_QUERIES:
        return {"coin": coin, "status": "unknown-coin"}
    query, listing = COIN_QUERIES[coin]
    end = end or datetime.now(timezone.utc)
    out_root = out_root or NEWS_OUT_ROOT
    start = max(listing, GDELT_T0)

    windows = _month_windows(start, end)
    rows_total = 0
    months_written = 0
    months_skipped = 0
    started = time.time()

    import pandas as pd  # heavy import deferred until actually backfilling

    for w_start, w_end in windows:
        ymm = w_start.strftime("%Y-%m")
        out_dir = out_root / f"yyyymm={ymm}"
        out_path = out_dir / f"{coin}.parquet"
        if out_path.exists():
            months_skipped += 1
            continue
        articles = fetch_window(query, w_start, w_end)
        records = _to_records(articles, coin)
        if records:
            out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame.from_records(records)
            # Sort + dedup by URL within the month.
            df = df.sort_values("ts").drop_duplicates(subset=["url"]).reset_index(drop=True)
            try:
                df.to_parquet(out_path, index=False, compression="snappy")
                rows_total += len(df)
                months_written += 1
            except Exception as exc:
                logger.warning("%s %s: parquet write failed: %s", coin, ymm, exc)
        else:
            # Empty month — write a sentinel zero-row parquet so we don't
            # re-fetch on the next run (idempotency).
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                pd.DataFrame(columns=["ts", "title", "url", "source",
                                       "language", "tone", "coin", "domain"]).to_parquet(
                    out_path, index=False, compression="snappy"
                )
                months_written += 1
            except Exception:
                pass
        if progress:
            progress({"phase": "month", "coin": coin, "yyyymm": ymm,
                      "rows": len(records),
                      "elapsed_s": time.time() - started})
        time.sleep(SLEEP_BETWEEN)

    return {
        "coin":           coin,
        "status":         "ok",
        "windows":        len(windows),
        "months_written": months_written,
        "months_skipped": months_skipped,
        "rows_total":     rows_total,
        "elapsed_s":      round(time.time() - started, 1),
    }


def backfill_all(
    coins: tuple[str, ...] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    coins = coins or tuple(COIN_QUERIES.keys())
    out: dict[str, dict] = {}
    for i, coin in enumerate(coins):
        if progress:
            progress({"phase": "coin_start", "coin": coin,
                      "i": i, "total": len(coins)})
        try:
            out[coin] = backfill_coin(coin, progress=progress)
        except Exception as exc:
            out[coin] = {"_error": f"{type(exc).__name__}: {exc}"}
        if progress:
            progress({"phase": "coin_done", "coin": coin,
                      "i": i + 1, "total": len(coins),
                      "result": out[coin]})
    return out


if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="GDELT crypto news backfill")
    ap.add_argument("--coin", help="Single coin (BTC, ETH, ...). Omit for all.")
    args = ap.parse_args()
    def _cb(ev): sys.stderr.write(json.dumps(ev) + "\n")
    if args.coin:
        print(json.dumps(backfill_coin(args.coin.upper(), progress=_cb), default=str, indent=2))
    else:
        print(json.dumps(backfill_all(progress=_cb), default=str, indent=2))
