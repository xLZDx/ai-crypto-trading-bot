"""
cryptocompare_news_backfill — pull crypto-specific historical news from
CryptoCompare and write monthly parquet partitions.

Why this complements GDELT (already running):
  - CryptoCompare aggregates from crypto-native publishers (CoinDesk,
    Cointelegraph, Decrypt, The Block, BeInCrypto, ...). Lower false-
    positive rate than GDELT on coin keywords (no "Cardano = football
    coach" issue).
  - Free, no API key required for the news endpoint.
  - Pagination via lTs= (lower-bound unix-seconds), 50 items per call,
    so depth requires many calls but each one is cheap.

Output: data/parquet/_NEWS/news/yyyymm=YYYY-MM/cc_<COIN>.parquet (or
        cc_ALL.parquet for the multi-coin pull). Schema mirrors GDELT:
  ts, title, url, source, language, tone, coin, domain.

Tone is approximated from CryptoCompare's `categories` (e.g. positive
words like "MARKET" alone is neutral; "REGULATION" + "BAN" trends
negative). Without a tone field we fall back to the headline VADER score
(if vaderSentiment is installed) — otherwise tone=0.0 and the sentiment
feature relies on title-text NLP downstream.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
NEWS_OUT_ROOT = PROJECT_ROOT / "data" / "parquet" / "_NEWS" / "news"

CC_API        = "https://min-api.cryptocompare.com/data/v2/news/"
SLEEP_BETWEEN = 1.5     # ~40 req/min — well under CryptoCompare's free limit
MAX_PAGES_PER_RUN = 2000   # ≈ 100k articles cap per coin (deep history)

# Tone proxy — CryptoCompare doesn't return a sentiment score on the free
# tier so we lean on a small lexicon over the title. Better than nothing;
# trainers still own the heavy NLP downstream via add_news_sentiment().
_NEG_WORDS = {
    "ban", "banned", "crash", "drop", "plunge", "hack", "hacked", "exploit",
    "scam", "fraud", "rug", "rugpull", "bear", "bearish", "lawsuit",
    "investigation", "indicted", "arrested", "down", "loss", "warning",
    "fail", "fails", "failed", "collapsed", "collapse", "selloff", "dump",
    "fud", "regulation", "regulator", "fine", "fined", "delist", "delisted",
}
_POS_WORDS = {
    "surge", "surges", "rally", "rallies", "soar", "soars", "bull", "bullish",
    "breakout", "ath", "all-time", "approved", "approval", "etf", "adoption",
    "partnership", "launch", "launched", "milestone", "growth", "rises", "rise",
    "gains", "gain", "up", "high", "record", "boost", "boosts",
}


def _crude_tone(title: str) -> float:
    """Tone in [-1, +1] from a small word-bag.

    Phase B preference: when finbert_scorer is available + a real model
    loaded (CryptoBERT / FinBERT), defer to it. Falls back to the
    lexicon when the model can't be loaded (no transformers, GPU OOM,
    network issue on first download). The output range stays the same
    [-1, +1] so existing parquet readers don't change.
    """
    if not title:
        return 0.0
    try:
        from src.analysis.finbert_scorer import score_one, is_ready
        if is_ready():
            return score_one(title)
    except Exception:
        pass
    words = {w.strip(".,!?:;'\"()").lower() for w in title.split()}
    n = sum(1 for w in words if w in _NEG_WORDS)
    p = sum(1 for w in words if w in _POS_WORDS)
    if not (n or p):
        return 0.0
    return round((p - n) / max(p + n, 1), 3)


def _coin_keywords(coin: str) -> set[str]:
    """Per-coin keywords for filtering articles (lowercase). Mirrors the
    GDELT scraper's coin map but free-form so we can match titles."""
    base = {coin.lower()}
    EXTRA = {
        "BTC": {"bitcoin"},
        "ETH": {"ethereum", "ether"},
        "SOL": {"solana"},
        "ADA": {"cardano"},
        "BNB": {"binance coin", "bnb chain"},
        "XRP": {"ripple"},
        "DOGE": {"dogecoin"},
        "TRX": {"tron"},
        "AVAX": {"avalanche"},
        "SHIB": {"shiba inu"},
        "DOT": {"polkadot"},
        "LINK": {"chainlink"},
        "NEAR": {"near protocol"},
        "UNI": {"uniswap"},
        "LTC": {"litecoin"},
        "APT": {"aptos"},
        "ATOM": {"cosmos"},
        "HBAR": {"hedera"},
        "ICP": {"internet computer"},
        "SUI": {"sui network", "sui blockchain"},
    }
    return base | EXTRA.get(coin.upper(), set())


def _api_key() -> str | None:
    """Return CRYPTOCOMPARE_API_KEY from env if present (loads .env on first
    call). Free-tier key still required as of 2025; without it the news
    endpoint returns 'valid auth key required'."""
    key = os.environ.get("CRYPTOCOMPARE_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            key = os.environ.get("CRYPTOCOMPARE_API_KEY")
        except Exception:
            pass
    if key:
        # Strip surrounding quotes that .env may carry
        key = key.strip().strip('"').strip("'")
    return key or None


def fetch_page(l_ts: int | None = None,
               categories: str | None = None,
               timeout: int = 30) -> list[dict]:
    """Fetch one page of news from CryptoCompare. Returns the article
    list (newest first). Pass lTs=<unix-seconds> to fetch articles
    older than that timestamp (paginate to deeper history)."""
    params: dict[str, str | int] = {"sortOrder": "latest"}
    if l_ts is not None:
        params["lTs"] = int(l_ts)
    if categories:
        params["categories"] = categories
    headers: dict[str, str] = {}
    key = _api_key()
    if key:
        # CryptoCompare accepts the key as a header OR as ?api_key=
        # — header is cleaner and doesn't appear in proxy logs.
        headers["authorization"] = f"Apikey {key}"
    try:
        r = requests.get(CC_API, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            logger.warning("CryptoCompare HTTP %d: %s", r.status_code, r.text[:80])
            return []
        body = r.json()
        if body.get("Response") == "Error":
            logger.warning("CryptoCompare error: %s",
                           body.get("Message", "")[:120])
            return []
        return body.get("Data") or []
    except Exception as exc:
        logger.warning("CryptoCompare fetch lTs=%s: %s", l_ts, exc)
        return []


def _bucket_yyyymm(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


def _to_record(article: dict, coin: str = "ALL") -> dict | None:
    title = (article.get("title") or "").strip()
    url   = article.get("url") or ""
    if not title or not url:
        return None
    pub = article.get("published_on")
    if not pub:
        return None
    try:
        ts = datetime.fromtimestamp(int(pub), tz=timezone.utc)
    except (TypeError, ValueError):
        return None
    return {
        "ts":        ts,
        "title":     title[:500],
        "url":       url[:500],
        "source":    "cryptocompare",
        "language":  (article.get("lang") or "EN")[:8],
        "tone":      _crude_tone(title),
        "coin":      coin,
        "domain":    (article.get("source_info", {}).get("name") or
                      article.get("source") or "")[:120],
    }


def backfill(
    coins: tuple[str, ...] | None = None,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """One-shot historical pull. Iterates pages backwards in time from
    `end` (default: now) until either `start` is crossed or pages run out.

    Articles are bucketed per yyyymm and split per coin via title keyword
    matching — one article can land in multiple coin buckets if it
    mentions e.g. both BTC and ETH. Idempotent: writes per (yyyymm, coin)
    parquet, dedupe-by-url within each.

    coins=None means we still bucket per coin (matched on title), but
    also write a `cc_ALL.parquet` per yyyymm with everything. Useful for
    'show me all crypto news in March 2024' queries.
    """
    import pandas as pd
    coins = coins or (
        "BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "TRX",
        "AVAX", "SHIB", "DOT", "LINK", "NEAR", "UNI", "LTC", "APT",
        "ATOM", "HBAR", "ICP", "SUI",
    )
    end = end or datetime.now(timezone.utc)
    start = start or datetime(2017, 1, 1, tzinfo=timezone.utc)
    cur_l_ts = int(end.timestamp())
    coin_keys = {c: _coin_keywords(c) for c in coins}

    started_at = time.time()
    pages = 0
    records_total = 0
    # Per (yyyymm, coin) → list of records buffered before write.
    by_bucket: dict[tuple[str, str], list[dict]] = {}

    while pages < MAX_PAGES_PER_RUN:
        articles = fetch_page(l_ts=cur_l_ts)
        pages += 1
        if not articles:
            break
        oldest_ts = None
        for a in articles:
            rec = _to_record(a)
            if rec is None:
                continue
            ts: datetime = rec["ts"]
            if ts < start:
                # below our floor — stop the whole loop
                cur_l_ts = 0
                break
            oldest_ts = ts if (oldest_ts is None or ts < oldest_ts) else oldest_ts
            ymm = _bucket_yyyymm(ts)
            tl  = rec["title"].lower()
            matched: list[str] = []
            for c, kws in coin_keys.items():
                if any(kw in tl for kw in kws):
                    matched.append(c)
            # Always keep an ALL bucket so the operator can query
            # everything regardless of coin tagging.
            by_bucket.setdefault((ymm, "ALL"), []).append({**rec, "coin": "ALL"})
            for c in matched:
                by_bucket.setdefault((ymm, c), []).append({**rec, "coin": c})
            records_total += 1
        if cur_l_ts == 0 or oldest_ts is None:
            break
        cur_l_ts = int(oldest_ts.timestamp()) - 1
        if progress and pages % 5 == 0:
            progress({"phase": "page", "pages": pages,
                      "records_total": records_total,
                      "cur_ts": datetime.fromtimestamp(cur_l_ts,
                                                       tz=timezone.utc).isoformat(),
                      "elapsed_s": time.time() - started_at})
        time.sleep(SLEEP_BETWEEN)

    # Write per (yyyymm, coin) with dedup-by-url.
    written: dict[str, int] = {}
    for (ymm, coin), recs in by_bucket.items():
        out_dir = NEWS_OUT_ROOT / f"yyyymm={ymm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cc_{coin}.parquet"
        df_new = pd.DataFrame.from_records(recs).sort_values("ts").drop_duplicates(subset=["url"])
        if out_path.exists():
            try:
                df_old = pd.read_parquet(out_path)
                df_new = pd.concat([df_old, df_new]).drop_duplicates(subset=["url"]).sort_values("ts")
            except Exception:
                pass
        try:
            df_new.to_parquet(out_path, index=False, compression="snappy")
            written[f"{ymm}/cc_{coin}"] = len(df_new)
        except Exception as exc:
            logger.warning("write %s failed: %s", out_path, exc)

    if progress:
        progress({"phase": "done", "pages": pages,
                  "records_total": records_total,
                  "buckets_written": len(written),
                  "elapsed_s": time.time() - started_at})
    return {
        "status":          "ok",
        "pages":           pages,
        "records_total":   records_total,
        "buckets_written": len(written),
        "elapsed_s":       round(time.time() - started_at, 1),
    }


if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="CryptoCompare historical news")
    ap.add_argument("--start", default="2017-01-01",
                    help="Lower bound (YYYY-MM-DD). Default 2017-01-01.")
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    def _cb(ev): sys.stderr.write(json.dumps(ev) + "\n")
    print(json.dumps(backfill(start=start, progress=_cb), default=str, indent=2))
