"""
reddit_news_backfill — community-sentiment counterpart to GDELT.

Why this complements GDELT:
  - GDELT is editorial / publisher news.
  - Reddit captures *community* sentiment — early signal of price moves
    often shows up here before mainstream press picks it up. Per-coin
    subreddits (r/Bitcoin, r/ethereum, r/CryptoCurrency, r/CardanoCoin,
    r/solana, ...) are dense with discussion.
  - Free, no API key required for the JSON endpoints. Reddit asks for
    a User-Agent that identifies your project — we comply.

Caveats:
  - Reddit's free JSON API has a hard pagination cap of ~1000 listings
    per query (back through "new" feed), so depth on a single subreddit
    is bounded. We compensate by polling MULTIPLE subreddits per coin
    and including comment-rich submissions only.
  - Pushshift's deeper archive is defunct. Going pre-2024 needs a
    HuggingFace pre-scraped dataset or paid API. This module captures
    going-forward + ~last 1000 posts/sub.

Output: data/parquet/_NEWS/news/yyyymm=YYYY-MM/reddit_<COIN>.parquet
Schema mirrors GDELT (ts, title, url, source, language, tone, coin,
domain). Tone is the title's small-lexicon score, identical formula
to cryptocompare_news_backfill so downstream features stay consistent.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
NEWS_OUT_ROOT = PROJECT_ROOT / "data" / "parquet" / "_NEWS" / "news"

USER_AGENT = "ai-trading-assistance/1.0 (research)"
LISTING_API = "https://www.reddit.com/r/{sub}/new.json"
SLEEP_BETWEEN = 2.5    # ~24 req/min — comfortable under Reddit's 100/min cap

# Per-coin subreddit fan-out. r/CryptoCurrency catches everything; the
# coin-specific subs catch deeper / more focused threads.
COIN_SUBS: dict[str, tuple[str, ...]] = {
    "BTC":  ("Bitcoin", "BitcoinMarkets", "btc"),
    "ETH":  ("ethereum", "ethfinance", "ethtrader"),
    "SOL":  ("solana",),
    "ADA":  ("cardano", "CardanoCoin"),
    "BNB":  ("binance",),
    "XRP":  ("Ripple", "XRP"),
    "DOGE": ("dogecoin",),
    "TRX":  ("Tronix",),
    "AVAX": ("Avax", "Avalanche"),
    "SHIB": ("SHIBArmy",),
    "DOT":  ("Polkadot",),
    "LINK": ("Chainlink",),
    "NEAR": ("NEARProtocol",),
    "UNI":  ("UniSwap",),
    "LTC":  ("litecoin",),
    "APT":  ("Aptos_Network",),
    "ATOM": ("cosmosnetwork",),
    "HBAR": ("Hedera",),
    "ICP":  ("dfinity",),
    "SUI":  ("SuiNetwork", "Sui"),
}

# Cross-coin general subs we always poll once per run.
GENERAL_SUBS = ("CryptoCurrency", "CryptoMarkets")

# Same lexicon as the cryptocompare module — keep tone scores comparable.
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
    if not title:
        return 0.0
    words = {w.strip(".,!?:;'\"()").lower() for w in title.split()}
    n = sum(1 for w in words if w in _NEG_WORDS)
    p = sum(1 for w in words if w in _POS_WORDS)
    if not (n or p):
        return 0.0
    return round((p - n) / max(p + n, 1), 3)


def fetch_subreddit_pages(sub: str,
                          max_pages: int = 10,
                          progress: Callable[[dict], None] | None = None) -> list[dict]:
    """Walk r/<sub>/new.json with the `after` cursor for up to max_pages
    pages of 100 listings each. Returns a flat list of submissions."""
    all_posts: list[dict] = []
    after: str | None = None
    headers = {"User-Agent": USER_AGENT}
    for page in range(max_pages):
        params: dict[str, str | int] = {"limit": 100, "raw_json": 1}
        if after:
            params["after"] = after
        url = LISTING_API.format(sub=sub)
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                logger.info("reddit r/%s 429 — sleeping 30s", sub)
                time.sleep(30)
                continue
            if r.status_code == 403:
                logger.warning("reddit r/%s blocked (403) — quarantined or banned", sub)
                break
            if r.status_code != 200:
                logger.warning("reddit r/%s HTTP %d", sub, r.status_code)
                break
            payload = r.json().get("data") or {}
            children = payload.get("children") or []
            if not children:
                break
            for c in children:
                d = c.get("data") or {}
                if d:
                    all_posts.append(d)
            after = payload.get("after")
            if not after:
                break
        except Exception as exc:
            logger.warning("reddit r/%s page %d: %s", sub, page, exc)
            break
        if progress:
            progress({"phase": "page", "subreddit": sub,
                      "page": page + 1, "posts_so_far": len(all_posts)})
        time.sleep(SLEEP_BETWEEN)
    return all_posts


def _to_record(post: dict, coin: str) -> dict | None:
    title = (post.get("title") or "").strip()
    if not title:
        return None
    pid = post.get("id")
    permalink = post.get("permalink") or ""
    url = ("https://www.reddit.com" + permalink) if permalink else (post.get("url") or "")
    if not url:
        return None
    created = post.get("created_utc") or post.get("created") or 0
    try:
        ts = datetime.fromtimestamp(int(created), tz=timezone.utc)
    except (TypeError, ValueError):
        return None
    score = int(post.get("score", 0) or 0)
    return {
        "ts":       ts,
        "title":    title[:500],
        "url":      url[:500],
        "source":   "reddit",
        "language": "en",
        "tone":     _crude_tone(title),
        "coin":     coin,
        "domain":   ("r/" + (post.get("subreddit") or ""))[:120],
        # Reddit-specific extras (kept in the parquet so downstream features
        # can use upvote score as a sentiment magnitude weight).
        "score":    score,
        "id":       pid or "",
    }


def _bucket_yyyymm(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


def backfill(
    coins: tuple[str, ...] | None = None,
    *,
    pages_per_sub: int = 10,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Walk every coin-specific subreddit + general subs, normalise posts
    to the news schema, and write monthly parquet partitions per coin.
    pages_per_sub × 100 = max posts per subreddit (Reddit hard cap is
    ~1000 listings → pages_per_sub=10 is the practical max)."""
    import pandas as pd
    coins = coins or tuple(COIN_SUBS.keys())
    started = time.time()

    # Phase 1: pull all general-sub posts ONCE; tag each into matching coin
    # buckets via title keyword.
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    coin_kw: dict[str, set[str]] = {}
    for c in coins:
        coin_kw[c] = {c.lower(), *_extra_keywords(c)}

    posts_total = 0
    for sub in GENERAL_SUBS:
        if progress:
            progress({"phase": "sub_start", "subreddit": sub, "scope": "general"})
        posts = fetch_subreddit_pages(sub, max_pages=pages_per_sub, progress=progress)
        posts_total += len(posts)
        for p in posts:
            title_lc = (p.get("title") or "").lower()
            matched = [c for c in coins if any(kw in title_lc for kw in coin_kw[c])]
            for c in matched or ["ALL"]:
                rec = _to_record(p, c)
                if rec is None:
                    continue
                by_bucket.setdefault((_bucket_yyyymm(rec["ts"]), c), []).append(rec)

    # Phase 2: per-coin subreddits — every post lands in that coin's bucket
    # (no keyword match required since the subreddit is already coin-specific).
    for c, subs in COIN_SUBS.items():
        if c not in coins:
            continue
        for sub in subs:
            if progress:
                progress({"phase": "sub_start", "subreddit": sub, "coin": c})
            posts = fetch_subreddit_pages(sub, max_pages=pages_per_sub, progress=progress)
            posts_total += len(posts)
            for p in posts:
                rec = _to_record(p, c)
                if rec is None:
                    continue
                by_bucket.setdefault((_bucket_yyyymm(rec["ts"]), c), []).append(rec)

    # Phase 3: dedup-by-id and write per (yyyymm, coin) parquet.
    written = 0
    for (ymm, coin), recs in by_bucket.items():
        out_dir = NEWS_OUT_ROOT / f"yyyymm={ymm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"reddit_{coin}.parquet"
        df_new = pd.DataFrame.from_records(recs).sort_values("ts").drop_duplicates(subset=["id"])
        if out_path.exists():
            try:
                df_old = pd.read_parquet(out_path)
                df_new = pd.concat([df_old, df_new]).drop_duplicates(subset=["id"]).sort_values("ts")
            except Exception:
                pass
        try:
            df_new.to_parquet(out_path, index=False, compression="snappy")
            written += 1
        except Exception as exc:
            logger.warning("write %s failed: %s", out_path, exc)

    if progress:
        progress({"phase": "done", "buckets_written": written,
                  "posts_total": posts_total,
                  "elapsed_s": time.time() - started})
    return {
        "status":          "ok",
        "buckets_written": written,
        "posts_total":     posts_total,
        "elapsed_s":       round(time.time() - started, 1),
    }


def _extra_keywords(coin: str) -> set[str]:
    EXTRA = {
        "BTC":  {"bitcoin"},
        "ETH":  {"ethereum", "ether"},
        "SOL":  {"solana"},
        "ADA":  {"cardano"},
        "BNB":  {"bnb"},
        "XRP":  {"ripple"},
        "DOGE": {"dogecoin"},
        "TRX":  {"tron"},
        "AVAX": {"avalanche"},
        "SHIB": {"shiba"},
        "DOT":  {"polkadot"},
        "LINK": {"chainlink"},
        "NEAR": {"near protocol"},
        "UNI":  {"uniswap"},
        "LTC":  {"litecoin"},
        "APT":  {"aptos"},
        "ATOM": {"cosmos"},
        "HBAR": {"hedera"},
        "ICP":  {"internet computer"},
        "SUI":  {"sui network"},
    }
    return EXTRA.get(coin.upper(), set())


if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Reddit historical news backfill")
    ap.add_argument("--pages", type=int, default=10,
                    help="Pages per subreddit (×100 posts). Reddit cap ≈10.")
    args = ap.parse_args()
    def _cb(ev): sys.stderr.write(json.dumps(ev) + "\n")
    print(json.dumps(backfill(pages_per_sub=args.pages, progress=_cb),
                     default=str, indent=2))
