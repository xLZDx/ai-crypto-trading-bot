"""Download all data needed for model training: funding rates, OHLCV, news."""
import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('download')

# ── Load watchlist ────────────────────────────────────────────────────────────
wl_path = os.path.join(project_root, 'data', 'watchlist.json')
if os.path.exists(wl_path):
    with open(wl_path, encoding='utf-8') as f:
        SYMBOLS = json.load(f)
else:
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
log.info("Watchlist: %s", SYMBOLS)

# ── 1. Funding rates ──────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("[1/4] Downloading funding rate history (2 years)...")
log.info("=" * 60)
try:
    from src.data_ingestion.funding_rate_downloader import download_funding_rates
    download_funding_rates(days=365 * 2)
    log.info("Funding rates: DONE")
except Exception as e:
    log.warning("Funding rates failed (non-fatal): %s", e)

# ── 2. OHLCV historical backfill ─────────────────────────────────────────────
log.info("=" * 60)
log.info("[2/4] Historical OHLCV backfill (1h, 730 days per symbol)...")
log.info("=" * 60)
try:
    from src.data_ingestion.historical_backfill import backfill_history
    for sym in SYMBOLS:
        log.info("  Backfilling %s 1h ...", sym)
        try:
            backfill_history(symbol=sym, timeframe='1h', days=730)
            log.info("  %s 1h: DONE", sym)
        except Exception as e:
            log.warning("  %s 1h failed: %s", sym, e)
    log.info("OHLCV backfill: DONE")
except Exception as e:
    log.warning("OHLCV backfill error: %s", e)

# ── 3. News ───────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("[3/4] News scraper...")
log.info("=" * 60)
try:
    from src.data_ingestion.news_scraper import scrape_news
    scrape_news()
    log.info("News: DONE")
except Exception as e:
    try:
        import src.data_ingestion.news_scraper as ns
        if hasattr(ns, 'main'):
            ns.main()
        log.info("News: DONE")
    except Exception as e2:
        log.warning("News scraper failed (non-fatal): %s", e2)

# ── 4. Binance live sync ─────────────────────────────────────────────────────
log.info("=" * 60)
log.info("[4/4] Binance downloader (live candles)...")
log.info("=" * 60)
try:
    from src.data_ingestion.binance_downloader import download_recent
    download_recent()
    log.info("Binance sync: DONE")
except Exception as e:
    try:
        import src.data_ingestion.binance_downloader as bd
        if hasattr(bd, 'main'):
            bd.main()
        log.info("Binance sync: DONE")
    except Exception as e2:
        log.warning("Binance downloader failed (non-fatal): %s", e2)

log.info("=" * 60)
log.info("ALL DOWNLOADS COMPLETE")
log.info("=" * 60)
