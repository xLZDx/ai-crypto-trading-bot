"""
Phase 8 tests — data governance + rate limiter + binance_sync + connectors.

Coverage:
  - rate_limiter: token bucket budget tracking, react_to_response
  - binance_archive_downloader: HEAD probe + listing-cache helpers
  - binance_sync: orchestrator structure + helpers
  - data_governance: config load/save, registry, base contract
  - connectors: every registered source has correct META, instantiates,
    is_available() returns bool

Run:
    python tests/test_phase8.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
results = {"pass": 0, "fail": 0, "skip": 0}


def check(name, ok, detail=""):
    if ok is None:
        results["skip"] += 1
        print(f"  {SKIP} {name} (skipped)")
    elif ok:
        results["pass"] += 1
        print(f"  {PASS} {name}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name}{': ' + detail if detail else ''}")


# ─── Rate limiter ──────────────────────────────────────────────────────────

def test_rate_limiter():
    print("\n[Rate Limiter]")
    try:
        from src.data_ingestion.rate_limiter import (
            RateLimiter, get_limiter, rate_limited, stats,
        )
    except Exception as exc:
        check("import rate_limiter", False, str(exc))
        return
    check("import rate_limiter", True)

    lim = RateLimiter("test.host", weight_per_min=10, req_per_min=5)
    # Use up budget
    for i in range(5):
        with lim.acquire(weight=1):
            pass
    snap = lim._budget_used(__import__("time").time())
    check("budget after 5 calls: weight==5, req==5",
          snap == (5, 5), f"got {snap}")

    # Singleton check
    a = get_limiter("api.binance.com")
    b = get_limiter("api.binance.com")
    check("get_limiter returns singleton", a is b)

    # stats() shape
    s = stats()
    check("stats() returns dict",
          isinstance(s, dict) and "api.binance.com" in s)

    # react_to_response — synthesise a 429
    class FakeResp:
        status_code = 429
        headers = {"Retry-After": "0.05"}
    lim.react_to_response(FakeResp())
    check("429 sets banned_until in the future",
          lim._banned_until > __import__("time").time())


# ─── Archive downloader improvements ───────────────────────────────────────

def test_archive_improvements():
    print("\n[Archive Downloader Improvements]")
    try:
        from src.data_ingestion.binance_archive_downloader import (
            _zip_url, _zip_exists, _load_listing_cache, _save_listing_cache,
            _filter_months_by_listing, download_all_timeframes_parallel,
            MAX_WORKERS,
        )
    except Exception as exc:
        check("import improvements", False, str(exc))
        return
    check("import improvements", True)

    # URL builder
    u = _zip_url("BTCUSDT", 2024, 1, "1m")
    check("_zip_url format", u.endswith("BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"))

    # MAX_WORKERS default raised
    check("MAX_WORKERS >= 8", MAX_WORKERS >= 8)

    # Listing cache helpers
    with tempfile.TemporaryDirectory() as tmp:
        # Monkey-patch the cache path
        import src.data_ingestion.binance_archive_downloader as ad
        old_path = ad.LISTING_CACHE
        ad.LISTING_CACHE = Path(tmp) / "listing.json"
        try:
            cache = {"BTC/USDT": "2017-08", "SUI/USDT": "2023-04"}
            _save_listing_cache(cache)
            check("listing cache writes JSON",
                  ad.LISTING_CACHE.exists() and "BTC/USDT" in ad.LISTING_CACHE.read_text())

            loaded = _load_listing_cache()
            check("listing cache roundtrip", loaded == cache)

            # Filter pre-listing months
            months = [(2017, 1), (2017, 7), (2017, 8), (2017, 9), (2018, 1)]
            filt = _filter_months_by_listing("BTC/USDT", months, cache)
            check("filter drops months before 2017-08",
                  filt == [(2017, 8), (2017, 9), (2018, 1)])
        finally:
            ad.LISTING_CACHE = old_path

    # download_all_timeframes_parallel must exist
    check("download_all_timeframes_parallel() defined",
          callable(download_all_timeframes_parallel))


# ─── binance_sync ──────────────────────────────────────────────────────────

def test_binance_sync():
    print("\n[binance_sync]")
    try:
        from src.data_ingestion import binance_sync as bs
    except Exception as exc:
        check("import binance_sync", False, str(exc))
        return
    check("import binance_sync", True)
    check("step_archive() defined",     hasattr(bs, "step_archive"))
    check("step_rest_topup() defined",  hasattr(bs, "step_rest_topup"))
    check("step_cross_check() defined", hasattr(bs, "step_cross_check"))
    check("run() orchestrates 3 steps", hasattr(bs, "run"))
    check("interval map covers 1m/1h/1d",
          all(k in bs._BINANCE_INTERVAL for k in ("1m", "1h", "1d", "1mo")))


# ─── Data governance: config + registry + base ─────────────────────────────

def test_governance_framework():
    print("\n[Data Governance Framework]")
    try:
        from src.data_governance import (
            DataSourceConnector, GovernanceConfig, REGISTRY, list_sources,
        )
        from src.data_governance.base import ConnectorMeta
    except Exception as exc:
        check("import framework", False, str(exc))
        return
    check("import framework", True)

    # Config save/load roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "g.json"
        cfg = GovernanceConfig.default()
        cfg.config_path = cfg_path
        cfg.save()
        check("config save() writes file", cfg_path.exists())
        loaded = GovernanceConfig.load(cfg_path)
        check("config load() roundtrip",
              "bybit" in loaded.sources and loaded.sources["bybit"].enabled)

    # Registry must have all the connectors we just wrote
    from src.data_governance import connectors  # ensure side-effect imports
    expected = {"bybit", "okx", "coinbase", "kraken", "coingecko",
                "fear_greed", "fred", "defillama",
                "cryptocompare_news", "coinglass", "reddit"}
    actual = set(REGISTRY.keys())
    missing = expected - actual
    check("registry contains all 11 expected connectors",
          not missing, f"missing: {missing}")

    # list_sources returns useful metadata
    src_list = list_sources()
    check("list_sources returns >= 11 entries", len(src_list) >= 11)
    check("each entry has required keys",
          all({"name", "host", "priority", "category", "requires_auth"} <= set(s)
              for s in src_list))


# ─── Each connector instantiates and reports availability ──────────────────

def test_connectors_instantiate():
    print("\n[Connectors instantiate + is_available]")
    try:
        from src.data_governance import REGISTRY
        from src.data_governance import connectors  # noqa
    except Exception as exc:
        check("import connectors package", False, str(exc))
        return
    check("import connectors package", True)

    for name, cls in REGISTRY.items():
        try:
            inst = cls()
            avail = inst.is_available()
            check(f"  {name}: instantiates + is_available={avail}",
                  isinstance(avail, bool))
        except Exception as exc:
            check(f"  {name}: instantiates", False, str(exc))


# ─── Orchestrator structure ────────────────────────────────────────────────

def test_orchestrator():
    print("\n[Orchestrator]")
    try:
        from src.data_governance import orchestrator as o
    except Exception as exc:
        check("import orchestrator", False, str(exc))
        return
    check("import orchestrator", True)
    check("run_history() defined",  hasattr(o, "run_history"))
    check("run_forever() defined",  hasattr(o, "run_forever"))
    check("--list flag in main",    "--list" in (PROJECT_ROOT / "src" / "data_governance" /
                                                  "orchestrator.py").read_text(encoding="utf-8"))


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 8 -- Data Governance + Rate Limiting + Sync")
    print("=" * 60)
    test_rate_limiter()
    test_archive_improvements()
    test_binance_sync()
    test_governance_framework()
    test_connectors_instantiate()
    test_orchestrator()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
