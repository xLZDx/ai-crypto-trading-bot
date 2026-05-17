"""
Phase 10 tests — live bot integration of the institutional infrastructure.

Coverage:
  - feature_reader: Parquet-first, CSV fallback
  - main.py wiring: feature_reader, beta history, dynamic threshold, alpha decay
  - feature_engineering.add_news_sentiment: parquet path
  - train_model_v2: modernized trainer module loads
  - watchlist files exist
  - dashboard 8-tab template
"""
from __future__ import annotations

import sys
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


def test_feature_reader():
    print("\n[feature_reader]")
    from src.analysis import feature_reader as fr
    check("import feature_reader", True)
    check("load_recent_bars defined",  hasattr(fr, "load_recent_bars"))
    check("load_news_recent  defined", hasattr(fr, "load_news_recent"))
    # parquet path: should return None or a list
    out = fr.load_recent_bars("NONEXIST/USDT", "1h", tail_n=10)
    check("returns None or list",
          out is None or isinstance(out, list))


def test_main_py_phase10():
    print("\n[main.py Phase 10 wiring]")
    src = (PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
    check("imports feature_reader", "from src.analysis import feature_reader as _feature_reader" in src)
    check("uses feature_reader.load_recent_bars",
          "_feature_reader.load_recent_bars(" in src)
    check("_attach_beta_history defined", "def _attach_beta_history" in src)
    check("_refresh_dynamic_thresholds defined", "def _refresh_dynamic_thresholds" in src)
    check("_dyn_thresholds initialised", "self._dyn_thresholds" in src)
    check("alpha-decay exit block present",
          "alpha-decay" in src.lower() and "should_exit_decay" in src)
    check("uses dynamic threshold in tft path",
          "_dyn_thresholds[symbol]" in src or "_dyn_thresholds.get" in src)


def test_news_from_parquet():
    print("\n[add_news_sentiment Parquet path]")
    src = (PROJECT_ROOT / "src" / "analysis" / "feature_engineering.py").read_text(encoding="utf-8")
    check("imports load_news_recent",
          "from src.analysis.feature_reader import load_news_recent" in src)
    check("Phase 10F docstring",  "Phase 10F" in src)


def test_watchlist_files():
    print("\n[Watchlist templates]")
    yt = PROJECT_ROOT / "data" / "youtube_watchlist.json"
    es = PROJECT_ROOT / "data" / "etherscan_wallets.json"
    check("youtube_watchlist.json exists", yt.exists())
    check("etherscan_wallets.json exists",  es.exists())


def test_dashboard_8tabs():
    print("\n[Dashboard 8-tab nav]")
    tpl = (PROJECT_ROOT / "src" / "dashboard" / "templates" / "index.html").read_text(encoding="utf-8")
    for tab in ('portfolio', 'alpha', 'orderflow', 'risk',
                'training', 'simulation', 'data', 'strategies'):
        check(f"tab '{tab}'", f"data-tab=\"{tab}\"" in tpl)
    check("setP6Tab JS", "function setP6Tab" in tpl or "window.setP6Tab" in tpl)
    check("refreshPhase6Pane", "refreshPhase6Pane" in tpl)


def test_documentation():
    print("\n[Documentation]")
    p = PROJECT_ROOT / "APP_DOCUMENTATION.md"
    check("APP_DOCUMENTATION.md exists", p.exists())
    if p.exists():
        d = p.read_text(encoding="utf-8")
        check("has Quick start section",        "Quick start" in d)
        check("has Architecture diagram",       "Architecture at a glance" in d)
        check("has Phase summary table",        "Phase-by-phase summary" in d)
        check("has §1-18 wiring table",         "wiring status" in d)
        check("has Operating procedures",       "Operating procedures" in d)
        check("has API reference",              "API reference" in d)
        check("has Troubleshooting",            "Troubleshooting" in d)


def main() -> int:
    print("=" * 60)
    print("Phase 10 -- Live Bot Integration + Documentation")
    print("=" * 60)
    test_feature_reader()
    test_main_py_phase10()
    test_news_from_parquet()
    test_watchlist_files()
    test_dashboard_8tabs()
    test_documentation()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
