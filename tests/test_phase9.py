"""
Phase 9 tests — main.py integration + analytic layer + dual-balance + connectors + joint training.

Run:
    python tests/test_phase9.py
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


# ─── InstitutionalGate ─────────────────────────────────────────────────────

def test_institutional_gate():
    print("\n[InstitutionalGate]")
    from src.engine.institutional_gate import InstitutionalGate

    class FakeOM:
        def circuit_breaker_check(self, **kw):
            # Force-trigger drawdown for the test
            dd = (kw["peak_equity"] - kw["current_equity"]) / kw["peak_equity"]
            if dd > kw.get("max_daily_drawdown_pct", 0.05):
                return {"ok": False, "trigger": "max_daily_drawdown",
                        "reason": f"{dd*100:.1f}%"}
            return {"ok": True, "trigger": None, "reason": None}

    gate = InstitutionalGate(FakeOM(), peak_equity=1000.0)
    res = gate.pre_trade_check("BTC/USDT", "buy", 100.0,
                                current_equity=999.0, api_latency_ms=10)
    check("ok=True under normal conditions",
          res["ok"], f"reasons={res['reasons']}")

    # Trigger drawdown
    res2 = gate.pre_trade_check("BTC/USDT", "buy", 100.0,
                                 current_equity=900.0, api_latency_ms=10)
    check("circuit breaker triggers on drawdown",
          not res2["ok"] and any("circuit_breaker" in r for r in res2["reasons"]))

    # CVaR sizing fallback when scenarios are empty
    sizes = gate.cvar_size(["BTC/USDT", "ETH/USDT"],
                            scenario_returns=[[0.01, 0.02], [-0.005, 0.005]],
                            p_wins=[0.6, 0.55], capital_usd=1000.0)
    check("cvar_size returns one entry per symbol", set(sizes) == {"BTC/USDT", "ETH/USDT"})

    # executed_price slippage
    p_buy = gate.executed_price(100.0, "buy", 1.0, 100.0,
                                 fee_bps=10, lambda_impact=0.5)
    check("executed_price (buy) > mid",  p_buy > 100.0)
    p_sell = gate.executed_price(100.0, "sell", 1.0, 100.0,
                                  fee_bps=10, lambda_impact=0.5)
    check("executed_price (sell) < mid", p_sell < 100.0)

    # Decay exit
    check("should_exit triggers on long time", gate.should_exit_decay(1.0, 50.0))


# ─── DataLens / DecisionMetrics ────────────────────────────────────────────

def test_analytics():
    print("\n[Analytics]")
    from src.analytics import DataLens, DecisionMetrics, DecisionSummary
    check("DataLens importable", DataLens() is not None)
    dm = DecisionMetrics()
    s = dm.summarize(symbol="BTC/USDT", timeframe="1h")
    check("DecisionMetrics returns DecisionSummary",
          isinstance(s, DecisionSummary))
    js = s.to_dict()
    check("DecisionSummary serializable",
          all(k in js for k in ("symbol", "timeframe", "go", "blockers")))


# ─── Dual balance ──────────────────────────────────────────────────────────

def test_dual_balance():
    print("\n[Dual Balance]")
    from src.engine import dual_balance as db
    s_v = db.reset_virtual(initial_cash=12345.67)
    check("reset_virtual writes file", db.VIRTUAL_PATH.exists())
    check("virtual snapshot has cash 12345.67",
          abs(s_v["cash_usdt"] - 12345.67) < 1e-6)
    s_r = db.read_real()
    check("read_real returns dict with mode='real'",
          isinstance(s_r, dict) and s_r.get("mode") == "real")


# ─── Joint OFT+RL training script ─────────────────────────────────────────

def test_joint_training_module():
    print("\n[Joint OFT+RL training]")
    from src.training import joint_oft_rl as j
    check("train_oft defined", hasattr(j, "train_oft"))
    check("train_sac defined", hasattr(j, "train_sac"))
    check("main() defined",    hasattr(j, "main"))


# ─── New connectors registered ─────────────────────────────────────────────

def test_new_connectors():
    print("\n[New Connectors registered]")
    from src.data_governance import REGISTRY
    from src.data_governance import connectors  # ensure registration
    expected_new = {"glassnode", "santiment", "newsapi",
                    "youtube", "etherscan", "theblock_rss"}
    actual = set(REGISTRY.keys())
    missing = expected_new - actual
    check("all 6 new connectors registered", not missing,
          f"missing: {missing}")


# ─── main.py wiring ────────────────────────────────────────────────────────

def test_main_py_wiring():
    print("\n[main.py -- gate wiring]")
    src = (PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
    check("imports InstitutionalGate",
          "from src.engine.institutional_gate import InstitutionalGate" in src)
    check("imports dual_balance",
          "from src.engine import dual_balance" in src)
    check("instantiates self.gate",
          "self.gate = InstitutionalGate(" in src)
    check("calls gate.pre_trade_check",
          "gate.pre_trade_check(" in src or "_gate_ok(" in src)
    check("calls gate.executed_price",
          "gate.executed_price(" in src)
    check("calls gate.mark_data_tick",
          "gate.mark_data_tick(" in src)


# ─── Dashboard new routes ─────────────────────────────────────────────────

def test_dashboard_new_routes():
    print("\n[Dashboard -- new Phase 9 routes]")
    src = (PROJECT_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    for route in ['/api/balance/real', '/api/balance/virtual',
                  '/api/news', '/api/oft_signal/',
                  '/api/orchestrator/sources', '/api/retention/stats',
                  '/api/rate_limiter/stats', '/api/decision_summary/']:
        check(f"{route} route exists", route in src)


# ─── Mode switcher in template ────────────────────────────────────────────

def test_mode_switcher():
    print("\n[Dashboard -- mode switcher]")
    tpl = (PROJECT_ROOT / "src" / "dashboard" / "templates" / "index.html").read_text(encoding="utf-8")
    check("mode-switcher div present", 'id="mode-switcher"' in tpl)
    check("setMode() JS function defined", "function setMode(" in tpl or "window.setMode" in tpl)
    check("REAL button", "mode-real" in tpl)
    check("TEST/TRAIN button", "mode-test" in tpl)


# ─── News-in-DB migration script ──────────────────────────────────────────

def test_news_migration():
    print("\n[News CSV migration]")
    p = PROJECT_ROOT / "scripts" / "migrate_news_to_parquet.py"
    check("migrate_news_to_parquet.py exists", p.exists())
    src = p.read_text(encoding="utf-8")
    check("uses parquet_store with timeframe='news'",
          "timeframe=\"news\"" in src or "timeframe='news'" in src or 'NEWS_TF' in src)


# ─── Model cleanup script ─────────────────────────────────────────────────

def test_model_cleanup():
    print("\n[Model cleanup script]")
    p = PROJECT_ROOT / "scripts" / "cleanup_models.py"
    check("cleanup_models.py exists", p.exists())
    src = p.read_text(encoding="utf-8")
    check("--apply flag", "--apply" in src)
    check("archives instead of deleting", "_archived" in src)


# ─── Telegram persistor ───────────────────────────────────────────────────

def test_telegram_persistor():
    print("\n[Telegram persistor]")
    p = PROJECT_ROOT / "src" / "data_ingestion" / "telegram_persistor.py"
    check("telegram_persistor.py exists", p.exists())
    src = p.read_text(encoding="utf-8")
    check("writes to QuestDB news_sentiment", "write_news_sentiment(" in src)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 9 -- Integration + Analytics + Connectors + Joint Training")
    print("=" * 60)
    test_institutional_gate()
    test_analytics()
    test_dual_balance()
    test_joint_training_module()
    test_new_connectors()
    test_main_py_wiring()
    test_dashboard_new_routes()
    test_mode_switcher()
    test_news_migration()
    test_model_cleanup()
    test_telegram_persistor()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
