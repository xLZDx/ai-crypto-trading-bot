"""
Health check for AI Trading Bot.
Run with: python health_check.py
"""
import sys
import os
import time
import json

# Force UTF-8 output on Windows so emoji/arrows in log messages don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []

def check(name, fn):
    try:
        msg = fn()
        if not isinstance(msg, str):
            msg = ""
        results.append((True, name, msg))
        line = f"[  OK  ] {name}"
        if msg:
            line += f" -- {msg}"
        print(line)
    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode()
        results.append((False, name, err))
        print(f"[ FAIL ] {name} -- {err}")

print("=" * 60)
print("  AI Trading Bot -- Health Check")
print("=" * 60)
print()

# ── 1. Core dependencies ──────────────────────────────────────────────────────
print("[Imports]")

def import_check(mod):
    def fn():
        m = __import__(mod)
        ver = getattr(m, '__version__', 'ok')
        return str(ver)
    return fn

check("filelock",   import_check("filelock"))
check("defusedxml", import_check("defusedxml"))
check("pandas",     import_check("pandas"))
check("sklearn",    import_check("sklearn"))
check("joblib",     import_check("joblib"))
check("ccxt",       import_check("ccxt"))
check("websockets", import_check("websockets"))
check("flask",      import_check("flask"))

print()
print("[Bot modules]")

def module_check(mod):
    def fn():
        __import__(mod)
        return "imported"
    return fn

check("src.utils.safe_json",              module_check("src.utils.safe_json"))
check("src.utils.config",                 module_check("src.utils.config"))
check("src.analysis.feature_engineering", module_check("src.analysis.feature_engineering"))
check("src.analysis.elliott_waves",       module_check("src.analysis.elliott_waves"))
check("src.analysis.risk_manager",        module_check("src.analysis.risk_manager"))
check("src.analysis.ml_predictor",        module_check("src.analysis.ml_predictor"))
check("src.engine.trade_tracker",         module_check("src.engine.trade_tracker"))
check("src.engine.order_manager",         module_check("src.engine.order_manager"))
check("src.engine.agentic_llm",           module_check("src.engine.agentic_llm"))
check("src.data_ingestion.binance_downloader", module_check("src.data_ingestion.binance_downloader"))

# ── 2. safe_json atomic read/write ────────────────────────────────────────────
print()
print("[safe_json]")

def test_safe_json():
    from src.utils.safe_json import read_json, write_json
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        write_json(path, {"hello": "world", "num": 42})
        data = read_json(path)
        assert data == {"hello": "world", "num": 42}, f"Got: {data}"
        return "atomic write+read OK"
    finally:
        for p in [path, path + ".lock"]:
            try: os.unlink(p)
            except: pass

check("Atomic write and read", test_safe_json)

# ── 3. Feature engineering ────────────────────────────────────────────────────
print()
print("[Feature Engineering]")

def test_feature_engineering():
    import pandas as pd
    import numpy as np
    from src.analysis.feature_engineering import (
        add_rsi, add_macd, add_bollinger_bands, add_roc,
        add_time_features, add_taker_and_trade_features, add_adx
    )
    n = 250
    np.random.seed(42)
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates, "open": price * 0.999, "high": price * 1.001,
        "low": price * 0.998, "close": price,
        "volume": np.random.rand(n) * 1000 + 100,
        "taker_buy_base": np.random.rand(n) * 500,
        "trades_count": np.random.randint(100, 500, n),
    })
    df = add_rsi(df, 14)
    assert "rsi_14" in df.columns
    df = add_macd(df)
    assert all(c in df.columns for c in ["macd", "macd_signal", "macd_hist"])
    df = add_bollinger_bands(df, 20)
    assert "bb_pb" in df.columns
    df = add_roc(df, [3, 7])
    assert "roc_3" in df.columns and "roc_7" in df.columns
    df = add_time_features(df)
    assert "hour" in df.columns
    df = add_taker_and_trade_features(df)
    assert "taker_buy_ratio" in df.columns
    df = add_adx(df, 14)
    assert "adx_14" in df.columns
    return "all 7 indicator functions produce correct columns"

check("All feature functions produce correct columns", test_feature_engineering)

# ── 4. ML Predictor ───────────────────────────────────────────────────────────
print()
print("[ML Predictor]")

def test_ml_loads():
    from src.analysis.ml_predictor import MLPredictor
    p = MLPredictor()
    if p.is_loaded:
        return f"loaded, accuracy={p.accuracy:.1f}%"
    raise RuntimeError(f"Model not found: {p.last_error} -- run train_all_models.py first")

def test_ml_predict():
    import pandas as pd
    import numpy as np
    from src.analysis.ml_predictor import MLPredictor
    p = MLPredictor()
    if not p.is_loaded:
        raise RuntimeError(f"Model not loaded: {p.last_error}")
    n = 100
    np.random.seed(1)
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    data = [{"timestamp": str(d), "open": float(pr*0.999), "high": float(pr*1.001),
              "low": float(pr*0.998), "close": float(pr), "volume": float(v),
              "taker_buy_base": float(v*0.5), "trades_count": 200}
            for d, pr, v in zip(dates, price, np.random.rand(n)*1000+100)]
    result = p.predict(data)
    assert result in (0, 1, None), f"Unexpected result: {result}"
    label = {0: "DOWN", 1: "UP", None: "insufficient data"}[result]
    return f"predict() -> {result} ({label})"

check("MLPredictor: model file loads", test_ml_loads)
check("MLPredictor: predict() on synthetic 100-candle data", test_ml_predict)

# ── 5. Scalping + Futures + Trend models ──────────────────────────────────────
print()
print("[All 4 ML Models]")

model_files = [
    ("Base",     "btc_rf_model.joblib",         "base"),
    ("Scalping", "scalping_model.joblib",        "scalping"),
    ("Futures",  "futures_short_model.joblib",   "futures"),
    ("Trend",    "trend_model.joblib",           "trend"),
]

for label, fname, mtype in model_files:
    def _check(f=fname, t=mtype, l=label):
        from src.analysis.ml_predictor import MLPredictor
        p = MLPredictor(model_filename=f, model_type=t)
        if not p.is_loaded:
            raise RuntimeError(f"Not found -- run train_all_models.py")
        return f"accuracy={p.accuracy:.1f}%"
    check(f"{label} model ({fname})", _check)

# ── 6. TradeTracker ───────────────────────────────────────────────────────────
print()
print("[TradeTracker]")

def test_trade_tracker():
    import tempfile
    from src.engine.trade_tracker import TradeTracker
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    os.unlink(path)
    try:
        tt = TradeTracker(filepath=path)
        trade = tt.open_trade("BTC/USDT", 100.0, 50000.0, strategy="Test")
        assert isinstance(trade["id"], str) and len(trade["id"]) == 36, "ID not UUID"
        tt2 = TradeTracker(filepath=path)
        assert len(tt2.trades) == 1, "Trade not persisted after reload"
        closed = tt2.close_trade_by_id(trade["id"], 51000.0)
        assert closed is not None, "close_trade_by_id returned None"
        assert closed["status"] == "CLOSED"
        pnl = closed["pnl_usdt"]
        # bought 0.002 BTC at 50000, sold at 51000 -> profit = 0.002 * 1000 = 2.0
        assert abs(pnl - 2.0) < 0.01, f"PnL expected ~2.0, got {pnl}"
        return f"open->persist->close, PnL={pnl:.2f} USDT"
    finally:
        for p in [path, path + ".lock"]:
            try: os.unlink(p)
            except: pass

check("open->persist->reload->close, PnL correct", test_trade_tracker)

# ── 7. OrderManager ───────────────────────────────────────────────────────────
print()
print("[OrderManager]")

def test_futures_symbol():
    from src.engine.order_manager import OrderManager
    assert OrderManager.to_futures_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert OrderManager.to_futures_symbol("ETH/BTC") == "ETH/BTC:BTC"
    return "BTC/USDT -> BTC/USDT:USDT, ETH/BTC -> ETH/BTC:BTC"

check("Futures symbol conversion", test_futures_symbol)

# ── 8. Dashboard API ──────────────────────────────────────────────────────────
print()
print("[Dashboard (must be running on :5000)]")

def _api_get(path):
    import urllib.request, urllib.error
    api_key = os.getenv("DASHBOARD_API_KEY", "")
    req = urllib.request.Request(
        f"http://127.0.0.1:5000{path}",
        headers={"X-API-Key": api_key}
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}

def test_dashboard_state():
    code, data = _api_get("/api/state")
    if code == 401:
        return "UP (401 -- set DASHBOARD_API_KEY env var)"
    if code != 200:
        raise RuntimeError(f"HTTP {code}")
    status = str(data.get("status", "?")).encode('ascii', 'replace').decode()
    return f"HTTP 200, status={status!r}"

def test_dashboard_trades():
    code, data = _api_get("/api/trades")
    if code == 401:
        return "UP (401 -- set DASHBOARD_API_KEY env var)"
    if code != 200:
        raise RuntimeError(f"HTTP {code}")
    return f"HTTP 200, {len(data.get('trades', []))} trade(s)"

def test_dashboard_logs():
    code, data = _api_get("/api/logs")
    if code == 401:
        return "UP (401 -- set DASHBOARD_API_KEY env var)"
    if code != 200:
        raise RuntimeError(f"HTTP {code}")
    return f"HTTP 200, {len(data.get('logs', []))} log line(s)"

def _wrap_connection(fn):
    def inner():
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "10061" in msg or "Connection refused" in msg:
                raise RuntimeError("Dashboard not running on :5000 -- start it first")
            raise
    return inner

check("GET /api/state", _wrap_connection(test_dashboard_state))
check("GET /api/trades", _wrap_connection(test_dashboard_trades))
check("GET /api/logs", _wrap_connection(test_dashboard_logs))

# ── 9. Bot state file ─────────────────────────────────────────────────────────
print()
print("[Bot State (must be running)]")

def test_bot_state_file():
    from src.utils.safe_json import read_json
    state = read_json("data/state.json", default=None)
    if state is None:
        raise RuntimeError("data/state.json missing -- bot has not started")
    raw_status = state.get("status", "")
    status = raw_status.encode('ascii', 'replace').decode()
    if "STARTUP ERROR" in raw_status.upper() or "CRASH" in raw_status.upper():
        raise RuntimeError(f"Bot is in error state: {status}")
    return f"status={status!r}"

def test_bot_recently_active():
    from src.utils.safe_json import read_json
    state = read_json("data/state.json", default=None)
    if state is None:
        raise RuntimeError("data/state.json missing")
    ts = state.get("last_update", "")
    if not ts:
        raise RuntimeError("last_update field missing")
    try:
        age = time.time() - time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        raise RuntimeError(f"Cannot parse last_update: {ts!r}")
    if age > 300:
        raise RuntimeError(f"State is {age:.0f}s old -- bot may not be running")
    return f"state updated {age:.0f}s ago"

check("data/state.json readable and not in error", test_bot_state_file)
check("Bot state updated within last 5 minutes", test_bot_recently_active)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for ok, _, _ in results if ok)
failed = sum(1 for ok, _, _ in results if not ok)
print(f"  Result: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print()
    print("  Failed checks:")
    for ok, name, msg in results:
        if not ok:
            print(f"    - {name}: {msg}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
