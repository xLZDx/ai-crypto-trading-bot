"""
Phase 1 tests — Level 1 Data Layer (microstructure features).

Coverage:
  - kalman_smoother.smooth_price() — produces same-shape output, NaN-safe
  - orderbook_features  — exact formulas from arch plan §2
  - orderbook_collector — module imports, parser correctness, URL builder
  - feature_engineering.add_kalman_close / add_l2_features / causal_audit
  - feature_store.FeatureStore — applies Kalman, surfaces L2 features
  - triple_barrier.causal_t1_audit / purge_overlapping_train

Run:
    python tests/test_phase1.py
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


# ─── Kalman smoother ─────────────────────────────────────────────────────────

def test_kalman():
    print("\n[Kalman Smoother]")
    try:
        from src.analysis.kalman_smoother import smooth_price, smooth_dataframe
    except Exception as exc:
        check("import kalman_smoother", False, str(exc))
        return
    check("import kalman_smoother", True)

    import numpy as np
    rng = np.random.default_rng(42)
    n = 500
    truth = np.linspace(30000, 31000, n)
    noisy = truth + rng.normal(0, 50, size=n)
    smoothed = smooth_price(noisy)
    check("output shape matches input", smoothed.shape == noisy.shape)
    check("smoothed std < raw std (noise reduced)",
          float(np.std(np.diff(smoothed))) < float(np.std(np.diff(noisy))))
    check("no NaN in output", not np.any(np.isnan(smoothed)))

    # Empty input handling
    check("smooth_price([]) returns empty", smooth_price([]).size == 0)

    # NaN-safe: NaNs in input
    arr = noisy.copy()
    arr[10:20] = np.nan
    out = smooth_price(arr)
    check("smooth_price tolerates NaN input", out.shape == arr.shape)

    # DataFrame helper
    import pandas as pd
    df = pd.DataFrame({"close": noisy})
    df = smooth_dataframe(df)
    check("smooth_dataframe adds price_kalman column", "price_kalman" in df.columns)


# ─── Order book features ────────────────────────────────────────────────────

def test_orderbook_features():
    print("\n[Order Book Features]")
    try:
        from src.analysis.orderbook_features import (
            imbalance, microprice, order_flow_imbalance,
            add_orderbook_features, aggregate_levels,
        )
    except Exception as exc:
        check("import orderbook_features", False, str(exc))
        return
    check("import orderbook_features", True)

    # Imbalance: balanced book → 0
    import numpy as np
    check("imbalance(10,10) == 0", abs(float(imbalance([10.0], [10.0])[0])) < 1e-9)
    check("imbalance(20,5) > 0",   float(imbalance([20.0], [5.0])[0]) > 0.5)
    check("imbalance(5,20) < 0",   float(imbalance([5.0], [20.0])[0]) < -0.5)
    check("imbalance(0,0) == 0 (no NaN)",
          abs(float(imbalance([0.0], [0.0])[0])) < 1e-9)

    # Microprice: when V_bid = V_ask, equals midpoint
    p = float(microprice([100.0], [101.0], [10.0], [10.0])[0])
    check("microprice equals midpoint when volumes equal",
          abs(p - 100.5) < 1e-9, f"got {p}")
    # When V_bid >> V_ask, microprice leans toward ASK (per the formula)
    p2 = float(microprice([100.0], [101.0], [100.0], [1.0])[0])
    check("microprice biases toward ask side when V_bid > V_ask",
          p2 > 100.5, f"got {p2}")

    # OFI: causal first row = 0
    ofi = order_flow_imbalance([10.0, 12.0, 11.0], [8.0, 8.0, 9.0])
    check("OFI shape matches input", ofi.shape == (3,))
    check("OFI[0] == 0 (causal: no prev tick)",
          abs(float(ofi[0])) < 1e-9, f"got {ofi[0]}")
    # tick 1: ΔV_bid=+2, ΔV_ask=0  → OFI = +2
    check("OFI[1] == +2 (ΔV_bid=+2, ΔV_ask=0)",
          abs(float(ofi[1]) - 2.0) < 1e-9, f"got {ofi[1]}")

    # add_orderbook_features
    import pandas as pd
    df = pd.DataFrame({
        "p_bid": [100, 100, 101], "p_ask": [101, 101, 102],
        "v_bid": [10, 12, 11],     "v_ask": [8, 8, 9],
    })
    out = add_orderbook_features(df.copy())
    for col in ["imbalance", "microprice", "ofi"]:
        check(f"add_orderbook_features adds '{col}'", col in out.columns)

    # No-op when bid/ask cols missing
    df_no_ob = pd.DataFrame({"close": [100, 101, 102]})
    out_no = add_orderbook_features(df_no_ob.copy())
    check("no-op when bid/ask absent",
          "imbalance" not in out_no.columns and "ofi" not in out_no.columns)

    # aggregate_levels reduction
    snap = aggregate_levels({
        "symbol": "BTC/USDT", "timestamp": 1234567890000,
        "bids": [[100, 1.0], [99, 2.0], [98, 3.0]],
        "asks": [[101, 0.5], [102, 1.5], [103, 2.5]],
    }, depth=3)
    check("aggregate_levels p_bid == top bid", snap.get("p_bid") == 100.0)
    check("aggregate_levels p_ask == top ask", snap.get("p_ask") == 101.0)
    check("aggregate_levels v_bid == sum top-3", snap.get("v_bid") == 6.0)
    check("aggregate_levels v_ask == sum top-3", snap.get("v_ask") == 4.5)


# ─── Order book collector module ────────────────────────────────────────────

def test_orderbook_collector():
    print("\n[Order Book Collector]")
    try:
        from src.data_ingestion import orderbook_collector as obc
    except Exception as exc:
        check("import orderbook_collector", False, str(exc))
        return
    check("import orderbook_collector", True)
    check("_stream_name() canonical form",
          obc._stream_name("BTC/USDT", 20, "100ms") == "btcusdt@depth20@100ms")
    url = obc._binance_url(["BTC/USDT", "ETH/USDT"], 20, "100ms")
    check("_binance_url builds combined-stream URL",
          "btcusdt@depth20@100ms" in url and "ethusdt@depth20@100ms" in url)
    check("URL uses /stream endpoint", "/stream?streams=" in url)

    # Parser correctness
    snap = obc._parse_depth_event({
        "stream": "btcusdt@depth20@100ms",
        "data": {"E": 1700000000000, "bids": [["50000", "1.0"]], "asks": [["50001", "2.0"]]},
    }, depth=5)
    check("_parse_depth_event yields aggregated snapshot",
          snap.get("p_bid") == 50000.0 and snap.get("p_ask") == 50001.0)


# ─── feature_engineering Phase 1 helpers ─────────────────────────────────────

def test_feature_engineering_phase1():
    print("\n[feature_engineering — Phase 1 helpers]")
    try:
        from src.analysis.feature_engineering import (
            add_kalman_close, add_l2_features, causal_audit,
        )
    except Exception as exc:
        check("import phase1 helpers", False, str(exc))
        return
    check("import phase1 helpers", True)

    import pandas as pd, numpy as np
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=300, freq="1min"),
        "close": 30000 + rng.normal(0, 25, size=300).cumsum(),
    })
    df = add_kalman_close(df)
    check("add_kalman_close adds price_kalman", "price_kalman" in df.columns)
    check("price_kalman not NaN-only",
          df["price_kalman"].notna().all())

    # add_l2_features no-op without bid/ask
    df2 = add_l2_features(df.copy())
    check("add_l2_features no-op without bid/ask",
          "ob_imbalance" not in df2.columns)

    # add_l2_features works when columns present
    df3 = pd.DataFrame({
        "p_bid": [100, 101], "p_ask": [101, 102],
        "v_bid": [10, 11], "v_ask": [8, 9],
    })
    df3 = add_l2_features(df3)
    for c in ("ob_imbalance", "ob_microprice", "ob_ofi"):
        check(f"add_l2_features adds '{c}'", c in df3.columns)

    # causal_audit on monotone, valid frame
    audit = causal_audit(df)
    check("causal_audit returns ok=True on valid frame",
          audit["ok"] is True, f"warnings={audit['warnings']}")

    # causal_audit detects non-monotone timestamps
    df_bad = df.iloc[::-1].reset_index(drop=True)
    audit_bad = causal_audit(df_bad)
    check("causal_audit catches non-monotone timestamps",
          audit_bad["ok"] is False)


# ─── FeatureStore integration ────────────────────────────────────────────────

def test_feature_store_phase1():
    print("\n[FeatureStore Phase 1 integration]")
    try:
        from src.analysis.feature_store import FeatureStore
    except Exception as exc:
        check("import FeatureStore", False, str(exc))
        return
    check("import FeatureStore", True)

    import pandas as pd, numpy as np
    rng = np.random.default_rng(3)
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min"),
        "open":  30000 + rng.normal(0, 10, size=n),
        "high":  30050 + rng.normal(0, 10, size=n),
        "low":   29950 + rng.normal(0, 10, size=n),
        "close": 30000 + rng.normal(0, 10, size=n).cumsum(),
        "volume": rng.uniform(1, 10, size=n),
    })
    fs = FeatureStore()
    fs.update_features("BTC/USDT", df, sentiment=0.1, volatility=0.5, ou_signal=0.2)
    out = fs.get_latest_data("BTC/USDT")
    check("FeatureStore stored data", len(out) > 0)
    check("FeatureStore added price_kalman", "price_kalman" in out.columns)
    check("FeatureStore added sentiment_score", "sentiment_score" in out.columns)
    check("FeatureStore retains close (for execution)", "close" in out.columns)

    # When bid/ask are present, L2 features should appear
    df["p_bid"] = df["close"] - 0.5
    df["p_ask"] = df["close"] + 0.5
    df["v_bid"] = rng.uniform(1, 10, size=n)
    df["v_ask"] = rng.uniform(1, 10, size=n)
    fs2 = FeatureStore()
    fs2.update_features("BTC/USDT", df, sentiment=0, volatility=0, ou_signal=0)
    out2 = fs2.get_latest_data("BTC/USDT")
    for c in ("imbalance", "microprice", "ofi"):
        check(f"FeatureStore added L2 '{c}' when bid/ask present",
              c in out2.columns)


# ─── Triple Barrier causal t1 audit ──────────────────────────────────────────

def test_t1_audit():
    print("\n[Triple Barrier causal t1 audit]")
    try:
        from src.analysis.triple_barrier import causal_t1_audit, purge_overlapping_train
    except Exception as exc:
        check("import causal_t1_audit", False, str(exc))
        return
    check("import causal_t1_audit", True)

    import pandas as pd
    # Train ends 2025-01-15. Some labels resolve into the test set.
    n = 30
    base = pd.date_range("2025-01-01", periods=n, freq="1D")
    # Most labels resolve same-day; last 3 resolve AFTER train_end (leak).
    t1_times = pd.Series(base + pd.Timedelta(hours=12))
    t1_times.index = base
    train_end = pd.Timestamp("2025-01-15")
    audit = causal_t1_audit(t1_times, train_end=train_end, test_start=pd.Timestamp("2025-01-16"))
    check("audit on clean split → ok", audit["ok"] is True,
          f"violations={audit['n_violations']}")

    # Force a violation
    bad_t1 = t1_times.copy()
    bad_t1.iloc[10] = pd.Timestamp("2025-02-01")  # resolves deep in test
    bad_audit = causal_t1_audit(bad_t1, train_end=train_end,
                                 test_start=pd.Timestamp("2025-01-16"))
    check("audit catches violation", bad_audit["ok"] is False)
    check("audit reports first_violation",
          bad_audit["first_violation"] is not None)

    # purge_overlapping_train drops bad rows
    df = pd.DataFrame({"x": range(n)}, index=base)
    purged = purge_overlapping_train(df, bad_t1, train_end=train_end,
                                     test_start=pd.Timestamp("2025-01-16"))
    check("purge_overlapping_train drops leaking rows",
          len(purged) < len(df))


# ─── requirements.txt deps ───────────────────────────────────────────────────

def test_requirements():
    print("\n[requirements.txt — Phase 1]")
    req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    check("pykalman in requirements.txt", "pykalman" in req)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 1 — Level 1 Data Layer Tests")
    print("=" * 60)
    test_requirements()
    test_kalman()
    test_orderbook_features()
    test_orderbook_collector()
    test_feature_engineering_phase1()
    test_feature_store_phase1()
    test_t1_audit()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
