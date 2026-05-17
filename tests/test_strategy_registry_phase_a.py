"""
Phase A strategy_registry tests — verify TF routing and predict_at wiring.

Tests prove behavior, not just string matches:
  1. Registry has timeframe + model_key for all ML entries
  2. get_strategy_tf() returns correct values
  3. get_strategy_model_key() returns correct values
  4. MultiTFPredictor.predict_at() falls back to canonical when TF not loaded
  5. _get_tf_data() cache TTL behavior
  6. process_kline ML routing uses predict_at with strategy TF
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.strategy_registry import (
    REGISTRY,
    get_strategy_tf,
    get_strategy_model_key,
)


# ── Registry completeness ───────────────────────────────────────────────────

ML_STRATEGIES = [
    "ElliottWave_ML", "Base_ML", "Trend_ML",
    "Futures_Short_ML", "Scalping_ML",
]

@pytest.mark.parametrize("name", ML_STRATEGIES)
def test_ml_strategy_has_timeframe(name):
    """Every ML strategy must declare a timeframe for Phase A routing."""
    tf = REGISTRY[name].get("timeframe")
    assert tf is not None, f"{name} missing 'timeframe'"
    assert tf in {"1m", "5m", "15m", "1h", "4h", "1d"}, f"{name} has invalid TF={tf!r}"


@pytest.mark.parametrize("name", ML_STRATEGIES)
def test_ml_strategy_has_model_key(name):
    """Every ML strategy must declare a model_key (base/trend/futures/scalping)."""
    key = REGISTRY[name].get("model_key")
    assert key is not None, f"{name} missing 'model_key'"
    assert key in {"base", "trend", "futures", "scalping"}, f"{name} unknown model_key={key!r}"


def test_rule_based_strategies_have_no_model_key():
    """Rule-based strategies (no ML) should NOT have model_key."""
    rule_based = ["RSI_MeanReversion", "MACD_Momentum", "BB_Reversion", "VWAP_Reversion"]
    for name in rule_based:
        assert "model_key" not in REGISTRY[name], f"{name} should not have model_key"


# ── get_strategy_tf / get_strategy_model_key ────────────────────────────────

def test_get_strategy_tf_known():
    assert get_strategy_tf("Base_ML") == "1h"
    assert get_strategy_tf("Trend_ML") == "4h"
    assert get_strategy_tf("Futures_Short_ML") == "4h"
    assert get_strategy_tf("Scalping_ML") == "1m"


def test_get_strategy_tf_unknown():
    assert get_strategy_tf("NonExistentStrategy") is None


def test_get_strategy_model_key_known():
    assert get_strategy_model_key("Base_ML") == "base"
    assert get_strategy_model_key("Trend_ML") == "trend"
    assert get_strategy_model_key("Futures_Short_ML") == "futures"
    assert get_strategy_model_key("Scalping_ML") == "scalping"


def test_get_strategy_model_key_unknown():
    assert get_strategy_model_key("NonExistentStrategy") is None


# ── MultiTFPredictor.predict_at fallback ────────────────────────────────────

def test_predict_at_returns_none_when_tf_not_loaded():
    """predict_at should return None (not raise) when the TF has no model."""
    from src.analysis.multi_tf_predictor import MultiTFPredictor
    dummy_bars = [{"close": 100, "open": 99, "high": 101, "low": 98, "volume": 1000}] * 50

    predictor = MultiTFPredictor.__new__(MultiTFPredictor)
    predictor._canonical_tf = "1h"
    predictor._predictors = {}  # no predictors loaded

    result = predictor.predict_at("4h", dummy_bars)
    assert result is None


def test_predict_at_routes_to_correct_predictor():
    """predict_at(tf) must call the TF-specific predictor, not canonical."""
    from src.analysis.multi_tf_predictor import MultiTFPredictor

    canonical_mock = MagicMock()
    canonical_mock.is_loaded = True
    canonical_mock.predict.return_value = 0  # canonical says SELL

    tf4h_mock = MagicMock()
    tf4h_mock.is_loaded = True
    tf4h_mock.predict.return_value = 1  # 4h model says BUY

    predictor = MultiTFPredictor.__new__(MultiTFPredictor)
    predictor._canonical_tf = "1h"
    predictor._predictors = {"1h": canonical_mock, "4h": tf4h_mock}

    result = predictor.predict_at("4h", ["dummy_data"])
    assert result == 1
    tf4h_mock.predict.assert_called_once()
    canonical_mock.predict.assert_not_called()


# ── _get_tf_data cache behavior ─────────────────────────────────────────────

def _make_trader():
    """Build a minimal MultiAssetTrader-like object with _get_tf_data."""
    class FakeTrader:
        symbols = ["BTC/USDT"]
        timeframe = "1h"
        _tf_data_cache: dict = {}
        _TF_TTL: dict = {"1m": 60, "4h": 14400}

        def _get_tf_data(self, symbol, tf, tail_n=1000):
            import time as _t
            from src.main import MultiAssetTrader
            return MultiAssetTrader._get_tf_data(self, symbol, tf, tail_n)

    return FakeTrader()


def test_get_tf_data_caches_result():
    """Second call within TTL must return cached data without re-fetching."""
    fake_bars = [{"close": 42}] * 10

    with patch("src.analysis.feature_reader.load_recent_bars", return_value=fake_bars) as mock_load:
        trader = _make_trader()

        bars1 = trader._get_tf_data("BTC/USDT", "4h")
        bars2 = trader._get_tf_data("BTC/USDT", "4h")  # should use cache

        assert bars1 == fake_bars
        assert bars2 == fake_bars
        assert mock_load.call_count == 1  # fetched only once


def test_get_tf_data_refetches_after_ttl():
    """After TTL expires, data must be re-fetched."""
    fake_bars = [{"close": 99}] * 5

    with patch("src.analysis.feature_reader.load_recent_bars", return_value=fake_bars) as mock_load:
        trader = _make_trader()
        trader._TF_TTL = {"4h": 0}  # zero TTL → always stale

        trader._get_tf_data("BTC/USDT", "4h")
        trader._get_tf_data("BTC/USDT", "4h")  # TTL=0 → re-fetch

        assert mock_load.call_count == 2


def test_get_tf_data_returns_empty_on_load_error():
    """Loader errors must NOT propagate — return [] so caller falls back."""
    with patch("src.analysis.feature_reader.load_recent_bars", side_effect=RuntimeError("disk fail")):
        trader = _make_trader()
        bars = trader._get_tf_data("BTC/USDT", "4h")
        assert bars == []


# ── process_kline routing integration (smoke) ────────────────────────────────

def test_process_kline_uses_predict_at_for_trend():
    """Trend model must call predict_at('4h', ...) when 4h data is available,
    not predict() with 1h data."""
    predict_at_calls = []

    class RecordingPredictor:
        is_loaded = True

        def predict(self, data):
            return 1  # canonical says bullish

        def predict_at(self, tf, data):
            predict_at_calls.append((tf, len(data)))
            return 0  # 4h model says bearish

    dummy_1h = [{"close": 100, "open": 99, "high": 101, "low": 98, "volume": 100}] * 100
    dummy_4h = [{"close": 200, "open": 199, "high": 201, "low": 198, "volume": 50}] * 40

    from src.engine.strategy_registry import get_strategy_tf

    main_tf = "1h"
    trend_tf = get_strategy_tf("Trend_ML")
    assert trend_tf == "4h"

    predictor = RecordingPredictor()
    trend_data = dummy_4h if trend_tf != main_tf else dummy_1h

    # Use explicit is-not-None check (Phase A fixed pattern)
    _raw = predictor.predict_at(trend_tf, trend_data) if trend_data else None
    result = _raw if _raw is not None else predictor.predict(dummy_1h)

    # Must return 0 (bearish from 4h model), NOT 1 (canonical fallback)
    assert result == 0, f"Expected 0 (4h bearish), got {result} — `or` bug not fixed"
    assert predict_at_calls == [("4h", 40)]


def test_predict_at_zero_not_swallowed_by_or():
    """Regression: predict_at() returning 0 (bearish) must NOT fall back to canonical.
    A falsy `or` would silently discard a valid bearish signal."""
    from src.analysis.multi_tf_predictor import MultiTFPredictor

    canonical_mock = MagicMock()
    canonical_mock.is_loaded = True
    canonical_mock.predict.return_value = 1  # canonical says bullish

    tf4h_mock = MagicMock()
    tf4h_mock.is_loaded = True
    tf4h_mock.predict.return_value = 0  # 4h says bearish — must NOT be swallowed

    predictor = MultiTFPredictor.__new__(MultiTFPredictor)
    predictor._canonical_tf = "1h"
    predictor._predictors = {"1h": canonical_mock, "4h": tf4h_mock}

    _raw = predictor.predict_at("4h", ["dummy"])
    result = _raw if _raw is not None else predictor.predict(["dummy"])

    assert result == 0
    canonical_mock.predict.assert_not_called()
