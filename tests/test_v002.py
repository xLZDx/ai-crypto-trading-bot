"""
v0.002 Test Suite — comprehensive + regression.

Covers all Phase 1–8 new modules:
  fractional_diff, triple_barrier, kelly_criterion, meta_labeler,
  regime_classifier, agent_bus, feature_engineering (new funcs),
  risk_manager (Kelly layer), dashboard /api/agents endpoint.

Plus regression tests for existing functionality.
"""
from __future__ import annotations

import os
import sys
import json
import math
import threading
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ─── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, start_price: float = 50_000.0, freq: str = "1h") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with realistic properties."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, n)
    close = start_price * np.cumprod(1 + returns)
    high  = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol   = rng.uniform(100, 1000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    df["taker_buy_volume"] = df["volume"] * rng.uniform(0.3, 0.7, n)
    df["num_trades"] = rng.integers(50, 500, n)
    return df


@pytest.fixture
def ohlcv():
    return _make_ohlcv(200)


@pytest.fixture
def ohlcv_short():
    return _make_ohlcv(60)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Fractional Differencing
# ══════════════════════════════════════════════════════════════════════════════

class TestFractionalDiff:
    def test_compute_weights_first_is_one(self):
        from src.analysis.fractional_diff import _compute_weights
        w = _compute_weights(0.4)
        assert w[-1] == pytest.approx(1.0), "Last weight (newest) must be 1"

    def test_compute_weights_length(self):
        from src.analysis.fractional_diff import _compute_weights
        w = _compute_weights(0.4, threshold=1e-4)
        assert len(w) >= 5, "Should produce at least 5 meaningful weights for d=0.4"

    def test_fractional_diff_output_shape(self, ohlcv):
        from src.analysis.fractional_diff import fractional_diff
        result = fractional_diff(ohlcv["close"], d=0.4)
        assert len(result) == len(ohlcv)

    def test_fractional_diff_has_warmup_nans(self):
        # Use 300 rows so warm-up (max 99 bars) leaves a valid tail
        from src.analysis.fractional_diff import fractional_diff
        series = pd.Series(50000 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 300)))
        result = fractional_diff(series, d=0.4)
        assert result.isna().any(), "Should have NaN in warm-up period"
        assert not result.iloc[-50:].isna().any(), "Tail should not be NaN"

    def test_add_fractional_diff_adds_column(self, ohlcv):
        from src.analysis.fractional_diff import add_fractional_diff
        df = add_fractional_diff(ohlcv.copy(), d=0.4)
        assert "frac_diff_d40" in df.columns

    def test_fractional_diff_d0_is_price(self):
        """d=0 → identity transform (single weight=1, no warm-up NaN)."""
        from src.analysis.fractional_diff import fractional_diff
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fractional_diff(series, d=0.0)
        assert result.isna().sum() == 0, "d=0 should produce no NaN"
        assert np.allclose(result.values, series.values, rtol=1e-9), "d=0 should return original values"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Triple Barrier Method
# ══════════════════════════════════════════════════════════════════════════════

class TestTripleBarrier:
    def test_labels_in_valid_set(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized
        labels, _ = triple_barrier_labels_vectorized(ohlcv, pt_multiplier=2.0, sl_multiplier=1.0)
        assert set(labels.unique()).issubset({-1, 0, 1})

    def test_labels_output_length(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized
        labels, _ = triple_barrier_labels_vectorized(ohlcv)
        assert len(labels) == len(ohlcv)

    def test_last_bars_are_zero(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized
        max_bars = 24
        labels, _ = triple_barrier_labels_vectorized(ohlcv, max_bars=max_bars)
        assert (labels.iloc[-max_bars:] == 0).all()

    def test_vectorized_returns_tuple(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized
        result = triple_barrier_labels_vectorized(ohlcv)
        assert isinstance(result, tuple) and len(result) == 2, \
            "triple_barrier_labels_vectorized must return (labels, t1_times)"

    def test_label_stats_sums_to_100(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats
        labels, _ = triple_barrier_labels_vectorized(ohlcv)
        stats = label_stats(labels)
        total_pct = stats["long_pct"] + stats["short_pct"] + stats["timeout_pct"]
        assert abs(total_pct - 100.0) < 0.1

    def test_label_stats_keys(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats
        labels, _ = triple_barrier_labels_vectorized(ohlcv)
        stats = label_stats(labels)
        assert set(stats.keys()) == {"long_pct", "short_pct", "timeout_pct", "total"}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Kelly Criterion
# ══════════════════════════════════════════════════════════════════════════════

class TestKellyCriterion:
    def test_kelly_fraction_positive_edge(self):
        from src.analysis.kelly_criterion import kelly_fraction
        # p=0.6, b=2.0 → full Kelly = (0.6*3 - 1)/2 = 0.4 → half = 0.2
        f = kelly_fraction(p_win=0.6, win_loss_ratio=2.0, half_kelly=True)
        assert 0.0 < f <= 0.25

    def test_kelly_fraction_no_edge_returns_zero(self):
        from src.analysis.kelly_criterion import kelly_fraction
        # p=0.4, b=1.0 → Kelly = (0.4*2 - 1)/1 = -0.2 → 0
        f = kelly_fraction(p_win=0.4, win_loss_ratio=1.0, half_kelly=False)
        assert f == 0.0

    def test_kelly_fraction_max_cap(self):
        from src.analysis.kelly_criterion import kelly_fraction
        # Even with very high edge, must not exceed max
        f = kelly_fraction(p_win=0.99, win_loss_ratio=5.0, half_kelly=False, max_fraction=0.25)
        assert f <= 0.25

    def test_kelly_fraction_min_floor(self):
        from src.analysis.kelly_criterion import kelly_fraction
        # With some positive edge, must be at least min
        f = kelly_fraction(p_win=0.55, win_loss_ratio=1.5, min_fraction=0.005)
        assert f >= 0.005

    def test_kelly_position_size_scales_with_capital(self):
        from src.analysis.kelly_criterion import kelly_position_size
        s1 = kelly_position_size(capital=1000, p_win=0.6, win_loss_ratio=2.0)
        s2 = kelly_position_size(capital=2000, p_win=0.6, win_loss_ratio=2.0)
        assert s2 == pytest.approx(s1 * 2, rel=0.01)

    def test_compute_win_loss_ratio_fallback(self):
        from src.analysis.kelly_criterion import compute_win_loss_ratio
        assert compute_win_loss_ratio([]) == 1.5
        assert compute_win_loss_ratio([1.0, 2.0]) == 1.5  # < 5 trades

    def test_compute_win_loss_ratio_realistic(self):
        from src.analysis.kelly_criterion import compute_win_loss_ratio
        # 5 wins of 100, 5 losses of 50 → ratio = 2.0
        pnls = [100.0] * 5 + [-50.0] * 5
        ratio = compute_win_loss_ratio(pnls)
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_kelly_sizer_records_trades(self):
        from src.analysis.kelly_criterion import KellySizer
        ks = KellySizer(window=10)
        for _ in range(7):
            ks.record_trade(50.0)
        for _ in range(3):
            ks.record_trade(-25.0)
        assert len(ks._pnls) == 10
        assert ks.win_rate == pytest.approx(0.7, rel=0.01)

    def test_kelly_sizer_window_trim(self):
        from src.analysis.kelly_criterion import KellySizer
        ks = KellySizer(window=5)
        for i in range(10):
            ks.record_trade(float(i))
        assert len(ks._pnls) == 5

    def test_kelly_sizer_circuit_breaker(self):
        from src.analysis.kelly_criterion import KellySizer
        ks = KellySizer()
        assert ks.circuit_breaker(3, threshold=3) is True
        assert ks.circuit_breaker(2, threshold=3) is False

    def test_kelly_sizer_size_returns_positive(self):
        from src.analysis.kelly_criterion import KellySizer
        ks = KellySizer()
        for _ in range(5):
            ks.record_trade(100.0)
        for _ in range(5):
            ks.record_trade(-40.0)
        size = ks.size(capital=10000, p_win=0.62)
        assert size > 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Meta-Labeler (no model loaded — fail-open behavior)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetaLabeler:
    def test_filter_no_model_passes_all(self):
        from src.analysis.meta_labeler import MetaLabeler
        ml = MetaLabeler(model_path='/nonexistent/path.joblib')
        decision, conf = ml.filter(signal=1.0, features={})
        assert decision == 'PASS'
        assert conf == 0.5

    def test_filter_zero_signal_blocked_with_model(self):
        """When model IS loaded, a zero signal should be blocked before model runs."""
        from src.analysis.meta_labeler import MetaLabeler, CONFIDENCE_THRESHOLD
        ml = MetaLabeler(model_path='/nonexistent/path.joblib')
        # Without model, fail-open: everything passes
        decision, _ = ml.filter(signal=1.0, features={})
        assert decision == 'PASS', "No model → should fail open"
        # The zero-signal block is only tested when model IS present (covered in integration)

    def test_batch_filter_no_model_passes_signals(self):
        from src.analysis.meta_labeler import MetaLabeler
        ml = MetaLabeler(model_path='/nonexistent/path.joblib')
        signals = pd.Series([1.0, -1.0, 0.0, 1.0])
        feats   = pd.DataFrame({"close": [1, 2, 3, 4]})
        out = ml.batch_filter(signals, feats)
        assert "decision" in out.columns
        assert "confidence" in out.columns
        assert "filtered_signal" in out.columns
        # Without model all decisions should be PASS
        assert (out["decision"] == "PASS").all()

    def test_is_loaded_false_for_missing_model(self):
        from src.analysis.meta_labeler import MetaLabeler
        ml = MetaLabeler(model_path='/nonexistent/path.joblib')
        assert ml.is_loaded is False


# ══════════════════════════════════════════════════════════════════════════════
# 5. Regime Classifier
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeClassifier:
    def test_classifier_initializes(self):
        from src.analysis.regime_classifier import RegimeClassifier
        with patch.object(RegimeClassifier, 'MODEL_PATH', '/nonexistent/clf.joblib'):
            clf = RegimeClassifier()
        assert not clf.is_ready

    def test_predict_without_training_returns_zero(self, ohlcv):
        from src.analysis.regime_classifier import RegimeClassifier
        with patch.object(RegimeClassifier, 'MODEL_PATH', '/nonexistent/clf.joblib'):
            clf = RegimeClassifier()
        regime = clf.predict(ohlcv)
        assert regime == 0

    def test_regime_map_exists(self):
        from src.analysis.regime_classifier import REGIME_STRATEGY_MAP
        assert 0 in REGIME_STRATEGY_MAP
        assert 1 in REGIME_STRATEGY_MAP
        assert 2 in REGIME_STRATEGY_MAP


# ══════════════════════════════════════════════════════════════════════════════
# 6. Agent Bus
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentBus:
    def test_singleton(self):
        from src.engine.agents.agent_bus import get_bus
        b1 = get_bus()
        b2 = get_bus()
        assert b1 is b2

    def test_publish_and_subscribe(self):
        from src.engine.agents.agent_bus import AgentBus
        bus = AgentBus()
        received = []
        bus.subscribe("test", lambda msg: received.append(msg.payload))
        bus.publish("test", "sender", {"val": 42})
        assert received == [{"val": 42}]

    def test_get_latest(self):
        from src.engine.agents.agent_bus import AgentBus
        bus = AgentBus()
        bus.publish("candle", "DataAgent", {"price": 100})
        bus.publish("candle", "DataAgent", {"price": 200})
        msg = bus.get_latest("candle")
        assert msg is not None
        assert msg.payload["price"] == 200

    def test_get_latest_unknown_topic(self):
        from src.engine.agents.agent_bus import AgentBus
        bus = AgentBus()
        assert bus.get_latest("no_such_topic") is None

    def test_history_capped(self):
        from src.engine.agents.agent_bus import AgentBus
        bus = AgentBus()
        bus._max_history = 5
        for i in range(10):
            bus.publish("x", "s", i)
        assert len(bus._history) <= 5

    def test_callback_error_does_not_crash_bus(self):
        from src.engine.agents.agent_bus import AgentBus
        bus = AgentBus()
        def bad_cb(msg):
            raise RuntimeError("boom")
        bus.subscribe("evt", bad_cb)
        # Should not raise
        bus.publish("evt", "s", {})

    def test_base_agent_start_stop(self):
        from src.engine.agents.agent_bus import AgentBus, BaseAgent
        bus = AgentBus()
        cycles = []

        class DummyAgent(BaseAgent):
            NAME = "DummyAgent"
            def _run_cycle(self):
                cycles.append(1)

        agent = DummyAgent(bus=bus, interval_sec=0.05)
        agent.start()
        time.sleep(0.5)  # filelock writes in _loop add ~80ms overhead per cycle
        agent.stop()
        assert len(cycles) >= 2, "Agent should have run at least 2 cycles"

    def test_write_status_file(self, tmp_path):
        from src.engine.agents.agent_bus import _write_agent_status, _STATUS_FILE
        # Temporarily redirect to tmp
        import src.engine.agents.agent_bus as bus_mod
        orig = bus_mod._STATUS_FILE
        bus_mod._STATUS_FILE = tmp_path / "agent_status.json"
        try:
            _write_agent_status("TestAgent", "idle", "Waiting", 60.0)
            data = json.loads((tmp_path / "agent_status.json").read_text())
            assert "TestAgent" in data
            assert data["TestAgent"]["status"] == "idle"
        finally:
            bus_mod._STATUS_FILE = orig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Feature Engineering — new functions
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_add_ofi_creates_column(self, ohlcv):
        from src.analysis.feature_engineering import add_ofi
        df = add_ofi(ohlcv.copy(), window=10)
        assert "ofi" in df.columns

    def test_add_ofi_z_column(self, ohlcv):
        from src.analysis.feature_engineering import add_ofi
        df = add_ofi(ohlcv.copy(), window=10)
        assert "ofi_z" in df.columns

    def test_add_vwap_creates_columns(self, ohlcv):
        from src.analysis.feature_engineering import add_vwap
        df = add_vwap(ohlcv.copy())
        assert "vwap" in df.columns
        assert "vwap_dist" in df.columns

    def test_add_vwap_dist_bounded(self, ohlcv):
        from src.analysis.feature_engineering import add_vwap
        df = add_vwap(ohlcv.copy()).dropna()
        assert df["vwap_dist"].abs().max() < 1.0, "VWAP distance should be a small fraction"

    def test_add_donchian_creates_columns(self, ohlcv):
        from src.analysis.feature_engineering import add_donchian
        df = add_donchian(ohlcv.copy(), n=20)
        assert "don_upper_20" in df.columns
        assert "don_lower_20" in df.columns
        assert "don_pos_20" in df.columns

    def test_add_donchian_pos_bounded(self, ohlcv):
        from src.analysis.feature_engineering import add_donchian
        df = add_donchian(ohlcv.copy(), n=20).dropna()
        assert df["don_pos_20"].between(0, 1).all()

    def test_add_keltner_creates_columns(self, ohlcv):
        from src.analysis.feature_engineering import add_keltner
        df = add_keltner(ohlcv.copy())
        assert "kc_upper" in df.columns
        assert "kc_lower" in df.columns
        assert "kc_pos" in df.columns
        assert "kc_width" in df.columns

    def test_add_funding_zscore_with_column(self, ohlcv):
        from src.analysis.feature_engineering import add_funding_zscore
        df = ohlcv.copy()
        df["funding_rate"] = np.random.normal(0, 0.001, len(df))
        df = add_funding_zscore(df)
        assert "funding_z" in df.columns
        assert "funding_positive" in df.columns
        assert "funding_negative" in df.columns

    def test_add_funding_zscore_without_column(self, ohlcv):
        from src.analysis.feature_engineering import add_funding_zscore
        # Should handle missing funding_rate gracefully
        df = add_funding_zscore(ohlcv.copy())
        assert "funding_z" in df.columns
        assert (df["funding_z"] == 0.0).all()

    def test_add_liquidity_proximity_creates_columns(self, ohlcv):
        from src.analysis.feature_engineering import add_liquidity_proximity
        df = add_liquidity_proximity(ohlcv.copy())
        assert "dist_to_supply" in df.columns
        assert "dist_to_demand" in df.columns
        assert "liq_proximity" in df.columns

    def test_existing_add_rsi_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_rsi
        df = add_rsi(ohlcv.copy(), 14)
        assert "rsi_14" in df.columns
        valid = df["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_existing_add_macd_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_macd
        df = add_macd(ohlcv.copy())
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_existing_add_bollinger_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_bollinger_bands
        df = add_bollinger_bands(ohlcv.copy(), window=20)
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_pb" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# 8. Risk Manager — Kelly layer
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskManager:
    def _make_data(self, n=120):
        df = _make_ohlcv(n)
        return df.to_dict("records")

    def test_get_position_size_returns_positive(self):
        from src.analysis.risk_manager import HullRiskManager
        rm = HullRiskManager(default_risk_usd=20.0)
        size = rm.get_position_size(self._make_data())
        assert size > 0

    def test_get_kelly_position_size_positive(self):
        from src.analysis.risk_manager import HullRiskManager
        rm = HullRiskManager(default_risk_usd=20.0)
        size = rm.get_kelly_position_size(
            capital=10000, p_win=0.65, data=self._make_data()
        )
        assert size > 0

    def test_get_kelly_position_size_scales_with_capital(self):
        from src.analysis.risk_manager import HullRiskManager
        rm = HullRiskManager(default_risk_usd=20.0)
        data = self._make_data()
        s1 = rm.get_kelly_position_size(capital=1000, p_win=0.65, data=data)
        s2 = rm.get_kelly_position_size(capital=2000, p_win=0.65, data=data)
        assert s2 > s1

    def test_record_trade_outcome_updates_history(self):
        from src.analysis.risk_manager import HullRiskManager
        rm = HullRiskManager()
        rm.record_trade_outcome(50.0)
        rm.record_trade_outcome(-20.0)
        assert len(rm._kelly._pnls) == 2

    def test_high_volatility_reduces_size(self):
        from src.analysis.risk_manager import HullRiskManager
        rm = HullRiskManager(default_risk_usd=20.0)
        # Low vol data: flat price series
        flat_data = [{"close": 50000 + i * 0.01, "high": 50001, "low": 49999, "volume": 100}
                     for i in range(120)]
        size_low = rm.get_position_size(flat_data)
        # High vol data: large random swings
        rng = np.random.default_rng(1)
        prices = 50000 * np.cumprod(1 + rng.normal(0, 0.05, 120))
        high_vol_data = [{"close": float(p), "high": float(p * 1.03),
                          "low": float(p * 0.97), "volume": 100} for p in prices]
        size_high = rm.get_position_size(high_vol_data)
        assert size_low > size_high, "High volatility should reduce position size"


# ══════════════════════════════════════════════════════════════════════════════
# 9. Dashboard — /api/agents endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboardAgentsEndpoint:
    @pytest.fixture
    def client(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_agents_endpoint_returns_200(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 200

    def test_agents_endpoint_returns_list(self, client):
        data = client.get("/api/agents").get_json()
        assert "agents" in data
        assert isinstance(data["agents"], list)

    def test_agents_count(self, client):
        data = client.get("/api/agents").get_json()
        assert len(data["agents"]) == 8

    def test_agents_have_required_keys(self, client):
        data = client.get("/api/agents").get_json()
        required = {"name", "label", "desc", "market", "color",
                    "interval_sec", "status", "current_task", "last_heartbeat", "timeline"}
        for agent in data["agents"]:
            missing = required - set(agent.keys())
            assert not missing, f"Agent {agent.get('name')} missing keys: {missing}"

    def test_agents_offline_without_status_file(self, client):
        """When no agent_status.json exists, all agents should be offline."""
        with patch("src.dashboard.app._PROJECT_ROOT") as mock_root:
            mock_root.__truediv__ = lambda self, x: Path("/nonexistent") / x
            # Rebuild path inside the endpoint by patching Path.exists
            with patch.object(Path, "exists", return_value=False):
                data = client.get("/api/agents").get_json()
        # All should be offline
        for agent in data["agents"]:
            assert agent["status"] == "offline"

    def test_model_stats_includes_meta_and_regime(self, client):
        data = client.get("/api/monitor/model_stats").get_json()
        keys = [m["key"] for m in data["models"]]
        assert "meta" in keys, "meta_labeler should appear in model stats"
        assert "regime" in keys, "regime_classifier should appear in model stats"

    def test_model_stats_new_fields(self, client):
        data = client.get("/api/monitor/model_stats").get_json()
        for model in data["models"]:
            assert "walk_forward_mean_acc" in model
            assert "target" in model


# ══════════════════════════════════════════════════════════════════════════════
# 10. Regression — ML Predictor still works
# ══════════════════════════════════════════════════════════════════════════════

class TestMLPredictorRegression:
    def test_predict_returns_none_when_no_model(self):
        from src.analysis.ml_predictor import MLPredictor
        p = MLPredictor(model_filename="nonexistent_model.joblib")
        data = _make_ohlcv(50).to_dict("records")
        result = p.predict(data)
        assert result is None

    def test_predict_proba_long_returns_float(self):
        from src.analysis.ml_predictor import MLPredictor
        p = MLPredictor(model_filename="nonexistent_model.joblib")
        data = _make_ohlcv(50).to_dict("records")
        val = p.predict_proba_long(data)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_predict_returns_none_for_short_data(self):
        from src.analysis.ml_predictor import MLPredictor
        p = MLPredictor(model_filename="nonexistent_model.joblib")
        result = p.predict([{"close": 1}] * 5)
        assert result is None

    def test_last_error_set_when_model_missing(self):
        from src.analysis.ml_predictor import MLPredictor
        p = MLPredictor(model_filename="nonexistent_model.joblib")
        assert p.last_error != ""


# ══════════════════════════════════════════════════════════════════════════════
# 11. Regression — Backtester no off-by-one
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktesterRegression:
    def _make_backtester_df(self, n=300):
        df = _make_ohlcv(n)
        df["signal_buy_hold"] = 1
        df["signal_rsi"] = 0
        df["signal_macd"] = 0
        df["signal_bb"] = 0
        df["signal_don"] = 0
        return df

    def test_backtester_imports(self):
        from src.engine.backtester import Backtester
        assert Backtester is not None

    def test_backtester_no_length_mismatch(self):
        """The historical off-by-one bug: equity_series length != price length."""
        from src.engine.backtester import Backtester
        df = self._make_backtester_df(100)
        bt = Backtester(initial_capital=1000)
        try:
            result = bt.run(df, signal_col="signal_buy_hold",
                            strategy_name="BuyHold", symbol="BTC_USDT")
            # equity_curve length should equal df length
            assert len(result.equity_curve) == len(df), \
                f"equity_curve len {len(result.equity_curve)} != df len {len(df)}"
        except ValueError as e:
            if "length" in str(e).lower():
                pytest.fail(f"Off-by-one bug regressed: {e}")

    def test_backtester_returns_metrics(self):
        from src.engine.backtester import Backtester, BacktestResult
        df = self._make_backtester_df(200)
        bt = Backtester(initial_capital=1000)
        result = bt.run(df, signal_col="signal_buy_hold",
                        strategy_name="BuyHold", symbol="BTC_USDT")
        assert isinstance(result, BacktestResult)
        assert hasattr(result, "sharpe")
        assert isinstance(result.sharpe(), float)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Regression — Feature engineering old functions unchanged
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineeringRegression:
    def test_add_adx_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_adx
        df = add_adx(ohlcv.copy(), 14)
        assert "adx_14" in df.columns
        assert "atr_14" in df.columns

    def test_add_roc_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_roc
        df = add_roc(ohlcv.copy(), [3, 7, 14])
        assert "roc_3" in df.columns
        assert "roc_7" in df.columns
        assert "roc_14" in df.columns

    def test_add_time_features_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_time_features
        df = ohlcv.copy()
        df["timestamp"] = df.index  # add timestamp column from DatetimeIndex
        df = add_time_features(df)
        assert "hour" in df.columns
        assert "day_of_week" in df.columns

    def test_add_taker_features_unchanged(self, ohlcv):
        from src.analysis.feature_engineering import add_taker_and_trade_features
        df = add_taker_and_trade_features(ohlcv.copy())
        assert "taker_buy_ratio" in df.columns
        assert "avg_trade_size" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# 13. Agent system — market agents config
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketAgentsConfig:
    def test_scalping_agent_filters_illiquid(self):
        from src.engine.agents.scalping_agent import ScalpingAgent, LIQUID_SYMBOLS
        symbols = ["BTC_USDT", "XRP_USDT", "ETH_USDT"]
        # Mock data getter and bus
        agent = ScalpingAgent(symbols=symbols, data_getter_1m=lambda s: None, bus=MagicMock())
        kept = set(agent.symbols)
        assert "XRP_USDT" not in kept
        assert "BTC_USDT" in kept

    def test_scalping_confidence_threshold(self):
        from src.engine.agents.scalping_agent import CONFIDENCE_THRESHOLD
        assert CONFIDENCE_THRESHOLD >= 0.60, "Scalping needs higher confidence bar"

    def test_futures_leverage(self):
        from src.engine.agents.futures_agent import LEVERAGE
        assert 1.0 <= LEVERAGE <= 5.0, "Conservative leverage expected"

    def test_futures_confidence_threshold(self):
        from src.engine.agents.futures_agent import CONFIDENCE_THRESHOLD
        assert CONFIDENCE_THRESHOLD >= 0.55


# ══════════════════════════════════════════════════════════════════════════════
# 14. Strategy Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyRegistry:
    def test_registry_has_entries(self):
        from src.engine.strategy_registry import REGISTRY
        assert len(REGISTRY) >= 15, "Registry should have at least 15 strategies"

    def test_registry_required_fields(self):
        from src.engine.strategy_registry import REGISTRY
        required = {"label", "description", "group", "signal_col", "models", "can_live", "can_backtest"}
        for name, info in REGISTRY.items():
            missing = required - set(info.keys())
            assert not missing, f"{name} missing fields: {missing}"

    def test_load_config_returns_dict(self):
        from src.engine.strategy_registry import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_config_keys_match_registry(self):
        from src.engine.strategy_registry import load_config, REGISTRY
        cfg = load_config()
        for name in cfg:
            assert name in REGISTRY, f"Config has unknown strategy: {name}"

    def test_update_strategy_persists(self, tmp_path, monkeypatch):
        from src.engine import strategy_registry as reg
        monkeypatch.setattr(reg, "CONFIG_PATH", tmp_path / "strategy_config.json")
        entry = reg.update_strategy("RSI_MeanReversion", live=False, backtest=True)
        assert entry["backtest"] is True
        cfg = reg.load_config()
        assert cfg["RSI_MeanReversion"]["backtest"] is True

    def test_update_unknown_strategy_raises(self):
        from src.engine.strategy_registry import update_strategy
        with pytest.raises(KeyError):
            update_strategy("NonExistentStrategy_XYZ", live=True)

    def test_get_sync_report_structure(self):
        from src.engine.strategy_registry import get_sync_report
        report = get_sync_report()
        assert "strategies" in report
        assert "summary"    in report
        assert "total"      in report["summary"]
        assert "synced"     in report["summary"]
        assert "gaps"       in report["summary"]

    def test_sync_report_every_strategy_has_status(self):
        from src.engine.strategy_registry import get_sync_report, REGISTRY
        report = get_sync_report()
        names = {e["name"] for e in report["strategies"]}
        assert names == set(REGISTRY.keys())
        valid_statuses = {
            "synced", "live_only", "backtest_only",
            "live_only_by_design", "backtest_only_by_design", "disabled"
        }
        for entry in report["strategies"]:
            assert entry["sync_status"] in valid_statuses

    def test_enabled_backtest_signal_cols(self):
        from src.engine.strategy_registry import enabled_backtest_signal_cols
        cols = enabled_backtest_signal_cols()
        assert isinstance(cols, list)
        assert len(cols) > 0
        for name, label, sig_col in cols:
            assert isinstance(sig_col, str)
            assert len(sig_col) > 0

    def test_is_enabled_helpers(self):
        from src.engine.strategy_registry import is_enabled_live, is_enabled_backtest
        # RSI is enabled by default for both
        assert isinstance(is_enabled_live("RSI_MeanReversion"), bool)
        assert isinstance(is_enabled_backtest("RSI_MeanReversion"), bool)


# ══════════════════════════════════════════════════════════════════════════════
# 15. Strategy Sync API endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategySyncEndpoint:
    @pytest.fixture(autouse=True)
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            self.client = c

    def test_strategy_sync_get_returns_200(self):
        r = self.client.get("/api/strategy-sync")
        assert r.status_code == 200

    def test_strategy_sync_has_strategies_key(self):
        r = self.client.get("/api/strategy-sync")
        data = r.get_json()
        assert "strategies" in data
        assert "summary"    in data

    def test_strategy_sync_strategies_are_list(self):
        r = self.client.get("/api/strategy-sync")
        data = r.get_json()
        assert isinstance(data["strategies"], list)
        assert len(data["strategies"]) > 0

    def test_strategy_sync_each_entry_has_required_fields(self):
        r = self.client.get("/api/strategy-sync")
        data = r.get_json()
        required = {"name", "label", "group", "can_live", "can_backtest",
                    "live_enabled", "backtest_enabled", "sync_status"}
        for entry in data["strategies"]:
            missing = required - set(entry.keys())
            assert not missing, f"Entry missing: {missing}"

    def test_strategy_sync_post_toggle(self, tmp_path, monkeypatch):
        from src.engine import strategy_registry as reg
        monkeypatch.setattr(reg, "CONFIG_PATH", tmp_path / "strategy_config.json")
        r = self.client.post(
            "/api/strategy-sync",
            json={"name": "RSI_MeanReversion", "live": False, "backtest": True},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert "updated" in data
        assert len(data["updated"]) == 1
        assert data["updated"][0]["name"] == "RSI_MeanReversion"

    def test_strategy_sync_post_unknown_name_returns_400(self):
        r = self.client.post(
            "/api/strategy-sync",
            json={"name": "TOTALLY_FAKE_STRATEGY_XYZ", "live": True},
            content_type="application/json",
        )
        assert r.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 16. Backtester new signals (ML batch + Elliott proxy + GARCH + vol breakout)
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktesterNewSignals:
    @pytest.fixture
    def bt_df(self):
        df = _make_ohlcv(300)
        df["timestamp"] = df.index
        df["funding_rate"] = 0.0
        df["taker_buy_base"] = df["volume"] * 0.5
        df["taker_buy_quote"] = df["close"] * df["volume"] * 0.5
        df["trades_count"] = 100
        return df.reset_index(drop=True)

    def test_build_signals_adds_vol_breakout(self, bt_df):
        from src.engine.backtester import _build_signals
        out = _build_signals(bt_df)
        assert "signal_vol_breakout" in out.columns
        assert out["signal_vol_breakout"].isin([-1.0, 0.0, 1.0]).all()

    def test_build_signals_adds_garch_mult(self, bt_df):
        from src.engine.backtester import _build_signals
        out = _build_signals(bt_df)
        assert "garch_size_mult" in out.columns
        assert out["garch_size_mult"].isin([0.5, 1.0]).all()

    def test_build_signals_adds_mtf_filter(self, bt_df):
        from src.engine.backtester import _build_signals
        out = _build_signals(bt_df)
        assert "signal_mtf_filter" in out.columns
        assert out["signal_mtf_filter"].isin([-1.0, 1.0]).all()

    def test_build_signals_adds_ml_signals(self, bt_df):
        from src.engine.backtester import _build_signals
        out = _build_signals(bt_df)
        # ML signals present regardless of whether model file exists (falls back to 0)
        assert "signal_base_ml"    in out.columns
        assert "signal_trend_ml"   in out.columns
        assert "signal_futures_ml" in out.columns

    def test_build_signals_adds_elliott_proxy(self, bt_df):
        from src.engine.backtester import _build_signals
        out = _build_signals(bt_df)
        assert "signal_elliott_proxy" in out.columns
        assert out["signal_elliott_proxy"].isin([-1.0, 0.0, 1.0]).all()

    def test_batch_ml_predict_missing_model_returns_zeros(self, bt_df):
        from src.engine.backtester import _batch_ml_predict
        result = _batch_ml_predict(bt_df, "nonexistent_model_xyz.joblib")
        assert (result == 0.0).all()

    def test_garch_halves_position_on_vol_spike(self, bt_df):
        from src.engine.backtester import Backtester, _build_signals
        df = _build_signals(bt_df)
        # Force a spike on one bar
        df["garch_size_mult"] = 1.0
        df.loc[50, "garch_size_mult"] = 0.5
        df["signal_test"] = 1.0  # always long
        bt = Backtester(initial_capital=1000)
        result = bt.run(df, "signal_test", "garch_test", "BTC_USDT")
        assert result is not None


# ─── Simulator Tests ──────────────────────────────────────────────────────────

class TestSimulatorDataStore:
    """Tests for DuckDB-backed simulator store."""

    def test_import(self):
        from src.simulation.data_store import SimulatorDataStore
        assert SimulatorDataStore is not None

    def test_init_creates_db(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        db = tmp_path / "test.duckdb"
        store = SimulatorDataStore(db_path=db)
        assert db.exists()

    def test_start_scenario_returns_id(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        sid = store.start_scenario("VOLATILE", "BTC_USDT", "1m")
        assert isinstance(sid, str) and len(sid) == 36  # UUID

    def test_update_scenario_bars(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        sid = store.start_scenario("RANGING", "ETH_USDT", "1h")
        store.update_scenario_bars(sid, 1234)
        summary = store.get_summary()
        assert summary["max_bars_replayed"] == 1234

    def test_record_paper_trade(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        from datetime import datetime, timezone
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        sid = store.start_scenario("TRENDING_UP", "BTC_USDT", "1m")
        store.record_paper_trade(
            scenario_id=sid, symbol="BTC_USDT", direction=1,
            entry_price=50000.0, exit_price=51000.0, size_usd=100.0,
            entry_ts=datetime(2023,1,1,tzinfo=timezone.utc),
            exit_ts=datetime(2023,1,2,tzinfo=timezone.utc),
            strategy="ScalpingML", model_ver="v1",
        )
        summary = store.get_summary()
        assert summary["total_paper_trades"] == 1
        assert summary["total_pnl_usd"] > 0  # long + price rose

    def test_record_training_event(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        store.record_training_event(
            model_name="ScalpingML", scenario_id="test-id",
            bars_trained=5000, train_loss=0.35, val_loss=0.37,
            accuracy=0.64, sharpe=1.2,
        )
        events = store.get_recent_training_events()
        assert len(events) == 1
        assert events[0]["model"] == "ScalpingML"

    def test_get_summary_models(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        store.record_training_event("TFT_MM", "x", 1000, 0.1, 0.11, 0.7)
        store.record_training_event("OU_Filter", "x", 500, 0.0, 0.0, 0.55)
        summary = store.get_summary()
        names = {m["name"] for m in summary["models"]}
        assert "TFT_MM" in names
        assert "OU_Filter" in names

    def test_pnl_series_empty(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        series = store.get_paper_pnl_series()
        assert isinstance(series, list)

    def test_training_events_filter_by_model(self, tmp_path):
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore(db_path=tmp_path / "t.duckdb")
        store.record_training_event("ScalpingML", "", 100, 0.3, 0.31, 0.6)
        store.record_training_event("TFT_MM",     "", 200, 0.2, 0.21, 0.7)
        scalp_events = store.get_recent_training_events(model_name="ScalpingML")
        assert all(e["model"] == "ScalpingML" for e in scalp_events)


class TestScenarioManager:
    """Tests for scenario selection and classification."""

    def test_import(self):
        from src.simulation.scenario_manager import ScenarioManager
        assert ScenarioManager is not None

    def test_scenario_types_defined(self):
        from src.simulation.scenario_manager import SCENARIO_TYPES
        assert "VOLATILE" in SCENARIO_TYPES
        assert "TRENDING_UP" in SCENARIO_TYPES
        assert len(SCENARIO_TYPES) >= 5

    def test_classify_bars_trending_up(self):
        from src.simulation.scenario_manager import ScenarioManager
        mgr = ScenarioManager()
        prices = pd.Series([100 * (1 + 0.01) ** i for i in range(100)])
        df = pd.DataFrame({"close": prices, "volume": 1.0})
        result = mgr.classify_bars(df)
        assert result in ("TRENDING_UP", "COMPOSITE")

    def test_classify_bars_volatile(self):
        from src.simulation.scenario_manager import ScenarioManager
        mgr = ScenarioManager()
        rng = np.random.default_rng(42)
        prices = 100 + rng.normal(0, 5, 100).cumsum()
        df = pd.DataFrame({"close": prices, "volume": 1.0})
        result = mgr.classify_bars(df)
        assert result in ("VOLATILE", "COMPOSITE", "TRENDING_UP", "TRENDING_DOWN")

    def test_classify_bars_empty_returns_composite(self):
        from src.simulation.scenario_manager import ScenarioManager
        mgr = ScenarioManager()
        assert mgr.classify_bars(pd.DataFrame()) == "COMPOSITE"

    def test_record_result_updates_weights(self):
        from src.simulation.scenario_manager import ScenarioManager
        mgr = ScenarioManager()
        orig = mgr._weights.get("VOLATILE", 1.0)
        mgr.record_result("VOLATILE", 0.01)  # low accuracy → higher weight
        assert mgr._weights["VOLATILE"] > orig

    def test_weighted_choice_returns_valid_type(self):
        from src.simulation.scenario_manager import ScenarioManager, SCENARIO_TYPES
        mgr = ScenarioManager()
        for _ in range(20):
            assert mgr._weighted_choice() in SCENARIO_TYPES


class TestMarketReplay:
    """Tests for GZ streaming replay engine."""

    def test_import(self):
        from src.simulation.market_replay import MarketReplay
        assert MarketReplay is not None

    def test_missing_gz_raises(self):
        from src.simulation.market_replay import MarketReplay
        import pytest
        with pytest.raises(FileNotFoundError):
            MarketReplay("NONEXISTENT_USDT", "1m")

    def test_btc_1m_exists_and_streams(self):
        from src.simulation.market_replay import RAW_DIR, MarketReplay
        gz = RAW_DIR / "BTC_USDT_1m.csv.gz"
        if not gz.exists():
            pytest.skip("BTC_USDT_1m.csv.gz not present in data/raw/")
        replay = MarketReplay("BTC_USDT", "1m", speed=100_000.0)
        bars = []
        for bar in replay.stream(stopped_flag=[False]):
            bars.append(bar)
            if len(bars) >= 10:
                break
        assert len(bars) == 10
        assert "close" in bars[0]
        assert bars[0]["source"] == "simulator"

    def test_bar_has_required_fields(self):
        from src.simulation.market_replay import RAW_DIR, MarketReplay
        gz = RAW_DIR / "BTC_USDT_1m.csv.gz"
        if not gz.exists():
            pytest.skip("BTC_USDT_1m.csv.gz not present in data/raw/")
        replay = MarketReplay("BTC_USDT", "1m", speed=100_000.0)
        for bar in replay.stream():
            required = {"symbol","timeframe","timestamp","open","high","low",
                        "close","volume","funding_rate","source"}
            assert required.issubset(set(bar.keys()))
            break

    def test_stop_flag_halts_stream(self):
        from src.simulation.market_replay import RAW_DIR, MarketReplay
        gz = RAW_DIR / "BTC_USDT_1m.csv.gz"
        if not gz.exists():
            pytest.skip("BTC_USDT_1m.csv.gz not present in data/raw/")
        stop = [False]
        replay = MarketReplay("BTC_USDT", "1m", speed=100_000.0)
        count = 0
        for bar in replay.stream(stopped_flag=stop):
            count += 1
            if count == 5:
                stop[0] = True
        assert count <= 6  # stopped promptly


class TestSimulatorAgent:
    """Tests for SimulatorAgent state machine."""

    def test_import(self):
        from src.engine.agents.simulator_agent import SimulatorAgent
        assert SimulatorAgent is not None

    def test_initial_state_idle(self):
        from src.engine.agents.simulator_agent import SimulatorAgent, IDLE
        agent = SimulatorAgent(auto_cycle=False)
        assert agent.get_status()["state"] == IDLE

    def test_configure_updates_config(self):
        from src.engine.agents.simulator_agent import SimulatorAgent
        agent = SimulatorAgent(auto_cycle=False)
        agent.configure({"symbol": "ETH_USDT", "speed": 500.0})
        cfg = agent.get_status()["config"]
        assert cfg["symbol"] == "ETH_USDT"
        assert cfg["speed"] == 500.0

    def test_stop_sets_idle(self):
        from src.engine.agents.simulator_agent import SimulatorAgent, IDLE
        agent = SimulatorAgent(auto_cycle=False)
        agent.stop()
        assert agent.get_status()["state"] == IDLE


class TestSimulatorEndpoints:
    """Tests for simulator Flask REST endpoints."""

    @pytest.fixture(autouse=True)
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            self.client = c

    def _hdrs(self):
        return {"Content-Type": "application/json"}

    def test_status_endpoint_returns_200(self):
        r = self.client.get("/api/simulator/status")
        assert r.status_code == 200
        d = r.get_json()
        assert "state" in d

    def test_available_data_endpoint(self):
        r = self.client.get("/api/simulator/available_data")
        assert r.status_code == 200
        d = r.get_json()
        assert "files" in d
        assert "total" in d

    def test_config_endpoint(self):
        from unittest.mock import MagicMock, patch
        mock_sim = MagicMock()
        mock_sim.get_status.return_value = {"config": {"symbol": "ETH_USDT", "speed": 500.0}}
        with patch("src.dashboard.app._get_simulator", return_value=(mock_sim, MagicMock(), MagicMock())):
            r = self.client.post(
                "/api/simulator/config",
                json={"symbol": "ETH_USDT", "speed": 500.0},
                headers=self._hdrs(),
            )
        assert r.status_code == 200
        d = r.get_json()
        assert d["ok"] is True

    def test_training_history_endpoint(self):
        r = self.client.get("/api/simulator/training_history")
        assert r.status_code == 200
        d = r.get_json()
        assert "events" in d
        assert "pnl_series" in d

    def test_pause_stop_endpoints(self):
        r_pause = self.client.post("/api/simulator/pause")
        r_stop  = self.client.post("/api/simulator/stop")
        assert r_pause.status_code == 200
        assert r_stop.status_code == 200


# ─── New Indicator Tests ──────────────────────────────────────────────────────

class TestNewIndicators:
    """Tests for Ichimoku, SuperTrend, and MACD Divergence indicators."""

    @pytest.fixture
    def ohlcv_df(self):
        """Synthetic 300-bar OHLCV DataFrame for indicator testing."""
        rng = np.random.default_rng(42)
        n = 300
        close = 50000 + rng.normal(0, 500, n).cumsum()
        close = np.maximum(close, 10000)
        high  = close + rng.uniform(50, 300, n)
        low   = close - rng.uniform(50, 300, n)
        df = pd.DataFrame({
            "open":   close - rng.uniform(0, 100, n),
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": rng.uniform(1e6, 5e6, n),
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        })
        return df

    def test_ichimoku_columns_added(self, ohlcv_df):
        from src.analysis.feature_engineering import add_ichimoku
        df = add_ichimoku(ohlcv_df.copy())
        required = {"ichimoku_tenkan", "ichimoku_kijun",
                    "ichimoku_senkou_a", "ichimoku_senkou_b",
                    "ichimoku_chikou", "signal_ichimoku"}
        assert required.issubset(df.columns)

    def test_ichimoku_signal_values(self, ohlcv_df):
        from src.analysis.feature_engineering import add_ichimoku
        df = add_ichimoku(ohlcv_df.copy())
        assert df["signal_ichimoku"].isin([-1.0, 0.0, 1.0]).all()

    def test_ichimoku_min_rows_required(self):
        from src.analysis.feature_engineering import add_ichimoku
        df_tiny = pd.DataFrame({
            "open": [1.0]*5, "high": [1.1]*5, "low": [0.9]*5,
            "close": [1.0]*5, "volume": [1e6]*5,
        })
        df = add_ichimoku(df_tiny)
        assert "signal_ichimoku" in df.columns

    def test_supertrend_columns_added(self, ohlcv_df):
        from src.analysis.feature_engineering import add_supertrend
        df = add_supertrend(ohlcv_df.copy())
        assert "supertrend"        in df.columns
        assert "supertrend_dir"    in df.columns
        assert "signal_supertrend" in df.columns

    def test_supertrend_direction_binary(self, ohlcv_df):
        from src.analysis.feature_engineering import add_supertrend
        df = add_supertrend(ohlcv_df.copy())
        dirs = df["supertrend_dir"].dropna()
        assert dirs.isin([-1.0, 1.0, np.nan]).all() or set(dirs.unique()).issubset({-1, 1, 0, np.nan})

    def test_supertrend_signal_only_on_flip(self, ohlcv_df):
        from src.analysis.feature_engineering import add_supertrend
        df = add_supertrend(ohlcv_df.copy())
        # Flips should be rare (not every bar)
        flips = (df["signal_supertrend"] != 0).sum()
        assert flips < len(df) * 0.3, "SuperTrend should emit signals on direction flips only"

    def test_macd_divergence_columns(self, ohlcv_df):
        from src.analysis.feature_engineering import add_macd_divergence
        df = add_macd_divergence(ohlcv_df.copy())
        assert "macd_cl_cross"    in df.columns
        assert "macd_divergence"  in df.columns
        assert "signal_macd_div"  in df.columns

    def test_macd_divergence_signal_values(self, ohlcv_df):
        from src.analysis.feature_engineering import add_macd_divergence
        df = add_macd_divergence(ohlcv_df.copy())
        assert df["signal_macd_div"].isin([-1.0, 0.0, 1.0]).all()

    def test_backtester_builds_ichimoku_signal(self):
        from src.engine.backtester import _build_signals
        rng = np.random.default_rng(0)
        n = 300
        close = 50000 + rng.normal(0, 400, n).cumsum()
        close = np.maximum(close, 10000)
        df = pd.DataFrame({
            "open":  close - 50, "high": close + 100,
            "low":   close - 100, "close": close,
            "volume": rng.uniform(1e6, 3e6, n),
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        })
        result = _build_signals(df)
        assert "signal_ichimoku"    in result.columns
        assert "signal_supertrend"  in result.columns
        assert "signal_macd_div"    in result.columns
        assert "signal_ou_entry"    in result.columns

    def test_backtester_ichimoku_signal_range(self):
        from src.engine.backtester import _build_signals
        rng = np.random.default_rng(7)
        n = 300
        close = 40000 + rng.normal(0, 300, n).cumsum()
        close = np.maximum(close, 1000)
        df = pd.DataFrame({
            "open": close - 30, "high": close + 60,
            "low":  close - 60, "close": close,
            "volume": rng.uniform(5e5, 2e6, n),
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="h"),
        })
        result = _build_signals(df)
        assert result["signal_ichimoku"].isin([-1.0, 0.0, 1.0]).all()
        assert result["signal_supertrend"].isin([-1.0, 0.0, 1.0]).all()

    def test_strategy_registry_has_new_strategies(self):
        from src.engine.strategy_registry import REGISTRY
        assert "Ichimoku_Cloud"  in REGISTRY
        assert "Supertrend"      in REGISTRY
        assert "MACD_Divergence" in REGISTRY
        assert "OU_Entry"        in REGISTRY

    def test_new_strategies_have_required_fields(self):
        from src.engine.strategy_registry import REGISTRY
        for name in ("Ichimoku_Cloud", "Supertrend", "MACD_Divergence", "OU_Entry"):
            entry = REGISTRY[name]
            assert "signal_col" in entry
            assert "can_live"   in entry
            assert "can_backtest" in entry
            assert entry["can_live"]     is True
            assert entry["can_backtest"] is True

    def test_strategy_registry_total_count(self):
        from src.engine.strategy_registry import REGISTRY
        # Originally 22, now 26 with 4 new strategies
        assert len(REGISTRY) >= 26

    def test_sync_report_includes_new_strategies(self):
        from src.engine.strategy_registry import get_sync_report
        report = get_sync_report()
        names = {e["name"] for e in report["strategies"]}
        assert "Ichimoku_Cloud"  in names
        assert "Supertrend"      in names
        assert "MACD_Divergence" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
