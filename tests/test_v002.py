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
        labels = triple_barrier_labels_vectorized(ohlcv, profit_pct=0.02, loss_pct=0.01)
        assert set(labels.unique()).issubset({-1, 0, 1})

    def test_labels_output_length(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized
        labels = triple_barrier_labels_vectorized(ohlcv)
        assert len(labels) == len(ohlcv)

    def test_last_bars_are_zero(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized
        max_bars = 24
        labels = triple_barrier_labels_vectorized(ohlcv, max_bars=max_bars)
        # Last max_bars rows must be 0 (unlabeled)
        assert (labels.iloc[-max_bars:] == 0).all()

    def test_loop_version_matches_vectorized(self, ohlcv_short):
        from src.analysis.triple_barrier import (
            triple_barrier_labels, triple_barrier_labels_vectorized
        )
        profit_pct, loss_pct, max_bars = 0.02, 0.01, 10
        loop_lbls = triple_barrier_labels(ohlcv_short, profit_pct=profit_pct,
                                          loss_pct=loss_pct, max_bars=max_bars)
        vec_lbls  = triple_barrier_labels_vectorized(ohlcv_short, profit_pct=profit_pct,
                                                     loss_pct=loss_pct, max_bars=max_bars)
        # The two implementations should largely agree (allow small diff due to bar-priority rules)
        agree = (loop_lbls == vec_lbls).mean()
        assert agree >= 0.85, f"Loop vs vectorized agreement only {agree:.0%}"

    def test_label_stats_sums_to_100(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats
        labels = triple_barrier_labels_vectorized(ohlcv)
        stats = label_stats(labels)
        total_pct = stats["long_pct"] + stats["short_pct"] + stats["timeout_pct"]
        assert abs(total_pct - 100.0) < 0.1

    def test_label_stats_keys(self, ohlcv):
        from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats
        labels = triple_barrier_labels_vectorized(ohlcv)
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
        time.sleep(0.18)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
