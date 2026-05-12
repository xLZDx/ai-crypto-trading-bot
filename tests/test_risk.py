"""
Phase G tests — risk management behavioral coverage.

Covers:
  G1 - calc_liquidation_price  (long/short, fees, funding, cross-margin assertion)
  G2 - HullRiskManager.size_from_stop_distance  (sizing, cap, edge cases)
  G3 - live_funding TTL cache  (hit/miss, expiry, concurrent threads, fail-closed)
  G4 - FuturesAgent._on_signal gates  (liq-too-close, funding unavailable, adverse funding)
"""
from __future__ import annotations

import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch


# ─── G1: calc_liquidation_price ──────────────────────────────────────────────

class TestCalcLiquidationPrice(unittest.TestCase):

    def _fn(self):
        from src.analysis.risk_manager import calc_liquidation_price
        return calc_liquidation_price

    # ── correctness (no fees, no funding) ────────────────────────────────────

    def test_long_no_fees_basic(self):
        fn = self._fn()
        # entry=100, 10×, no fees/funding: liq ≈ 100*(1-0.1+0) / (1-0.005-0) ≈ 90.45
        liq = fn(100.0, 10.0, "long", maint_margin_rate=0.005,
                 taker_fee_rate=0.0, accumulated_funding=0.0)
        expected = 100.0 * (1 - 1 / 10) / (1 - 0.005)
        self.assertAlmostEqual(liq, expected, places=6)

    def test_short_no_fees_basic(self):
        fn = self._fn()
        liq = fn(100.0, 10.0, "short", maint_margin_rate=0.005,
                 taker_fee_rate=0.0, accumulated_funding=0.0)
        expected = 100.0 * (1 + 1 / 10) / (1 + 0.005)
        self.assertAlmostEqual(liq, expected, places=6)

    def test_long_liq_below_entry(self):
        fn = self._fn()
        liq = fn(100.0, 2.0, "long")
        self.assertLess(liq, 100.0, "Long liq must be below entry")

    def test_short_liq_above_entry(self):
        fn = self._fn()
        liq = fn(100.0, 2.0, "short")
        self.assertGreater(liq, 100.0, "Short liq must be above entry")

    # ── fee impact ────────────────────────────────────────────────────────────

    def test_fees_move_liq_closer(self):
        """Fees consume margin → liq moves closer to entry."""
        fn = self._fn()
        liq_nofee = fn(100.0, 10.0, "long", taker_fee_rate=0.0)
        liq_fee   = fn(100.0, 10.0, "long", taker_fee_rate=0.0004)
        # With fees, available margin is smaller → liq closer to entry
        self.assertGreater(liq_fee, liq_nofee,
                           "Fees should move long liq closer (higher liq price)")

    # ── funding impact ────────────────────────────────────────────────────────

    def test_positive_funding_moves_long_liq_closer(self):
        """Longs paying positive funding have less margin → liq closer."""
        fn = self._fn()
        liq_no  = fn(100.0, 10.0, "long", accumulated_funding=0.0)
        liq_acc = fn(100.0, 10.0, "long", accumulated_funding=0.5)
        self.assertGreater(liq_acc, liq_no)

    def test_positive_funding_moves_short_liq_farther(self):
        """Shorts receiving positive funding (positive acc_funding) have MORE margin buffer.
        Short liq is above entry; more buffer pushes liq even higher (farther from entry).
        """
        fn = self._fn()
        liq_no  = fn(100.0, 10.0, "short", accumulated_funding=0.0)
        liq_acc = fn(100.0, 10.0, "short", accumulated_funding=0.5)
        # More margin buffer → liq farther above entry → higher liq price
        self.assertGreater(liq_acc, liq_no)

    # ── leverage sensitivity ──────────────────────────────────────────────────

    def test_higher_leverage_brings_liq_closer_long(self):
        fn = self._fn()
        liq_2x  = fn(100.0, 2.0,  "long")
        liq_10x = fn(100.0, 10.0, "long")
        # Both are below entry; 10× liq is closer (higher price)
        self.assertGreater(liq_10x, liq_2x)

    def test_higher_leverage_brings_liq_closer_short(self):
        fn = self._fn()
        liq_2x  = fn(100.0, 2.0,  "short")
        liq_10x = fn(100.0, 10.0, "short")
        # Both are above entry; 10× liq is closer (lower price)
        self.assertLess(liq_10x, liq_2x)

    # ── case-insensitive side ─────────────────────────────────────────────────

    def test_side_case_insensitive(self):
        fn = self._fn()
        self.assertAlmostEqual(fn(100.0, 5.0, "LONG"),  fn(100.0, 5.0, "long"),  places=8)
        self.assertAlmostEqual(fn(100.0, 5.0, "SHORT"), fn(100.0, 5.0, "short"), places=8)

    # ── cross-margin assertion ────────────────────────────────────────────────

    def test_cross_margin_raises_value_error(self):
        fn = self._fn()
        with self.assertRaises(ValueError):
            fn(100.0, 5.0, "long", margin_type="cross")

    # ── bad input raises ValueError ───────────────────────────────────────────

    def test_entry_zero_raises(self):
        fn = self._fn()
        with self.assertRaises(ValueError):
            fn(0.0, 5.0, "long")

    def test_entry_negative_raises(self):
        fn = self._fn()
        with self.assertRaises(ValueError):
            fn(-1.0, 5.0, "long")

    def test_leverage_zero_raises(self):
        fn = self._fn()
        with self.assertRaises(ValueError):
            fn(100.0, 0.0, "long")

    def test_bad_side_raises(self):
        fn = self._fn()
        with self.assertRaises(ValueError):
            fn(100.0, 5.0, "buy")

    # ── realistic scenario ────────────────────────────────────────────────────

    def test_btc_2x_isolated_realistic(self):
        """2× leverage on BTC at $60k with default rates.
        Long liq should be roughly $30k (50% from entry before fees/maint).
        """
        fn = self._fn()
        liq = fn(60_000.0, 2.0, "long")
        # Without any fees: 60000*(1-0.5)/(1-0.005) = ~30150
        self.assertGreater(liq, 28_000.0)
        self.assertLess(liq, 32_000.0)


# ─── G2: size_from_stop_distance ─────────────────────────────────────────────

class TestSizeFromStopDistance(unittest.TestCase):

    def _hrm(self):
        from src.analysis.risk_manager import HullRiskManager
        return HullRiskManager()

    def test_basic_sizing(self):
        """entry=100, stop=95, risk=20 → units=4, notional=400."""
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(100.0, 95.0, 20.0)
        self.assertAlmostEqual(notional, 400.0, places=2)

    def test_short_sizing(self):
        """Short side: entry=100, stop=105 (above), risk=20 → same formula."""
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(100.0, 105.0, 20.0)
        # stop_distance=5, units=4, notional=400
        self.assertAlmostEqual(notional, 400.0, places=2)

    def test_max_notional_cap(self):
        """max_notional_usd caps the return value."""
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(100.0, 95.0, 20.0, max_notional_usd=200.0)
        self.assertAlmostEqual(notional, 200.0, places=2)

    def test_cap_not_triggered(self):
        """Cap above computed value → no effect."""
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(100.0, 95.0, 20.0, max_notional_usd=1000.0)
        self.assertAlmostEqual(notional, 400.0, places=2)

    def test_stop_equal_entry_returns_zero(self):
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(100.0, 100.0, 20.0)
        self.assertEqual(notional, 0.0)

    def test_zero_risk_returns_zero(self):
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(100.0, 95.0, 0.0)
        self.assertEqual(notional, 0.0)

    def test_zero_entry_returns_zero(self):
        hrm = self._hrm()
        notional = hrm.size_from_stop_distance(0.0, 95.0, 20.0)
        self.assertEqual(notional, 0.0)

    def test_returns_float(self):
        hrm = self._hrm()
        result = hrm.size_from_stop_distance(100.0, 90.0, 50.0)
        self.assertIsInstance(result, float)

    def test_larger_stop_distance_smaller_notional(self):
        """Wider stop → fewer units → smaller notional for same risk."""
        hrm = self._hrm()
        n_tight = hrm.size_from_stop_distance(100.0, 99.0, 20.0)  # $1 stop
        n_wide  = hrm.size_from_stop_distance(100.0, 95.0, 20.0)  # $5 stop
        self.assertGreater(n_tight, n_wide)

    def test_proportional_to_risk(self):
        """Doubling risk_usd doubles the notional (all else equal)."""
        hrm = self._hrm()
        n1 = hrm.size_from_stop_distance(100.0, 95.0, 20.0)
        n2 = hrm.size_from_stop_distance(100.0, 95.0, 40.0)
        self.assertAlmostEqual(n2 / n1, 2.0, places=6)


# ─── G3: live_funding TTL cache ───────────────────────────────────────────────

class TestLiveFundingCache(unittest.TestCase):

    def setUp(self):
        from src.analysis import live_funding
        self.mod = live_funding
        self.mod.clear_cache()
        self.mod._exchange = None  # ensure fresh exchange on each test

    def tearDown(self):
        self.mod.clear_cache()
        self.mod._exchange = None

    def _mock_exchange(self, rate: float):
        """Return a mock ccxt exchange whose fetch_funding_rate returns rate."""
        ex = MagicMock()
        ex.fetch_funding_rate.return_value = {"fundingRate": rate}
        return ex

    def test_cache_hit_avoids_second_fetch(self):
        ex = self._mock_exchange(0.0001)
        with patch.object(self.mod, "_get_exchange", return_value=ex):
            r1 = self.mod.fetch_funding_rate("BTC/USDT")
            r2 = self.mod.fetch_funding_rate("BTC/USDT")
        self.assertEqual(r1, 0.0001)
        self.assertEqual(r2, 0.0001)
        # Only one actual network call despite two fetch calls
        self.assertEqual(ex.fetch_funding_rate.call_count, 1)

    def test_cache_miss_after_expiry(self):
        ex = self._mock_exchange(0.0002)
        with patch.object(self.mod, "_get_exchange", return_value=ex):
            r1 = self.mod.fetch_funding_rate("ETH/USDT", ttl_sec=0.05)
            time.sleep(0.1)
            r2 = self.mod.fetch_funding_rate("ETH/USDT", ttl_sec=0.05)
        self.assertEqual(r1, 0.0002)
        self.assertEqual(r2, 0.0002)
        # TTL expired → second fetch triggered
        self.assertEqual(ex.fetch_funding_rate.call_count, 2)

    def test_different_symbols_cached_separately(self):
        ex = MagicMock()
        ex.fetch_funding_rate.side_effect = lambda sym: {
            "fundingRate": 0.0001 if "BTC" in sym else 0.0003
        }
        with patch.object(self.mod, "_get_exchange", return_value=ex):
            btc = self.mod.fetch_funding_rate("BTC/USDT")
            eth = self.mod.fetch_funding_rate("ETH/USDT")
        self.assertAlmostEqual(btc, 0.0001)
        self.assertAlmostEqual(eth, 0.0003)
        self.assertEqual(ex.fetch_funding_rate.call_count, 2)

    def test_fail_closed_returns_none_on_exception(self):
        ex = MagicMock()
        ex.fetch_funding_rate.side_effect = RuntimeError("network error")
        with patch.object(self.mod, "_get_exchange", return_value=ex):
            result = self.mod.fetch_funding_rate("BTC/USDT")
        self.assertIsNone(result)

    def test_clear_cache_forces_refetch(self):
        ex = self._mock_exchange(0.0005)
        with patch.object(self.mod, "_get_exchange", return_value=ex):
            self.mod.fetch_funding_rate("BTC/USDT")
            self.mod.clear_cache()
            self.mod.fetch_funding_rate("BTC/USDT")
        self.assertEqual(ex.fetch_funding_rate.call_count, 2)

    def test_concurrent_threads_single_fetch(self):
        """N concurrent threads should produce exactly 1 network call (lock prevents stampede)."""
        call_count = {"n": 0}
        fetch_lock = threading.Lock()

        def slow_fetch(sym):
            with fetch_lock:
                call_count["n"] += 1
            time.sleep(0.02)
            return {"fundingRate": 0.0001}

        ex = MagicMock()
        ex.fetch_funding_rate.side_effect = slow_fetch

        with patch.object(self.mod, "_get_exchange", return_value=ex):
            threads = [
                threading.Thread(target=self.mod.fetch_funding_rate, args=("BTC/USDT",))
                for _ in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # All 10 threads hit the lock; only 1 gets through (cache is warm for rest)
        self.assertEqual(call_count["n"], 1)

    def test_returns_float_on_success(self):
        ex = self._mock_exchange(0.00015)
        with patch.object(self.mod, "_get_exchange", return_value=ex):
            result = self.mod.fetch_funding_rate("SOL/USDT")
        self.assertIsInstance(result, float)


# ─── G4: FuturesAgent._on_signal gates ───────────────────────────────────────

class TestFuturesAgentGates(unittest.TestCase):
    """Tests exercise _on_signal() via a synthetic bus message."""

    def _make_agent(self, symbols=None):
        from src.engine.agents.futures_agent import FuturesAgent

        fake_df = MagicMock()
        fake_df.__len__ = lambda s: 5
        fake_df.__getitem__ = lambda s, k: MagicMock(iloc=MagicMock(
            __getitem__=lambda ss, i: 45_000.0  # current price
        ))
        data_getter = MagicMock(return_value=fake_df)

        agent = FuturesAgent(
            symbols=symbols or ["BTC/USDT"],
            data_getter=data_getter,
            bus=None,
        )
        # Silence model load errors in tests
        agent._futures_model = None
        agent.publish = MagicMock()
        return agent

    def _msg(self, payload: dict):
        msg = MagicMock()
        msg.payload = payload
        return msg

    def _good_payload(self, direction=1, confidence=0.75, price=45_000.0):
        return {
            "symbol": "BTC/USDT",
            "direction": direction,
            "confidence": confidence,
            "meta_pass": True,
            "regime": 0,
            "raw_signals": {"liq_proximity": 0.0, "signal_funding": 0.0},
            "price": price,
        }

    def test_signal_passes_with_healthy_conditions(self):
        agent = self._make_agent()
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(self._good_payload()))
        agent.publish.assert_called_once()

    def test_blocked_when_funding_unavailable(self):
        """Fail-closed: None funding rate must block the signal."""
        agent = self._make_agent()
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=None):
            agent._on_signal(self._msg(self._good_payload()))
        agent.publish.assert_not_called()

    def test_blocked_on_adverse_funding_long(self):
        """High positive funding → headwind for longs → block."""
        agent = self._make_agent()
        # 0.4% funding is above _MAX_ADVERSE_FUNDING=0.003 for a long trade
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.004):
            agent._on_signal(self._msg(self._good_payload(direction=1)))
        agent.publish.assert_not_called()

    def test_not_blocked_on_favorable_funding_long(self):
        """Negative funding (longs receive) → favorable for longs → allow."""
        agent = self._make_agent()
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=-0.001):
            agent._on_signal(self._msg(self._good_payload(direction=1)))
        agent.publish.assert_called_once()

    def test_blocked_on_liq_too_close(self):
        """If liq price is within 3% of entry, block the signal."""
        agent = self._make_agent()
        # At 100× leverage the liq is very close to entry
        payload = self._good_payload(price=45_000.0)
        # Patch calc_liquidation_price to return a price within 1% of entry
        close_liq = 45_000.0 * 0.98  # 2% away — below the 3% threshold
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001), \
             patch("src.engine.agents.futures_agent.calc_liquidation_price",
                   return_value=close_liq):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_not_called()

    def test_not_blocked_when_liq_safely_far(self):
        """Liq 20% from entry → allow."""
        agent = self._make_agent()
        payload = self._good_payload(price=45_000.0)
        safe_liq = 45_000.0 * 0.75  # 25% away — well above the 3% threshold
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001), \
             patch("src.engine.agents.futures_agent.calc_liquidation_price",
                   return_value=safe_liq):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_called_once()

    def test_blocked_on_meta_pass_false(self):
        """meta_pass=False should block regardless of funding."""
        agent = self._make_agent()
        payload = self._good_payload()
        payload["meta_pass"] = False
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_not_called()

    def test_blocked_on_low_confidence(self):
        agent = self._make_agent()
        payload = self._good_payload(confidence=0.50)
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_not_called()

    def test_published_payload_includes_leverage(self):
        """Published signal must carry leverage field."""
        agent = self._make_agent()
        safe_liq = 45_000.0 * 0.5
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001), \
             patch("src.engine.agents.futures_agent.calc_liquidation_price",
                   return_value=safe_liq):
            agent._on_signal(self._msg(self._good_payload()))
        call_kwargs = agent.publish.call_args
        published_payload = call_kwargs[0][1]
        self.assertIn("leverage", published_payload)
        from src.engine.agents.futures_agent import LEVERAGE
        self.assertEqual(published_payload["leverage"], LEVERAGE)

    def test_blocked_on_adverse_funding_short(self):
        """High negative funding → headwind for shorts → block."""
        agent = self._make_agent()
        # -0.4% funding is below -_MAX_ADVERSE_FUNDING=-0.003 for a short trade
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=-0.004):
            agent._on_signal(self._msg(self._good_payload(direction=-1)))
        agent.publish.assert_not_called()

    def test_symbol_not_in_symbols_list_ignored(self):
        """Signals for symbols the agent doesn't track must be silently ignored."""
        agent = self._make_agent(symbols=["ETH/USDT"])
        # BTC/USDT is NOT in agent.symbols
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(self._good_payload()))  # symbol="BTC/USDT"
        agent.publish.assert_not_called()

    def test_blocked_on_direction_zero(self):
        """direction=0 (no signal) must not trigger a publish."""
        agent = self._make_agent()
        payload = self._good_payload(direction=0)
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_not_called()

    def test_blocked_on_high_liq_proximity(self):
        """liq_proximity > 0.90 in raw_signals triggers the liquidity-sweep guard."""
        agent = self._make_agent()
        payload = self._good_payload()
        payload["raw_signals"]["liq_proximity"] = 0.95
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_not_called()

    def test_blocked_in_volatile_regime_without_funding_signal(self):
        """regime=2 (VOLATILE) with weak funding signal must be blocked."""
        agent = self._make_agent()
        payload = self._good_payload()
        payload["regime"] = 2
        payload["raw_signals"]["signal_funding"] = 0.3  # abs < 0.5 → no funding arb
        with patch("src.engine.agents.futures_agent.fetch_funding_rate",
                   return_value=0.0001):
            agent._on_signal(self._msg(payload))
        agent.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
