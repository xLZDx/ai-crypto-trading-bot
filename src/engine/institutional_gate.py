"""
InstitutionalGate — Phase 9 unified wrapper around §1-18 of the architecture plan.

The live `main.py` calls this gate at four points:

    1. Before every trade:   gate.pre_trade_check(symbol, side, notional)
    2. When sizing:          gate.compute_position_size(symbol, p_win, scenarios)
    3. When placing order:   gate.executed_price(mid, side, size, book_volume)
    4. After entry, periodically:
                             gate.should_exit(signal, time_in_trade)

Each method is a thin facade over the modules built in Phases 1-5:

    §11 inventory hedging             — applied via shaped_reward (RL only)
    §12 alpha decay                   — alpha_decay.should_exit()
    §13 CVaR optimizer                — cvar_optimizer.CVaROptimizer
    §14 risk parity / confidence       — cvar_optimizer.risk_parity_weights
    §15 dynamic threshold             — dynamic_threshold.find_best_threshold
    §16 slippage / execution cost     — slippage_model.real_price
    §17 beta-neutrality               — beta_neutrality.BetaNeutralityFilter
    §18 circuit breakers              — order_manager.circuit_breaker_check

If a module isn't initialised yet (e.g. β-filter has no history), the
gate fails-OPEN — the existing legacy logic still runs. This makes the
integration risk-free: at worst the bot behaves as it did before.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


class InstitutionalGate:
    """Single entry-point used by the live trader for every trade decision."""

    def __init__(
        self,
        order_manager,
        *,
        peak_equity: float = 0.0,
        max_daily_drawdown_pct: float = 0.05,
        max_api_latency_ms: float = 500.0,
        max_data_staleness_sec: float = 30.0,
        decay_rate: float = 0.1,
        decay_exit_threshold: float = 0.2,
    ):
        self.om = order_manager
        self.peak_equity = float(peak_equity)
        self.max_daily_dd = float(max_daily_drawdown_pct)
        self.max_latency = float(max_api_latency_ms)
        self.max_staleness = float(max_data_staleness_sec)
        self.decay_rate = float(decay_rate)
        self.decay_exit = float(decay_exit_threshold)

        self._last_data_ts_unix: float = time.time()
        self._beta_filter = None  # set by attach_beta_filter()

        self._cvar = None  # lazy

    # ── Hot-path setters from main.py loop ──────────────────────────────

    def update_peak_equity(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, float(equity))

    def mark_data_tick(self) -> None:
        self._last_data_ts_unix = time.time()

    def attach_beta_filter(self, history_returns, *, factor: str = "BTC/USDT",
                           max_beta_exposure: float = 1.0) -> None:
        try:
            from src.analysis.beta_neutrality import BetaNeutralityFilter
            self._beta_filter = BetaNeutralityFilter(
                history_returns, factor=factor, max_beta_exposure=max_beta_exposure,
            )
            logger.info("[gate] beta filter attached, factor=%s cap=%.2f",
                        factor, max_beta_exposure)
        except Exception as exc:
            logger.warning("[gate] could not attach beta filter: %s", exc)

    def update_position(self, symbol: str, side: str, notional: float) -> None:
        if self._beta_filter is not None:
            try:
                self._beta_filter.update_position(symbol, side, notional)
            except Exception:
                pass

    # ── §18 circuit breakers + §17 beta neutrality combined gate ────────

    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        notional: float,
        *,
        current_equity: float,
        api_latency_ms: float = 0.0,
    ) -> dict:
        """Return {ok, reasons[], details}. Call before every order."""
        reasons: list[str] = []

        # §18 circuit breakers
        try:
            cb = self.om.circuit_breaker_check(
                peak_equity=self.peak_equity,
                current_equity=current_equity,
                api_latency_ms=api_latency_ms,
                last_data_ts_unix=self._last_data_ts_unix,
                now_unix=time.time(),
                max_daily_drawdown_pct=self.max_daily_dd,
                max_api_latency_ms=self.max_latency,
                max_data_staleness_sec=self.max_staleness,
            )
            if not cb["ok"]:
                reasons.append(f"circuit_breaker:{cb['trigger']}:{cb['reason']}")
        except Exception as exc:
            logger.debug("[gate] circuit_breaker_check unavailable: %s", exc)

        # §17 beta neutrality
        if self._beta_filter is not None:
            try:
                if self._beta_filter.would_breach(symbol, side.lower(), notional):
                    snap = self._beta_filter.snapshot()
                    reasons.append(f"beta_neutrality:would_push_to_{snap.aggregate_beta:+.2f}")
            except Exception as exc:
                logger.debug("[gate] beta check failed: %s", exc)

        return {"ok": not reasons, "reasons": reasons}

    # ── §13-14 sizing ────────────────────────────────────────────────────

    def cvar_size(
        self,
        symbols: list[str],
        scenario_returns: np.ndarray,
        p_wins: list[float],
        *,
        capital_usd: float,
        alpha: float = 0.05,
        lam: float = 1.0,
        leverage_cap: float = 1.0,
        box_max: float = 0.4,
    ) -> dict[str, float]:
        """Return {symbol -> notional_usd} via CVaR optimisation. Falls back
        to equal-weight if CVXPY fails (e.g. solver unavailable).
        """
        try:
            from src.analysis.cvar_optimizer import CVaROptimizer, risk_parity_weights
            from src.analysis.kelly_criterion import kelly_weight_prior
        except Exception as exc:
            logger.debug("[gate] cvar deps unavailable: %s", exc)
            return {s: capital_usd / max(len(symbols), 1) for s in symbols}

        try:
            scen = np.asarray(scenario_returns, dtype=float)
            asset_vol = scen.std(axis=0) + 1e-9
            import pandas as pd
            corr = pd.DataFrame(scen).corr().to_numpy()
            prior = risk_parity_weights(np.asarray(p_wins, dtype=float),
                                        asset_vol, corr)
            kelly = kelly_weight_prior(p_wins)
            prior = (prior + kelly * np.sign(prior)) / 2.0

            opt = CVaROptimizer(alpha=alpha, lam=lam,
                                leverage_cap=leverage_cap, box_max=box_max)
            res = opt.fit(scen, prior_weights=prior)
            return {sym: float(w) * capital_usd
                    for sym, w in zip(symbols, res.weights)}
        except Exception as exc:
            logger.warning("[gate] CVaR sizing failed (%s) — equal weight", exc)
            return {s: capital_usd / max(len(symbols), 1) for s in symbols}

    # ── §15 dynamic threshold ────────────────────────────────────────────

    def best_threshold(self, probs, returns) -> float:
        try:
            from src.analysis.dynamic_threshold import find_best_threshold
            return float(find_best_threshold(probs, returns).best_threshold)
        except Exception:
            return 0.5

    # ── §16 slippage-aware executed price ────────────────────────────────

    def executed_price(self, p_mid: float, side: str,
                       size: float, book_volume: float,
                       *, fee_bps: float = 10.0,
                       lambda_impact: float = 0.5) -> float:
        try:
            from src.analysis.slippage_model import real_price
            return real_price(p_mid, side, size, book_volume,
                              fee_bps=fee_bps, lambda_impact=lambda_impact)
        except Exception:
            return p_mid * (1 + (fee_bps / 1e4)) if side == "buy" \
                else p_mid * (1 - (fee_bps / 1e4))

    # ── §12 alpha decay ──────────────────────────────────────────────────

    def should_exit_decay(self, signal_strength: float,
                          time_in_trade_bars: float) -> bool:
        try:
            from src.analysis.alpha_decay import should_exit
            return should_exit(signal_strength, time_in_trade_bars,
                               decay_rate=self.decay_rate,
                               exit_threshold=self.decay_exit)
        except Exception:
            return False


__all__ = ["InstitutionalGate"]
