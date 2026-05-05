import math
import logging
import pandas as pd
from typing import List, Dict
from src.analysis.volatility import VolatilityForecaster
from src.analysis.kelly_criterion import KellySizer

logger = logging.getLogger(__name__)

class HullRiskManager:
    """
    Risk management module based on John Hull's concepts.
    ("Options, Futures, and Other Derivatives").
    """
    
    def __init__(self, default_risk_usd: float = 20.0):
        self.default_risk_usd = default_risk_usd
        self.vol_forecaster = VolatilityForecaster()
        self._kelly = KellySizer(window=50, half_kelly=True)

    def calculate_historical_volatility(self, data: List[Dict], periods: int = 30) -> float:
        """
        Calculates the historical volatility of the asset (Historical Volatility).
        According to Hull, volatility is the standard deviation of logarithmic returns.
        """
        if len(data) < periods + 1:
            return 0.01 # Default value
            
        recent_data = data[-(periods+1):]
        log_returns = []
        
        for i in range(1, len(recent_data)):
            prev_close = recent_data[i-1]['close']
            curr_close = recent_data[i]['close']
            if prev_close > 0:
                log_return = math.log(curr_close / prev_close)
                log_returns.append(log_return)
                
        if not log_returns:
            return 0.01
            
        mean_return = sum(log_returns) / len(log_returns)
        
        variance_sum = sum((r - mean_return) ** 2 for r in log_returns)
        variance = variance_sum / (len(log_returns) - 1)
        
        daily_volatility = math.sqrt(variance)
        
        # Annualized volatility (assuming 1h candles, 24*365 = 8760 hours in a year)
        annual_volatility = daily_volatility * math.sqrt(8760)
        return annual_volatility

    def get_position_size(self, data: List[Dict]) -> float:
        """
        Dynamic position size based on volatility (Value at Risk concept).
        The higher the volatility (stormy market), the smaller the position to protect capital.
        The lower the volatility, the larger the position.
        """
        volatility = self.calculate_historical_volatility(data)
        
        # Check for GARCH volatility spike
        garch_reduction = 1.0
        if len(data) >= 100:
            returns = pd.Series([d['close'] for d in data]).pct_change().dropna()
            garch_res = self.vol_forecaster.forecast_garch(returns)
            if garch_res.get('volatility_spike', False):
                logger.warning("GARCH predicts sharp volatility spike! Reducing position size by 50%.")
                garch_reduction = 0.5
        
        # Base logic: if volatility is 50% (0.5), risk the base amount.
        # If volatility is 100% (1.0), risk 2x less ($10).
        # If volatility is 25% (0.25), risk 2x more ($40).
        
        baseline_volatility = 0.50 # 50% annual
        
        # Protection against division by zero or abnormally low volatility
        volatility = max(volatility, 0.05) 
        
        scaling_factor = baseline_volatility / volatility
        
        # Limit scaling from 0.25x to 3.0x
        scaling_factor = max(0.25, min(scaling_factor, 3.0))
        
        adjusted_position = self.default_risk_usd * scaling_factor * garch_reduction
        return round(adjusted_position, 2)

    def get_kelly_position_size(
        self,
        capital: float,
        p_win: float,
        data: List[Dict],
    ) -> float:
        """
        Full position sizing: Kelly criterion × volatility scaling × GARCH.
        This is the production-grade sizing method combining all three layers:
          - Kelly: maximizes geometric growth given model confidence
          - Volatility scaling: reduces size when market is turbulent (Hull)
          - GARCH: emergency 50% cut on volatility spike detection

        Args:
            capital: Current account equity in USDT.
            p_win:   Model's P(win) from predict_proba (0–1).
            data:    Recent OHLCV candle list for volatility calculation.

        Returns:
            Position size in USDT.
        """
        # Volatility scaling factor from Hull (0.25–3.0)
        volatility = self.calculate_historical_volatility(data)
        volatility = max(volatility, 0.05)
        baseline_volatility = 0.50
        vol_scale = max(0.25, min(baseline_volatility / volatility, 3.0))

        # GARCH spike reduction
        garch_reduction = 1.0
        if len(data) >= 100:
            returns = pd.Series([d['close'] for d in data]).pct_change().dropna()
            garch_res = self.vol_forecaster.forecast_garch(returns)
            if garch_res.get('volatility_spike', False):
                garch_reduction = 0.5

        # Kelly sizing with combined scale
        combined_scale = vol_scale * garch_reduction
        size = self._kelly.size(capital=capital, p_win=p_win, volatility_scale=combined_scale)
        logger.debug("Kelly size: %.2f USDT | p_win=%.2f | vol_scale=%.2f | garch=%.1f",
                     size, p_win, vol_scale, garch_reduction)
        return size

    def record_trade_outcome(self, pnl: float) -> None:
        """Update Kelly sizer with trade outcome for dynamic win/loss ratio."""
        self._kelly.record_trade(pnl)

    # ── Phase 4: CVaR-driven portfolio sizing ───────────────────────────────

    def cvar_position_weights(
        self,
        symbols: list,
        scenario_returns,                  # (n_scenarios, n_symbols)
        p_wins: list,
        *,
        alpha: float = 0.05,
        lam: float = 1.0,
        leverage_cap: float = 1.0,
        box_max: float = 0.4,
    ) -> dict:
        """Return CVaR-optimised position weights for a basket of assets.

        Per updated_architecture_plan_en.md §13-14, Kelly+vol-scaling becomes
        the *prior* for the CVaR optimizer. The optimizer then shrinks /
        rotates the prior to respect tail-risk and realised correlation.

        Args:
            symbols:          List of symbol identifiers (used for output dict keys).
            scenario_returns: Historical return matrix (rows = time, cols = assets).
            p_wins:           Per-asset model P(win), aligned with `symbols`.

        Returns: {symbol -> weight} (signed; sum |w| ≤ leverage_cap).
        """
        from src.analysis.cvar_optimizer import CVaROptimizer, risk_parity_weights
        from src.analysis.kelly_criterion import kelly_weight_prior
        import numpy as np
        import pandas as pd

        if len(symbols) != len(p_wins):
            raise ValueError("symbols and p_wins length mismatch")

        scenario_returns = np.asarray(scenario_returns, dtype=float)
        # Build the risk-parity / Kelly prior. Vol per asset comes from
        # historical scenarios; correlation penalty from the same.
        asset_vol = scenario_returns.std(axis=0) + 1e-9
        corr = pd.DataFrame(scenario_returns).corr().to_numpy()
        prior = risk_parity_weights(np.asarray(p_wins, dtype=float),
                                    asset_vol, corr)
        kelly = kelly_weight_prior(p_wins)
        prior = (prior + kelly * np.sign(prior)) / 2.0  # blend

        opt = CVaROptimizer(alpha=alpha, lam=lam,
                            leverage_cap=leverage_cap, box_max=box_max)
        result = opt.fit(scenario_returns, prior_weights=prior)
        return {sym: float(w) for sym, w in zip(symbols, result.weights)}
