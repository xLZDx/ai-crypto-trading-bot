"""
Mean Reversion & Cointegration (Pairs Trading) module.
Implements Ornstein-Uhlenbeck (OU) process and Engle-Granger tests.
"""
import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

logger = logging.getLogger(__name__)

class MeanReversionCore:
    def __init__(self):
        pass

    def calibrate_ou_process(self, prices: np.ndarray, dt: float = 1.0):
        """
        Calibrates the Ornstein-Uhlenbeck (OU) parameters using Maximum Likelihood Estimation (MLE).
        dX_t = theta * (mu - X_t)dt + sigma * dW_t
        """
        if len(prices) < 2:
            return None
            
        x = prices[:-1]
        y = prices[1:]
        
        # Linear regression: y_t = a + b * x_{t-1} + error
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        
        a, b = model.params
        std_resid = np.std(model.resid)
        
        # Avoid division by zero or negative log for b >= 1
        if b >= 1 or b <= 0:
            logger.warning("Prices do not strictly exhibit mean-reversion (b >= 1). Returning flat OU parameters.")
            return {"theta": 0.0, "mu": np.mean(prices), "sigma": np.std(prices), "signal": 0}
            
        theta = -np.log(b) / dt
        mu = a / (1 - b)
        sigma = std_resid / np.sqrt((1 - b**2) / (2 * theta))
        
        current_price = prices[-1]
        deviation = current_price - mu
        
        signal = 0
        if deviation > 2 * sigma:
            signal = -1  # Overbought -> Sell
        elif deviation < -2 * sigma:
            signal = 1   # Oversold -> Buy
            
        return {"theta": theta, "mu": mu, "sigma": sigma, "signal": signal}

    def check_cointegration(self, asset_a: np.ndarray, asset_b: np.ndarray):
        """
        Runs Engle-Granger cointegration test to determine if the spread between two assets
        is stationary and mathematically bound to return to zero.
        """
        if len(asset_a) != len(asset_b) or len(asset_a) < 30:
            return {"is_cointegrated": False, "spread": None}
            
        score, p_value, _ = coint(asset_a, asset_b)
        
        # Calculate hedge ratio via OLS
        model = sm.OLS(asset_a, sm.add_constant(asset_b)).fit()
        hedge_ratio = model.params[1]
        spread = asset_a - hedge_ratio * asset_b
        
        # Signal logic on spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        z_score = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        signal = 0
        if z_score > 2:
            signal = -1 # Asset A overvalued relative to B -> Short A, Long B
        elif z_score < -2:
            signal = 1  # Asset A undervalued relative to B -> Long A, Short B
            
        return {
            "is_cointegrated": p_value < 0.05,
            "p_value": p_value,
            "hedge_ratio": hedge_ratio,
            "z_score": z_score,
            "signal": signal
        }
