import math
import logging
import pandas as pd
from typing import List, Dict
from src.analysis.volatility import VolatilityForecaster

logger = logging.getLogger(__name__)

class HullRiskManager:
    """
    Risk management module based on John Hull's concepts.
    ("Options, Futures, and Other Derivatives").
    """
    
    def __init__(self, default_risk_usd: float = 20.0):
        self.default_risk_usd = default_risk_usd
        self.vol_forecaster = VolatilityForecaster()

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
