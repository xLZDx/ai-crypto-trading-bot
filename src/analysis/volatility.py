"""
Volatility forecasting using GARCH(1,1).
Used to predict sharp volatility spikes to adapt risk and leverage.
"""
import pandas as pd
import numpy as np
import logging
from arch import arch_model

logger = logging.getLogger(__name__)

class VolatilityForecaster:
    def __init__(self):
        pass

    def forecast_garch(self, returns: pd.Series, horizon: int = 1) -> dict:
        """
        Fits a GARCH(1,1) model on returns and forecasts the next step.
        """
        if len(returns) < 100:
            logger.warning("Not enough data to fit GARCH (need 100+ points).")
            return {"forecast_volatility": np.std(returns) if len(returns)>0 else 0, "volatility_spike": False, "status": "insufficient_data"}
            
        # Rescale returns for better optimizer convergence (standard practice for arch)
        returns_scaled = returns * 100
        
        try:
            model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
            res = model.fit(disp='off')
            
            forecast = res.forecast(horizon=horizon)
            # arch returns variance, take sqrt to get volatility, then descale
            pred_var = forecast.variance.iloc[-1].values[0]
            pred_vol = np.sqrt(pred_var) / 100.0
            
            current_vol = np.std(returns)
            
            # Detect if forecasted vol is significantly higher than historical (volatility spike)
            volatility_spike = bool(pred_vol > current_vol * 1.5)
            
            return {
                "forecast_volatility": pred_vol,
                "historical_volatility": current_vol,
                "volatility_spike": volatility_spike,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"GARCH fitting failed: {e}")
            return {"forecast_volatility": np.std(returns), "volatility_spike": False, "status": "failed"}
