import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Centralized data store for preprocessed features (OFI, Sentiment, GARCH volatility, OU mean reversion).
    Provides the InferenceEngine and ML predictors with clean, normalized DataFrames.
    """
    def __init__(self):
        self.data = {}

    def update_features(self, symbol: str, ohlcv_df: pd.DataFrame, sentiment: float, volatility: float, ou_signal: float):
        """Updates the store with the latest market data and analytical signals."""
        try:
            df = ohlcv_df.copy()
            
            # Add external signals as columns
            df['sentiment_score'] = sentiment
            df['garch_volatility'] = volatility
            df['ou_mean_reversion'] = ou_signal
            
            # Calculate basic OFI (Order Flow Imbalance) proxy based on volume & price action
            if 'volume' in df.columns:
                df['volume_delta'] = df['volume'].diff().fillna(0)
                df['ofi'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                            np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()
            else:
                df['ofi'] = 0.0

            self.data[symbol] = df.tail(1000) # Maintain the required window for TFT (1000 chunks)
        except Exception as e:
            logger.error(f"Error updating FeatureStore for {symbol}: {e}")

    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        return self.data.get(symbol, pd.DataFrame())