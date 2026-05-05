import pandas as pd
import numpy as np
import logging

from src.analysis.kalman_smoother import smooth_price
from src.analysis.orderbook_features import add_orderbook_features

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Centralized data store for preprocessed features (OFI, Sentiment, GARCH volatility, OU mean reversion).
    Provides the InferenceEngine and ML predictors with clean, normalized DataFrames.

    Phase 1 upgrade: applies a Kalman filter to `close` (per
    updated_architecture_plan_en.md §3) and surfaces L2 order-book features
    (imbalance, microprice, OFI per §2) when bid/ask columns are present.
    """
    def __init__(self, *, apply_kalman: bool = True):
        self.data = {}
        self.apply_kalman = apply_kalman

    def update_features(self, symbol: str, ohlcv_df: pd.DataFrame, sentiment: float, volatility: float, ou_signal: float):
        """Updates the store with the latest market data and analytical signals."""
        try:
            df = ohlcv_df.copy()

            # Phase 1: Kalman-smoothed close — used as the noise-cleaned base
            # for any downstream technical indicators that need price level.
            # Original `close` is preserved for execution / PnL accounting.
            if self.apply_kalman and 'close' in df.columns and len(df) >= 2:
                df['price_kalman'] = smooth_price(df['close'].values)

            # Phase 1: L2/L3 order-book features (no-op when bid/ask absent)
            df = add_orderbook_features(df)

            # Add external signals as columns
            df['sentiment_score'] = sentiment
            df['garch_volatility'] = volatility
            df['ou_mean_reversion'] = ou_signal

            # Calculate basic OFI (Order Flow Imbalance) proxy based on volume & price action.
            # Note: this is the kline-level proxy retained for legacy ML models. The
            # canonical L2 OFI lives in `add_orderbook_features` (column `ofi`), which
            # overrides this when real bid/ask data is present in the frame.
            if 'ofi' not in df.columns:
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