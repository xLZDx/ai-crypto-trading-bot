"""Shared feature engineering functions used by all ML models and the predictor."""
import pandas as pd


def add_taker_and_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'taker_buy_base' in df.columns:
        df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, 0.0001)
    else:
        df['taker_buy_ratio'] = 0.5
    if 'trades_count' in df.columns:
        df['avg_trade_size'] = df['volume'] / df['trades_count'].replace(0, 1)
    else:
        df['avg_trade_size'] = 0.0
    return df


def add_rsi(df: pd.DataFrame, period: int, col_name: str = None) -> pd.DataFrame:
    col_name = col_name or f'rsi_{period}'
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    df[col_name] = 100 - (100 / (1 + gain / loss))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
             prefix: str = '') -> pd.DataFrame:
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    df[f'{prefix}macd'] = exp1 - exp2
    df[f'{prefix}macd_signal'] = df[f'{prefix}macd'].ewm(span=signal, adjust=False).mean()
    df[f'{prefix}macd_hist'] = df[f'{prefix}macd'] - df[f'{prefix}macd_signal']
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, prefix: str = '') -> pd.DataFrame:
    sma_col = f'sma_{window}'
    std_col = f'std_{window}'
    df[sma_col] = df['close'].rolling(window=window).mean()
    df[std_col] = df['close'].rolling(window=window).std()
    df[f'{prefix}bb_upper'] = df[sma_col] + 2 * df[std_col]
    df[f'{prefix}bb_lower'] = df[sma_col] - 2 * df[std_col]
    bb_range = (df[f'{prefix}bb_upper'] - df[f'{prefix}bb_lower']).replace(0, 0.0001)
    df[f'{prefix}bb_pb'] = (df['close'] - df[f'{prefix}bb_lower']) / bb_range
    return df


def add_roc(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    for p in periods:
        df[f'roc_{p}'] = df['close'].pct_change(periods=p)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['tr'] = tr
    df[f'atr_{period}'] = tr.rolling(window=period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = add_atr(df, period)
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = df[f'atr_{period}'].replace(0, 0.0001)
    df['plus_di'] = 100 * (plus_dm.rolling(window=period).mean() / atr)
    df['minus_di'] = 100 * (minus_dm.rolling(window=period).mean() / atr)
    di_sum = (df['plus_di'] + df['minus_di']).replace(0, 0.0001)
    df['dx'] = 100 * (df['plus_di'] - df['minus_di']).abs() / di_sum
    df[f'adx_{period}'] = df['dx'].rolling(window=period).mean()
    return df

def add_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Order Flow Imbalance (OFI).
    Uses taker_buy_base vs total volume as an OFI proxy on kline data.
    """
    if 'taker_buy_base' in df.columns and 'volume' in df.columns:
        taker_sell_base = df['volume'] - df['taker_buy_base']
        df['ofi'] = df['taker_buy_base'] - taker_sell_base
        df['ofi_cumulative'] = df['ofi'].cumsum()
    else:
        df['ofi'] = 0.0
        df['ofi_cumulative'] = 0.0
    return df

def normalize_tensors(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Normalizes features using Z-score for TFT/LSTM Deep Learning ingestion.
    """
    if not columns:
        # Normalize all numeric columns except target variables or raw timestamps if they exist
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ('timestamp', 'close_target')]
        
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[f'{col}_norm'] = (df[col] - mean) / std
            else:
                df[f'{col}_norm'] = 0.0
    return df
