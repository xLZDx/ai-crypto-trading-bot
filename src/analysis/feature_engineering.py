"""Shared feature engineering functions used by all ML models and the predictor."""
import os
import pandas as pd
import logging


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

def add_telegram_signal(df: pd.DataFrame, telegram_data_path: str) -> pd.DataFrame:
    """
    Merges signals from a Telegram analytics channel into the feature DataFrame.
    """
    df = df.copy()
    if not os.path.exists(telegram_data_path):
        df['telegram_signal'] = 0.0
        return df

    try:
        tg_df = pd.read_csv(telegram_data_path)
        tg_df['timestamp'] = pd.to_datetime(tg_df['timestamp'])
        tg_df.set_index('timestamp', inplace=True)

        # Simple keyword-based signal extraction (EN & RU)
        bullish_keywords = ['buy', 'long', 'pump', 'bull', 'moon', 'up', 'strong buy', 'лонг', 'покупка', 'бычий', 'рост']
        bearish_keywords = ['sell', 'short', 'dump', 'bear', 'crash', 'down', 'strong sell', 'шорт', 'продажа', 'медвежий', 'падение']

        def get_signal(text):
            text_lower = str(text).lower()
            if any(kw in text_lower for kw in bullish_keywords):
                return 1.0
            if any(kw in text_lower for kw in bearish_keywords):
                return -1.0
            return 0.0

        tg_df['telegram_signal'] = tg_df['text'].apply(get_signal)

        # Merge and forward-fill the signal
        df = pd.merge_asof(df.sort_values('timestamp'), tg_df[['telegram_signal']], on='timestamp', direction='backward')
        df['telegram_signal'] = df['telegram_signal'].fillna(0.0)

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to process Telegram signals: {e}")
        df['telegram_signal'] = 0.0

    return df


def add_news_sentiment(df: pd.DataFrame, news_path: str) -> pd.DataFrame:
    """
    Merges hourly news sentiment from cryptocompare_news.csv into the feature DataFrame.
    Score = (bullish_hits - bearish_hits) / total_hits, resampled to 1h and forward-filled.
    """
    df = df.copy()
    if not os.path.exists(news_path):
        df['news_sentiment'] = 0.0
        return df

    _BULL = ['buy', 'bull', 'pump', 'moon', 'breakout', 'rally', 'surge', 'long',
             'gain', 'rise', 'recover', 'support', 'bullish', 'ath', 'upside']
    _BEAR = ['sell', 'bear', 'crash', 'dump', 'short', 'drop', 'fall', 'decline',
             'fear', 'loss', 'breakdown', 'risk', 'bearish', 'correction', 'downside']

    def _score(text: str) -> float:
        t = str(text).lower()
        b = sum(1 for kw in _BULL if kw in t)
        s = sum(1 for kw in _BEAR if kw in t)
        total = b + s
        return float(b - s) / total if total > 0 else 0.0

    try:
        news = pd.read_csv(news_path, usecols=['published_at', 'title', 'summary'])
        news['published_at'] = pd.to_datetime(news['published_at'], utc=True, errors='coerce')
        news = news.dropna(subset=['published_at'])
        news['published_at'] = news['published_at'].dt.tz_convert(None)
        news['text'] = news['title'].fillna('') + ' ' + news['summary'].fillna('')
        news['sentiment'] = news['text'].apply(_score)

        hourly = (news.set_index('published_at')['sentiment']
                  .resample('1h').mean()
                  .reset_index()
                  .rename(columns={'published_at': 'timestamp', 'sentiment': 'news_sentiment'}))

        df_ts = pd.to_datetime(df['timestamp'])
        if df_ts.dt.tz is not None:
            df_ts = df_ts.dt.tz_convert(None)
        df = df.copy()
        df['timestamp'] = df_ts
        df = pd.merge_asof(df.sort_values('timestamp'),
                           hourly.sort_values('timestamp'),
                           on='timestamp', direction='backward')
        df['news_sentiment'] = df['news_sentiment'].fillna(0.0)

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to process news sentiment: {e}")
        df['news_sentiment'] = 0.0

    return df


def add_finbert_sentiment(df: pd.DataFrame, news_path: str, batch_size: int = 64) -> pd.DataFrame:
    """
    Replaces keyword-based sentiment with FinBERT scores (ProsusAI/finbert).
    Falls back to add_news_sentiment() if transformers/torch are unavailable.
    Requires: pip install transformers torch sentencepiece
    Model is ~500MB and downloaded once to ~/.cache/huggingface/
    Score mapped to [-1, 1]: positive→+1, neutral→0, negative→-1
    """
    try:
        from transformers import pipeline as _pipe
        import torch as _torch
        device = 0 if _torch.cuda.is_available() else -1
        _finbert = _pipe(
            "text-classification",
            model="ProsusAI/finbert",
            device=device,
            truncation=True,
            max_length=512,
            top_k=None,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "FinBERT unavailable (%s) — falling back to keyword sentiment.", e
        )
        return add_news_sentiment(df, news_path)

    df = df.copy()
    if not os.path.exists(news_path):
        df['news_sentiment'] = 0.0
        return df

    _LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def _finbert_score(texts):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            try:
                preds = _finbert(batch)
                for pred_list in preds:
                    best = max(pred_list, key=lambda x: x['score'])
                    results.append(_LABEL_MAP.get(best['label'], 0.0))
            except Exception:
                results.extend([0.0] * len(batch))
        return results

    try:
        news = pd.read_csv(news_path, usecols=['published_at', 'title', 'summary'])
        news['published_at'] = pd.to_datetime(news['published_at'], utc=True, errors='coerce')
        news = news.dropna(subset=['published_at'])
        news['published_at'] = news['published_at'].dt.tz_convert(None)
        news['text'] = (news['title'].fillna('') + ' ' + news['summary'].fillna('')).str[:512]

        logging.getLogger(__name__).info("Running FinBERT on %d articles ...", len(news))
        news['sentiment'] = _finbert_score(news['text'].tolist())

        hourly = (news.set_index('published_at')['sentiment']
                  .resample('1h').mean()
                  .reset_index()
                  .rename(columns={'published_at': 'timestamp', 'sentiment': 'news_sentiment'}))

        df_ts = pd.to_datetime(df['timestamp'])
        if df_ts.dt.tz is not None:
            df_ts = df_ts.dt.tz_convert(None)
        df['timestamp'] = df_ts
        df = pd.merge_asof(df.sort_values('timestamp'),
                           hourly.sort_values('timestamp'),
                           on='timestamp', direction='backward')
        df['news_sentiment'] = df['news_sentiment'].fillna(0.0)
        logging.getLogger(__name__).info("FinBERT sentiment merged successfully.")

    except Exception as e:
        logging.getLogger(__name__).error("FinBERT pipeline failed (%s) — falling back.", e)
        return add_news_sentiment(df, news_path)

    return df


def resample_1s_to_1m(filepath_1s: str, max_days: int = 365) -> pd.DataFrame:
    """
    Reads 1s OHLCV data in chunks, keeps only the last max_days days,
    and resamples to 1m candles. Avoids loading multi-GB files into RAM.
    Default 365 days covers Binance archive lag of up to 3 months.
    """
    agg = {
        'open':          'first',
        'high':          'max',
        'low':           'min',
        'close':         'last',
        'volume':        'sum',
        'quote_volume':  'sum',
        'trades_count':  'sum',
        'taker_buy_base': 'sum',
        'taker_buy_quote': 'sum',
    }

    from datetime import datetime, timedelta, timezone as _tz
    cutoff = datetime.now(_tz.utc).replace(tzinfo=None) - timedelta(days=max_days)
    kept_chunks = []

    for chunk in pd.read_csv(filepath_1s, chunksize=500_000):
        if 'timestamp' not in chunk.columns:
            # Try common Binance alternative column names
            for alt in ['open_time', 'Open time', 'date']:
                if alt in chunk.columns:
                    chunk = chunk.rename(columns={alt: 'timestamp'})
                    break
        if 'timestamp' not in chunk.columns:
            continue
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
        chunk = chunk.dropna(subset=['timestamp'])
        if chunk['timestamp'].dt.tz is not None:
            chunk['timestamp'] = chunk['timestamp'].dt.tz_convert(None)
        recent = chunk[chunk['timestamp'] >= cutoff]
        if len(recent) > 0:
            kept_chunks.append(recent)

    if not kept_chunks:
        return pd.DataFrame()

    df = pd.concat(kept_chunks, ignore_index=True).sort_values('timestamp').set_index('timestamp')
    existing_agg = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample('1min').agg(existing_agg).dropna(subset=['close'])
    return resampled.reset_index()
