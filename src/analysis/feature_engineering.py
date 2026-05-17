"""Shared feature engineering functions used by all ML models and the predictor."""
import os
import pandas as pd
import logging


def add_l2_features(df: pd.DataFrame, symbol: str, freq_ms: int) -> pd.DataFrame:
    """Attach L2/microstructure features to a bar-indexed dataframe.

    2026-05-15 operator request — feed L2 microstructure into the GBT
    trainers (base/trend/futures/scalping/meta). When the L2 parquet store
    is empty for `symbol`, all columns are filled with 0.0 so the trainer
    keeps a stable feature schema and can opportunistically benefit as the
    store accumulates."""
    df = df.copy()
    try:
        from src.analysis.l2_feature_loader import (
            load_bar_aligned, L2_FEATURE_COLUMNS,
        )
    except Exception:
        return df
    ts_col = 'timestamp' if 'timestamp' in df.columns else None
    if ts_col is None:
        for c in L2_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
        return df
    try:
        ts_ms = (pd.to_datetime(df[ts_col]).view('int64') // 1_000_000).tolist()
        closes = df['close'].tolist() if 'close' in df.columns else None
        feats = load_bar_aligned(symbol, ts_ms, freq_ms, closes=closes)
        for c in L2_FEATURE_COLUMNS:
            df[c] = feats[c].values if c in feats.columns else 0.0
    except Exception:
        for c in L2_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
    return df


def add_news_features(df: pd.DataFrame, symbol: str, freq_ms: int) -> pd.DataFrame:
    """Attach bar-aligned news sentiment features from
    data/parquet/_NEWS/news/. Stable-schema contract: every column is
    filled with 0.0 when no data is present."""
    df = df.copy()
    try:
        from src.analysis.news_feature_loader import (
            load_bar_aligned, NEWS_FEATURE_COLUMNS,
        )
    except Exception:
        return df
    ts_col = 'timestamp' if 'timestamp' in df.columns else None
    if ts_col is None:
        for c in NEWS_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
        return df
    try:
        ts_ms = (pd.to_datetime(df[ts_col]).view('int64') // 1_000_000).tolist()
        feats = load_bar_aligned(symbol, ts_ms, freq_ms)
        for c in NEWS_FEATURE_COLUMNS:
            df[c] = feats[c].values if c in feats.columns else 0.0
    except Exception:
        for c in NEWS_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
    return df


def add_tick_features(df: pd.DataFrame, symbol: str, freq_ms: int) -> pd.DataFrame:
    """Attach 1s-derived microstructure features to a bar-indexed dataframe.

    2026-05-15 — the project's archives are at 1-second resolution; this
    function aggregates them into the target-bar timeline as tick-like
    intensity features (taker imbalance, volume concentration, intra-bar
    range, signed volume drift). Stable-schema contract."""
    df = df.copy()
    try:
        from src.analysis.tick_feature_loader import (
            load_bar_aligned, TICK_FEATURE_COLUMNS,
        )
    except Exception:
        return df
    ts_col = 'timestamp' if 'timestamp' in df.columns else None
    if ts_col is None:
        for c in TICK_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
        return df
    try:
        ts_ms = (pd.to_datetime(df[ts_col]).view('int64') // 1_000_000).tolist()
        feats = load_bar_aligned(symbol, ts_ms, freq_ms)
        for c in TICK_FEATURE_COLUMNS:
            df[c] = feats[c].values if c in feats.columns else 0.0
    except Exception:
        for c in TICK_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
    return df


def freq_to_ms(tf: str) -> int:
    """Canonical TF token → ms duration."""
    return {
        '1m':  60_000,
        '5m':  300_000,
        '15m': 900_000,
        '30m': 1_800_000,
        '1h':  3_600_000,
        '4h':  14_400_000,
        '1d':  86_400_000,
        '1w':  604_800_000,
        '1mo': 2_592_000_000,
    }.get(tf, 3_600_000)


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

def add_ofi(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Order Flow Imbalance — normalized rolling OFI signal in [-1, 1].
    Uses taker_buy_base vs total volume as a kline-level OFI proxy.
    The rolling Z-score form makes it stationary and model-friendly.
    """
    import numpy as np
    if 'taker_buy_base' in df.columns and 'volume' in df.columns:
        taker_sell_base = df['volume'] - df['taker_buy_base']
        raw_ofi = df['taker_buy_base'] - taker_sell_base
        df['ofi'] = raw_ofi
        df['ofi_cumulative'] = raw_ofi.cumsum()
        # Rolling Z-score normalisation — the form used as a model feature
        roll_mean = raw_ofi.rolling(window).mean()
        roll_std = raw_ofi.rolling(window).std().replace(0, 1e-9)
        df['ofi_z'] = ((raw_ofi - roll_mean) / roll_std).clip(-3, 3)
    else:
        df['ofi'] = 0.0
        df['ofi_cumulative'] = 0.0
        df['ofi_z'] = 0.0
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session VWAP anchored to midnight UTC.
    vwap_dist = (close - vwap) / vwap — directional distance from institutional reference.
    Positive → price above VWAP (buyers in control), Negative → below (sellers).
    """
    import numpy as np
    if 'timestamp' not in df.columns:
        df['vwap'] = df['close']
        df['vwap_dist'] = 0.0
        return df

    ts = pd.to_datetime(df['timestamp'])
    session = ts.dt.floor('D')  # anchor to day

    typical = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = typical * df['volume']

    df['vwap'] = (tp_vol.groupby(session).cumsum() /
                  df['volume'].groupby(session).cumsum().replace(0, 1e-9))
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'].replace(0, 1e-9)
    return df


def add_donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Donchian channel breakout signals.
    dn_pos = position of close within the channel [0, 1].
    Classic turtle-trading breakout: buy new N-day high, sell new N-day low.
    """
    df[f'don_upper_{n}'] = df['high'].rolling(n).max()
    df[f'don_lower_{n}'] = df['low'].rolling(n).min()
    dn_range = (df[f'don_upper_{n}'] - df[f'don_lower_{n}']).replace(0, 1e-9)
    df[f'don_pos_{n}'] = (df['close'] - df[f'don_lower_{n}']) / dn_range
    return df


def add_keltner(df: pd.DataFrame, ema_period: int = 20, atr_mult: float = 2.0,
                atr_period: int = 10) -> pd.DataFrame:
    """
    Keltner Channel — ATR-based volatility envelope around EMA.
    kc_pos = position of close within channel [-1, 1].
    Breakout above 1.0 → strong momentum; below -1.0 → strong downside.
    """
    import numpy as np
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    df['kc_upper'] = ema + atr_mult * atr
    df['kc_lower'] = ema - atr_mult * atr
    kc_range = (df['kc_upper'] - df['kc_lower']).replace(0, 1e-9)
    df['kc_pos'] = (df['close'] - df['kc_lower']) / kc_range  # 0=lower, 1=upper
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / ema.replace(0, 1e-9)  # volatility proxy
    return df


def add_funding_zscore(df: pd.DataFrame, window: int = 168,
                       funding_col: str = 'funding_rate') -> pd.DataFrame:
    """
    Funding rate Z-score (rolling 168h = 1 week window).
    High positive Z → longs are overcrowded → fade the long, short bias.
    High negative Z → shorts are overcrowded → fade the short, long bias.
    Used as a regime/crowding signal in all futures models.
    """
    if funding_col in df.columns:
        roll_mean = df[funding_col].rolling(window, min_periods=1).mean()
        roll_std = df[funding_col].rolling(window, min_periods=1).std().replace(0, 1e-9)
        df['funding_z'] = ((df[funding_col] - roll_mean) / roll_std).clip(-4, 4)
        df['funding_positive'] = (df[funding_col] > 0.001).astype(int)  # >0.1% → short bias
        df['funding_negative'] = (df[funding_col] < -0.0005).astype(int)  # < -0.05% → long bias
    else:
        df['funding_z'] = 0.0
        df['funding_positive'] = 0
        df['funding_negative'] = 0
    return df


def add_liquidity_proximity(df: pd.DataFrame, atr_period: int = 14,
                            lookback: int = 48) -> pd.DataFrame:
    """
    Liquidity proximity — approximates distance to nearest liquidation cluster.

    In the absence of a live order book, we use a price-action proxy:
    Large wicks (high-low >> ATR) near round-number levels indicate stop clusters.
    dist_to_liq_zone = (close - nearest_swing_extreme) / ATR

    Negative value → price is very close to a demand zone (buy stops below).
    Positive value → price is very close to a supply zone (sell stops above).
    """
    # ATR-normalised wick size as proxy for stop cluster density
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().replace(0, 1e-9)

    swing_high = df['high'].rolling(lookback).max()
    swing_low = df['low'].rolling(lookback).min()

    dist_to_swing_high = (swing_high - df['close']) / atr
    dist_to_swing_low = (df['close'] - swing_low) / atr

    # Proximity score: how close (in ATR) are we to the nearest extreme?
    df['dist_to_supply'] = dist_to_swing_high.clip(0, 10)   # 0 = at resistance
    df['dist_to_demand'] = dist_to_swing_low.clip(0, 10)    # 0 = at support
    df['liq_proximity'] = (1.0 / (1.0 + df[['dist_to_supply', 'dist_to_demand']].min(axis=1)))
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
    Merges hourly news sentiment into the feature DataFrame.

    Phase 10F: prefers the Parquet news partition (`_NEWS/news/yyyymm=*/`)
    over the legacy `cryptocompare_news.csv` when available. Falls back to
    CSV when Parquet is empty or unreachable.

    Score = (bullish_hits - bearish_hits) / total_hits, resampled to 1h and forward-filled.
    """
    df = df.copy()

    # ── Phase 10F: try Parquet first ──────────────────────────────────────
    # Sentiment-source preference, in order:
    #   1. `tone` column         — populated by GDELT / Reddit /
    #                              CryptoCompare scrapers (our new schema)
    #   2. `score` column        — legacy field
    #   3. `sentiment` column    — older legacy
    #   4. fall back to lexicon-scoring the title text
    # All four cases coexist because partitions can be from different
    # eras / sources; we use whichever signal each row carries.
    try:
        # Phase D — prefer in-memory live news buffer when the bot started
        # one (skips the per-call DuckDB cold-start ~100-500ms). Fall back
        # to the parquet query for trainers / backtests that don't run a
        # buffer thread.
        news = None
        try:
            from src.analysis.live_news_buffer import get_active_buffer
            buf = get_active_buffer()
            if buf is not None:
                snap = buf.get_snapshot()
                if snap is not None and not snap.empty:
                    news = snap.copy()
        except Exception:
            news = None

        if news is None:
            from src.analysis.feature_reader import load_news_recent
            news_rows = load_news_recent(hours=24 * 365)   # 1y window
            if news_rows:
                news = pd.DataFrame(news_rows)
        if news is not None and not news.empty:
            # Pick whichever timestamp the row has (ts / published_at / timestamp)
            for ts_col in ("ts", "published_at", "timestamp"):
                if ts_col in news.columns and pd.api.types.is_numeric_dtype(news[ts_col]):
                    news["_t"] = pd.to_datetime(news[ts_col], unit="ms",
                                                 errors="coerce", utc=True)
                    break
                elif ts_col in news.columns:
                    news["_t"] = pd.to_datetime(news[ts_col],
                                                 errors="coerce", utc=True)
                    break
            else:
                news["_t"] = pd.NaT
            news = news.dropna(subset=["_t"])
            if news.empty:
                raise ValueError("no parseable timestamps in news partition")
            news["_t"] = news["_t"].dt.tz_convert(None)
            # Pick the best sentiment column. tone is float-coded directly,
            # score may be int (Reddit upvotes) → normalise via tanh later.
            if "tone" in news.columns and pd.to_numeric(news["tone"], errors="coerce").abs().sum() > 0:
                news["sentiment"] = pd.to_numeric(news["tone"], errors="coerce").fillna(0)
            elif "score" in news.columns and pd.to_numeric(news["score"], errors="coerce").abs().sum() > 0:
                news["sentiment"] = pd.to_numeric(news["score"], errors="coerce").fillna(0)
                # Reddit scores can be huge — squash to ±1 range so they
                # don't dominate the merged-asof feature.
                import numpy as _np
                news["sentiment"] = _np.tanh(news["sentiment"] / 50.0)
            elif "sentiment" in news.columns and pd.to_numeric(news["sentiment"], errors="coerce").abs().sum() > 0:
                news["sentiment"] = pd.to_numeric(news["sentiment"], errors="coerce").fillna(0)
            else:
                text_col = "title" if "title" in news.columns else (
                    "summary" if "summary" in news.columns else None
                )
                if text_col:
                    news["sentiment"] = news[text_col].fillna("").apply(_score_kw)
                else:
                    news["sentiment"] = 0.0
            hourly = (news.set_index("_t")["sentiment"]
                      .resample("1h").mean()
                      .reset_index()
                      .rename(columns={"_t": "timestamp",
                                       "sentiment": "news_sentiment"}))
            df_ts = pd.to_datetime(df["timestamp"])
            if df_ts.dt.tz is not None:
                df_ts = df_ts.dt.tz_convert(None)
            df["timestamp"] = df_ts
            df = pd.merge_asof(df.sort_values("timestamp"),
                                hourly.sort_values("timestamp"),
                                on="timestamp", direction="backward")
            df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
            return df
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "news_sentiment: parquet path failed (%s) -- falling back to CSV.", exc)

    # ── Legacy CSV fallback ───────────────────────────────────────────────
    if not os.path.exists(news_path):
        df['news_sentiment'] = 0.0
        return df

    return df


_BULL = ['buy', 'bull', 'pump', 'moon', 'breakout', 'rally', 'surge', 'long',
         'gain', 'rise', 'recover', 'support', 'bullish', 'ath', 'upside']
_BEAR = ['sell', 'bear', 'crash', 'dump', 'short', 'drop', 'fall', 'decline',
         'fear', 'loss', 'breakdown', 'risk', 'bearish', 'correction', 'downside']


def _score_kw(text) -> float:
    t = str(text).lower()
    b = sum(1 for kw in _BULL if kw in t)
    s = sum(1 for kw in _BEAR if kw in t)
    total = b + s
    return float(b - s) / total if total > 0 else 0.0


def add_news_sentiment_csv_fallback(df: pd.DataFrame, news_path: str) -> pd.DataFrame:
    """Original CSV-only path, kept for the FinBERT pipeline below."""
    df = df.copy()
    if not os.path.exists(news_path):
        df['news_sentiment'] = 0.0
        return df

    def _score(text: str) -> float:
        return _score_kw(text)

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
            "FinBERT unavailable (%s) -- falling back to keyword sentiment.", e
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
        logging.getLogger(__name__).error("FinBERT pipeline failed (%s) -- falling back.", e)
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


def add_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Ichimoku Kinko Hyo — complete cloud system.

    Columns added:
      ichimoku_tenkan   — Conversion Line (9-bar midpoint)
      ichimoku_kijun    — Base Line (26-bar midpoint)
      ichimoku_senkou_a — Leading Span A (avg of tenkan+kijun, shifted +26)
      ichimoku_senkou_b — Leading Span B (52-bar midpoint, shifted +26)
      ichimoku_chikou   — Lagging Span (close shifted -26)
      signal_ichimoku   — +1 (bullish), -1 (bearish), 0 (neutral)

    Bull signal:
      close > cloud AND tenkan > kijun (TK cross) AND close > chikou (cloud twist)
    Bear signal:
      close < cloud AND tenkan < kijun AND close < chikou
    """
    import numpy as np

    def _midpoint(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
        return (high.rolling(n).max() + low.rolling(n).min()) / 2.0

    df["ichimoku_tenkan"]   = _midpoint(df["high"], df["low"], tenkan)
    df["ichimoku_kijun"]    = _midpoint(df["high"], df["low"], kijun)
    df["ichimoku_senkou_a"] = ((df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2.0).shift(displacement)
    df["ichimoku_senkou_b"] = _midpoint(df["high"], df["low"], senkou_b).shift(displacement)
    df["ichimoku_chikou"]   = df["close"].shift(-displacement)

    close      = df["close"]
    span_a     = df["ichimoku_senkou_a"]
    span_b     = df["ichimoku_senkou_b"]
    cloud_top  = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot  = pd.concat([span_a, span_b], axis=1).min(axis=1)

    above_cloud = close > cloud_top
    below_cloud = close < cloud_bot
    tk_bull = df["ichimoku_tenkan"] > df["ichimoku_kijun"]
    tk_bear = df["ichimoku_tenkan"] < df["ichimoku_kijun"]

    signal = np.where(above_cloud & tk_bull,  1.0,
             np.where(below_cloud & tk_bear, -1.0, 0.0))
    df["signal_ichimoku"] = signal
    return df


def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend — ATR-based trend-following indicator.

    Rules:
      Upper band = (high+low)/2 + multiplier × ATR(period)
      Lower band = (high+low)/2 - multiplier × ATR(period)
      Trend flips when close crosses either band.

    Columns added:
      supertrend        — the trailing stop line
      supertrend_dir    — +1 (uptrend), -1 (downtrend)
      signal_supertrend — +1 (buy), -1 (sell), 0 (hold — no flip this bar)
    """
    import numpy as np

    hl2 = (df["high"] + df["low"]) / 2.0
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Use numpy arrays to avoid pandas copy-on-write issues with iloc assignment
    n = len(df)
    bu = basic_upper.values.copy()
    bl = basic_lower.values.copy()
    fu = bu.copy()
    fl = bl.copy()
    close_arr = df["close"].values
    st  = np.full(n, np.nan)
    dir_arr = np.ones(n, dtype=float)

    for i in range(1, n):
        # Upper band: tighten only; reset if prev close broke above
        if np.isnan(bu[i]) or np.isnan(fu[i - 1]):
            fu[i] = bu[i] if not np.isnan(bu[i]) else fu[i - 1]
        elif bu[i] < fu[i - 1] or close_arr[i - 1] > fu[i - 1]:
            fu[i] = bu[i]
        else:
            fu[i] = fu[i - 1]

        # Lower band: tighten only; reset if prev close broke below
        if np.isnan(bl[i]) or np.isnan(fl[i - 1]):
            fl[i] = bl[i] if not np.isnan(bl[i]) else fl[i - 1]
        elif bl[i] > fl[i - 1] or close_arr[i - 1] < fl[i - 1]:
            fl[i] = bl[i]
        else:
            fl[i] = fl[i - 1]

        # Direction
        prev_d = dir_arr[i - 1]
        if not np.isnan(fu[i - 1]) and close_arr[i] > fu[i - 1]:
            dir_arr[i] = 1
        elif not np.isnan(fl[i - 1]) and close_arr[i] < fl[i - 1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = prev_d

        st[i] = fl[i] if dir_arr[i] == 1 else fu[i]

    df["supertrend"]     = st
    df["supertrend_dir"] = dir_arr

    # Signal: only emit on direction flip (avoids holding stale signal)
    prev_dir = df["supertrend_dir"].shift(1)
    df["signal_supertrend"] = np.where(
        (df["supertrend_dir"] == 1)  & (prev_dir == -1),  1.0,
        np.where(
        (df["supertrend_dir"] == -1) & (prev_dir ==  1), -1.0, 0.0)
    )
    return df


def add_macd_divergence(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                         signal_p: int = 9, lookback: int = 5) -> pd.DataFrame:
    """
    MACD centerline cross + price/MACD divergence signals.

    Columns added:
      macd_cl_cross     — +1 centerline bull cross, -1 bear cross, 0 otherwise
      macd_divergence   — +1 bullish divergence, -1 bearish divergence, 0 none
      signal_macd_div   — combined: +1 bull, -1 bear, 0 neutral
    """
    import numpy as np

    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    sig_line  = macd_line.ewm(span=signal_p, adjust=False).mean()

    # Centerline cross: MACD crosses zero
    prev_macd = macd_line.shift(1)
    cl_cross = np.where((macd_line > 0) & (prev_macd <= 0),  1.0,
               np.where((macd_line < 0) & (prev_macd >= 0), -1.0, 0.0))
    df["macd_cl_cross"] = cl_cross

    # Divergence: price makes new high/low but MACD does not (over lookback bars)
    price_high = df["close"].rolling(lookback).max()
    price_low  = df["close"].rolling(lookback).min()
    macd_high  = macd_line.rolling(lookback).max()
    macd_low   = macd_line.rolling(lookback).min()

    # Bearish: price at new high, MACD below prior high
    bear_div = (df["close"] >= price_high) & (macd_line < macd_high.shift(lookback))
    # Bullish: price at new low, MACD above prior low
    bull_div  = (df["close"] <= price_low)  & (macd_line > macd_low.shift(lookback))

    df["macd_divergence"] = np.where(bull_div, 1.0, np.where(bear_div, -1.0, 0.0))

    # Combined signal: centerline cross OR divergence
    df["signal_macd_div"] = np.where(
        (df["macd_cl_cross"] != 0), df["macd_cl_cross"],
        df["macd_divergence"]
    )
    return df


# ────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Level 1 Data Layer additions
#  Refer to updated_architecture_plan_en.md §2 (L2/L3 features) and §3 (Kalman)
# ────────────────────────────────────────────────────────────────────────────


def add_kalman_close(df: pd.DataFrame, out_col: str = "price_kalman") -> pd.DataFrame:
    """Apply the Kalman filter from the architecture plan to `close`.

    Adds a new column (default `price_kalman`) holding the noise-cleaned
    series. Original `close` is left intact for execution / PnL accounting.
    """
    if "close" not in df.columns or len(df) < 2:
        df[out_col] = df.get("close", 0.0)
        return df
    try:
        from src.analysis.kalman_smoother import smooth_price
        df[out_col] = smooth_price(df["close"].values)
    except Exception as exc:
        logging.getLogger(__name__).warning("Kalman smoothing skipped: %s", exc)
        df[out_col] = df["close"]
    return df


def add_l2_features(
    df: pd.DataFrame,
    *,
    p_bid_col: str = "p_bid",
    p_ask_col: str = "p_ask",
    v_bid_col: str = "v_bid",
    v_ask_col: str = "v_ask",
    prefix: str = "ob_",
) -> pd.DataFrame:
    """Attach `ob_imbalance`, `ob_microprice`, `ob_ofi` columns when bid/ask data is present.

    No-op when the bid/ask columns are absent — keeps candle-only training
    pipelines unchanged. The `ob_` prefix avoids colliding with the existing
    kline-level `ofi` column produced by `add_ofi`.
    """
    try:
        from src.analysis.orderbook_features import add_orderbook_features
    except Exception as exc:
        logging.getLogger(__name__).warning("orderbook_features unavailable: %s", exc)
        return df
    return add_orderbook_features(
        df,
        p_bid_col=p_bid_col,
        p_ask_col=p_ask_col,
        v_bid_col=v_bid_col,
        v_ask_col=v_ask_col,
        prefix=prefix,
    )


def causal_audit(df: pd.DataFrame) -> dict:
    """Detect lookahead-bias hazards in a feature DataFrame.

    Catches the three failure modes called out in the plan §4:
      1. cumulative VWAP without intraday reset
      2. OFI computed against future ticks
      3. timestamp not monotone (rolling windows would peek backward)

    Returns: {ok: bool, warnings: list[str]}
    """
    warnings: list[str] = []

    # 1) timestamps must be monotone non-decreasing
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.is_monotonic_increasing is False:
            warnings.append("timestamps are not monotonically increasing")

    # 2) VWAP must reset intraday — cumulative-from-zero is a leak.
    #    Heuristic: VWAP grows monotonically across day boundaries → suspicious.
    if "vwap" in df.columns and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any() and len(df) > 100:
            day = ts.dt.floor("D")
            day_changes = day.diff().fillna(pd.Timedelta(0)) > pd.Timedelta(0)
            if day_changes.any():
                # vwap right after midnight should drop close to typical price
                resets = (df["vwap"].diff().abs() / df["close"].replace(0, 1e-9))[day_changes]
                if (resets < 1e-4).all():
                    warnings.append("vwap appears cumulative — does not reset intraday")

    # 3) OFI sanity — must change sign at least sometimes
    for col in ("ofi", "ob_ofi"):
        if col in df.columns and df[col].nunique(dropna=True) <= 1 and len(df) > 100:
            warnings.append(f"{col} is constant across {len(df)} rows — possibly stale or non-causal")

    return {"ok": not warnings, "warnings": warnings}


__all__ = [name for name in globals() if name.startswith("add_")] + [
    "add_kalman_close", "add_l2_features", "causal_audit", "normalize_tensors",
]
