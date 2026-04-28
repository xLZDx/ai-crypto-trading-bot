import os
import sys
import json
import logging
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_base')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features, add_telegram_signal,
    add_news_sentiment
)


def prepare_data(filepath):
    log.info("Loading data from %s...", filepath)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    log.info("Engineering features...")
    df['return'] = df['close'].pct_change()
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['volatility'] = df['return'].rolling(window=7).std()
    df['dist_sma_7'] = df['close'] / df['sma_7'] - 1
    df['dist_sma_30'] = df['close'] / df['sma_30'] - 1

    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_bollinger_bands(df, window=20)
    df = add_roc(df, [3, 7, 14])
    df = add_time_features(df)
    df = add_taker_and_trade_features(df)

    df['vol_sma_14'] = df['volume'].rolling(window=14).mean()
    df['volume_momentum'] = df['volume'] / df['vol_sma_14']

    df['high_14'] = df['high'].rolling(window=14).max()
    df['low_14'] = df['low'].rolling(window=14).min()
    hl_diff = (df['high_14'] - df['low_14']).replace(0, 0.0001)
    df['stoch_k'] = (df['close'] - df['low_14']) / hl_diff * 100

    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['return_lag3'] = df['return'].shift(3)
    df['return_lag5'] = df['return'].shift(5)
    df['atr_pct'] = (df['high'] - df['low']) / df['close']

    # News sentiment from cryptocompare_news.csv (replaces always-zero telegram_analytics)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    news_path = os.path.join(base_dir, 'data', 'raw', 'cryptocompare_news.csv')
    df = add_news_sentiment(df, news_path)

    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    df = df.dropna()
    return df

def train_model():
    # Construct absolute path to the data downloaded by historical_backfill.py
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        import json
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT', 'ETH_USDT']
        
    timeframe = '1h' # USING 1h, AS THE BOT TRADES ON HOURLY CANDLES!
    
    all_data = []
    for sym in symbols:
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_{timeframe}.csv.gz')
        # Prefer archive downloader format if standard file missing (more historical data)
        archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_{timeframe}.csv.gz')
        if not os.path.exists(full_data_path) and os.path.exists(archive_path):
            full_data_path = archive_path
            log.info("  Using archive data for %s: %s", sym, archive_path)
        if not os.path.exists(full_data_path):
            log.warning("Data for %s not found at %s. Auto-downloading...", sym, full_data_path)
            import sys
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe=timeframe, days=6*365)
            
        if os.path.exists(full_data_path):
            log.info("Processing %s...", sym)
            df = prepare_data(full_data_path)
            all_data.append(df)

    if not all_data:
        log.error("No data found to train the model even after attempted download.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    feature_columns = ['return', 'volatility', 'dist_sma_7', 'dist_sma_30', 'rsi_14', 'macd', 'macd_hist', 'volume_momentum', 'stoch_k', 'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5', 'atr_pct', 'taker_buy_ratio', 'avg_trade_size', 'hour', 'day_of_week', 'roc_14', 'roc_3', 'roc_7', 'bb_pb', 'news_sentiment']
    X = combined_df[feature_columns]
    y = combined_df['target']

    log.info("Dataset: %d total samples | %d features | symbols: %s | timeframe: %s", len(combined_df), len(feature_columns), symbols, timeframe)
    log.info("Splitting data (90%% train / 10%% test, no shuffle)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    log.info("Training HistGradientBoosting Classifier (max_iter=500, early_stopping)...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=500, max_depth=6, learning_rate=0.03, l2_regularization=0.5, early_stopping=True, class_weight='balanced')
    model.fit(X_train, y_train)
    n_iter = getattr(model, 'n_iter_', 500)
    log.info("Training complete — used %d iterations", n_iter)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100

    log.info("Base Model Accuracy: %.2f%%  |  Long precision: %.2f%%  |  Short precision: %.2f%%", accuracy * 100, long_acc, short_acc)

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'btc_rf_model.joblib')
    joblib.dump(model, model_path)
    log.info("Model saved -> %s", model_path)

    from src.utils.safe_json import write_json
    from datetime import datetime, timezone
    meta_path = os.path.join(models_dir, 'btc_rf_model_meta.json')
    write_json(meta_path, {
        "model": "Base (HistGBT)", "accuracy": accuracy * 100,
        "long_accuracy": long_acc, "short_accuracy": short_acc,
        "n_samples": len(combined_df), "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(feature_columns), "n_iterations": n_iter,
        "symbols": symbols, "timeframe": timeframe,
        "last_trained": datetime.now(timezone.utc).isoformat()
    })

if __name__ == "__main__":
    train_model()