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
log = logging.getLogger('train_scalping')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features, resample_1s_to_1m
)


def _engineer_scalping_features(df: pd.DataFrame) -> pd.DataFrame:
    df['return'] = df['close'].pct_change()
    df = add_rsi(df, period=7, col_name='rsi_7')
    df = add_macd(df, fast=5, slow=13, signal=3, prefix='')
    df.rename(columns={'macd': 'macd_fast', 'macd_signal': 'macd_fast_signal', 'macd_hist': 'macd_fast_hist'}, errors='ignore', inplace=True)
    df = add_bollinger_bands(df, window=10)
    df = add_roc(df, [3, 5, 10])
    df = add_time_features(df)
    df = add_taker_and_trade_features(df)

    df['vol_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_5'] * 2.0).astype(int)

    df['low_15'] = df['low'].rolling(15).min()
    df['dist_to_micro_supp'] = (df['close'] - df['low_15']) / df['close']

    df['target_scalp'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna()


def prepare_scalping_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return _engineer_scalping_features(df)


def prepare_scalping_data(filepath):
    print(f"Loading data for Scalping Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return _engineer_scalping_features(df)

def train_scalping_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        import json
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']
    
    all_data = []
    for sym in symbols:
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_1m.csv.gz')
        archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1m.csv.gz')
        # 1s data paths — resampled to 1m for richer features and longer history
        path_1s = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1s.csv.gz')
        path_1s_v2 = os.path.join(base_dir, 'data', 'raw', f'{sym}_1s.csv.gz')

        if not os.path.exists(full_data_path) and os.path.exists(archive_path):
            full_data_path = archive_path
            print(f"  Using archive 1m data for {sym}: {archive_path}")

        # Try resampling 1s → 1m when 1s data exists
        df_1s_resampled = None
        for p1s in [path_1s, path_1s_v2]:
            if os.path.exists(p1s):
                try:
                    print(f"  Found 1s data for {sym} at {p1s} — resampling to 1m (last 90 days)...")
                    df_1s_resampled = resample_1s_to_1m(p1s)
                    if df_1s_resampled is None or len(df_1s_resampled) == 0:
                        print(f"  Warning: 1s resample returned no data for {sym}, skipping.")
                        df_1s_resampled = None
                    else:
                        print(f"  Resampled: {len(df_1s_resampled):,} 1m candles from 1s source")
                        break
                except Exception as e:
                    print(f"  Warning: Could not resample 1s data for {sym}: {e}")
                    df_1s_resampled = None

        if not os.path.exists(full_data_path):
            if df_1s_resampled is not None:
                print(f"  Using resampled 1s data as primary source for {sym}")
            else:
                print(f"Warning: 1m Data for {sym} not found. Auto-downloading...")
                import sys
                if base_dir not in sys.path:
                    sys.path.insert(0, base_dir)
                from src.data_ingestion.historical_backfill import backfill_history
                backfill_history(symbol=sym.replace('_', '/'), timeframe='1m', days=70)

        # Build final DataFrame — prefer 1s-resampled if it has more data
        df_combined = None
        if os.path.exists(full_data_path):
            try:
                df_1m = prepare_scalping_data(full_data_path)
            except Exception as e:
                print(f"  Warning: failed to prepare 1m data for {sym}: {e}")
                df_1m = None

            if df_1m is not None and df_1s_resampled is not None:
                try:
                    df_1s_feat = prepare_scalping_data_from_df(df_1s_resampled)
                    if len(df_1s_feat) >= len(df_1m):
                        print(f"  Using 1s-resampled data ({len(df_1s_feat):,} rows) for {sym}")
                        df_combined = df_1s_feat
                    else:
                        print(f"  Using standard 1m data ({len(df_1m):,} rows) for {sym}")
                        df_combined = df_1m
                except Exception as e:
                    print(f"  Warning: 1s feature engineering failed for {sym}: {e}, using 1m")
                    df_combined = df_1m
            elif df_1m is not None:
                df_combined = df_1m

        if df_combined is None and df_1s_resampled is not None:
            try:
                df_combined = prepare_scalping_data_from_df(df_1s_resampled)
            except Exception as e:
                print(f"  Warning: 1s-only feature engineering failed for {sym}: {e}")

        if df_combined is not None:
            df_combined = df_combined.tail(500000)
            all_data.append(df_combined)
            
    if not all_data:
        print("Error: No 1m data found even after attempted download.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    feature_columns = ['return', 'rsi_7', 'macd_fast', 'volume_surge', 'dist_to_micro_supp', 'taker_buy_ratio', 'avg_trade_size', 'hour', 'roc_3', 'roc_5', 'roc_10', 'bb_pb']
    X = combined_df[feature_columns]
    y = combined_df['target_scalp']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    log.info("Scalping dataset: %d total | train %d | test %d | features %d | symbols %s | timeframe 1m",
             len(combined_df), len(X_train), len(X_test), len(feature_columns), symbols)
    log.info("Training Scalping AI Model...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=400, max_depth=5, learning_rate=0.05, early_stopping=True, class_weight='balanced')
    model.fit(X_train, y_train)
    n_iter = getattr(model, 'n_iter_', 400)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc  = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100

    log.info("Scalping Model Accuracy: %.2f%%  |  Long: %.2f%%  |  Short: %.2f%%  |  Iterations: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'scalping_model.joblib')
    joblib.dump(model, model_path)
    log.info("Scalping Model saved -> %s", model_path)

    from datetime import datetime, timezone
    meta_path = os.path.join(models_dir, 'scalping_model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({"model": "Scalping (HistGBT)", "accuracy": accuracy * 100,
                   "long_accuracy": long_acc, "short_accuracy": short_acc,
                   "n_samples": len(combined_df), "n_train": len(X_train), "n_test": len(X_test),
                   "n_features": len(feature_columns), "n_iterations": n_iter,
                   "symbols": symbols, "timeframe": "1m",
                   "last_trained": datetime.now(timezone.utc).isoformat()}, f)

if __name__ == "__main__":
    train_scalping_model()