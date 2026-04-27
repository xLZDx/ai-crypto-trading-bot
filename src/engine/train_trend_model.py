import os
import sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import add_macd, add_adx, add_time_features


def prepare_trend_data(filepath):
    print(f"Loading data for Trend Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    df['return'] = df['close'].pct_change()
    df = add_macd(df)
    df = add_adx(df, period=14)
    df = add_time_features(df)

    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['trend_alignment'] = (df['sma_50'] > df['sma_200']).astype(int)

    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_20'] * 1.5).astype(int)

    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df = df.dropna()
    return df

def train_trend_model():
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
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_1h.csv.gz')
        archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1h.csv.gz')
        if not os.path.exists(full_data_path) and os.path.exists(archive_path):
            full_data_path = archive_path
            print(f"  Using archive data for {sym}: {archive_path}")
        if not os.path.exists(full_data_path):
            print(f"Warning: Data for {sym} not found at {full_data_path}. Auto-downloading...")
            import sys
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe='1h', days=6*365)

        if os.path.exists(full_data_path):
            df = prepare_trend_data(full_data_path)
            all_data.append(df)
            
    if not all_data:
        print("Error: No data found even after attempted download.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    feature_columns = ['return', 'macd', 'macd_signal', 'macd_hist', 'trend_alignment', 'volume_surge', 'atr_14', 'adx_14']
    X = combined_df[feature_columns]
    y = combined_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    print("Training Trend-Following AI Model...")
    # Use more conservative settings for macro-trends
    model = HistGradientBoostingClassifier(random_state=42, max_iter=500, max_depth=5, learning_rate=0.02, early_stopping=True, l2_regularization=0.5)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100

    print(f"\nTrend Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Long (UP) Precision: {long_acc:.2f}% | Short (DOWN) Precision: {short_acc:.2f}%")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'trend_model.joblib')
    joblib.dump(model, model_path)
    print(f"Trend Model saved to {model_path}")
    
    meta_path = os.path.join(models_dir, 'trend_model_meta.json')
    import json
    from datetime import datetime, timezone
    with open(meta_path, 'w') as f:
        json.dump({"accuracy": accuracy * 100, "long_accuracy": long_acc, "short_accuracy": short_acc,
                   "last_trained": datetime.now(timezone.utc).isoformat()}, f)

if __name__ == "__main__":
    train_trend_model()