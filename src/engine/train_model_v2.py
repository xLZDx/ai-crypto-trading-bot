"""
Modernized training pipeline — Phase 10.

Uses Parquet instead of CSV.gz, event-time labels instead of bar-shift
targets, and the canonical feature pipeline (Kalman + L2 + funding + news).

Compatible with the same model output path / meta JSON shape that the
inference engine reads, so it's a drop-in replacement for the old
`train_model.py` once you're ready to switch.

Usage:
    python -m src.engine.train_model_v2 --symbol BTC/USDT --tf 1h
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("train_model_v2")
MODELS_DIR = PROJECT_ROOT / "models"


def build_dataset(symbol: str, timeframe: str, *, lookback_days: int = 365):
    from src.analytics.data_lens import DataLens
    from src.analysis.event_time_labeler import label_event_time
    from src.analysis.feature_engineering import (
        add_kalman_close, add_rsi, add_macd, add_atr, add_bollinger_bands,
        add_l2_features, add_news_sentiment, add_funding_zscore,
    )
    lens = DataLens()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    df = lens.training_frame(
        symbol=symbol, timeframe=timeframe, start=start, end=end,
        include_funding=True, include_news_24h=True, include_macro=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol}/{timeframe}")

    df = add_kalman_close(df)
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_atr(df, 14)
    df = add_bollinger_bands(df)
    df = add_l2_features(df)
    if "funding_rate" in df.columns:
        df = add_funding_zscore(df)
    df = df.dropna().reset_index(drop=True)

    labels = label_event_time(df, k_tp=2.0, k_sl=2.0, max_horizon_bars=24)
    keep = labels.labels != 0
    X = df[keep].reset_index(drop=True)
    y = labels.binary_y.reset_index(drop=True).astype(np.int8)
    return X, y, labels


def feature_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pick the numeric feature columns we'll feed to the classifier."""
    candidates = [
        "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14",
        "bb_pb", "ob_imbalance", "ob_microprice", "ob_ofi",
        "funding_z", "news_sentiment_24h",
        "fred_dxy", "fred_vix", "fear_greed",
    ]
    cols = [c for c in candidates if c in X.columns]
    if not cols:
        raise RuntimeError("No usable feature columns built")
    return X[cols].fillna(0.0), cols


def train(symbol: str, timeframe: str, *, lookback_days: int = 365,
          out_name: str = "btc_rf_model.joblib") -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    X, y, labels = build_dataset(symbol, timeframe, lookback_days=lookback_days)
    Xf, cols = feature_matrix(X)
    if len(Xf) < 200:
        raise RuntimeError(f"Too few labelled rows ({len(Xf)}) — abort")

    # 70/30 chronological split (no shuffle — time-series).
    split = int(len(Xf) * 0.7)
    X_tr, X_te = Xf.iloc[:split].to_numpy(), Xf.iloc[split:].to_numpy()
    y_tr, y_te = y.iloc[:split].to_numpy(), y.iloc[split:].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6, random_state=42,
    )
    model.fit(X_tr, y_tr)
    acc  = float(accuracy_score(y_te, model.predict(X_te)))
    preds = model.predict(X_te)
    long_acc  = float(np.mean(preds[y_te == 1] == 1)) if (y_te == 1).any() else 0.0
    short_acc = float(np.mean(preds[y_te == 0] == 0)) if (y_te == 0).any() else 0.0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / out_name
    joblib.dump({"model": model, "feature_cols": cols}, out_path)

    meta = {
        "symbol":         symbol,
        "timeframe":      timeframe,
        "trainer":        "train_model_v2 (Parquet + event-time)",
        "feature_cols":   cols,
        "n_features":     len(cols),
        "n_samples":      int(len(Xf)),
        "n_train":        int(split),
        "n_test":         int(len(Xf) - split),
        "accuracy":       round(acc * 100, 2),
        "long_accuracy":  round(long_acc * 100, 2),
        "short_accuracy": round(short_acc * 100, 2),
        "label_stats":    labels.stats,
        "last_trained":   datetime.now(timezone.utc).isoformat(),
        "model_path":     str(out_path),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("[train_v2] saved %s (acc=%.3f)", out_path.name, acc)
    return meta


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--tf",     default="1h")
    p.add_argument("--days",   type=int, default=365)
    p.add_argument("--out",    default="btc_rf_model.joblib")
    args = p.parse_args()
    res = train(args.symbol, args.tf, lookback_days=args.days, out_name=args.out)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
