"""Temporal Fusion Transformer training pipeline for Phase 3."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.feature_engineering import add_ofi, add_taker_and_trade_features, add_time_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_SYMBOLS = ["BTC_USDT", "SOL_USDT", "ADA_USDT", "ETH_USDT"]


def load_symbols() -> list[str]:
    watchlist_path = PROJECT_ROOT / "data" / "watchlist.json"
    if watchlist_path.exists():
        with watchlist_path.open("r", encoding="utf-8") as handle:
            return [symbol.replace("/", "_") for symbol in json.load(handle)]
    return DEFAULT_SYMBOLS


def load_frame(symbol: str, timeframe: str, history_days: int) -> pd.DataFrame | None:
    data_path = RAW_DIR / f"{symbol}_{timeframe}.csv.gz"
    if not data_path.exists():
        logger.warning("Missing %s. Triggering historical backfill.", data_path)
        from src.data_ingestion.historical_backfill import backfill_history

        backfill_history(symbol=symbol.replace("_", "/"), timeframe=timeframe, days=history_days)

    if not data_path.exists():
        logger.error("Data still missing after backfill: %s", data_path)
        return None

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def engineer_frame(df: pd.DataFrame, asset_id: int, freq: str, symbol: str = "") -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote", "sentiment_score", "funding_rate"]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Merge funding rates if available (adds 'funding_rate' column)
    if symbol and "funding_rate" not in df.columns:
        try:
            from src.data_ingestion.funding_rate_downloader import merge_funding_into_ohlcv
            df = merge_funding_into_ohlcv(df, symbol.replace("_", "/"))
        except Exception:
            df["funding_rate"] = 0.0
    elif "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0

    df = df.set_index("timestamp").asfreq(freq).ffill().reset_index()
    df["return"] = df["close"].pct_change().fillna(0.0)
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df = add_taker_and_trade_features(df)
    df = add_ofi(df)
    df = add_time_features(df)

    df["asset_id"] = float(asset_id)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    return df.dropna(subset=["close"])


def build_series_bundle(df: pd.DataFrame, freq: str):
    try:
        from darts import TimeSeries
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Darts is required for TFT training. Install project dependencies first."
        ) from exc

    target = TimeSeries.from_dataframe(df, time_col="timestamp", value_cols="close", fill_missing_dates=True, freq=freq)
    past_cov_cols = ["return", "volume", "taker_buy_ratio", "avg_trade_size", "ofi", "funding_rate"]
    if "sentiment_score" in df.columns:
        past_cov_cols.append("sentiment_score")
    past_covariates = TimeSeries.from_dataframe(
        df,
        time_col="timestamp",
        value_cols=past_cov_cols,
        fill_missing_dates=True,
        freq=freq,
    )
    future_covariates = TimeSeries.from_dataframe(
        df,
        time_col="timestamp",
        value_cols=["hour_sin", "hour_cos", "dow_sin", "dow_cos", "asset_id"],
        fill_missing_dates=True,
        freq=freq,
    )
    return target, past_covariates, future_covariates


def train_tft_model(
    timeframe: str = "1h",
    input_chunk_length: int = 168,
    output_chunk_length: int = 24,
    n_epochs: int = 30,
    history_days: int = 365 * 2,
    dry_run: bool = False,
):
    freq = "1h" if timeframe == "1h" else "1min"
    symbols = load_symbols()
    series_bundle = []
    dry_run_summary: dict[str, int] = {}

    for asset_id, symbol in enumerate(symbols):
        frame = load_frame(symbol, timeframe, history_days)
        if frame is None or len(frame) < input_chunk_length + output_chunk_length + 50:
            logger.warning("Skipping %s due to insufficient data.", symbol)
            continue

        engineered = engineer_frame(frame, asset_id, freq, symbol=symbol)

        if dry_run:
            dry_run_summary[symbol] = len(engineered)
            continue

        series_bundle.append((symbol, *build_series_bundle(engineered, freq)))

    if dry_run:
        if not dry_run_summary:
            raise RuntimeError("No training series were prepared. Check raw data availability.")
        logger.info("Dry-run summary: %s", dry_run_summary)
        return dry_run_summary

    try:
        from darts.dataprocessing.transformers import Scaler
        from darts.models import TFTModel
    except Exception as exc:  # pragma: no cover - import guard
        logger.error("TFT dependencies are unavailable: %s", exc)
        raise

    if not series_bundle:
        raise RuntimeError("No training series were prepared. Check raw data availability.")

    target_scaler = Scaler()
    past_scaler = Scaler()
    future_scaler = Scaler()

    targets = [bundle[1] for bundle in series_bundle]
    past_covariates = [bundle[2] for bundle in series_bundle]
    future_covariates = [bundle[3] for bundle in series_bundle]

    scaled_targets = target_scaler.fit_transform(targets)
    scaled_past = past_scaler.fit_transform(past_covariates)
    scaled_future = future_scaler.fit_transform(future_covariates)

    # Apply CPU/GPU optimisations (TF32, cuDNN benchmark, thread count)
    try:
        from src.utils.hw_config import configure as _hw_cfg
        _hw_cfg(verbose=True)
    except Exception:
        pass

    # Auto-detect GPU (RTX 3080 Ti will be ~20-50x faster than CPU)
    try:
        import torch
        _use_gpu = torch.cuda.is_available()
        if _use_gpu:
            gpu_count = torch.cuda.device_count()
            logger.info("CUDA GPU detected: %d device(s) found. Primary: %s — training on all GPUs.", gpu_count, torch.cuda.get_device_name(0))
        else:
            logger.warning("No CUDA GPU detected — training on CPU (slow). Run install_cuda_torch.ps1 to enable GPU.")
    except Exception:
        _use_gpu = False

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=256 if _use_gpu else 32,
        lstm_layers=2 if _use_gpu else 1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=256 if _use_gpu else 32,
        n_epochs=n_epochs,
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "strategy": "auto", "enable_progress_bar": True} if _use_gpu else {"accelerator": "cpu", "enable_progress_bar": True},
    )

    logger.info(
        "Training TFT on %d series with input=%d output=%d epochs=%d device=%s",
        len(scaled_targets),
        input_chunk_length,
        output_chunk_length,
        n_epochs,
        "GPU" if _use_gpu else "CPU",
    )
    model.fit(
        series=scaled_targets,
        past_covariates=scaled_past,
        future_covariates=scaled_future,
        verbose=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "tft_model.pt"
    model.save(str(model_path))

    from datetime import datetime, timezone as _tz
    meta_path = MODEL_DIR / "tft_model_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "symbols": [symbol for symbol, *_ in series_bundle],
                "timeframe": timeframe,
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "n_epochs": n_epochs,
                "model_path": str(model_path),
                "device": "GPU" if _use_gpu else "CPU",
                "last_trained": datetime.now(_tz.utc).isoformat(),
            },
            handle,
            indent=2,
        )

    logger.info("TFT model saved to %s", model_path)
    return {"model_path": str(model_path), "series_count": len(series_bundle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TFT neural forecasting model.")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "1m"])
    parser.add_argument("--input-chunk-length", type=int, default=168)
    parser.add_argument("--output-chunk-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--history-days", type=int, default=365 * 2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_tft_model(
        timeframe=args.timeframe,
        input_chunk_length=args.input_chunk_length,
        output_chunk_length=args.output_chunk_length,
        n_epochs=args.epochs,
        history_days=args.history_days,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
