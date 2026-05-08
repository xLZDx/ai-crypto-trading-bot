"""Temporal Fusion Transformer training pipeline for Phase 3."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.analysis.feature_engineering import add_ofi, add_taker_and_trade_features, add_time_features

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_PATH / "data" / "raw"
MODEL_DIR = ROOT_PATH / "models"
DEFAULT_SYMBOLS = ["BTC_USDT", "SOL_USDT", "ADA_USDT", "ETH_USDT"]


def load_symbols() -> list[str]:
    watchlist_path = ROOT_PATH / "data" / "watchlist.json"
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

    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.asfreq(freq).ffill().reset_index()
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


def _tf_to_pandas_freq(tf: str) -> str:
    """Map our canonical TF tokens to pandas asfreq strings.

    The previous one-line `freq = "1h" if timeframe=="1h" else "1min"` was
    a 1B-fix root cause: passing timeframe='4h' silently used '1min' freq,
    which `df.asfreq('1min').ffill()` expanded into 240× rows per 4h bar
    and then mis-aligned with darts' fill_missing_dates path, producing
    "cannot reindex on an axis with duplicate labels" downstream.
    """
    return {
        '1m':  '1min',
        '5m':  '5min',
        '15m': '15min',
        '30m': '30min',
        '1h':  '1h',
        '2h':  '2h',
        '4h':  '4h',
        '1d':  '1D',
        '1w':  '1W',
    }.get(tf, '1h')


def _dedupe_for_darts(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy of df with unique ascending timestamps.

    darts.TimeSeries.from_dataframe with fill_missing_dates=True calls
    pandas reindex(), which raises "cannot reindex on an axis with
    duplicate labels" if the same timestamp appears twice. Duplicates
    leak in via the market_data UNION (new ParquetClient store + legacy
    data/parquet/) when the two stores overlap; this guard is also
    defensive against any intermediate mutation between the three
    from_dataframe() calls in build_series_bundle().
    """
    return (df.sort_values(time_col)
              .drop_duplicates(subset=[time_col], keep="last")
              .reset_index(drop=True))


def build_series_bundle(df: pd.DataFrame, freq: str):
    try:
        from darts import TimeSeries
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Darts is required for TFT training. Install project dependencies first."
        ) from exc

    # Triple dedupe — once at the top is the primary line of defence, and
    # then immediately before each of the three from_dataframe() calls
    # (target / past_covariates / future_covariates) so any intermediate
    # mutation can't reintroduce duplicates. Each call gets its own clean
    # frame — drop_duplicates returns a new DataFrame so this is cheap.
    df = _dedupe_for_darts(df)

    target_df = _dedupe_for_darts(df)
    target = TimeSeries.from_dataframe(target_df, time_col="timestamp",
                                       value_cols="close",
                                       fill_missing_dates=True, freq=freq)

    past_cov_cols = ["return", "volume", "taker_buy_ratio", "avg_trade_size", "ofi", "funding_rate"]
    if "sentiment_score" in df.columns:
        past_cov_cols.append("sentiment_score")
    past_df = _dedupe_for_darts(df)
    past_covariates = TimeSeries.from_dataframe(
        past_df,
        time_col="timestamp",
        value_cols=past_cov_cols,
        fill_missing_dates=True,
        freq=freq,
    )
    future_df = _dedupe_for_darts(df)
    future_covariates = TimeSeries.from_dataframe(
        future_df,
        time_col="timestamp",
        value_cols=["hour_sin", "hour_cos", "dow_sin", "dow_cos", "asset_id"],
        fill_missing_dates=True,
        freq=freq,
    )
    return target, past_covariates, future_covariates


def load_frame_from_db(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load OHLCV from ParquetClient. Returns None if DB unavailable or data insufficient."""
    try:
        from src.database.parquet_client import get_client
        db = get_client()
        if not db.is_available():
            return None
        df = db.query_df(
            f"SELECT ts as timestamp, open, high, low, close, volume, funding_rate "
            f"FROM market_data "
            f"WHERE symbol='{symbol}' AND timeframe='{timeframe}' "
            f"ORDER BY ts"
        )
        if df.empty or len(df) < 500:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        logger.debug("DB load skipped for %s/%s: %s", symbol, timeframe, exc)
        return None


def train_tft_model(
    timeframe: str = "1h",
    input_chunk_length: int = 168,
    output_chunk_length: int = 24,
    n_epochs: int = 30,
    history_days: int = 365 * 2,
    dry_run: bool = False,
):
    freq = _tf_to_pandas_freq(timeframe)
    symbols = load_symbols()
    series_bundle = []
    dry_run_summary: dict[str, int] = {}

    for asset_id, symbol in enumerate(symbols):
        # Try QuestDB first (faster), fall back to CSV
        frame = load_frame_from_db(symbol, timeframe)
        if frame is not None:
            logger.info("Loaded %s/%s from QuestDB (%d rows)", symbol, timeframe, len(frame))
        else:
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

    # Split each series 80/20 train/val (chronological — no shuffle)
    VAL_FRAC = 0.20
    targets_all     = [bundle[1] for bundle in series_bundle]
    past_cov_all    = [bundle[2] for bundle in series_bundle]
    future_cov_all  = [bundle[3] for bundle in series_bundle]

    split_idx = [max(input_chunk_length + output_chunk_length, int(len(t) * (1 - VAL_FRAC)))
                 for t in targets_all]

    train_targets = [t[:s] for t, s in zip(targets_all, split_idx)]
    val_targets   = [t[s:] for t, s in zip(targets_all, split_idx)]
    train_past    = [p[:s] for p, s in zip(past_cov_all, split_idx)]
    val_past      = [p[s:] for p, s in zip(past_cov_all, split_idx)]
    train_future  = [f[:s] for f, s in zip(future_cov_all, split_idx)]
    val_future    = [f[s:] for f, s in zip(future_cov_all, split_idx)]

    target_scaler = Scaler()
    past_scaler   = Scaler()
    future_scaler = Scaler()

    scaled_train_tgt    = target_scaler.fit_transform(train_targets)
    scaled_val_tgt      = target_scaler.transform(val_targets)
    scaled_train_past   = past_scaler.fit_transform(train_past)
    scaled_val_past     = past_scaler.transform(val_past)
    scaled_train_future = future_scaler.fit_transform(train_future)
    scaled_val_future   = future_scaler.transform(val_future)

    # Apply CPU/GPU optimisations (TF32, cuDNN benchmark, thread count)
    try:
        from src.utils.hw_config import configure as _hw_cfg, get_tft_config
        _hw_cfg(verbose=True)
        _hw = get_tft_config(verbose=True)
    except Exception:
        _hw = {"batch_size": 32, "hidden_size": 32, "lstm_layers": 1, "device": "cpu", "vram_gb": 0.0}

    # Auto-detect GPU
    try:
        import torch
        _use_gpu = torch.cuda.is_available()
        if _use_gpu:
            logger.info("CUDA GPU detected: %s  VRAM=%.1f GB", torch.cuda.get_device_name(0), _hw["vram_gb"])
            torch.cuda.empty_cache()
        else:
            logger.warning("No CUDA GPU detected — training on CPU. Run install_cuda_torch.ps1 to enable GPU.")
    except Exception:
        _use_gpu = False

    # ── Phase 0 fix: filter val series that are too short to produce any validation
    # window. A val series needs ≥ input_chunk_length + output_chunk_length bars for
    # Darts to form even one validation batch; shorter ones silently produce NaN
    # val_loss, which prevents EarlyStopping from ever firing.
    MIN_VAL_BARS = input_chunk_length + output_chunk_length
    valid_val_mask = [len(v) >= MIN_VAL_BARS for v in scaled_val_tgt]

    if any(valid_val_mask):
        _fit_val_tgt    = [v for v, ok in zip(scaled_val_tgt,    valid_val_mask) if ok]
        _fit_val_past   = [v for v, ok in zip(scaled_val_past,   valid_val_mask) if ok]
        _fit_val_future = [v for v, ok in zip(scaled_val_future, valid_val_mask) if ok]
        _monitor_metric = "val_loss"
        if not all(valid_val_mask):
            logger.warning(
                "[TFT] %d/%d val series long enough (≥%d bars) — using only valid ones.",
                sum(valid_val_mask), len(valid_val_mask), MIN_VAL_BARS,
            )
    else:
        # All val series are too short → skip validation and monitor train_loss instead
        logger.warning(
            "[TFT] All val series shorter than %d bars — EarlyStopping will monitor train_loss.",
            MIN_VAL_BARS,
        )
        _fit_val_tgt = _fit_val_past = _fit_val_future = None
        _monitor_metric = "train_loss"

    # EarlyStopping: check_finite=True stops if metric is NaN (guards against empty
    # val loops); strict=True raises if the metric key is never logged (catches
    # Darts version skew where the key name differs).
    try:
        from pytorch_lightning.callbacks import EarlyStopping
        _early_stop_cb = [
            EarlyStopping(
                monitor=_monitor_metric,
                patience=5,
                mode="min",
                check_finite=True,
                min_delta=1e-6,
                verbose=True,
            )
        ]
    except Exception:
        _early_stop_cb = []

    _trainer_kw: dict = {"enable_progress_bar": True}
    if _use_gpu:
        _trainer_kw.update({"accelerator": "gpu", "devices": 1, "strategy": "auto"})
    else:
        _trainer_kw["accelerator"] = "cpu"
    if _early_stop_cb:
        _trainer_kw["callbacks"] = _early_stop_cb

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=_hw["hidden_size"],
        lstm_layers=_hw["lstm_layers"],
        num_attention_heads=4,
        dropout=0.1,
        batch_size=_hw["batch_size"],
        n_epochs=n_epochs,
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs=_trainer_kw,
    )

    logger.info(
        "Training TFT on %d series  input=%d output=%d epochs=%d  "
        "device=%s VRAM=%.1fGB  batch=%d hidden=%d  val_monitor=%s",
        len(scaled_train_tgt),
        input_chunk_length,
        output_chunk_length,
        n_epochs,
        "GPU" if _use_gpu else "CPU",
        _hw["vram_gb"],
        _hw["batch_size"],
        _hw["hidden_size"],
        _monitor_metric,
    )
    model.fit(
        series=scaled_train_tgt,
        past_covariates=scaled_train_past,
        future_covariates=scaled_train_future,
        val_series=_fit_val_tgt,
        val_past_covariates=_fit_val_past,
        val_future_covariates=_fit_val_future,
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
    return {"model_path": str(model_path), "series_count": len(series_bundle), "val_split": VAL_FRAC}


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
