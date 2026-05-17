"""Evaluate the trained TFT model on a held-out window and write classification-
style metrics into models/tft_model_meta.json so the dashboard model card can
render Accuracy / Bull / Bear like the gradient-boosting models do.

TFT is a regression model with quantile loss. We convert its forecasts into
direction (UP if forecast_close > last_observed_close, else DOWN) and compare
against actual direction over a rolling validation window.

Metrics produced:
    - accuracy           : % of forecasts whose direction matched actual
    - long_accuracy      : recall on UP moves (tp / (tp + fn))
    - short_accuracy     : recall on DOWN moves (tn / (tn + fp))
    - n_samples          : forecast points evaluated
    - n_features         : count of past+future covariates the model uses
    - val_loss / train_loss: parsed from logs/tft_3epoch.log if present
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH    = PROJECT_ROOT / "logs" / "tft_3epoch.log"
MODEL_PATH  = PROJECT_ROOT / "models" / "tft_model.pt"
META_PATH   = PROJECT_ROOT / "models" / "tft_model_meta.json"
RAW_DIR     = PROJECT_ROOT / "data" / "raw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_tft")


def parse_losses_from_log() -> tuple[float | None, float | None]:
    """Read the (UTF-16) TFT log and find the final train/val loss."""
    if not LOG_PATH.exists():
        return None, None
    raw = LOG_PATH.read_bytes()
    for enc in ("utf-16", "utf-8"):
        try:
            txt = raw.decode(enc); break
        except UnicodeDecodeError:
            txt = ""
    last_train = last_val = None
    rgx = re.compile(r"train_loss=([0-9.]+)(?:.*?val_loss=([0-9.]+))?")
    for line in txt.splitlines():
        m = rgx.search(line)
        if m:
            last_train = float(m.group(1))
            if m.group(2):
                last_val = float(m.group(2))
    return last_train, last_val


def main() -> int:
    if not MODEL_PATH.exists():
        logger.error("TFT model file missing at %s", MODEL_PATH)
        return 1

    # Defer heavy imports so help/etc. stay snappy
    import warnings; warnings.filterwarnings("ignore")
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TFTModel

    from src.engine.train_tft_model import engineer_frame, build_series_bundle

    logger.info("Loading TFT model from %s", MODEL_PATH)
    model = TFTModel.load(str(MODEL_PATH))
    icl = int(model.input_chunk_length)
    ocl = int(model.output_chunk_length)

    # Load BTC 1h history (TFT was trained on 1h)
    csv = RAW_DIR / "BTC_USDT_1h.csv.gz"
    if not csv.exists():
        logger.error("BTC raw frame missing at %s", csv)
        return 1

    df = pd.read_csv(csv, compression="gzip")
    df = df.dropna(subset=["close", "timestamp"]).reset_index(drop=True)
    if len(df) < icl + ocl + 200:
        logger.error("Not enough rows for evaluation: %d", len(df))
        return 1

    # Same feature pipeline the trainer used
    eng = engineer_frame(df, asset_id=0, freq="1h", symbol="BTC_USDT")
    target, past_cov, future_cov = build_series_bundle(eng, freq="1h")

    # 80/20 split (same as trainer) — refit scalers on TRAIN HALF ONLY,
    # then transform the val half. Mirrors trainer's logic.
    split = max(icl + ocl, int(len(target) * 0.8))
    tgt_train, tgt_val = target[:split], target[split:]
    past_train, past_val = past_cov[:split], past_cov[split:]
    fut_train, fut_val   = future_cov[:split], future_cov[split:]

    tgt_scaler  = Scaler(); past_scaler = Scaler(); fut_scaler = Scaler()
    tgt_scaler.fit(tgt_train); past_scaler.fit(past_train); fut_scaler.fit(fut_train)

    s_tgt_train = tgt_scaler.transform(tgt_train)
    s_tgt_val   = tgt_scaler.transform(tgt_val)
    s_past_full = past_scaler.transform(past_cov)
    s_fut_full  = fut_scaler.transform(future_cov)

    n_val = len(s_tgt_val)
    if n_val < ocl + 5:
        logger.error("Validation segment too short: %d bars", n_val)
        return 1

    # Sample N forecast points across the val window
    SAMPLE_POINTS = 50
    starts = np.linspace(0, max(1, n_val - ocl - 1), num=min(SAMPLE_POINTS, n_val - ocl), dtype=int)
    starts = sorted(set(starts.tolist()))
    logger.info("Running %d rolling forecasts (val len=%d, ocl=%d)", len(starts), n_val, ocl)

    correct = 0; total = 0
    tp = fn = tn = fp = 0

    for s in starts:
        # series ending at split + s (inclusive, + ocl-1 for val target windows)
        series_end = split + s
        history_target = s_tgt_train.append(s_tgt_val[:s]) if s > 0 else s_tgt_train
        try:
            forecast = model.predict(
                n=ocl,
                series=history_target,
                past_covariates=s_past_full,
                future_covariates=s_fut_full,
            )
        except Exception as exc:
            logger.debug("forecast at s=%d failed: %s", s, exc)
            continue

        # Inverse-scale to real prices
        try:
            pred_real = tgt_scaler.inverse_transform(forecast).pd_dataframe()
            actual_real = tgt_scaler.inverse_transform(s_tgt_val[s:s + ocl]).pd_dataframe()
        except Exception:
            continue
        if pred_real.empty or actual_real.empty:
            continue

        # Last observed price = end of history
        last_obs = float(tgt_scaler.inverse_transform(history_target).pd_dataframe().iloc[-1, 0])
        pred_end = float(pred_real.iloc[-1, 0])
        act_end  = float(actual_real.iloc[-1, 0])

        pred_dir = 1 if pred_end > last_obs else 0
        act_dir  = 1 if act_end  > last_obs else 0

        total += 1
        if pred_dir == act_dir: correct += 1
        if act_dir == 1 and pred_dir == 1: tp += 1
        if act_dir == 1 and pred_dir == 0: fn += 1
        if act_dir == 0 and pred_dir == 0: tn += 1
        if act_dir == 0 and pred_dir == 1: fp += 1

    if total == 0:
        logger.error("No forecasts produced -- aborting")
        return 1

    acc       = 100.0 * correct / total
    long_acc  = 100.0 * tp / max(1, tp + fn)
    short_acc = 100.0 * tn / max(1, tn + fp)
    train_loss, val_loss = parse_losses_from_log()

    # Approximate feature count: past + future covariate columns
    n_features = past_cov.n_components + future_cov.n_components

    # Update meta — preserve existing keys, add new ones
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta.update({
        "model":              "TFT (Darts Quantile)",
        "accuracy":           round(acc, 2),
        "long_accuracy":      round(long_acc, 2),
        "short_accuracy":     round(short_acc, 2),
        "directional_accuracy_pct": round(acc, 2),
        "n_samples":          int(total),
        "n_features":         int(n_features),
        "train_loss":         train_loss,
        "val_loss":           val_loss,
        "evaluated_at":       datetime.now(timezone.utc).isoformat(),
        "evaluation_method":  "rolling_24h_directional_on_BTC_USDT_val_split",
    })
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print("=== TFT evaluation ===")
    print(f"  forecasts:          {total}")
    print(f"  directional acc:    {acc:.2f}%")
    print(f"  long  recall (BULL): {long_acc:.2f}%   ({tp}/{tp+fn})")
    print(f"  short recall (BEAR): {short_acc:.2f}%   ({tn}/{tn+fp})")
    print(f"  train_loss / val_loss: {train_loss}  /  {val_loss}")
    print(f"  meta saved -> {META_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
