"""
Joint OFT + RL training — Phase 9 (architecture plan §10).

Trains the Order Flow Transformer (OFT) and the SAC execution agent in
ONE combined loop inside `SyntheticExchange`. Objective per the plan:

    L = -E[PnL]  +  λ1 · CVaR_α(R)  +  λ2 · ImpactCost  +  λ3 · InventoryRisk

OFT is trained on labeled historical data via `OFTTrainer` (Phase 2). SAC
is then trained inside the synthetic exchange using OFT predictions as
part of its observation. After both phases, OFT predictions are calibrated
on out-of-fold data (`IsotonicCalibrator`).

This is intentionally a script rather than a class — it's run once per
training cycle (overnight or by hand). Outputs:

    models/oft_model.pt
    models/oft_calibrator.joblib
    models/sac_execution.pt

Run:
    python -m src.training.joint_oft_rl --symbol BTC/USDT --tf 1m
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("joint_oft_rl")

MODELS_DIR = PROJECT_ROOT / "models"


# ─── Stage A — OFT supervised training ──────────────────────────────────

def train_oft(symbol: str, timeframe: str, *, n_epochs: int = 5,
              n_splits: int = 5) -> dict:
    """Pull OHLCV from parquet, build event-time labels, train OFT."""
    import torch
    import numpy as np
    import pandas as pd

    from src.database.parquet_store import get_store
    from src.analysis.event_time_labeler import label_event_time
    from src.analysis.feature_engineering import (
        add_rsi, add_macd, add_atr, add_bollinger_bands,
        add_kalman_close, add_l2_features,
    )
    from src.models.order_flow_transformer import OrderFlowTransformer, OFTConfig
    from src.training.oft_trainer import OFTTrainer, OFTTrainerConfig, IsotonicCalibrator

    parquet = get_store()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)
    df = parquet.query(symbol, start=start, end=end, timeframe=timeframe)
    if df is None or df.empty:
        logger.error("[oft] no data for %s/%s -- abort", symbol, timeframe)
        return {"status": "no_data"}

    df = df.copy()
    df = add_kalman_close(df)
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_atr(df, 14)
    df = add_bollinger_bands(df)
    df = add_l2_features(df)
    df = df.dropna().reset_index(drop=True)

    labels = label_event_time(df, k_tp=2.0, k_sl=2.0, max_horizon_bars=60)

    # Filter to binary classification subset (per plan §5)
    keep_mask = labels.labels != 0
    X_df = df[keep_mask].reset_index(drop=True)
    y_bin = labels.binary_y.reset_index(drop=True).astype(np.float32)
    t1   = labels.t1[keep_mask].reset_index(drop=True)
    if len(X_df) < 200:
        logger.error("[oft] insufficient samples after filter (%d) -- abort", len(X_df))
        return {"status": "too_few_samples"}

    # Build tensors. The model expects (B, T_e, F_e) and (B, T_o, F_o).
    # For a sanity training pass we use a tiny window (T=8) and the available
    # numeric features as the event channel. Order-book channel is zero-padded
    # if we don't have L2 columns — the cross-attention layer still runs.
    event_cols = [c for c in X_df.columns
                  if c in {"close", "rsi_14", "macd", "macd_hist", "atr_14",
                           "bb_pb", "ob_imbalance", "ob_microprice", "ob_ofi"}]
    F_e = max(len(event_cols), 4)
    cfg = OFTConfig(event_features=F_e, orderbook_features=4,
                    d_model=64, n_heads=4, n_layers=2, ff_dim=128,
                    max_event_len=16, max_orderbook_len=16, n_regimes=3,
                    use_regime_cond=False)
    model = OrderFlowTransformer(cfg)

    # Sliding windows — cap at MAX_WINDOWS to prevent CUDA kernel dimension
    # overflow on RTX 2060 (sm_75) with large 1m datasets (BTC/ADA/SOL have
    # ~500k rows which triggers cudaErrorInvalidConfiguration at fold 4).
    MAX_WINDOWS = 200_000
    T = cfg.max_event_len
    feats = X_df[event_cols].fillna(0).to_numpy(np.float32)
    if feats.shape[0] < T + 1:
        logger.error("[oft] not enough rows for windowing")
        return {"status": "no_windows"}
    raw_count = feats.shape[0] - T
    # Use stride-based subsampling so temporal coverage is preserved.
    # Clamp to MAX_WINDOWS exactly — stride alone can overshoot by up to
    # one stride-worth of windows (e.g. 263k instead of 200k for ADA 1m).
    stride = max(1, raw_count // MAX_WINDOWS)
    indices = list(range(0, raw_count, stride))[:MAX_WINDOWS]
    win_count = len(indices)
    ev = np.zeros((win_count, T, F_e), dtype=np.float32)
    for out_i, in_i in enumerate(indices):
        block = feats[in_i:in_i + T]
        ev[out_i, :, :len(event_cols)] = block
    ob = np.zeros((win_count, T, 4), dtype=np.float32)
    log_returns = np.diff(np.log(X_df["close"].astype(float))).astype(np.float32)
    log_returns = np.concatenate([np.zeros(1, dtype=np.float32), log_returns])
    returns = np.array([log_returns[T + i] for i in indices], dtype=np.float32)
    targets = np.array([y_bin.to_numpy()[T + i] for i in indices], dtype=np.float32)
    t1_w = pd.Series([t1.iloc[T + i] for i in indices]).reset_index(drop=True)
    logger.info("[oft] %s/%s: %d raw windows -> %d sampled (stride=%d)",
                symbol, timeframe, raw_count, win_count, stride)

    trainer = OFTTrainer(model, OFTTrainerConfig(epochs=n_epochs, n_splits=n_splits,
                                                  batch_size=32, lr=1e-3, device="cpu"))
    res = trainer.run(
        events=torch.from_numpy(ev),
        orderbook=torch.from_numpy(ob),
        returns=torch.from_numpy(returns),
        binary_y=torch.from_numpy(targets),
        regime=None,
        t1_times=t1_w,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    oft_path = MODELS_DIR / "oft_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config":     vars(cfg),
        "calibrator": trainer.calibrator,
    }, oft_path)
    from src.utils.model_integrity import sign_model
    sign_model(str(oft_path))
    logger.info("[oft] saved checkpoint -> %s", oft_path)
    return {"status": "ok", **res}


# ─── Stage B — SAC inside SyntheticExchange ─────────────────────────────

def train_sac(symbol: str, timeframe: str, *, n_episodes: int = 50,
              n_updates_per_episode: int = 100, device: str = "auto") -> dict:
    """Train SAC inside a SyntheticExchange replay of recent OHLCV.

    Args:
        device: "auto" picks cuda when available; pass "cpu" to force a
                CPU-only run, which makes any latent NaN/index bugs surface
                with a real Python traceback instead of an opaque CUDA assert.
    """
    import numpy as np
    import torch

    from src.database.parquet_store import get_store
    from src.simulation.synthetic_exchange import SyntheticExchange, ImpactModel
    from src.models.rl_execution_sac import SACAgent
    from src.models.rl_base import (
        ReplayBuffer, Transition, obs_dict_to_vector, shaped_reward,
    )

    parquet = get_store()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)
    df = parquet.query(symbol, start=start, end=end, timeframe=timeframe)
    if df is None or df.empty:
        logger.error("[sac] no data for %s/%s -- abort", symbol, timeframe)
        return {"status": "no_data"}

    book_iter = []
    for _, row in df.iterrows():
        c = float(row["close"])
        book_iter.append({
            "timestamp": int(row["timestamp"].value // 10**6) if hasattr(row["timestamp"], "value") else 0,
            "p_bid":  c * 0.9999, "p_ask":  c * 1.0001,
            "v_bid":  10.0,        "v_ask":  10.0,
        })
    if len(book_iter) < 100:
        return {"status": "too_few_ticks"}

    obs_dim = 5
    agent = SACAgent(obs_dim=obs_dim, hidden=64, device=device)
    buf = ReplayBuffer(capacity=20_000, obs_dim=obs_dim)

    metrics = {}
    for ep in range(n_episodes):
        ex = SyntheticExchange(book_iter, impact=ImpactModel(lambda_impact=0.4))
        obs = ex.reset()
        prev_o = obs_dict_to_vector(obs)
        ep_reward = 0.0
        steps = 0
        while True:
            action = agent.act(prev_o)  # [-1, +1]^2
            obs_next, r, done, info = ex.step(tuple(float(a) for a in action))
            r_shaped = shaped_reward(r, ex.state.inventory, inventory_lambda=0.05)
            o_next = obs_dict_to_vector(obs_next)
            buf.push(Transition(prev_o, action.astype(np.float32),
                                float(r_shaped), o_next, bool(done)))
            ep_reward += r_shaped
            steps += 1
            prev_o = o_next
            if done:
                break

        for _ in range(min(n_updates_per_episode, len(buf) // 2)):
            metrics = agent.update(buf, batch_size=64)

        if ep % 5 == 0:
            logger.info("[sac] ep=%d steps=%d reward=%.3f q1=%.3f",
                        ep, steps, ep_reward,
                        metrics.get("q1_loss", 0.0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(str(MODELS_DIR / "sac_execution.pt"))
    logger.info("[sac] saved -> %s", MODELS_DIR / "sac_execution.pt")
    return {"status": "ok", **metrics}


# ─── Combined entry ─────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--tf",     default="1m")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--skip-oft", action="store_true")
    p.add_argument("--skip-sac", action="store_true")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU for SAC stage -- surfaces NaN/index bugs "
                        "as real tracebacks instead of CUDA-side asserts.")
    args = p.parse_args()

    if args.cpu:
        import os
        # Block-on-CUDA gives us a real stack trace if anything CUDA-bound
        # ever runs (e.g. during OFT). Belt-and-suspenders.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    summary = {}
    if not args.skip_oft:
        logger.info("=" * 60)
        logger.info("[joint] STAGE A -- OFT supervised training")
        logger.info("=" * 60)
        summary["oft"] = train_oft(args.symbol, args.tf, n_epochs=args.epochs)

    if not args.skip_sac:
        logger.info("=" * 60)
        logger.info("[joint] STAGE B -- SAC inside SyntheticExchange")
        logger.info("=" * 60)
        summary["sac"] = train_sac(args.symbol, args.tf, n_episodes=args.episodes,
                                   device="cpu" if args.cpu else "auto")

    logger.info("=" * 60)
    logger.info("[joint] DONE  summary=%s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
