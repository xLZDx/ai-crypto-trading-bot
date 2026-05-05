"""Central configuration — all tunable constants in one place."""

# ── Trading ──────────────────────────────────────────────────────────────────
MIN_TRADE_USDT = 55.0          # Binance MIN_NOTIONAL safety floor
SCALPING_TRADE_FRACTION = 0.25 # Scalping position size relative to base size
DEFAULT_TRAILING_STOP_PCT = 2.0

# ── Elliott Wave ──────────────────────────────────────────────────────────────
WAVE_DEVIATION_DEFAULT = 1.5   # Starting ZigZag deviation %
WAVE_DEVIATION_MIN = 0.3       # Minimum ZigZag deviation before giving up
WAVE_DEVIATION_STEP = 0.3      # Step size for auto-scaling

# ── Signals ───────────────────────────────────────────────────────────────────
SENTIMENT_BOOST_THRESHOLD = 0.15   # Score above this = bullish news
SENTIMENT_DRAG_THRESHOLD = -0.15   # Score below this = bearish news
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SCALPING_RSI_OVERBOUGHT = 65
SCALPING_RSI_OVERSOLD = 35
FUNDING_SQUEEZE_THRESHOLD = 0.015  # 1.5% funding rate triggers contrarian signal
VOLATILITY_BREAKOUT_VOLUME_MULT = 1.5  # Volume must be N× SMA to confirm breakout

# ── Risk & Volatility ─────────────────────────────────────────────────────────
ANNUALIZATION_FACTOR = 8760    # Hours per year (for hourly candle volatility)
BASELINE_VOLATILITY = 0.5      # Normalisation baseline for position sizing
VOLATILITY_FLOOR = 0.05        # Minimum volatility to avoid division by zero

# ── Market context refresh intervals (seconds) ───────────────────────────────
MTF_SMA200_REFRESH = 3600      # Re-fetch 1D SMA200 every hour
FUNDING_RATE_REFRESH = 300     # Re-fetch funding rates every 5 minutes
NEWS_CACHE_TTL = 900           # Sentiment cache: 15 minutes

# ── Data ingestion ────────────────────────────────────────────────────────────
DEFAULT_CANDLE_LIMIT = 1000
WEBSOCKET_RECONNECT_DELAY = 5  # Seconds before WebSocket reconnect attempt

# ── OFT (Order Flow Transformer) live integration ─────────────────────────────
# Used by the OFT_Microstructure strategy in src/main.py. Both gates run in
# series: first the entry filter blocks trades on weak signals, then the
# confidence weight shrinks the surviving trade's notional.
OFT_GATE_P_MOVE_MIN   = 0.50   # Block entry when p_move_calibrated below this
OFT_GATE_LIQ_RISK_MAX = 0.70   # Block entry when liquidity_risk above this
OFT_WEIGHT_FLOOR      = 0.25   # Minimum size multiplier even on weak signals
OFT_WEIGHT_CEILING    = 1.00   # Maximum size multiplier (don't oversize)

# ── Database (DuckDB + Parquet — replaces QuestDB, no daemon required) ───────
# All time-series workloads land as partitioned Parquet on D:. DuckDB is
# the query engine (in-process, file-based, no separate server). Writes
# go through src.database.parquet_client (singleton, append-buffered).
PARQUET_DB_DIR        = "data/db"   # relative to project root
PARQUET_DB_FLUSH_S    = 30.0        # write buffer flush cadence
PARQUET_DB_FLUSH_ROWS = 5000        # or flush sooner if the buffer fills

# ── Dashboard ────────────────────────────────────────────────────────────────
LOG_TAIL_BYTES = 51200         # 50 KB tail read for log endpoint
LOG_MAX_LINES = 500
