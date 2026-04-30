"""
QuestDB schema — CREATE TABLE statements for all bot data.

Run:  python -m src.database.schema
      (Idempotent — safe to re-run; uses CREATE TABLE IF NOT EXISTS)

Tables
------
market_data          — OHLCV bars (1s / 1m / 1h / 1d) for all symbols
trade_events         — every live bot trade (entry + exit)
model_signals        — per-bar signals from all strategies + ML models
training_telemetry   — epoch-level ML training metrics (loss, accuracy…)
strategy_performance — periodic snapshots of paper/live strategy stats
news_sentiment       — scraped headlines + VADER/Gemini sentiment scores
agent_heartbeats     — agent health pings (for monitoring tab)
backtest_results     — stored backtests for comparison over time
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_TABLES: list[tuple[str, str]] = [

    ("market_data", """
CREATE TABLE IF NOT EXISTS market_data (
    ts            TIMESTAMP,
    symbol        SYMBOL  CAPACITY 64 CACHE,
    timeframe     SYMBOL  CAPACITY 16 CACHE,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    volume        DOUBLE,
    funding_rate  DOUBLE
) TIMESTAMP(ts)
PARTITION BY MONTH
DEDUP UPSERT KEYS(ts, symbol, timeframe);
"""),

    ("trade_events", """
CREATE TABLE IF NOT EXISTS trade_events (
    ts            TIMESTAMP,
    trade_id      VARCHAR,
    symbol        SYMBOL  CAPACITY 64 CACHE,
    strategy      SYMBOL  CAPACITY 64 CACHE,
    market        SYMBOL  CAPACITY 8  CACHE,
    direction     INT,
    entry_price   DOUBLE,
    exit_price    DOUBLE,
    size_usd      DOUBLE,
    pnl_usd       DOUBLE,
    fees_usd      DOUBLE,
    bars_held     INT,
    is_live       BOOLEAN
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("model_signals", """
CREATE TABLE IF NOT EXISTS model_signals (
    ts                  TIMESTAMP,
    symbol              SYMBOL  CAPACITY 64 CACHE,
    signal_rsi          DOUBLE,
    signal_macd         DOUBLE,
    signal_bb           DOUBLE,
    signal_ensemble     DOUBLE,
    signal_vwap         DOUBLE,
    signal_donchian     DOUBLE,
    signal_keltner      DOUBLE,
    signal_funding      DOUBLE,
    signal_vol_breakout DOUBLE,
    signal_supertrend   DOUBLE,
    signal_ml_long      DOUBLE,
    signal_ml_short     DOUBLE,
    signal_scalping     DOUBLE,
    signal_trend        DOUBLE,
    signal_meta         DOUBLE,
    regime              INT,
    meta_label          INT,
    meta_prob           DOUBLE,
    garch_vol           DOUBLE,
    ou_zscore           DOUBLE,
    close               DOUBLE
) TIMESTAMP(ts)
PARTITION BY WEEK
DEDUP UPSERT KEYS(ts, symbol);
"""),

    ("training_telemetry", """
CREATE TABLE IF NOT EXISTS training_telemetry (
    ts            TIMESTAMP,
    model         SYMBOL  CAPACITY 32 CACHE,
    run_id        SYMBOL  CAPACITY 256 CACHE,
    hardware      SYMBOL  CAPACITY 32 CACHE,
    epoch         INT,
    train_loss    DOUBLE,
    val_loss      DOUBLE,
    accuracy      DOUBLE,
    sharpe        DOUBLE,
    learning_rate DOUBLE,
    batch_size    INT,
    seq_len       INT,
    n_samples     LONG
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("strategy_performance", """
CREATE TABLE IF NOT EXISTS strategy_performance (
    ts            TIMESTAMP,
    strategy      SYMBOL  CAPACITY 64 CACHE,
    symbol        SYMBOL  CAPACITY 64 CACHE,
    is_live       BOOLEAN,
    balance       DOUBLE,
    total_pnl     DOUBLE,
    pnl_pct       DOUBLE,
    win_rate      DOUBLE,
    n_trades      INT,
    n_wins        INT,
    sharpe        DOUBLE,
    wf_mean_sharpe DOUBLE,
    wf_consistency DOUBLE
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("news_sentiment", """
CREATE TABLE IF NOT EXISTS news_sentiment (
    ts              TIMESTAMP,
    source          SYMBOL  CAPACITY 32 CACHE,
    sentiment       SYMBOL  CAPACITY 8  CACHE,
    coins           SYMBOL  CAPACITY 64 CACHE,
    score           DOUBLE,
    headline        VARCHAR,
    url             VARCHAR
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("agent_heartbeats", """
CREATE TABLE IF NOT EXISTS agent_heartbeats (
    ts            TIMESTAMP,
    agent         SYMBOL  CAPACITY 32 CACHE,
    status        SYMBOL  CAPACITY 16 CACHE,
    current_task  VARCHAR,
    cpu_pct       DOUBLE,
    mem_mb        DOUBLE
) TIMESTAMP(ts)
PARTITION BY WEEK;
"""),

    ("backtest_results", """
CREATE TABLE IF NOT EXISTS backtest_results (
    ts              TIMESTAMP,
    run_id          VARCHAR,
    strategy        SYMBOL  CAPACITY 64 CACHE,
    symbol          SYMBOL  CAPACITY 64 CACHE,
    total_pnl       DOUBLE,
    gross_pnl       DOUBLE,
    total_fees      DOUBLE,
    sharpe          DOUBLE,
    win_rate        DOUBLE,
    max_drawdown    DOUBLE,
    n_trades        INT,
    wf_mean_sharpe  DOUBLE,
    wf_consistency  DOUBLE,
    wf_decay        DOUBLE
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("csv_ingestion_log", """
CREATE TABLE IF NOT EXISTS csv_ingestion_log (
    ts              TIMESTAMP,
    filename        VARCHAR,
    source_path     VARCHAR,
    symbol          SYMBOL  CAPACITY 64 CACHE,
    timeframe       SYMBOL  CAPACITY 16 CACHE,
    rows_written    LONG,
    file_size_bytes LONG,
    first_bar_ts    TIMESTAMP,
    last_bar_ts     TIMESTAMP
) TIMESTAMP(ts)
PARTITION BY YEAR;
"""),

    ("training_runs", """
CREATE TABLE IF NOT EXISTS training_runs (
    ts                TIMESTAMP,
    run_id            VARCHAR,
    model_name        SYMBOL  CAPACITY 64  CACHE,
    strategy          SYMBOL  CAPACITY 64  CACHE,
    symbol            SYMBOL  CAPACITY 64  CACHE,
    timeframe         SYMBOL  CAPACITY 16  CACHE,
    trigger           SYMBOL  CAPACITY 32  CACHE,
    start_ts          TIMESTAMP,
    end_ts            TIMESTAMP,
    duration_secs     DOUBLE,
    train_rows        LONG,
    val_rows          LONG,
    n_wf_folds        INT,
    best_epoch        INT,
    final_train_loss  DOUBLE,
    final_val_loss    DOUBLE,
    early_stopped     BOOLEAN,
    oos_sharpe        DOUBLE,
    oos_win_rate      DOUBLE,
    oos_max_drawdown  DOUBLE,
    n_oos_trades      INT,
    hyperparams_json  VARCHAR,
    feature_list_json VARCHAR,
    notes             VARCHAR
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("model_wf_folds", """
CREATE TABLE IF NOT EXISTS model_wf_folds (
    ts           TIMESTAMP,
    run_id       VARCHAR,
    model_name   SYMBOL  CAPACITY 64 CACHE,
    fold_index   INT,
    train_start  TIMESTAMP,
    train_end    TIMESTAMP,
    test_start   TIMESTAMP,
    test_end     TIMESTAMP,
    train_rows   LONG,
    test_rows    LONG,
    oos_sharpe   DOUBLE,
    oos_pnl      DOUBLE,
    oos_win_rate DOUBLE,
    oos_max_dd   DOUBLE,
    n_trades     INT
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("testnet_trades", """
CREATE TABLE IF NOT EXISTS testnet_trades (
    ts                  TIMESTAMP,
    trade_id            VARCHAR,
    symbol              SYMBOL  CAPACITY 64 CACHE,
    strategy            SYMBOL  CAPACITY 64 CACHE,
    model               SYMBOL  CAPACITY 64 CACHE,
    exit_reason         SYMBOL  CAPACITY 16 CACHE,
    direction           INT,
    is_live             BOOLEAN,
    entry_ts            TIMESTAMP,
    exit_ts             TIMESTAMP,
    entry_price         DOUBLE,
    exit_price          DOUBLE,
    size_usd            DOUBLE,
    pnl_usd             DOUBLE,
    fees_usd            DOUBLE,
    funding_pnl         DOUBLE,
    net_pnl             DOUBLE,
    bars_held           INT,
    meta_label          INT,
    regime              INT,
    garch_vol_at_entry  DOUBLE,
    stop_loss           DOUBLE,
    take_profit         DOUBLE,
    meta_prob           DOUBLE,
    signal_strength     DOUBLE
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),

    ("testnet_session_stats", """
CREATE TABLE IF NOT EXISTS testnet_session_stats (
    ts                TIMESTAMP,
    session_id        VARCHAR,
    strategy          SYMBOL  CAPACITY 64 CACHE,
    symbol            SYMBOL  CAPACITY 64 CACHE,
    balance           DOUBLE,
    total_pnl         DOUBLE,
    unrealized_pnl    DOUBLE,
    n_open_trades     INT,
    n_closed_trades   INT,
    win_rate          DOUBLE,
    sharpe            DOUBLE,
    max_drawdown      DOUBLE,
    funding_collected DOUBLE
) TIMESTAMP(ts)
PARTITION BY MONTH;
"""),
]


def create_all(client=None) -> bool:
    """Create all tables. Returns True if all succeeded."""
    if client is None:
        from src.database.questdb_client import get_client
        client = get_client()

    if not client.is_available():
        logger.error("QuestDB not available — run: docker-compose up -d questdb")
        return False

    ok = True
    for name, ddl in _TABLES:
        success = client.exec_ddl(ddl.strip())
        if success:
            logger.info("  ✓ %s", name)
        else:
            logger.error("  ✗ %s — DDL failed", name)
            ok = False
    return ok


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(levelname)s %(message)s")
    logger.info("Creating QuestDB tables…")
    success = create_all()
    if success:
        logger.info("All tables ready.")
    else:
        logger.error("Some tables failed — check QuestDB is running.")
        sys.exit(1)
