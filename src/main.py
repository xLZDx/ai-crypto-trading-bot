import time
import logging
import json
import os
import asyncio
import sys
import subprocess
import traceback

# Add the root folder to the Python path to avoid the "No module named 'src'" error
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Global startup error interceptor
try:
    import websockets
    from src.data_ingestion.binance_downloader import download_history
    from src.analysis.elliott_waves import ElliottWaveAnalyzer
    from src.analysis.risk_manager import HullRiskManager
    from src.engine.order_manager import OrderManager
    from src.engine.trade_tracker import TradeTracker
    from src.analysis.sentiment import NewsSentimentAnalyzer
    from src.analysis.ml_predictor import MLPredictor
    from src.engine.agentic_llm import AgenticLLM
    from src.analysis.feature_store import FeatureStore
    from src.analysis.telegram_monitor import TelegramMonitor
    from src.engine.inference_engine import InferenceEngine
    from src.engine.market_maker import AvellanedaStoikov
    from src.analysis.mean_reversion import MeanReversionCore
    from src.analysis.momentum import CrossSectionalMomentum
    from src.analysis.telegram_monitor import TelegramMonitor
    from src.utils.safe_json import read_json, write_json
    # Phase 9 — institutional gate wraps Phases 1-5 (§11-18 of arch plan)
    from src.engine.institutional_gate import InstitutionalGate
    from src.engine import dual_balance
    # Phase 10 — Parquet-first data, replaces CSV.gz reads (with fallback)
    from src.analysis import feature_reader as _feature_reader
    import numpy as np
    from src.utils.config import (
        MIN_TRADE_USDT, SCALPING_TRADE_FRACTION, MTF_SMA200_REFRESH,
        FUNDING_RATE_REFRESH, SENTIMENT_BOOST_THRESHOLD, SENTIMENT_DRAG_THRESHOLD,
        RSI_OVERBOUGHT, RSI_OVERSOLD, SCALPING_RSI_OVERBOUGHT, SCALPING_RSI_OVERSOLD,
        FUNDING_SQUEEZE_THRESHOLD, VOLATILITY_BREAKOUT_VOLUME_MULT,
        WAVE_DEVIATION_DEFAULT, WAVE_DEVIATION_MIN, WAVE_DEVIATION_STEP,
        WEBSOCKET_RECONNECT_DELAY,
        OFT_GATE_P_MOVE_MIN, OFT_GATE_LIQ_RISK_MAX,
        OFT_WEIGHT_FLOOR, OFT_WEIGHT_CEILING,
    )
    from src.utils import runtime_overrides as _runtime_overrides
    import pandas as pd
    import ccxt
    try:
        import debugpy  # optional remote-debugger; harmless if missing
    except ImportError:
        pass
except Exception as e:
    error_trace = traceback.format_exc()
    os.makedirs('data', exist_ok=True)
    with open('data/state.json', 'w', encoding='utf-8') as f:
        json.dump({
            "status": "❌ BOT STARTUP ERROR",
            "last_signal": "CRASH",
            "reason": f"Startup Error: {e} | {error_trace[-500:]}",
            "balance_usdt": 0.0, "balance_btc": 0.0, "balance_sol": 0.0, "balance_ada": 0.0,
            "volatility": 0.0, "recommended_trade_size": 0.0,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment_score": 0.0
        }, f, indent=4, ensure_ascii=False)
    raise

# Configure logging to file and console
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove old handlers, if any
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler('logs/trading.log', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class MultiAssetTrader:
    # Tracks consecutive close failures per trade ID; force-close after this many failures
    _CLOSE_FAIL_LIMIT = 3
    _close_fail_counts: dict = {}

    def __init__(self, symbols=['BTC/USDT', 'SOL/USDT', 'ADA/USDT', 'ETH/USDT'], timeframe='1h'):
        self.symbols = symbols
        self.timeframe = timeframe
        self.analyzers = {sym: ElliottWaveAnalyzer(deviation_percent=WAVE_DEVIATION_DEFAULT) for sym in symbols}
        self.risk_managers = {sym: HullRiskManager(default_risk_usd=20.0) for sym in symbols}
        self.engine = OrderManager()
        self.tracker = TradeTracker()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        # Phase G — multi-TF inference. Replaces the single-TF MLPredictor
        # with a wrapper that auto-loads every per-TF artifact present on
        # disk. `.predict(data)` still routes to the canonical TF (1h for
        # base/trend/futures, 1m for scalping) so existing call sites work
        # unchanged. Per-TF inference is exposed via .predict_at(tf, data)
        # — used by strategy_registry once Phase A pins per-strategy TFs.
        from src.analysis.multi_tf_predictor import MultiTFPredictor
        self.ml_predictor       = MultiTFPredictor('base')
        self.scalping_predictor = MultiTFPredictor('scalping')
        self.futures_predictor  = MultiTFPredictor('futures')
        self.trend_predictor    = MultiTFPredictor('trend')
        self.agent = AgenticLLM()
        self.telegram_monitor = TelegramMonitor()
        self.state_file = 'data/state.json'

        # --- Strategy registry (controls which strategies are active) ---
        from src.engine.strategy_registry import load_config as _load_strat_cfg
        self._strat_cfg = _load_strat_cfg()
        self._strat_cfg_mtime = 0.0

        # --- Live news buffer (Phase D) ---
        # Background thread that maintains an in-memory rolling cache of
        # recent news. Without it, add_news_sentiment() pays a 100-500 ms
        # DuckDB cold-start on every signal cycle. With it, the lookup is
        # O(1) and the cache refreshes every 5 minutes.
        try:
            from src.analysis.live_news_buffer import start_buffer
            self.live_news_buffer = start_buffer(window_hours=48,
                                                 refresh_seconds=300)
        except Exception as exc:
            import logging as _l
            _l.getLogger(__name__).warning(
                "live_news_buffer failed to start: %s — sentiment will use parquet path", exc)
            self.live_news_buffer = None

        # --- Regime Classifier ---
        try:
            from src.analysis.regime_classifier import RegimeClassifier
            self.regime_clf = RegimeClassifier()
        except Exception as e:
            logger.error(
                "[init] RegimeClassifier load FAILED — regime will default to TRENDING "
                "for all signals; size_mult will be unscaled. Reason: %s",
                e, exc_info=True,
            )
            self.regime_clf = None
        self._regime_cache: dict = {}  # symbol → {regime, regime_name, size_mult}

        # --- Meta-Labeler ---
        try:
            from src.analysis.meta_labeler import MetaLabeler
            self.meta_labeler = MetaLabeler()
        except Exception as e:
            logger.error(
                "[init] MetaLabeler load FAILED — second-layer filter is DISABLED. "
                "Trades will pass without meta-labeler confidence check. Reason: %s",
                e, exc_info=True,
            )
            self.meta_labeler = None

        # --- Advanced Quant Modules ---
        self.feature_store = FeatureStore()
        self.inference_engine = InferenceEngine(feature_store=self.feature_store, update_interval=60)
        self.market_makers = {sym: AvellanedaStoikov(gamma=0.1, k=1.5) for sym in symbols}
        self.mean_reversion = MeanReversionCore()
        self.ou_results = {sym: {'signal': 0, 'mu': 0.0, 'sigma': 0.0} for sym in symbols}
        self.momentum_engine = CrossSectionalMomentum(lookback=20, top_pct=0.30, bottom_pct=0.30, rebalance_period=24)
        self.momentum_signals = {sym: 0.0 for sym in symbols}
        # VilarsoPro and mr_mozart are secondary sources — used as extra context for Gemini veto
        self.telegram_monitor = TelegramMonitor(channels=['VilarsoPro', 'vilarsofree', 'mr_mozart'])
        
        # Components for advanced strategies (MTF and Derivatives)
        self.futures_exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
        self.daily_sma200 = {sym: None for sym in symbols}
        self.daily_sma_last_update = 0
        self.funding_rates = {sym: 0.0 for sym in symbols}
        self.funding_last_update = 0
        # Periodic refresh of data/balance_real.json so the dashboard's
        # testnet/mainnet balance cards reflect actual exchange state.
        # Pre-fix: only refreshed once at init, then went stale (operator
        # complaint 2026-05-13: "testnet balance is not being updated").
        self.balance_last_refresh = 0
        self._BALANCE_REFRESH_SEC = 60
        
        self.current_state = {
            "status": "Initializing...",
            "prices": {sym: 0.0 for sym in symbols},
            "balance_usdt": 0.0,
            "balance_btc": 0.0,
            "balance_sol": 0.0,
            "balance_ada": 0.0,
            "last_update": "",
            "sentiment_score": 0.0,
            "market_data": {
                "SPOT": { sym: self.get_default_market_state() for sym in symbols },
                "FUTURES": { sym: self.get_default_market_state() for sym in symbols },
                "SCALPING": { sym: self.get_default_market_state() for sym in symbols },
            }
        }

        # Phase 9 — institutional gate (§11-18 of arch plan).
        # Fail-OPEN: if any module is unavailable, the legacy logic still runs.
        try:
            self.gate = InstitutionalGate(
                self.engine,
                peak_equity=10_000.0,
                max_daily_drawdown_pct=0.05,
                max_api_latency_ms=500.0,
                max_data_staleness_sec=120.0,
                decay_rate=0.10,
                decay_exit_threshold=0.20,
            )
            logger.info("[gate] InstitutionalGate ready (§11-18 wired).")
        except Exception as e:
            self.gate = None
            logger.critical(
                "[gate] InstitutionalGate init FAILED — §11-18 risk controls "
                "(beta-neutrality, circuit breakers, CVaR sizing, alpha-decay) "
                "are ALL DISABLED. Reason: %s",
                e, exc_info=True,
            )
            try:
                from src.utils.error_state import push_error
                push_error(
                    component='institutional_gate',
                    severity='critical',
                    message=f'InstitutionalGate init failed — all §11-18 risk controls disabled: {e}',
                )
            except Exception:
                pass  # never let error-state writer block init

        # Phase 9 — keep dual-balance state files fresh.
        try:
            dual_balance.refresh_real_from_binance(self.engine)
        except Exception as e:
            logger.warning("[init] dual_balance refresh_real_from_binance failed: %s", e)

        # Phase 10E — attach β-history so §17 beta-neutrality is actually live.
        try:
            self._attach_beta_history()
        except Exception as e:
            logger.debug("[gate] beta history attach skipped: %s", e)

        # Phase 10D — cache for dynamic-threshold lookups (per symbol)
        self._dyn_thresholds = {sym: 0.01 for sym in symbols}
        self._dyn_threshold_last_refresh = 0.0

        # Phase A — per-TF data cache for strategy_registry TF routing.
        # Stores {symbol: {tf: (list_of_bars, fetch_epoch)}} so we only
        # re-fetch a TF when its TTL has expired.  1h data is always the
        # main `data` from process_kline so we do NOT double-cache it here.
        self._tf_data_cache: dict[str, dict[str, tuple[list, float]]] = {
            sym: {} for sym in symbols
        }
        _TF_TTL = {"1m": 60, "5m": 300, "15m": 900, "4h": 14400}
        self._TF_TTL: dict[str, float] = _TF_TTL
        # Precompute per-strategy TFs once at init so process_kline (hot-path)
        # never does module-level attribute lookups on every WebSocket tick.
        from src.engine.strategy_registry import get_strategy_tf as _gtf
        self._base_tf    = _gtf("Base_ML")          or self.timeframe
        self._trend_tf   = _gtf("Trend_ML")         or self.timeframe
        self._fut_tf     = _gtf("Futures_Short_ML") or self.timeframe

        # Phase 10B — track signal_strength + entry time per open trade so
        # alpha-decay can close stale positions.
        self._signal_strength_at_entry: dict = {}      # trade_id -> float
        self._signal_entry_ts:         dict = {}       # trade_id -> unix ts

        # Phase 10 — unified pre-trade safety gate + position sizing gate.
        try:
            from src.risk.pre_trade_gate import PreTradeGate
            from src.risk.position_sizing import PositionSizingGate
            from src.risk.kill_switch import get_kill_switch
            _ks = get_kill_switch()
            self.pre_trade_gate = PreTradeGate(warmup_bars_required=14, kill_switch=_ks)
            self.sizing_gate = PositionSizingGate()
            logger.info("[gate] PreTradeGate + PositionSizingGate ready.")
        except Exception as _e:
            self.pre_trade_gate = None
            self.sizing_gate = None
            logger.warning("[gate] PreTradeGate init failed — safety gate DISABLED: %s", _e)

        self._update_state()

    def _attach_beta_history(self) -> None:
        """Build a simple per-symbol returns DataFrame from Parquet and pass
        it to the institutional gate so the β-neutrality filter is non-noop.
        """
        if self.gate is None:
            return
        try:
            import pandas as pd
            from src.database.parquet_store import get_store
            from datetime import datetime, timedelta, timezone as _tz
            store = get_store()
            end = datetime.now(_tz.utc)
            start = end - timedelta(days=180)
            cols = {}
            for sym in self.symbols:
                df = store.query(sym, start=start, end=end, timeframe="1d")
                if df is None or df.empty or "close" not in df.columns:
                    continue
                ser = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
                cols[sym] = ser.reset_index(drop=True)
            if not cols:
                return
            history = pd.DataFrame(cols).dropna()
            if "BTC/USDT" not in history.columns or len(history) < 100:
                return
            self.gate.attach_beta_filter(history, factor="BTC/USDT",
                                          max_beta_exposure=1.5)
        except Exception as e:
            logger.debug("[gate] _attach_beta_history failed: %s", e)

    def _check_pre_trade(
        self,
        symbol: str,
        action: str = "open",
        trade_usdt: float = 0.0,
        bankroll_usdt: float = 0.0,
        has_nan_inf: bool = False,
    ) -> tuple:
        """Run PreTradeGate + PositionSizingGate before placing a new position.

        Returns (allow: bool, sized_usdt: float).
        Falls through with (True, trade_usdt) if either gate is not initialised.
        """
        if self.pre_trade_gate is None:
            return True, trade_usdt
        from src.risk.pre_trade_gate import GateContext
        ctx = GateContext(symbol=symbol, action=action, has_nan_inf=has_nan_inf)
        with self.pre_trade_gate.trading_lock:
            gate_result = self.pre_trade_gate.check(ctx)
            if not gate_result.allow:
                logger.warning("[pre_trade] %s blocked (%s): %s",
                               symbol, action, gate_result.reason)
                return False, 0.0
            if action == "open" and self.sizing_gate is not None and trade_usdt > 0:
                open_count = sum(
                    1 for t in self.tracker.trades if t.get("status") == "OPEN"
                )
                sizing = self.sizing_gate.check(
                    trade_usdt=trade_usdt,
                    bankroll_usdt=bankroll_usdt or float(
                        self.get_real_or_sim_balance('USDT') or 0
                    ),
                    open_position_count=open_count,
                )
                if not sizing.allow:
                    logger.warning("[pre_trade] %s sizing blocked: %s",
                                   symbol, sizing.reason)
                    return False, 0.0
                return True, sizing.sized_usdt
        return True, trade_usdt

    def _refresh_dynamic_thresholds(self) -> None:
        """Phase 10D — every ~1h refit the entry threshold per symbol from
        recent (probs, returns) history. Cheap; runs on a timer from
        process_kline. No-op if the institutional gate is missing.
        """
        if self.gate is None:
            return
        now = time.time()
        if now - self._dyn_threshold_last_refresh < 3600:
            return
        self._dyn_threshold_last_refresh = now
        try:
            import pandas as pd
            from src.database.parquet_store import get_store
            from datetime import datetime, timedelta, timezone as _tz
            store = get_store()
            end = datetime.now(_tz.utc)
            start = end - timedelta(days=14)
            for sym in self.symbols:
                df = store.query(sym, start=start, end=end, timeframe="1h")
                if df is None or df.empty:
                    continue
                close = pd.to_numeric(df["close"], errors="coerce").bfill()
                ret = close.pct_change().fillna(0).to_numpy()
                # Synthetic probs ∈ [0,1] from normalized return — placeholder
                # until the OFT is trained (then we use OFT.p_move directly).
                probs = (ret - ret.min()) / max(ret.max() - ret.min(), 1e-9)
                thr = self.gate.best_threshold(probs, ret)
                if 0.0 < thr < 1.0:
                    self._dyn_thresholds[sym] = float(thr)
            logger.debug("[gate] dynamic thresholds refreshed: %s", self._dyn_thresholds)
        except Exception as e:
            logger.debug("[gate] dyn-threshold refresh failed: %s", e)

    def _get_tf_data(self, symbol: str, tf: str, tail_n: int = 1000) -> list:
        """Return OHLCV bars at `tf` for `symbol`, using a TTL cache.

        Falls back to an empty list (not an exception) so callers can
        gracefully fall back to the canonical-TF prediction.
        On a transient fetch failure the result is cached for only 60 s
        (not the full TF TTL) so a recovering parquet store is retried soon.
        """
        now = time.time()
        cached = self._tf_data_cache.get(symbol, {}).get(tf)
        if cached is not None:
            bars, fetched_at = cached
            ttl = self._TF_TTL.get(tf, 3600.0)
            if now - fetched_at < ttl:
                return bars
        try:
            bars = _feature_reader.load_recent_bars(symbol, tf, tail_n=tail_n) or []
            if bars:
                self._tf_data_cache.setdefault(symbol, {})[tf] = (bars, now)
        except Exception as exc:
            logger.debug("[%s] _get_tf_data(%s) load failed: %s", symbol, tf, exc)
            bars = []
            # Cache empty result for 60 s only — retry after a short back-off
            # so a transient parquet failure doesn't lock out the 4h model for
            # its full 4-hour TTL.
            self._tf_data_cache.setdefault(symbol, {})[tf] = (bars, now - self._TF_TTL.get(tf, 3600.0) + 60.0)
        return bars

    def _strategy_enabled(self, name: str, scope: str = "live") -> bool:
        """Hot-reloads strategy_config.json if it changed, then checks the flag."""
        try:
            from src.engine import strategy_registry as _reg
            mtime = _reg.CONFIG_PATH.stat().st_mtime if _reg.CONFIG_PATH.exists() else 0.0
            if mtime != self._strat_cfg_mtime:
                self._strat_cfg = _reg.load_config()
                self._strat_cfg_mtime = mtime
        except Exception:
            pass
        return self._strat_cfg.get(name, {}).get(scope, False)

    def get_real_or_sim_balance(self, asset):
        """Returns the real Binance balance"""
        return self.engine.get_balance(asset)

    def get_default_market_state(self):
        return {"last_signal": "NONE", "reason": "Waiting for data...", "wave_stage": "Initializing...", "ml_prediction_text": "Analyzing...", "ml_accuracy": 0.0, "ml_accuracy_long": 0.0, "ml_accuracy_short": 0.0, "rsi": 50.0, "volatility": 0.0, "recommended_trade_size": 0.0, "recent_pivots": []}

    async def update_market_context(self):
        """Asynchronously updates the macro context: 1D SMA200 (once an hour) and Funding Rates (every 5 minutes)"""
        current_time = time.time()
        
        # 1. Multi-Timeframe: Update 1-day SMA200 every hour
        if current_time - self.daily_sma_last_update > MTF_SMA200_REFRESH:
            for sym in self.symbols:
                try:
                    ohlcv = await asyncio.to_thread(self.engine.exchange.fetch_ohlcv, sym, '1d', limit=200)
                    if len(ohlcv) == 200:
                        sma200 = sum([x[4] for x in ohlcv]) / 200
                        self.daily_sma200[sym] = sma200
                except Exception as e:
                    logger.debug(f"Error getting 1D data for MTF: {e}")
            self.daily_sma_last_update = current_time

        # 2. Derivatives: Update Funding Rates every 5 minutes
        if current_time - self.funding_last_update > FUNDING_RATE_REFRESH:
            for sym in self.symbols:
                try:
                    future_sym = f"{sym.split('/')[0]}/USDT:USDT"
                    funding = await asyncio.to_thread(self.futures_exchange.fetch_funding_rate, future_sym)
                    self.funding_rates[sym] = funding['fundingRate']
                except Exception as e:
                    # Silent-failure review: bare `pass` was swallowing every
                    # exception type including network timeouts, auth failures
                    # and symbol-not-found. The "coin has no futures" comment
                    # only justifies ccxt.BadSymbol; everything else needs to
                    # be visible. ccxt is not always importable in all
                    # environments so we match by class name string.
                    cls = e.__class__.__name__
                    if cls in ("BadSymbol", "ExchangeError"):
                        logger.debug(
                            "[main] funding fetch skipped for %s (BadSymbol): %s", sym, e)
                    else:
                        logger.warning(
                            "[main] funding fetch failed for %s (%s): %s", sym, cls, e)
            self.funding_last_update = current_time

        # 3. Balance refresh: keep data/balance_real.json fresh so the dashboard
        # testnet card updates. Once-per-init was insufficient — the snapshot
        # went stale within hours, leaving the operator's balance card showing
        # last-restart values.
        if current_time - self.balance_last_refresh > self._BALANCE_REFRESH_SEC:
            try:
                from src.engine import dual_balance as _db
                await asyncio.to_thread(_db.refresh_real_from_binance, self.engine)
            except Exception as e:
                logger.warning("[main] balance refresh failed: %s", e)
            self.balance_last_refresh = current_time

    def check_volatility_breakout(self, df):
        """Strategy: Volatility squeeze breakout (Bollinger Bands inside Keltner Channels)"""
        if len(df) < 20: return "HOLD"
        df['sma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
        df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
        
        # Safe vectorized True Range calculation (without using .apply and .loc)
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr_20'] = df['tr'].rolling(20).mean()
        df['kc_upper'] = df['sma_20'] + 1.5 * df['atr_20']
        df['kc_lower'] = df['sma_20'] - 1.5 * df['atr_20']
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        squeeze_active = (prev['bb_upper'] < prev['kc_upper']) and (prev['bb_lower'] > prev['kc_lower'])
        vol_surge = last['volume'] > (df['volume'].rolling(20).mean().iloc[-1] * VOLATILITY_BREAKOUT_VOLUME_MULT)
        
        if squeeze_active and last['close'] > last['bb_upper'] and vol_surge: return "BUY"
        if squeeze_active and last['close'] < last['bb_lower'] and vol_surge: return "SELL"
        return "HOLD"

    def calculate_rsi(self, data, period=14):
        """Calculation of Relative Strength Index (RSI)"""
        if len(data) < period:
            return 50.0
        try:
            df = pd.DataFrame(data)
            close = pd.to_numeric(df['close'])
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            ema_gain = gain.ewm(com=period-1, adjust=False).mean()
            ema_loss = loss.ewm(com=period-1, adjust=False).mean()
            rs = ema_gain / ema_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0

    def _update_state(self, **kwargs):
        self.current_state.update(kwargs)
        self.current_state['last_update'] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            write_json(self.state_file, self.current_state)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def analyze_market_state(self, pivots, sentiment_score, ml_prediction, rsi):
        """Heuristics based on Elliott Waves."""
        if len(pivots) < 4:
            return "HOLD", "Not enough data for wave analysis.", "Data Collection (Wave 1/2)", "Data_Collection"

        p1, p2, p3, p4 = pivots[-4:]
        
        sentiment_boost = sentiment_score > SENTIMENT_BOOST_THRESHOLD
        sentiment_drag = sentiment_score < SENTIMENT_DRAG_THRESHOLD
        
        ml_bullish = ml_prediction == 1
        ml_bearish = ml_prediction == 0
        
        wave_stage = "Flat / Consolidation"

        if p4['type'] == 'high' and p2['type'] == 'high' and p4['high'] > p2['high']:
            if p3['type'] == 'low' and p1['type'] == 'low' and p3['low'] > p1['low']:
                wave_stage = "Impulse: Wave 3 or 5 🚀"
                if sentiment_drag:
                    return "HOLD", "Bullish impulse, but negative news sentiment.", wave_stage, "Neutral"
                if ml_bearish:
                    return "HOLD", "Bullish impulse, but ML model expects drop.", wave_stage, "Neutral"
                if rsi > RSI_OVERBOUGHT:
                    return "HOLD", f"Bullish impulse, but asset is overbought (RSI: {rsi:.1f}).", wave_stage, "Neutral"
                
                # Determine what made the decisive contribution to the signal
                strategy_name = "ML_Trend_Following" if ml_bullish else "Elliott_Wave_Impulse"
                return "BUY", "Bullish impulse detected (Waves 3/5).", wave_stage, strategy_name

        if p4['type'] == 'low' and p2['type'] == 'low' and p4['low'] < p2['low']:
            if p3['type'] == 'high' and p1['type'] == 'high' and p3['high'] < p1['high']:
                wave_stage = "Correction: Wave A or C 📉"
                if sentiment_boost:
                    return "HOLD", "Bearish structure, but positive news sentiment.", wave_stage, "Neutral"
                if ml_bullish:
                    return "HOLD", "Bearish structure, but ML model expects rise.", wave_stage, "Neutral"
                if rsi < RSI_OVERSOLD:
                    return "HOLD", f"Bearish structure, but asset is oversold (RSI: {rsi:.1f}).", wave_stage, "Neutral"
                return "SELL", "Bearish structure detected (ABC correction).", wave_stage, "Elliott_Wave_Correction"

        return "HOLD", "Market is flat.", wave_stage, "Neutral"

    def evaluate_all_strategies(self, symbol, data, pivots, sentiment_score, ml_prediction, rsi_value, ou_signal=0):
        """Orchestrator: Evaluates all strategies and decides which one to apply."""
        df = pd.DataFrame(data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        current_price = df['close'].iloc[-1]

        # ── Regime classification ─────────────────────────────────────────────
        regime = 1  # default TRENDING
        regime_name = "TRENDING"
        size_mult = 1.0
        if self._strategy_enabled("RegimeClassifier_Router") and self.regime_clf is not None:
            try:
                regime = self.regime_clf.predict(df)
                regime_name = self.regime_clf.regime_name(regime)
                size_mult = self.regime_clf.size_multiplier(regime)
            except Exception as e:
                logger.debug("[%s] Regime predict failed: %s", symbol, e)
        # Cache regime so process_kline can apply size_mult and adjust TFT thresholds
        self._regime_cache[symbol] = {"regime": regime, "regime_name": regime_name, "size_mult": size_mult}
        if regime == 2:  # VOLATILE — hold all entries
            return "HOLD", f"Regime: {regime_name} — holding all entries, size_mult={size_mult:.2f}", "Regime_Volatile", "RegimeClassifier_Router"

        # ── Elliott Wave + Base ML (primary signal) ───────────────────────────
        signal, reason, wave_stage, strategy_used = "HOLD", "No signal", "Neutral", "Neutral"
        if self._strategy_enabled("ElliottWave_ML"):
            signal, reason, wave_stage, strategy_used = self.analyze_market_state(
                pivots, sentiment_score, ml_prediction, rsi_value)

        # ── MTF SMA-200 filter ────────────────────────────────────────────────
        sma200 = self.daily_sma200.get(symbol)
        if self._strategy_enabled("MTF_SMA200_Filter"):
            if signal == "BUY" and sma200 and current_price < sma200:
                signal, reason = "HOLD", f"{reason} [MTF Veto: below SMA200 ({sma200:.0f})]"
            elif signal == "SELL" and sma200 and current_price > sma200:
                signal, reason = "HOLD", f"{reason} [MTF Veto: above SMA200 ({sma200:.0f})]"

        # ── OU Entry: primary signal in RANGING regime ───────────────────────
        if self._strategy_enabled("OU_Entry") and regime == 0 and signal == "HOLD":
            ou_params = self.ou_results.get(symbol, {})
            ou_mu    = float(ou_params.get("mu", 0.0))
            ou_sigma = float(ou_params.get("sigma", 0.0))
            if ou_sigma > 0 and ou_mu > 0:
                deviation = (current_price - ou_mu) / ou_sigma
                if deviation < -1.5:
                    signal, reason, strategy_used = (
                        "BUY", f"OU fade: price {abs(deviation):.1f}σ below mean ({ou_mu:.2f})", "OU_Entry")
                elif deviation > 1.5:
                    signal, reason, strategy_used = (
                        "SELL", f"OU fade: price {deviation:.1f}σ above mean ({ou_mu:.2f})", "OU_Entry")

        # ── OU filter ─────────────────────────────────────────────────────────
        if self._strategy_enabled("OU_MeanReversion_Filter"):
            if ou_signal == -1 and signal == "BUY":
                return "HOLD", f"{reason} [OU Veto: price >2σ above OU mean]", wave_stage, "Neutral"
            elif ou_signal == 1 and signal == "SELL":
                return "HOLD", f"{reason} [OU Veto: price >2σ below OU mean]", wave_stage, "Neutral"
            elif (ou_signal == 1 and signal == "BUY") or (ou_signal == -1 and signal == "SELL"):
                reason += " [OU Confirmed]"

        # ── Secondary signals (run when primary = HOLD or as override) ────────
        funding = self.funding_rates.get(symbol, 0.0)

        # Volatility Breakout
        if self._strategy_enabled("Volatility_Breakout"):
            vol_signal = self.check_volatility_breakout(df)
            if vol_signal == "BUY" and (not sma200 or current_price > sma200) and regime != 2:
                signal, reason, wave_stage, strategy_used = (
                    "BUY", "TTM Squeeze breakout + volume surge", "Breakout", "Volatility_Breakout")
            elif vol_signal == "SELL" and (not sma200 or current_price < sma200) and regime != 2:
                signal, reason, wave_stage, strategy_used = (
                    "SELL", "TTM Squeeze breakdown + volume surge", "Breakdown", "Volatility_Breakout")

        # Funding Arb / Contrarian
        if self._strategy_enabled("Funding_Arb") and signal == "HOLD":
            if funding < -FUNDING_SQUEEZE_THRESHOLD and rsi_value < 40:
                signal, reason, wave_stage, strategy_used = (
                    "BUY", f"Short-squeeze: negative funding ({funding*100:.3f}%) + RSI low",
                    "Shorts Reversal", "Funding_Arb")
            elif funding > FUNDING_SQUEEZE_THRESHOLD and rsi_value > 60:
                signal, reason, wave_stage, strategy_used = (
                    "SELL", f"Long-squeeze: high funding ({funding*100:.3f}%) + RSI high",
                    "Longs Liquidation", "Funding_Arb")

        # Group B signals — only run when primary signal is still HOLD
        if signal == "HOLD":
            from src.analysis.feature_engineering import (
                add_vwap, add_donchian, add_keltner, add_ofi,
                add_ichimoku, add_supertrend, add_macd_divergence, add_macd,
                add_bollinger_bands,
            )
            try:
                df_feat = df.copy()
                df_feat = add_vwap(df_feat)
                df_feat = add_donchian(df_feat, n=20)
                df_feat = add_keltner(df_feat)
                df_feat = add_ofi(df_feat)
                df_feat = add_bollinger_bands(df_feat)
                df_feat = add_ichimoku(df_feat)
                df_feat = add_supertrend(df_feat)
                df_feat = add_macd_divergence(df_feat)
                df_feat = add_macd(df_feat)
                last = df_feat.iloc[-1]

                # ── RANGING regime: mean-reversion signals ──────────────────
                if regime == 0:
                    if self._strategy_enabled("VWAP_Reversion"):
                        vd = float(last.get("vwap_dist", 0) or 0)
                        if vd < -0.005:
                            signal, reason, strategy_used = "BUY",  f"VWAP reversion: {vd*100:.2f}% below VWAP", "VWAP_Reversion"
                        elif vd > 0.005:
                            signal, reason, strategy_used = "SELL", f"VWAP reversion: {vd*100:.2f}% above VWAP", "VWAP_Reversion"

                    if signal == "HOLD" and self._strategy_enabled("RSI_MeanReversion"):
                        if rsi_value < 30:
                            signal, reason, strategy_used = "BUY",  f"RSI oversold ({rsi_value:.1f})", "RSI_MeanReversion"
                        elif rsi_value > 70:
                            signal, reason, strategy_used = "SELL", f"RSI overbought ({rsi_value:.1f})", "RSI_MeanReversion"

                    if signal == "HOLD" and self._strategy_enabled("BB_Reversion"):
                        bb_pb = float(last.get("bb_pb", 0.5) or 0.5)
                        if bb_pb < 0.1:
                            signal, reason, strategy_used = "BUY",  f"BB lower band touch (pb={bb_pb:.2f})", "BB_Reversion"
                        elif bb_pb > 0.9:
                            signal, reason, strategy_used = "SELL", f"BB upper band touch (pb={bb_pb:.2f})", "BB_Reversion"

                # ── TRENDING regime: breakout/momentum signals ──────────────
                elif regime == 1:
                    # Ichimoku — strong trend confirmation
                    if self._strategy_enabled("Ichimoku_Cloud"):
                        ich = float(last.get("signal_ichimoku", 0) or 0)
                        if ich == 1.0:
                            signal, reason, strategy_used = "BUY",  "Ichimoku: TK bull cross above cloud", "Ichimoku_Cloud"
                        elif ich == -1.0:
                            signal, reason, strategy_used = "SELL", "Ichimoku: TK bear cross below cloud", "Ichimoku_Cloud"

                    # Supertrend — direction flip
                    if signal == "HOLD" and self._strategy_enabled("Supertrend"):
                        st = float(last.get("signal_supertrend", 0) or 0)
                        if st == 1.0:
                            signal, reason, strategy_used = "BUY",  "SuperTrend direction flip: uptrend", "Supertrend"
                        elif st == -1.0:
                            signal, reason, strategy_used = "SELL", "SuperTrend direction flip: downtrend", "Supertrend"

                    if signal == "HOLD" and self._strategy_enabled("Donchian_Breakout"):
                        dp = float(last.get("don_pos_20", 0.5) or 0.5)
                        if dp > 0.98:
                            signal, reason, strategy_used = "BUY",  "Donchian 20-bar high break", "Donchian_Breakout"
                        elif dp < 0.02:
                            signal, reason, strategy_used = "SELL", "Donchian 20-bar low break",  "Donchian_Breakout"

                    if signal == "HOLD" and self._strategy_enabled("Keltner_Breakout"):
                        kcp = float(last.get("kc_pos", 0.5) or 0.5)
                        if kcp > 1.0:
                            signal, reason, strategy_used = "BUY",  f"Keltner channel breakout up (pos={kcp:.2f})", "Keltner_Breakout"
                        elif kcp < 0.0:
                            signal, reason, strategy_used = "SELL", f"Keltner channel breakout down (pos={kcp:.2f})", "Keltner_Breakout"

                    if signal == "HOLD" and self._strategy_enabled("OFI_Momentum"):
                        ofi_z = float(last.get("ofi_z", 0) or 0)
                        if ofi_z > 1.5:
                            signal, reason, strategy_used = "BUY",  f"OFI buy pressure (z={ofi_z:.2f})", "OFI_Momentum"
                        elif ofi_z < -1.5:
                            signal, reason, strategy_used = "SELL", f"OFI sell pressure (z={ofi_z:.2f})", "OFI_Momentum"

                    # MACD Divergence — higher-quality MACD signals
                    if signal == "HOLD" and self._strategy_enabled("MACD_Divergence"):
                        md = float(last.get("signal_macd_div", 0) or 0)
                        if md == 1.0:
                            signal, reason, strategy_used = "BUY",  "MACD centerline cross or bullish divergence", "MACD_Divergence"
                        elif md == -1.0:
                            signal, reason, strategy_used = "SELL", "MACD centerline cross or bearish divergence", "MACD_Divergence"

                    if signal == "HOLD" and self._strategy_enabled("MACD_Momentum"):
                        hist = float(last.get("macd_hist", 0) or 0)
                        if hist > 0:
                            signal, reason, strategy_used = "BUY",  f"MACD momentum (hist={hist:.4f})", "MACD_Momentum"
                        elif hist < 0:
                            signal, reason, strategy_used = "SELL", f"MACD momentum (hist={hist:.4f})", "MACD_Momentum"

                # ── Ensemble fallback (any regime, last resort) ─────────────
                if signal == "HOLD":
                    votes = 0.0
                    n = 0
                    for col, en in [("signal_rsi", "RSI_MeanReversion"),
                                    ("signal_macd", "MACD_Momentum"),
                                    ("signal_bb", "BB_Reversion")]:
                        if self._strategy_enabled(en):
                            rsi_sig  = 1.0 if rsi_value < 30 else (-1.0 if rsi_value > 70 else 0.0)
                            macd_sig = 1.0 if float(last.get("macd_hist", 0) or 0) > 0 else -1.0
                            bb_sig   = (1.0 if float(last.get("bb_pb", 0.5) or 0.5) < 0.1
                                       else (-1.0 if float(last.get("bb_pb", 0.5) or 0.5) > 0.9 else 0.0))
                            sigs = {"signal_rsi": rsi_sig, "signal_macd": macd_sig, "signal_bb": bb_sig}
                            votes += sigs.get(col, 0.0)
                            n += 1
                    if n >= 2:
                        avg = votes / n
                        if self._strategy_enabled("Ensemble_A") and avg >= 0.67:
                            signal, reason, strategy_used = "BUY",  f"Ensemble A consensus (score={avg:.2f})", "Ensemble_A"
                        elif self._strategy_enabled("Ensemble_A") and avg <= -0.67:
                            signal, reason, strategy_used = "SELL", f"Ensemble A consensus (score={avg:.2f})", "Ensemble_A"

            except Exception as e:
                logger.warning("[%s] Group B signal computation failed: %s", symbol, e)

        # ── Meta-Labeler final gate ───────────────────────────────────────────
        if signal != "HOLD" and self._strategy_enabled("MetaLabeler_Filter") and self.meta_labeler is not None:
            try:
                if self.meta_labeler.is_loaded:
                    sig_num = 1.0 if signal == "BUY" else -1.0
                    last_row = df.iloc[-1].to_dict()
                    last_row.setdefault('prob_base', 0.5)
                    last_row.setdefault('prob_trend', 0.5)
                    last_row.setdefault('regime', 0)
                    result = self.meta_labeler.filter(sig_num, last_row)
                    if result[0] == "BLOCK":
                        signal, reason = "HOLD", f"[Meta-Labeler blocked] {reason}"
                        strategy_used = "MetaLabeler_Filter"
            except Exception as e:
                logger.debug("[%s] Meta-labeler filter failed: %s", symbol, e)

        return signal, reason, wave_stage, strategy_used

    def evaluate_scalping_strategy(self, symbol, data_1m):
        """Analyzes 1-minute data for scalping."""
        if not self.scalping_predictor.is_loaded:
            return "HOLD", "Scalping model not loaded", "Scalping_Model_Missing"

        # Runtime kill-list — set via dashboard Risk sub-tab. Lets the user
        # halt the 1m scalp path on specific symbols without restarting the
        # bot. The 1h macro path on these symbols continues normally.
        if _runtime_overrides.is_scalping_disabled(symbol):
            return "HOLD", f"Scalping disabled for {symbol} via runtime override", "Scalping_Disabled"

        if not data_1m or len(data_1m) < 20:
            return "HOLD", "Not enough 1m data for scalping", "Data_Collection"

        # Calculation of fast indicators for scalping
        df = pd.DataFrame(data_1m)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])
        
        delta = df['close'].diff()
        rsi_7 = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=6, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=6, adjust=False).mean())))
        rsi_7_val = rsi_7.iloc[-1]

        # Get prediction from the scalping model
        scalp_pred = self.scalping_predictor.predict(data_1m, symbol=symbol)

        if scalp_pred is None and hasattr(self.scalping_predictor, 'last_error') and self.scalping_predictor.last_error:
            return "HOLD", self.scalping_predictor.last_error, "ML_Error"

        if scalp_pred == 1 and rsi_7_val < SCALPING_RSI_OVERBOUGHT:
            return "BUY", f"Scalping signal UP (RSI: {rsi_7_val:.1f})", "Scalping_Long"

        if scalp_pred == 0 and rsi_7_val > SCALPING_RSI_OVERSOLD:
            return "SELL", f"Scalping signal DOWN (RSI: {rsi_7_val:.1f})", "Scalping_Short"

        return "HOLD", "No scalping signal", "Neutral"

    def _force_close_internally(self, trade, current_price, reason: str) -> bool:
        """Mark position closed locally without hitting the exchange (balance already 0 or unrecoverable)."""
        symbol = trade['symbol']
        market = trade.get('market', 'SPOT')
        closed_trade = self.tracker.close_trade_by_id(trade['id'], current_price)
        if closed_trade:
            logger.warning(
                f"[{symbol}] Force-closed {market} position ID {trade['id']} internally at {current_price:.8f}. "
                f"Reason: {reason}. PnL: {closed_trade['pnl_usdt']:.2f} USDT"
            )
        MultiAssetTrader._close_fail_counts.pop(trade['id'], None)
        return True

    def _execute_close(self, trade, current_price):
        """Physically closes the order on the exchange, then records it locally.

        Handles "insufficient balance" edge cases:
        - If the free balance for the base asset is 0, the position was already sold/dust
          on Binance — force-close it internally.
        - After _CLOSE_FAIL_LIMIT consecutive failures, force-close internally to prevent
          endless retry loops.
        """
        symbol = trade['symbol']
        amount_coin = trade['amount_coin']
        market = trade.get('market', 'SPOT')
        side = trade.get('side', 'LONG')
        trade_id = trade['id']
        order = None

        if market in ['SPOT', 'SPOT_SCALPING']:
            # Pre-check: does Binance actually hold the base asset?
            base_asset = symbol.split('/')[0]
            try:
                free_balance = self.engine.get_balance(base_asset)
            except Exception:
                free_balance = None  # proceed and let the order attempt surface the real error

            if free_balance is not None and free_balance < 1e-8:
                return self._force_close_internally(
                    trade, current_price,
                    f"free {base_asset} balance is 0 on Binance (dust/already sold)"
                )

            # Also guard against sub-minimum-notional situations (e.g. SHIB dust)
            try:
                self.engine.exchange.load_markets()
                market_info = self.engine.exchange.markets.get(symbol, {})
                min_notional = (market_info.get('limits', {}).get('cost', {}) or {}).get('min') or 0.0
                min_amount = (market_info.get('limits', {}).get('amount', {}) or {}).get('min') or 0.0
                notional_value = (free_balance or 0.0) * current_price
                if (free_balance is not None and free_balance < min_amount) or (min_notional and notional_value < min_notional):
                    return self._force_close_internally(
                        trade, current_price,
                        f"{base_asset} balance {free_balance:.2f} below exchange minimum "
                        f"(min_amount={min_amount}, notional={notional_value:.4f} < {min_notional})"
                    )
            except Exception:
                pass  # market info unavailable — let the order attempt proceed

            order = self.engine.execute_spot_order(symbol, 'SELL', amount_coin)

        elif market in ['FUTURES', 'SCALPING']:
            close_side = 'SELL' if side == 'LONG' else 'BUY'

            # Pre-check: does Binance actually hold an open futures position?
            # If exchange-side size is 0 (closed externally / liquidated /
            # SL hit on exchange), reduceOnly will deterministic-fail with
            # -2022. Force-close internally instead of cascading through 3
            # retry attempts × ~7 min apart.
            try:
                exch_pos = self.engine.get_futures_position_amount(symbol)
            except Exception:
                exch_pos = None  # treat as unknown → proceed with close attempt
            if exch_pos is not None and exch_pos < 1e-8:
                return self._force_close_internally(
                    trade, current_price,
                    "exchange has 0 contracts (already closed)"
                )

            order = self.engine.execute_futures_order(symbol, close_side, amount_coin, reduce_only=True)

            # Sentinel: order_manager returned the -2022 marker because the
            # exchange position vanished between the pre-check and the
            # market order. Same outcome: no retry, force-close locally.
            if isinstance(order, dict) and order.get('reduce_only_rejected'):
                return self._force_close_internally(
                    trade, current_price,
                    "Binance reduceOnly rejected (-2022) — position no longer open"
                )

        if order:
            real_price = order.get('average') or order.get('price') or current_price
            closed_trade = self.tracker.close_trade_by_id(trade_id, real_price)
            if closed_trade:
                logger.info(
                    f"[{symbol}] Closed {market} position ID {trade_id} on Binance. "
                    f"PnL: {closed_trade['pnl_usdt']:.2f} USDT"
                )
            MultiAssetTrader._close_fail_counts.pop(trade_id, None)
            return True
        else:
            # Count consecutive failures; force-close after limit
            fails = MultiAssetTrader._close_fail_counts.get(trade_id, 0) + 1
            MultiAssetTrader._close_fail_counts[trade_id] = fails
            if fails >= MultiAssetTrader._CLOSE_FAIL_LIMIT:
                logger.error(
                    f"[{symbol}] Close failed {fails}x for ID {trade_id} — force-closing internally."
                )
                return self._force_close_internally(
                    trade, current_price,
                    f"exchange SELL failed {fails} consecutive times"
                )
            logger.warning(
                f"[{symbol}] Failed to close {market} position ID {trade_id} on Binance "
                f"(attempt {fails}/{MultiAssetTrader._CLOSE_FAIL_LIMIT}). Will retry."
            )
            return False

    async def process_kline(self, symbol, current_price):
        logger.info(f"[{symbol}] WebSocket tick processed - Price: {current_price}")
        self.current_state["prices"][symbol] = current_price
        # Phase 9 — gate is informed of fresh data so the staleness breaker
        # (§18) doesn't fire spuriously on the next pre-trade check.
        if getattr(self, "gate", None) is not None:
            self.gate.mark_data_tick()

        # Cross-sectional momentum: update whenever any price arrives
        live_prices = {s: p for s, p in self.current_state["prices"].items() if p > 0}
        if len(live_prices) >= 2:
            updated = self.momentum_engine.update(live_prices)
            if updated:
                self.momentum_signals = updated

        await self.update_market_context()
        
        # 1. Risk priority: Check trailing stops. Execute on Binance before closing internally.
        triggered_stops = self.tracker.update_trailing_stops(symbol, current_price)
        for t in triggered_stops:
            logger.info(f"[{symbol}] 🛡️ TRAILING STOP TRIGGERED for ID {t['id']} ({t['market']}). Executing on Binance...")
            self._execute_close(t, current_price)

        # Phase 10B — alpha-decay exit: close trades whose signal has decayed
        # below threshold. `signal_strength` is recorded at entry by
        # _execute_close's counterpart (_record_entry); falls back to 1.0 for
        # legacy trades that pre-date the tracking dict.
        if self.gate is not None:
            try:
                open_trades = [
                    tr for tr in self.tracker.list_open()
                    if tr.get("symbol") == symbol
                ] if hasattr(self.tracker, "list_open") else []
                for tr in open_trades:
                    tid = tr.get("id")
                    s0 = float(self._signal_strength_at_entry.get(tid, 1.0))
                    t0 = float(self._signal_entry_ts.get(tid, time.time()))
                    bars_open = max(0.0, (time.time() - t0) / 3600.0)   # 1h-bar units
                    if self.gate.should_exit_decay(s0, bars_open):
                        logger.info(f"[{symbol}] ⏳ alpha-decay exit for {tid}")
                        self._execute_close(tr, current_price)
            except Exception as e:
                logger.debug(f"[{symbol}] alpha-decay check skipped: {e}")

        sentiment = self.sentiment_analyzer.get_average_sentiment()
        
        # --- MACRO ANALYSIS (1h timeframe for Spot & Futures) ---
        logger.info(f"[{symbol}][1h] Starting macro analysis...")
        # Phase 10 — Parquet-first read (falls back to CSV.gz if no parquet yet).
        data = _feature_reader.load_recent_bars(symbol, self.timeframe, tail_n=1000)
        if not data:
            data_filepath = f"data/raw/{symbol.replace('/', '_')}_{self.timeframe}.csv.gz"
            data = self.analyzers[symbol].load_data(data_filepath)

        # If the file is empty or data is insufficient (corrupted file), force download again
        if not data or len(data) < 50:
            logger.info(f"[{symbol}] Insufficient data, forcing history download...")
            download_history(symbol=symbol, timeframe=self.timeframe, limit=1000)
            data = (_feature_reader.load_recent_bars(symbol, self.timeframe, tail_n=1000)
                    or self.analyzers[symbol].load_data(
                        f"data/raw/{symbol.replace('/', '_')}_{self.timeframe}.csv.gz"))
            
        if not data:
            self._update_state(status=f"Data error {symbol}", reason="Unable to load candle history")
            return
            
        data[-1]['close'] = current_price
        
        pivots = self.analyzers[symbol].calculate_zigzag(data)
        
        # Automatic scaling: if there are less than 5 waves (flat market), look for micro-waves
        current_dev = WAVE_DEVIATION_DEFAULT
        while (not pivots or len(pivots) < 5) and current_dev >= WAVE_DEVIATION_MIN:
            current_dev -= WAVE_DEVIATION_STEP
            fallback_analyzer = ElliottWaveAnalyzer(deviation_percent=current_dev)
            pivots = fallback_analyzer.calculate_zigzag(data)
            
        if pivots is None:
            pivots = []
            
        volatility = self.risk_managers[symbol].calculate_historical_volatility(data)
        trade_amount = self.risk_managers[symbol].get_position_size(data)
        trade_amount = max(MIN_TRADE_USDT, trade_amount)

        # --- GARCH Volatility Spike Detection: halve position when regime is unstable ---
        garch_result = {}
        try:
            returns_series = pd.Series([
                float(np.log(float(data[i]['close']) / float(data[i-1]['close'])))
                for i in range(1, len(data))
                if float(data[i-1]['close']) > 0 and float(data[i]['close']) > 0
            ])
            garch_result = self.risk_managers[symbol].vol_forecaster.forecast_garch(returns_series)
            if garch_result.get('volatility_spike'):
                trade_amount = max(MIN_TRADE_USDT, trade_amount * 0.5)
                logger.warning(f"[{symbol}] ⚠️ GARCH spike! Position halved → {trade_amount:.2f} USDT")
        except Exception as e:
            logger.debug(f"[{symbol}] GARCH skipped: {e}")

        # Phase A — route each ML model to its pinned timeframe (precomputed
        # in __init__ as self._base_tf / _trend_tf / _fut_tf).
        # _get_tf_data() fetches+caches bars at that TF.  Falls back to
        # canonical .predict(data) when the per-TF model isn't on disk or
        # the data fetch returns empty.  Uses explicit `is not None` so a
        # valid bearish signal (0) is never discarded by a falsy short-circuit.
        _base_data  = self._get_tf_data(symbol, self._base_tf)  if self._base_tf  != self.timeframe else data
        _trend_data = self._get_tf_data(symbol, self._trend_tf) if self._trend_tf != self.timeframe else data
        _raw_ml    = self.ml_predictor.predict_at(self._base_tf, _base_data)    if _base_data  else None
        _raw_trend = self.trend_predictor.predict_at(self._trend_tf, _trend_data) if _trend_data else None
        ml_pred    = _raw_ml    if _raw_ml    is not None else self.ml_predictor.predict(data, symbol=symbol)
        trend_pred = _raw_trend if _raw_trend is not None else self.trend_predictor.predict(data, symbol=symbol)
        rsi_value = self.calculate_rsi(data)
        
        # --- 📈 OU Process (Ornstein-Uhlenbeck) + Feature Store ---
        try:
            closes = np.array([float(d['close']) for d in data[-200:]])
            ou_params = self.mean_reversion.calibrate_ou_process(closes)
            if ou_params:
                ou_signal = float(ou_params['signal'])
                self.ou_results[symbol] = ou_params
                logger.debug(f"[{symbol}] OU: signal={ou_signal} mu={ou_params['mu']:.2f} sigma={ou_params['sigma']:.4f}")
            else:
                ou_signal = 0.0
        except Exception as e:
            ou_signal = 0.0
            logger.debug(f"[{symbol}] OU skipped: {e}")
        self.feature_store.update_features(symbol, pd.DataFrame(data), sentiment, volatility, ou_signal)
        
        # --- 🧠 TFT Inference Engine & Avellaneda-Stoikov Market Maker ---
        # evaluate_all_strategies must run first to populate _regime_cache
        signal, reason, wave_stage, strategy_used = self.evaluate_all_strategies(symbol, data, pivots, sentiment, trend_pred, rsi_value, ou_signal=ou_signal)

        # Read regime result cached by evaluate_all_strategies
        _reg = self._regime_cache.get(symbol, {"regime": 1, "regime_name": "TRENDING", "size_mult": 1.0})
        _regime      = _reg["regime"]
        _regime_name = _reg["regime_name"]
        _size_mult   = _reg["size_mult"]

        # Apply regime size multiplier to position size
        # VOLATILE already returns HOLD above; RANGING gets 0.6× (mean-rev sizes smaller)
        trade_amount = max(MIN_TRADE_USDT, trade_amount * _size_mult)

        # Regime-aware TFT threshold: RANGING markets are noisy → raise threshold to reduce false signals
        _tft_threshold = {0: 0.02, 1: 0.01, 2: 0.03}.get(_regime, 0.01)  # RANGING=2%, TRENDING=1%, VOLATILE=3%
        # Phase 10D — overlay the data-driven threshold from gate.best_threshold()
        # when the dynamic refresh has produced a value for this symbol.
        try:
            self._refresh_dynamic_thresholds()
            if symbol in self._dyn_thresholds:
                _dyn = float(self._dyn_thresholds[symbol])
                # Blend: keep regime base, but lift floor toward learned threshold.
                _tft_threshold = max(_tft_threshold, min(_dyn / 100.0, 0.05))
        except Exception:
            pass

        tft_pred = self.inference_engine.get_latest_prediction(symbol)
        expected_return = tft_pred["expected_return"] if tft_pred else 0.0
        oft_pred = (tft_pred or {}).get("oft") or {}

        # ── OFT_Microstructure: filter + confidence-weight ────────────────────
        # Filter — block entries when the OFT model is unsure or the order
        #          book looks fragile.
        # Weight — when the trade survives the filter, scale notional by the
        #          calibrated p_move so stronger signals get bigger size.
        oft_block = False
        oft_block_reason = ""
        oft_weight = 1.0
        oft_active = (
            oft_pred
            and self._strategy_enabled("OFT_Microstructure")
        )
        if oft_active:
            p_move = float(oft_pred.get("p_move_calibrated",
                                        oft_pred.get("p_move", 0.5)))
            liq_risk = float(oft_pred.get("liquidity_risk", 0.0))
            if p_move < OFT_GATE_P_MOVE_MIN:
                oft_block = True
                oft_block_reason = f"OFT p_move {p_move:.2f} < {OFT_GATE_P_MOVE_MIN:.2f}"
            elif liq_risk > OFT_GATE_LIQ_RISK_MAX:
                oft_block = True
                oft_block_reason = f"OFT liquidity_risk {liq_risk:.2f} > {OFT_GATE_LIQ_RISK_MAX:.2f}"
            else:
                # Linear map p_move ∈ [GATE_MIN, 1.0] → weight ∈ [FLOOR, CEILING]
                span_p = max(1e-6, 1.0 - OFT_GATE_P_MOVE_MIN)
                norm   = max(0.0, min(1.0, (p_move - OFT_GATE_P_MOVE_MIN) / span_p))
                oft_weight = OFT_WEIGHT_FLOOR + norm * (OFT_WEIGHT_CEILING - OFT_WEIGHT_FLOOR)
            if oft_block:
                logger.info(f"[{symbol}] 🚫 OFT BLOCK: {oft_block_reason}")
            else:
                trade_amount = float(trade_amount) * oft_weight

        # Runtime risk override — hard cap on position notional. Set via
        # the dashboard's Risk sub-tab (data/runtime_overrides.json).
        # `None` means no cap. Applies AFTER Kelly + GARCH + OFT weight.
        _cap = _runtime_overrides.max_position_cap()
        if _cap is not None and float(trade_amount) > float(_cap):
            logger.info(f"[{symbol}] 🛑 Runtime cap: trimming {trade_amount:.2f} → {_cap:.2f} USDT")
            trade_amount = float(_cap)

        # Min-notional floor — Binance rejects orders smaller than ~$50 with
        # `code:-4164 Order's notional must be no smaller than 50`. Kelly +
        # GARCH halving + OFT weight can stack down to ~$13–25, so enforce
        # MIN_TRADE_USDT ($55) as a hard floor here. If the *intended* size
        # is below the floor we skip the trade entirely instead of submitting
        # a doomed order. The earlier MIN_TRADE_USDT check at signal time
        # only guards the BASE trade size — multipliers can shrink below it.
        _intended = float(trade_amount or 0)
        if 0 < _intended < MIN_TRADE_USDT:
            logger.info(
                f"[{symbol}] ⏭ Trade skipped — intended size {_intended:.2f} USDT "
                f"below Binance min-notional floor (MIN_TRADE_USDT={MIN_TRADE_USDT})"
            )
            return  # bail out of this evaluate_market call before order submission

        base_asset = symbol.split('/')[0]
        inventory_q = self.get_real_or_sim_balance(base_asset)
        mm_quotes = self.market_makers[symbol].calculate_quotes(current_price, inventory_q, volatility)

        # Phase 9 — pre-trade gate (§17 beta neutrality, §18 circuit breakers).
        # Returns ok=True when no Phase-1-5 module objects to the trade.
        def _gate_ok(side: str) -> bool:
            if self.gate is None:
                return True
            usdt_eq = float(self.get_real_or_sim_balance('USDT') or 0)
            self.gate.update_peak_equity(usdt_eq)
            res = self.gate.pre_trade_check(
                symbol=symbol, side=side, notional=float(trade_amount),
                current_equity=usdt_eq,
                api_latency_ms=float(getattr(self.engine, "last_api_latency_ms", 0) or 0),
            )
            if not res["ok"]:
                logger.warning(f"[{symbol}] 🛑 gate blocked {side}: {res['reasons']}")
            return res["ok"]

        if expected_return <= -_tft_threshold:
            logger.warning(f"[{symbol}] 📉 TFT predicts DROP ({expected_return*100:.1f}%) [{_regime_name}]. Shifting to SHORT.")
            if oft_block:
                logger.info(f"[{symbol}] 🚫 OFT vetoed SHORT entry: {oft_block_reason}")
            elif _gate_ok("sell"):
                _allow_mm, _sized_mm = self._check_pre_trade(
                    symbol, action="open", trade_usdt=float(trade_amount))
                if _allow_mm:
                    try:
                        self.engine.cancel_all_orders(symbol)
                        # §16 slippage-aware limit price (small adjustment)
                        ask = float(mm_quotes.get("optimal_ask", current_price))
                        if self.gate:
                            ask = self.gate.executed_price(
                                ask, "sell", float(_sized_mm / current_price),
                                book_volume=max(float(_sized_mm / current_price) * 5, 1.0),
                            )
                        self.engine.execute_limit_futures_order(symbol, "SELL", _sized_mm / current_price, ask)
                        if self.gate:
                            self.gate.update_position(symbol, "short", float(_sized_mm))
                    except Exception as e:
                        logger.error(f"MM Execution Error: {e}")
        elif expected_return >= _tft_threshold:
            logger.info(f"[{symbol}] 🚀 TFT predicts PUMP ({expected_return*100:.1f}%) [{_regime_name}]. Shifting to LONG.")
            if oft_block:
                logger.info(f"[{symbol}] 🚫 OFT vetoed LONG entry: {oft_block_reason}")
            elif _gate_ok("buy"):
                _allow_mm, _sized_mm = self._check_pre_trade(
                    symbol, action="open", trade_usdt=float(trade_amount))
                if _allow_mm:
                    try:
                        self.engine.cancel_all_orders(symbol)
                        bid = float(mm_quotes.get("optimal_bid", current_price))
                        if self.gate:
                            bid = self.gate.executed_price(
                                bid, "buy", float(_sized_mm / current_price),
                                book_volume=max(float(_sized_mm / current_price) * 5, 1.0),
                            )
                        self.engine.execute_limit_futures_order(symbol, "BUY", _sized_mm / current_price, bid)
                        if self.gate:
                            self.gate.update_position(symbol, "long", float(_sized_mm))
                    except Exception as e:
                        logger.error(f"MM Execution Error: {e}")

        # --- 📊 Push quant metrics to dashboard state (read by /api/state) ---
        _ou_r = self.ou_results.get(symbol, {})
        _ou_mu = float(_ou_r.get('mu', 0.0))
        _ou_sigma = float(_ou_r.get('sigma', 0.0))
        _ou_dev = round((current_price - _ou_mu) / _ou_sigma, 2) if _ou_sigma > 0 else 0.0
        self.current_state.setdefault("quant", {})[symbol] = {
            "ou_signal":      int(ou_signal),
            "ou_mu":          round(_ou_mu, 4),
            "ou_deviation":   _ou_dev,
            "garch_spike":    bool(garch_result.get("volatility_spike", False)),
            "garch_forecast": round(float(garch_result.get("forecast_volatility", 0.0)) * 100, 4),
            "garch_hist":     round(float(garch_result.get("historical_volatility", 0.0)) * 100, 4),
            "garch_status":   garch_result.get("status", "pending"),
            "tft_return":     round(expected_return * 100, 2),
            "tft_threshold":  round(_tft_threshold * 100, 1),
            "oft_active":     bool(oft_active),
            "oft_p_move":     round(float(oft_pred.get("p_move_calibrated",
                                                        oft_pred.get("p_move", 0))), 4) if oft_pred else None,
            "oft_liq_risk":   round(float(oft_pred.get("liquidity_risk", 0)), 4) if oft_pred else None,
            "oft_weight":     round(oft_weight, 3),
            "oft_blocked":    bool(oft_block),
            "regime":         _regime_name,
            "size_mult":      round(_size_mult, 2),
            "as_bid":         round(float(mm_quotes.get("optimal_bid", 0.0)), 2),
            "as_ask":         round(float(mm_quotes.get("optimal_ask", 0.0)), 2),
            "as_spread":      round(float(mm_quotes.get("optimal_spread", 0.0)), 2),
            "as_reservation": round(float(mm_quotes.get("reservation_price", 0.0)), 2),
            "momentum_signal": round(float(self.momentum_signals.get(symbol, 0.0)), 3),
        }
        
        if not self.ml_predictor.is_loaded:
            ml_text = "MODEL NOT FOUND"
        elif ml_pred is None:
            # last_status is the authoritative categorization. Only show
            # "ERROR" when something actually broke; "low_confidence" /
            # "no_data" / "not_loaded" each get their own user-visible label.
            _status = getattr(self.ml_predictor, 'last_status', '')
            if _status == 'error':
                ml_text = "ERROR"
                if self.ml_predictor.last_error:
                    reason += f" | {self.ml_predictor.last_error}"
            elif _status == 'low_confidence':
                _conf = getattr(self.ml_predictor, '_last_confidence', 0.5)
                ml_text = f"LOW CONF ({_conf:.2f})"
            elif _status == 'not_loaded':
                ml_text = "MODEL NOT FOUND"
            else:
                ml_text = "DATA COLLECTION"
        elif ml_pred == 1:
            ml_text = "UP 🔼 (Buy)"
        elif ml_pred == 0:
            ml_text = "DOWN 🔽 (Sell)"
        else:
            ml_text = "DATA COLLECTION"
        
        logger.info(f"[{symbol}] SIGNAL: [{signal}] | Reason: {reason} | Snt: {sentiment:.2f} | ML: {ml_pred} | RSI: {rsi_value:.1f}")
        
        formatted_pivots = [{"type": p.get("type", "unknown"), "close": p.get("close", 0), "timestamp": p.get("timestamp", "")} for p in pivots[-5:]] if pivots else []
        
        self.current_state["market_data"]["SPOT"][symbol] = {
            "last_signal": signal,
            "reason": reason,
            "wave_stage": wave_stage,
            "ml_prediction_text": ml_text,
            "ml_accuracy": round(self.ml_predictor.accuracy, 2),
            "ml_accuracy_long": round(self.ml_predictor.long_accuracy, 2),
            "ml_accuracy_short": round(self.ml_predictor.short_accuracy, 2),
            "rsi": round(rsi_value, 2),
            "volatility": round(volatility*100, 2), 
            "recommended_trade_size": trade_amount,
            "recent_pivots": formatted_pivots,
            "tft_prediction_pct": round(expected_return * 100, 3),
            "ou_mean_reversion": ou_signal,
            "funding_rate": round(self.funding_rates.get(symbol, 0.0) * 100, 4),
            "regime": _regime_name,
            "size_mult": round(_size_mult, 2),
        }
        
        # Build FUTURES state with specific Futures ML Model predictions
        _fut_data = self._get_tf_data(symbol, self._fut_tf) if self._fut_tf != self.timeframe else data
        _raw_fut  = self.futures_predictor.predict_at(self._fut_tf, _fut_data) if _fut_data else None
        fut_pred  = _raw_fut if _raw_fut is not None else self.futures_predictor.predict(data, symbol=symbol)
        if not self.futures_predictor.is_loaded:
            fut_ml_text = "MODEL NOT FOUND"
        elif fut_pred is None:
            _fs = getattr(self.futures_predictor, 'last_status', '')
            if _fs == 'error':
                fut_ml_text = "ERROR"
            elif _fs == 'low_confidence':
                _fc = getattr(self.futures_predictor, '_last_confidence', 0.5)
                fut_ml_text = f"LOW CONF ({_fc:.2f})"
            else:
                fut_ml_text = "DATA COLLECTION"
        elif fut_pred == 1:
            fut_ml_text = "DOWN 🔽 (Short)"
        else:
            fut_ml_text = "UP/HOLD 🔼"

        self.current_state["market_data"]["FUTURES"][symbol] = self.current_state["market_data"]["SPOT"][symbol].copy()
        self.current_state["market_data"]["FUTURES"][symbol].update({
            "ml_prediction_text": fut_ml_text,
            "ml_accuracy": round(self.futures_predictor.accuracy, 2),
            "ml_accuracy_long": round(self.futures_predictor.long_accuracy, 2),
            "ml_accuracy_short": round(self.futures_predictor.short_accuracy, 2)
        })

        # --- SCALPING ANALYSIS (1m timeframe) ---
        logger.info(f"[{symbol}][SCALPING] Starting 1m analysis...")
        # Phase 10 — Parquet-first
        data_1m = (_feature_reader.load_recent_bars(symbol, "1m", tail_n=500)
                   or self.analyzers[symbol].load_data(
                       f"data/raw/{symbol.replace('/', '_')}_1m.csv.gz", tail_n=500))
        if data_1m:
            # Add current candle to history
            data_1m.append({'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'open': current_price, 'high': current_price, 'low': current_price, 'close': current_price, 'volume': 0})
            
            scalp_signal, scalp_reason, scalp_strategy = self.evaluate_scalping_strategy(symbol, data_1m)
            logger.info(f"[{symbol}][SCALPING] SIGNAL: [{scalp_signal}] | Reason: {scalp_reason}")
            
            scalp_ml_text = "ERROR" if scalp_strategy == "ML_Error" else (
                "UP 🔼 (Buy)" if scalp_signal == "BUY" else ("DOWN 🔽 (Sell)" if scalp_signal == "SELL" else "NO SIGNAL")
            )
            
            self.current_state["market_data"]["SCALPING"][symbol] = {
                "last_signal": scalp_signal,
                "reason": scalp_reason,
                "wave_stage": "N/A", # Waves are not used in scalping
                "ml_prediction_text": scalp_ml_text,
                "ml_accuracy": round(self.scalping_predictor.accuracy, 2),
                "ml_accuracy_long": round(self.scalping_predictor.long_accuracy, 2),
                "ml_accuracy_short": round(self.scalping_predictor.short_accuracy, 2),
                "rsi": self.calculate_rsi(data_1m, period=7), # Fast RSI
                "volatility": self.risk_managers[symbol].calculate_historical_volatility(data_1m) * 100,
                "recommended_trade_size": max(MIN_TRADE_USDT, self.risk_managers[symbol].get_position_size(data_1m) * SCALPING_TRADE_FRACTION),
                "recent_pivots": []
            }
            
            if scalp_signal in ["BUY", "SELL"]:
                side = "LONG" if scalp_signal == "BUY" else "SHORT"
                scalp_trade_amount = self.current_state["market_data"]["SCALPING"][symbol]["recommended_trade_size"]
                amount_coin = scalp_trade_amount / current_price
                exec_side = 'BUY' if side == "LONG" else 'SELL'
                
                # 1. Scalping on Futures (LONG and SHORT)
                has_open_scalp_futures = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("market") == "SCALPING" for t in self.tracker.trades)
                if not has_open_scalp_futures:
                    _allow, _sized = self._check_pre_trade(
                        symbol, action="open", trade_usdt=scalp_trade_amount)
                    if _allow:
                        amount_coin = _sized / current_price
                        order = self.engine.execute_futures_order(symbol, exec_side, amount_coin)
                        if order:
                            real_price = order.get('average') or order.get('price') or current_price
                            self.tracker.open_trade(symbol, _sized, real_price, strategy=scalp_strategy, market="SCALPING", side=side)
                            logger.info(f"-> [SCALPING FUTURES] Trade {side} {_sized:.2f} USDT opened on Binance (Real Price: {real_price}).")

                # 2. Scalping on Spot (LONG only)
                if scalp_signal == "BUY":
                    has_open_scalp_spot = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("market") == "SPOT_SCALPING" for t in self.tracker.trades)
                    if not has_open_scalp_spot:
                        _allow_s, _sized_s = self._check_pre_trade(
                            symbol, action="open", trade_usdt=scalp_trade_amount)
                        if _allow_s:
                            amount_coin_s = _sized_s / current_price
                            order = self.engine.execute_spot_order(symbol, 'BUY', amount_coin_s)
                            if order:
                                real_price = order.get('average') or order.get('price') or current_price
                                self.tracker.open_trade(symbol, _sized_s, real_price, strategy=scalp_strategy, market="SPOT_SCALPING", side="LONG")
                                logger.info(f"-> [SCALPING SPOT] Trade LONG {_sized_s:.2f} USDT opened on Binance (Real Price: {real_price}).")
                            
                # 3. Scalping on Spot (Instant close LONG on SELL signal)
                elif scalp_signal == "SELL":
                    open_spot_scalps = self.tracker.get_open_trades(symbol=symbol, side="LONG", market="SPOT_SCALPING")
                    for t in open_spot_scalps: self._execute_close(t, current_price)
        
        self._update_state(
            status=f"Analysis for {symbol} complete",
            balance_usdt=self.get_real_or_sim_balance('USDT'),
            balance_btc=self.get_real_or_sim_balance('BTC'),
            balance_sol=self.get_real_or_sim_balance('SOL'),
            balance_ada=self.get_real_or_sim_balance('ADA'),
            sentiment_score=sentiment
        )

        if signal == "BUY":
            # 1. Close possible shorts on Futures (position reversal)
            open_shorts = self.tracker.get_open_trades(symbol=symbol, side="SHORT", market="FUTURES")
            for t in open_shorts: self._execute_close(t, current_price)
            
            # Protection against order spam: check if there is an open position for this coin
            has_open = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("side") == "LONG" and t.get("market") == "SPOT" for t in self.tracker.trades)
            if has_open:
                logger.info(f"[{symbol}] BUY signal ignored: position already open.")
            else:
                logger.info(f"[{symbol}] 🤖 Requesting Gemini Agent for macro check...")
                headlines = getattr(self.sentiment_analyzer, 'cached_headlines', [])[:10]
                decision, agent_reason = self.agent.evaluate_trade(symbol, "BUY", reason, headlines, self.telegram_monitor)
                
                if decision == "APPROVED":
                    logger.info(f"[{symbol}] ✅ AGENT APPROVED ({agent_reason}). Buying for {trade_amount} USDT...")
                    _allow, _sized = self._check_pre_trade(
                        symbol, action="open", trade_usdt=float(trade_amount))
                    if _allow:
                        amount_coin = _sized / current_price
                        order = self.engine.execute_spot_order(symbol, 'BUY', amount_coin)
                        if order:
                            real_price = order.get('average') or order.get('price') or current_price
                            self.tracker.open_trade(symbol, _sized, real_price, strategy=strategy_used, market="SPOT", side="LONG")
                            logger.info(f"-> [SPOT] Trade LONG {_sized:.2f} USDT opened on Binance (Real Price: {real_price:.2f}).")
                else:
                    logger.warning(f"[{symbol}] 🚫 GEMINI VETO (Trade cancelled): {agent_reason}")
                    self.current_state["market_data"]["SPOT"][symbol]["last_signal"] = "HOLD"
                    self.current_state["market_data"]["SPOT"][symbol]["reason"] = f"🚫 Gemini Veto: {agent_reason}"
                    self._update_state()
                
        elif signal == "SELL":
            # 1. Close macro-trend longs on Spot
            open_longs = self.tracker.get_open_trades(symbol=symbol, side="LONG", market="SPOT")
            for t in open_longs:
                self._execute_close(t, current_price)
                
            # 2. If it's a correction signal, open a SHORT on futures!
            if strategy_used in ["Elliott_Wave_Correction", "Funding_Contrarian"]:
                has_open_short = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("side") == "SHORT" and t.get("market") == "FUTURES" for t in self.tracker.trades)
                if not has_open_short:
                    _allow, _sized = self._check_pre_trade(
                        symbol, action="open", trade_usdt=float(trade_amount))
                    if _allow:
                        amount_coin = _sized / current_price
                        order = self.engine.execute_futures_order(symbol, 'SELL', amount_coin)
                        if order:
                            real_price = order.get('average') or order.get('price') or current_price
                            self.tracker.open_trade(symbol, _sized, real_price, strategy=strategy_used, market="FUTURES", side="SHORT")
                            logger.info(f"-> [FUTURES] Trade SHORT {_sized:.2f} USDT opened on Binance (Real Price: {real_price:.2f}).")

    async def binance_websocket(self):
        # Perform initial analysis before starting WebSocket so the dashboard updates instantly
        logger.info("Loading initial data for dashboard...")
        for sym in self.symbols:
            try:
                download_history(symbol=sym, timeframe=self.timeframe, limit=1000)
                download_history(symbol=sym, timeframe='1m', limit=1000) # Download 1-minute data too
                data_filepath = f"data/raw/{sym.replace('/', '_')}_{self.timeframe}.csv.gz"
                data = self.analyzers[sym].load_data(data_filepath)
                if data:
                    current_price = data[-1]['close']
                    await self.process_kline(sym, current_price)
                # 1 second pause between requests to protect against Binance REST API limits
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in initial analysis {sym}: {e}")

        streams = [f"{sym.replace('/', '').lower()}@kline_1m" for sym in self.symbols]
        stream_str = '/'.join(streams)
        # FIX: Binance uses /stream?streams= for combined streams
        url = f"wss://stream.binance.com:9443/stream?streams={stream_str}"
        
        logger.info(f"Connecting to Binance WebSocket: {url}")
        
        was_running = True
        # Reconnect with exponential backoff. The default keepalive raised
        # `sent 1011 keepalive ping timeout` under network jitter — explicit
        # ping_interval/ping_timeout/close_timeout makes recovery tighter.
        backoff = WEBSOCKET_RECONNECT_DELAY
        while True:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=15,
                    max_size=2**22,  # 4 MiB — combined kline streams stay well under this
                ) as ws:
                    backoff = WEBSOCKET_RECONNECT_DELAY  # reset on successful connect
                    if self.pre_trade_gate is not None:
                        with self.pre_trade_gate.flag_lock:
                            self.pre_trade_gate.ws_connected = True
                        logger.info("[gate] ws_connected=True on reconnect")
                    while True:
                        is_running = True
                        ctrl = read_json('data/control.json', default={})
                        is_running = ctrl.get('running', True)

                        if not is_running:
                            self._update_state(status="Stopped (Paused)")
                            was_running = False
                            await asyncio.sleep(5)
                            continue
                        elif not was_running:
                            self._update_state(status="Resuming (waiting for tick)...")
                            was_running = True

                        msg = await ws.recv()
                        raw_data = json.loads(msg)
                        
                        # In the combined stream, data is packed in the 'data' key
                        kline_data = raw_data.get('data', raw_data)
                        
                        if 'k' in kline_data:
                            kline = kline_data.get('k', {})
                            is_closed = kline.get('x', False)
                            symbol_raw = kline_data.get('s', '')
                            symbol = next((s for s in self.symbols if s.replace('/', '') == symbol_raw), None)

                            if symbol and is_closed:
                                try:
                                    current_price = float(kline['c'])
                                    if current_price <= 0:
                                        raise ValueError(f"Non-positive price: {current_price}")
                                except (KeyError, ValueError, TypeError) as e:
                                    logger.warning(f"Invalid kline price data: {e} — skipping tick.")
                                    continue
                                if self.pre_trade_gate is not None:
                                    self.pre_trade_gate.record_warmup_tick()
                                await self.process_kline(symbol, current_price)
                                
            except Exception as e:
                if self.pre_trade_gate is not None:
                    with self.pre_trade_gate.flag_lock:
                        self.pre_trade_gate.ws_connected = False
                    logger.warning("[gate] ws_connected=False on disconnect")
                logger.error(f"WebSocket disconnected or error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # cap at 60 s

    def run(self):
        logger.info("Checking Binance REST API connection...")
        self._update_state(status="Checking Binance connection...", reason="Waiting for server response...")

        # Retry the startup connectivity check with exponential backoff before
        # giving up. testnet.binance.vision regularly returns transient 502s
        # that recover within a minute — a single-attempt check would lock
        # the bot into "Bot stopped" for hours over a brief outage.
        # Backoff: 5s, 10s, 20s, 40s, 80s → up to ~2.5 min total before
        # accepting "really down" and entering the holding loop.
        max_attempts = 5
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                server_time = self.engine.exchange.fetch_time()
                logger.info(f"✅ Connection successful! Binance Server Time: {server_time}")

                usdt_balance = self.get_real_or_sim_balance('USDT')
                logger.info(f"✅ Available USDT balance: {usdt_balance} USDT")

                logger.info("Testing download of 1000 candles (Checking for IP ban)...")
                test_candles = self.engine.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
                logger.info(f"✅ Successfully downloaded {len(test_candles)} candles. IP is not blocked!")
                last_exc = None
                break  # success
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    wait = 5 * (2 ** (attempt - 1))   # 5, 10, 20, 40, 80
                    logger.warning(
                        f"Binance connectivity attempt {attempt}/{max_attempts} failed: {str(e)[:120]}. "
                        f"Retrying in {wait}s..."
                    )
                    self._update_state(
                        status=f"Binance unreachable — retry {attempt}/{max_attempts}",
                        reason=str(e)[:120],
                    )
                    time.sleep(wait)
                # else: fall through to permanent-failure block below

        if last_exc is not None:
            logger.error(f"❌ All {max_attempts} connectivity attempts failed; last error: {last_exc}")
            logger.error("Bot entering hold mode. Check internet/VPN/API key, then restart bot.")
            self._update_state(
                status="❌ ERROR: Binance unreachable after retries",
                reason=f"Details in logs: {str(last_exc)[:80]}...",
            )
            # Keep the process alive so the dashboard can read logs and show
            # the red error state. Periodically retry connectivity in the
            # background so the bot self-heals when testnet comes back.
            while True:
                time.sleep(60)
                try:
                    self.engine.exchange.fetch_time()
                    logger.info("Binance reachable again — exiting hold mode and restarting startup.")
                    self._update_state(
                        status="Binance reachable again — recovering...",
                        reason="Connectivity restored after outage",
                    )
                    break  # exit the hold loop, fall through to agent startup
                except Exception:
                    pass  # still down, keep waiting
                
        # Start all 8 core agents as daemon threads
        logger.info("Starting agent system...")
        self._start_agents()

        # Start Deep Learning Inference Engine in background
        logger.info("Starting TFT Inference Engine thread...")
        self.inference_engine.start(self.symbols)

        # Telegram Monitor is gated behind TELEGRAM_MONITOR_ENABLED (default off).
        # Telethon v1.43.2 has a known bug where the headless reconnect loop
        # crashes after the initial session restore fails — symptoms are 15+
        # CRITICAL banner entries (Event loop closed, Task destroyed, Future
        # exception never retrieved). Until we either upgrade Telethon or do
        # an interactive re-login to refresh trading_session.session, we keep
        # this off so the bot doesn't crashloop noise into the dashboard.
        # Re-enable with: $env:TELEGRAM_MONITOR_ENABLED='true' before launch.
        import os as _os
        _tg_enabled = _os.environ.get('TELEGRAM_MONITOR_ENABLED', 'false').lower() == 'true'
        if _tg_enabled:
            logger.info("Starting Telegram Monitor thread...")
            self.telegram_monitor.start()
        else:
            logger.info("Telegram Monitor disabled (set TELEGRAM_MONITOR_ENABLED=true to enable).")

        asyncio.run(self.binance_websocket())

    def _start_agents(self) -> None:
        """Start all 8 core agents as daemon threads. Safe to call once at boot."""
        try:
            from src.engine.agents.data_agent      import DataAgent
            from src.engine.agents.signal_agent    import SignalAgent
            from src.engine.agents.quant_agent     import QuantAgent
            from src.engine.agents.risk_agent      import RiskAgent
            from src.engine.agents.execution_agent import ExecutionAgent
            from src.engine.agents.spot_agent      import SpotAgent
            from src.engine.agents.futures_agent   import FuturesAgent
            from src.engine.agents.scalping_agent  import ScalpingAgent
        except ImportError as exc:
            logger.warning("Agent import error — agents not started: %s", exc)
            return

        # data_getter reads the last 200 rows of the 1h CSV for a symbol
        def _data_getter_1h(sym: str):
            safe = sym.replace('/', '_')
            path = os.path.join('data', 'raw', f'{safe}_1h.csv.gz')
            if not os.path.exists(path):
                return None
            try:
                df = pd.read_csv(path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.tail(200).reset_index(drop=True)
            except Exception:
                return None

        def _data_getter_1m(sym: str):
            safe = sym.replace('/', '_')
            path = os.path.join('data', 'raw', f'{safe}_1m.csv.gz')
            if not os.path.exists(path):
                return None
            try:
                df = pd.read_csv(path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.tail(500).reset_index(drop=True)
            except Exception:
                return None

        # Agent-friendly symbol list (underscore format expected by some agents)
        syms_safe = [s.replace('/', '_') for s in self.symbols]

        agents = [
            DataAgent(symbols=syms_safe, interval_sec=3600.0),
            SignalAgent(symbols=syms_safe, data_getter=_data_getter_1h, interval_sec=3600.0),
            QuantAgent(interval_sec=3600.0 * 4),
            RiskAgent(interval_sec=300.0),
            ExecutionAgent(exchange_client=self.engine.exchange, interval_sec=60.0),
            SpotAgent(symbols=syms_safe, data_getter=_data_getter_1h, interval_sec=3600.0),
            FuturesAgent(symbols=syms_safe, data_getter=_data_getter_1h, interval_sec=3600.0),
            ScalpingAgent(symbols=syms_safe, data_getter_1m=_data_getter_1m, interval_sec=60.0),
        ]
        for agent in agents:
            try:
                agent.start()
                logger.info("Agent started: %s", agent.NAME)
            except Exception as exc:
                logger.warning("Failed to start %s: %s", agent.NAME, exc)

if __name__ == "__main__":
    # Process registry: claim the 'bot' role BEFORE any heavy init. If another
    # live bot already owns it, exit cleanly — no duplicate. The previous
    # 2026-05-13 incident produced a 12.8-hour CPU runaway because
    # restart_all spawned a fresh bot on top of an old buggy one with no
    # arbitration. This block closes that mode of failure permanently.
    try:
        from src.utils.process_registry import claim_role, release_role, heartbeat
        import atexit, threading
        ok, existing = claim_role('bot', by='src.main')
        if not ok:
            import sys
            logger.error(
                "[startup] Another bot already running: PID=%s cmd=%s hb=%s. "
                "Exiting to avoid duplicate.",
                existing.get('pid'),
                (existing.get('cmdline') or '?')[:80],
                existing.get('last_heartbeat'),
            )
            sys.exit(0)
        atexit.register(lambda: release_role('bot', reason='atexit'))
        # Background heartbeat — refreshes the registry entry every 60s so
        # reap_zombies() doesn't mistake a long-running bot for stale.
        def _hb_loop():
            import time as _t
            consec_failures = 0
            while True:
                _t.sleep(60)
                try:
                    if not heartbeat('bot'):
                        # heartbeat() already logs WARNING on ownership loss.
                        # Bump a consecutive counter; after 3 misses, the
                        # bot has been evicted from the registry — best to
                        # exit so a clean restart can claim.
                        consec_failures += 1
                        if consec_failures >= 3:
                            logger.critical(
                                "[registry-hb] lost role ownership for 3 consecutive "
                                "heartbeats — exiting to let a clean restart take over"
                            )
                            os._exit(0)
                    else:
                        consec_failures = 0
                except Exception as exc:
                    consec_failures += 1
                    logger.warning("[registry-hb] heartbeat exception: %s", exc)
                    if consec_failures >= 5:
                        logger.error(
                            "[registry-hb] 5 consecutive heartbeat failures — "
                            "registry file likely unreachable"
                        )
        threading.Thread(target=_hb_loop, daemon=True, name='registry-hb').start()
    except Exception as exc:
        logger.warning("[startup] process_registry unavailable: %s", exc)

    try:
        from src.utils.hw_config import configure as _hw_cfg
        _hw_cfg(verbose=True)
    except Exception:
        pass

    wl_path = 'data/watchlist.json'
    if os.path.exists(wl_path):
        with open(wl_path, 'r', encoding='utf-8') as f:
            loaded_symbols = json.load(f)
    else:
        loaded_symbols = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT', 'ETH/USDT']

    trader = MultiAssetTrader(symbols=loaded_symbols)
    trader.run()
