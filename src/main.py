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
    import numpy as np
    from src.utils.config import (
        MIN_TRADE_USDT, SCALPING_TRADE_FRACTION, MTF_SMA200_REFRESH,
        FUNDING_RATE_REFRESH, SENTIMENT_BOOST_THRESHOLD, SENTIMENT_DRAG_THRESHOLD,
        RSI_OVERBOUGHT, RSI_OVERSOLD, SCALPING_RSI_OVERBOUGHT, SCALPING_RSI_OVERSOLD,
        FUNDING_SQUEEZE_THRESHOLD, VOLATILITY_BREAKOUT_VOLUME_MULT,
        WAVE_DEVIATION_DEFAULT, WAVE_DEVIATION_MIN, WAVE_DEVIATION_STEP,
        WEBSOCKET_RECONNECT_DELAY
    )
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
        self.ml_predictor = MLPredictor()
        self.scalping_predictor = MLPredictor(model_filename='scalping_model.joblib', model_type='scalping')
        self.futures_predictor = MLPredictor(model_filename='futures_short_model.joblib', model_type='futures')
        self.trend_predictor = MLPredictor(model_filename='trend_model.joblib', model_type='trend')
        self.agent = AgenticLLM()
        self.telegram_monitor = TelegramMonitor()
        self.state_file = 'data/state.json'

        # --- Strategy registry (controls which strategies are active) ---
        from src.engine.strategy_registry import load_config as _load_strat_cfg
        self._strat_cfg = _load_strat_cfg()
        self._strat_cfg_mtime = 0.0

        # --- Regime Classifier ---
        try:
            from src.analysis.regime_classifier import RegimeClassifier
            self.regime_clf = RegimeClassifier()
        except Exception:
            self.regime_clf = None

        # --- Meta-Labeler ---
        try:
            from src.analysis.meta_labeler import MetaLabeler
            self.meta_labeler = MetaLabeler()
        except Exception:
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
        self._update_state()

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
                    pass # Ignore if the coin has no futures
            self.funding_last_update = current_time

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
                if regime == 2:  # VOLATILE
                    return "HOLD", f"Regime: {regime_name} — holding all entries, size_mult={size_mult}", "Regime_Volatile", "RegimeClassifier_Router"
            except Exception as e:
                logger.debug("[%s] Regime predict failed: %s", symbol, e)

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
                    result = self.meta_labeler.filter_signal(sig_num, df.to_dict('records'))
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
        
        if not data_1m or len(data_1m) < 20:
            return "HOLD", "Not enough 1m data for scalping", "Data_Collection"

        # Calculation of fast indicators for scalping
        df = pd.DataFrame(data_1m)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])
        
        delta = df['close'].diff()
        rsi_7 = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=6, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=6, adjust=False).mean())))
        rsi_7_val = rsi_7.iloc[-1]

        # Get prediction from the scalping model
        scalp_pred = self.scalping_predictor.predict(data_1m)

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
            order = self.engine.execute_futures_order(symbol, close_side, amount_coin, reduce_only=True)

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

        sentiment = self.sentiment_analyzer.get_average_sentiment()
        
        # --- MACRO ANALYSIS (1h timeframe for Spot & Futures) ---
        logger.info(f"[{symbol}][1h] Starting macro analysis...")
        data_filepath = f"data/raw/{symbol.replace('/', '_')}_{self.timeframe}.csv.gz"
        data = self.analyzers[symbol].load_data(data_filepath)
        
        # If the file is empty or data is insufficient (corrupted file), force download again
        if not data or len(data) < 50:
            logger.info(f"[{symbol}] Insufficient data, forcing history download...")
            download_history(symbol=symbol, timeframe=self.timeframe, limit=1000)
            data = self.analyzers[symbol].load_data(data_filepath)
            
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

        ml_pred = self.ml_predictor.predict(data)
        trend_pred = self.trend_predictor.predict(data)
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
        tft_pred = self.inference_engine.get_latest_prediction(symbol)
        expected_return = tft_pred["expected_return"] if tft_pred else 0.0
        
        base_asset = symbol.split('/')[0]
        inventory_q = self.get_real_or_sim_balance(base_asset)
        mm_quotes = self.market_makers[symbol].calculate_quotes(current_price, inventory_q, volatility)
        
        if expected_return <= -0.01:
            logger.warning(f"[{symbol}] 📉 TFT predicts DROP ({expected_return*100:.1f}%). Disabling AvellanedaStoikov Buys, shifting to SHORT.")
            try:
                self.engine.cancel_all_orders(symbol)
                self.engine.execute_limit_futures_order(symbol, "SELL", trade_amount / current_price, mm_quotes["optimal_ask"])
            except Exception as e:
                logger.error(f"MM Execution Error: {e}")
        elif expected_return >= 0.01:
            logger.info(f"[{symbol}] 🚀 TFT predicts PUMP ({expected_return*100:.1f}%). Disabling AvellanedaStoikov Sells, shifting to LONG.")
            try:
                self.engine.cancel_all_orders(symbol)
                self.engine.execute_limit_futures_order(symbol, "BUY", trade_amount / current_price, mm_quotes["optimal_bid"])
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
            "as_bid":         round(float(mm_quotes.get("optimal_bid", 0.0)), 2),
            "as_ask":         round(float(mm_quotes.get("optimal_ask", 0.0)), 2),
            "as_spread":      round(float(mm_quotes.get("optimal_spread", 0.0)), 2),
            "as_reservation": round(float(mm_quotes.get("reservation_price", 0.0)), 2),
            "momentum_signal": round(float(self.momentum_signals.get(symbol, 0.0)), 3),
        }

        signal, reason, wave_stage, strategy_used = self.evaluate_all_strategies(symbol, data, pivots, sentiment, trend_pred, rsi_value, ou_signal=ou_signal)
        
        if not self.ml_predictor.is_loaded:
            ml_text = "MODEL NOT FOUND"
        elif ml_pred is None:
            if hasattr(self.ml_predictor, 'last_error') and self.ml_predictor.last_error:
                ml_text = "ERROR"
                reason += f" | {self.ml_predictor.last_error}"
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
            "funding_rate": round(self.funding_rates.get(symbol, 0.0) * 100, 4)
        }
        
        # Build FUTURES state with specific Futures ML Model predictions
        fut_pred = self.futures_predictor.predict(data)
        if not self.futures_predictor.is_loaded:
            fut_ml_text = "MODEL NOT FOUND"
        elif fut_pred is None:
            fut_ml_text = "ERROR" if self.futures_predictor.last_error else "DATA COLLECTION"
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
        scalping_data_filepath = f"data/raw/{symbol.replace('/', '_')}_1m.csv.gz"
        data_1m = self.analyzers[symbol].load_data(scalping_data_filepath, tail_n=500)
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
                    order = self.engine.execute_futures_order(symbol, exec_side, amount_coin)
                    if order:
                        real_price = order.get('average') or order.get('price') or current_price
                        self.tracker.open_trade(symbol, scalp_trade_amount, real_price, strategy=scalp_strategy, market="SCALPING", side=side)
                        logger.info(f"-> [SCALPING FUTURES] Trade {side} {scalp_trade_amount:.2f} USDT opened on Binance (Real Price: {real_price}).")
                        
                # 2. Scalping on Spot (LONG only)
                if scalp_signal == "BUY":
                    has_open_scalp_spot = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("market") == "SPOT_SCALPING" for t in self.tracker.trades)
                    if not has_open_scalp_spot:
                        order = self.engine.execute_spot_order(symbol, 'BUY', amount_coin)
                        if order:
                            real_price = order.get('average') or order.get('price') or current_price
                            self.tracker.open_trade(symbol, scalp_trade_amount, real_price, strategy=scalp_strategy, market="SPOT_SCALPING", side="LONG")
                            logger.info(f"-> [SCALPING SPOT] Trade LONG {scalp_trade_amount:.2f} USDT opened on Binance (Real Price: {real_price}).")
                            
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
                    amount_coin = trade_amount / current_price
                    order = self.engine.execute_spot_order(symbol, 'BUY', amount_coin)
                    if order:
                        real_price = order.get('average') or order.get('price') or current_price
                        self.tracker.open_trade(symbol, trade_amount, real_price, strategy=strategy_used, market="SPOT", side="LONG")
                        logger.info(f"-> [SPOT] Trade LONG {trade_amount:.2f} USDT opened on Binance (Real Price: {real_price:.2f}).")
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
                    amount_coin = trade_amount / current_price
                    order = self.engine.execute_futures_order(symbol, 'SELL', amount_coin)
                    if order:
                        real_price = order.get('average') or order.get('price') or current_price
                        self.tracker.open_trade(symbol, trade_amount, real_price, strategy=strategy_used, market="FUTURES", side="SHORT")
                        logger.info(f"-> [FUTURES] Trade SHORT {trade_amount:.2f} USDT opened on Binance (Real Price: {real_price:.2f}).")

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
        while True:
            try:
                async with websockets.connect(url) as ws:
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
                                await self.process_kline(symbol, current_price)
                                
            except Exception as e:
                logger.error(f"WebSocket disconnected or error: {e}. Reconnecting in {WEBSOCKET_RECONNECT_DELAY}s...")
                await asyncio.sleep(WEBSOCKET_RECONNECT_DELAY)

    def run(self):
        logger.info("Checking Binance REST API connection...")
        self._update_state(status="Checking Binance connection...", reason="Waiting for server response...")
        try:
            # Make one test request to the Binance server
            server_time = self.engine.exchange.fetch_time()
            logger.info(f"✅ Connection successful! Binance Server Time: {server_time}")
            
            # Check available balance
            usdt_balance = self.get_real_or_sim_balance('USDT')
            logger.info(f"✅ Available USDT balance: {usdt_balance} USDT")
            
            # TEST DOWNLOAD OF 1000 CANDLES (Check for IP ban)
            logger.info("Testing download of 1000 candles (Checking for IP ban)...")
            test_candles = self.engine.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
            logger.info(f"✅ Successfully downloaded {len(test_candles)} candles. IP is not blocked!")
        except Exception as e:
            logger.error(f"❌ Error connecting to Binance: {e}")
            logger.error("Bot stopped. Check your internet, VPN, or API key settings.")
            self._update_state(status="❌ ERROR: IP BAN OR NO NETWORK", reason=f"Details in logs: {str(e)[:80]}...")
            # Keep the process active so the dashboard can read logs and show the red error
            while True:
                time.sleep(10)
                
        # Start all 8 core agents as daemon threads
        logger.info("Starting agent system...")
        self._start_agents()

        # Start Deep Learning Inference Engine in background
        logger.info("Starting TFT Inference Engine thread...")
        self.inference_engine.start(self.symbols)

        # Start Telegram Monitor in background
        logger.info("Starting Telegram Monitor thread...")
        self.telegram_monitor.start()

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
