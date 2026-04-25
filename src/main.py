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
    import pandas as pd
    import ccxt
    import debugpy
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
    def __init__(self, symbols=['BTC/USDT', 'SOL/USDT', 'ADA/USDT'], timeframe='1h'):
        self.symbols = symbols
        self.timeframe = timeframe
        self.analyzers = {sym: ElliottWaveAnalyzer(deviation_percent=1.5) for sym in symbols}
        self.risk_managers = {sym: HullRiskManager(default_risk_usd=20.0) for sym in symbols}
        self.engine = OrderManager()
        self.tracker = TradeTracker()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.ml_predictor = MLPredictor()
        self.scalping_predictor = MLPredictor(model_filename='scalping_model.joblib', model_type='scalping')
        self.agent = AgenticLLM()
        self.state_file = 'data/state.json'
        
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

    def get_real_or_sim_balance(self, asset):
        """Returns the real Binance balance or dynamically simulated (accounting for PnL)"""
        if not self.engine.api_key or self.engine.api_key == 'your_api_key_here':
            if asset == 'USDT':
                bal = 10000.0
                for t in self.tracker.trades:
                    if t.get('status') == 'CLOSED':
                        bal += t.get('pnl_usdt', 0.0)
                    elif t.get('status') == 'OPEN':
                        bal -= t.get('invested_usdt', 0.0)
                return bal
            else:
                bal = 0.0
                for t in self.tracker.trades:
                    if t.get('status') == 'OPEN' and t.get('symbol', '').startswith(asset):
                        side = t.get('side', 'LONG')
                        bal += t.get('amount_coin', 0.0) if side == 'LONG' else -t.get('amount_coin', 0.0)
                return bal
        return self.engine.get_balance(asset)

    def get_default_market_state(self):
        return {"last_signal": "NONE", "reason": "Waiting for data...", "wave_stage": "Initializing...", "ml_prediction_text": "Analyzing...", "ml_accuracy": 0.0, "rsi": 50.0, "volatility": 0.0, "recommended_trade_size": 0.0, "recent_pivots": []}

    async def update_market_context(self):
        """Asynchronously updates the macro context: 1D SMA200 (once an hour) and Funding Rates (every 5 minutes)"""
        current_time = time.time()
        
        # 1. Multi-Timeframe: Update 1-day SMA200 every hour
        if current_time - self.daily_sma_last_update > 3600:
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
        if current_time - self.funding_last_update > 300:
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
        vol_surge = last['volume'] > (df['volume'].rolling(20).mean().iloc[-1] * 1.5)
        
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
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_state, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def analyze_market_state(self, pivots, sentiment_score, ml_prediction, rsi):
        """Heuristics based on Elliott Waves."""
        if len(pivots) < 4:
            return "HOLD", "Not enough data for wave analysis.", "Data Collection (Wave 1/2)", "Data_Collection"

        p1, p2, p3, p4 = pivots[-4:]
        
        sentiment_boost = sentiment_score > 0.15
        sentiment_drag = sentiment_score < -0.15
        
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
                if rsi > 70:
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
                if rsi < 30:
                    return "HOLD", f"Bearish structure, but asset is oversold (RSI: {rsi:.1f}).", wave_stage, "Neutral"
                return "SELL", "Bearish structure detected (ABC correction).", wave_stage, "Elliott_Wave_Correction"

        return "HOLD", "Market is flat.", wave_stage, "Neutral"

    def evaluate_all_strategies(self, symbol, data, pivots, sentiment_score, ml_prediction, rsi_value):
        """Orchestrator: Evaluates all 3 advanced strategies and decides which one to apply."""
        # 1. Base strategy (ML + Elliott)
        signal, reason, wave_stage, strategy_used = self.analyze_market_state(pivots, sentiment_score, ml_prediction, rsi_value)
        
        df = pd.DataFrame(data)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])
        current_price = df['close'].iloc[-1]
        
        # --- FILTER: Multi-Timeframe Confluence (Prohibit trading against the macro trend) ---
        sma200 = self.daily_sma200.get(symbol)
        if signal == "BUY" and sma200 and current_price < sma200:
            return "HOLD", f"{reason} [MTF Veto: Price below 1D SMA200 ({sma200:.0f})]", wave_stage, "Neutral"
        elif signal == "SELL" and sma200 and current_price > sma200:
            return "HOLD", f"{reason} [MTF Veto: Price above 1D SMA200 ({sma200:.0f})]", wave_stage, "Neutral"
            
        # --- STRATEGY 2: Volatility Breakout (Squeeze) ---
        vol_signal = self.check_volatility_breakout(df)
        if vol_signal == "BUY" and (not sma200 or current_price > sma200):
            return "BUY", "Volatility compression breakout (TTM Squeeze) + Volume", "Breakout Impulse 🚀", "Volatility_Breakout"
            
        # --- STRATEGY 3: Derivatives Sentiment (Counter-trend Squeeze on funding) ---
        funding = self.funding_rates.get(symbol, 0.0)
        if funding < -0.015 and rsi_value < 40:
            return "BUY", f"Short-squeeze: Negative funding ({funding*100:.3f}%) + RSI bottom", "Shorts Reversal 🚀", "Funding_Contrarian"
        elif funding > 0.015 and rsi_value > 60:
            return "SELL", f"Long-squeeze: Extreme positive funding ({funding*100:.3f}%) + Overheated", "Longs Liquidation 📉", "Funding_Contrarian"

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

        if scalp_pred == 1 and rsi_7_val < 65: # Buy only if not in hard overbought
            return "BUY", f"Scalping signal UP (RSI: {rsi_7_val:.1f})", "Scalping_Long"
        
        # Logic for short scalping (Playing the downside)
        if scalp_pred == 0 and rsi_7_val > 35:
            return "SELL", f"Scalping signal DOWN (RSI: {rsi_7_val:.1f})", "Scalping_Short"

        return "HOLD", "No scalping signal", "Neutral"

    def _execute_close(self, trade):
        """Helper method: physically closes the order on the exchange"""
        symbol = trade['symbol']
        amount_coin = trade['amount_coin']
        market = trade.get('market', 'SPOT')
        side = trade.get('side', 'LONG')
        if market in ['SPOT', 'SPOT_SCALPING']:
            self.engine.execute_spot_order(symbol, 'SELL', amount_coin)
        elif market in ['FUTURES', 'SCALPING']:
            close_side = 'SELL' if side == 'LONG' else 'BUY'
            self.engine.execute_futures_order(symbol, close_side, amount_coin, reduce_only=True)

    async def process_kline(self, symbol, current_price):
        logger.info(f"[{symbol}] WebSocket tick processed - Price: {current_price}")
        self.current_state["prices"][symbol] = current_price
        await self.update_market_context()
        
        # 1. Risk priority: Check trailing stops for open positions before new analysis
        closed_trades = self.tracker.update_trailing_stops(symbol, current_price)
        for t in closed_trades:
            logger.info(f"[{symbol}] 🛡️ TRAILING STOP: Closed position ID {t['id']} ({t['market']}). PnL: {t['pnl_usdt']:.2f} USDT")
            self._execute_close(t)

        sentiment = self.sentiment_analyzer.get_average_sentiment()
        
        # --- SPOT & FUTURES ANALYSIS (1h timeframe) ---
        logger.info(f"[{symbol}][SPOT] Starting 1h analysis...")
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
        current_dev = 1.5
        while (not pivots or len(pivots) < 5) and current_dev >= 0.3:
            current_dev -= 0.3
            fallback_analyzer = ElliottWaveAnalyzer(deviation_percent=current_dev)
            pivots = fallback_analyzer.calculate_zigzag(data)
            
        if pivots is None:
            pivots = []
            
        volatility = self.risk_managers[symbol].calculate_historical_volatility(data)
        trade_amount = self.risk_managers[symbol].get_position_size(data)
        trade_amount = max(55.0, trade_amount) # Protection against error "-4164 MIN_NOTIONAL" (Min order 50 USDT)
        
        ml_pred = self.ml_predictor.predict(data)
        rsi_value = self.calculate_rsi(data)
        
        signal, reason, wave_stage, strategy_used = self.evaluate_all_strategies(symbol, data, pivots, sentiment, ml_pred, rsi_value)
        
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
            "rsi": round(rsi_value, 2),
            "volatility": round(volatility*100, 2), 
            "recommended_trade_size": trade_amount,
            "recent_pivots": formatted_pivots
        }
        # Temporarily duplicate in FUTURES, since there is no separate logic for them
        self.current_state["market_data"]["FUTURES"][symbol] = self.current_state["market_data"]["SPOT"][symbol]

        # --- SCALPING ANALYSIS (1m timeframe) ---
        logger.info(f"[{symbol}][SCALPING] Starting 1m analysis...")
        scalping_data_filepath = f"data/raw/{symbol.replace('/', '_')}_1m.csv.gz"
        data_1m = self.analyzers[symbol].load_data(scalping_data_filepath)
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
                "rsi": self.calculate_rsi(data_1m, period=7), # Fast RSI
                "volatility": self.risk_managers[symbol].calculate_historical_volatility(data_1m) * 100,
                "recommended_trade_size": max(55.0, self.risk_managers[symbol].get_position_size(data_1m) * 0.25), # Account for Binance limits
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
                    if self.engine.execute_futures_order(symbol, exec_side, amount_coin):
                        self.tracker.open_trade(symbol, scalp_trade_amount, current_price, strategy=scalp_strategy, market="SCALPING", side=side)
                        logger.info(f"-> [SCALPING FUTURES] Trade {side} {scalp_trade_amount:.2f} USDT opened (Price: {current_price}).")
                        
                # 2. Scalping on Spot (LONG only)
                if scalp_signal == "BUY":
                    has_open_scalp_spot = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("market") == "SPOT_SCALPING" for t in self.tracker.trades)
                    if not has_open_scalp_spot:
                        if self.engine.execute_spot_order(symbol, 'BUY', amount_coin):
                            self.tracker.open_trade(symbol, scalp_trade_amount, current_price, strategy=scalp_strategy, market="SPOT_SCALPING", side="LONG")
                            logger.info(f"-> [SCALPING SPOT] Trade LONG {scalp_trade_amount:.2f} USDT opened (Price: {current_price}).")
                            
                # 3. Scalping on Spot (Instant close LONG on SELL signal)
                elif scalp_signal == "SELL":
                    closed_scalp_spot = self.tracker.close_trades(current_price, symbol=symbol, side="LONG", market="SPOT_SCALPING")
                    for t in closed_scalp_spot: self._execute_close(t)
        
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
            closed_shorts = self.tracker.close_trades(current_price, symbol=symbol, side="SHORT", market="FUTURES")
            for t in closed_shorts: self._execute_close(t)
            
            # Protection against order spam: check if there is an open position for this coin
            has_open = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("side") == "LONG" and t.get("market") == "SPOT" for t in self.tracker.trades)
            if has_open:
                logger.info(f"[{symbol}] BUY signal ignored: position already open.")
            else:
                logger.info(f"[{symbol}] 🤖 Requesting Gemini Agent for macro check...")
                headlines = getattr(self.sentiment_analyzer, 'cached_headlines', [])[:10]
                decision, agent_reason = self.agent.evaluate_trade(symbol, "BUY", reason, headlines)
                
                if decision == "APPROVED":
                    logger.info(f"[{symbol}] ✅ AGENT APPROVED ({agent_reason}). Buying for {trade_amount} USDT...")
                    amount_coin = trade_amount / current_price
                    if self.engine.execute_spot_order(symbol, 'BUY', amount_coin):
                        self.tracker.open_trade(symbol, trade_amount, current_price, strategy=strategy_used, market="SPOT", side="LONG")
                        logger.info(f"-> [SPOT] Trade LONG {trade_amount:.2f} USDT opened (Price: {current_price:.2f}).")
                else:
                    logger.warning(f"[{symbol}] 🚫 GEMINI VETO (Trade cancelled): {agent_reason}")
                    self.current_state["market_data"]["SPOT"][symbol]["last_signal"] = "HOLD"
                    self.current_state["market_data"]["SPOT"][symbol]["reason"] = f"🚫 Gemini Veto: {agent_reason}"
                    self._update_state()
                
        elif signal == "SELL":
            # 1. Close macro-trend longs on Spot
            closed_longs = self.tracker.close_trades(current_price, symbol=symbol, side="LONG", market="SPOT")
            for t in closed_longs:
                logger.info(f"[{symbol}] Closed SPOT position ID {t['id']}. PnL: {t['pnl_usdt']:.2f} USDT")
                self._execute_close(t)
                
            # 2. If it's a correction signal, open a SHORT on futures!
            if strategy_used in ["Elliott_Wave_Correction", "Funding_Contrarian"]:
                has_open_short = any(t["status"] == "OPEN" and t["symbol"] == symbol and t.get("side") == "SHORT" and t.get("market") == "FUTURES" for t in self.tracker.trades)
                if not has_open_short:
                    amount_coin = trade_amount / current_price
                    if self.engine.execute_futures_order(symbol, 'SELL', amount_coin):
                        self.tracker.open_trade(symbol, trade_amount, current_price, strategy=strategy_used, market="FUTURES", side="SHORT")
                        logger.info(f"-> [FUTURES] Trade SHORT {trade_amount:.2f} USDT opened (Price: {current_price:.2f}).")

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
                        if os.path.exists('data/control.json'):
                            with open('data/control.json', 'r', encoding='utf-8') as f:
                                ctrl = json.load(f)
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
                            kline = kline_data['k']
                            is_closed = kline['x']
                            symbol_raw = kline_data['s']
                            symbol = next((s for s in self.symbols if s.replace('/', '') == symbol_raw), None)
                            
                            if symbol and is_closed:
                                current_price = float(kline['c'])
                                await self.process_kline(symbol, current_price)
                                
            except Exception as e:
                logger.error(f"WebSocket disconnected or error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

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
            
        asyncio.run(self.binance_websocket())

if __name__ == "__main__":
    # ==============================================================================
    # IMPORTANT NOTE / AI INSTRUCTION:
    # AI ASSISTANT: Always remind the user to run `restart_all.bat` after any code modifications!
    # After making any code changes, always do a HARD RESTART!
    # For this, the AI ASSISTANT runs the restart_all.bat file in the root folder.
    # This will kill stuck background processes and guarantee new code loading.
    # ==============================================================================
    
    wl_path = 'data/watchlist.json'
    if os.path.exists(wl_path):
        with open(wl_path, 'r', encoding='utf-8') as f:
            loaded_symbols = json.load(f)
    else:
        loaded_symbols = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT']
        
    trader = MultiAssetTrader(symbols=loaded_symbols)
    trader.run()
