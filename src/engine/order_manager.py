import ccxt
import os
import time
import logging
import threading
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Local clock can drift several seconds vs Binance's server, which causes
# `code:-1021 "Timestamp for this request is outside of the recvWindow"` on
# every signed call. We solve this two ways:
#   1. recvWindow=60000 — widest Binance allows.
#   2. adjustForTimeDifference + periodic load_time_difference() — CCXT then
#      offsets every signed request by (server_time − local_time).
_TIME_SYNC_INTERVAL_S = 300.0


class OrderManager:
    """
    Trading engine for order and risk management.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        self.futures_api_key = os.getenv('FUTURES_API_KEY', self.api_key)
        self.futures_api_secret = os.getenv('FUTURES_API_SECRET', self.api_secret)
        self.use_testnet = os.getenv('USE_TESTNET', 'True').lower() in ('true', '1', 't')

        _common_options = {
            'recvWindow': 60000,
            'adjustForTimeDifference': True,
        }

        # Initialize Binance exchange (Spot)
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': dict(_common_options),
        })

        # Initialize Binance exchange (Futures)
        self.futures_exchange = ccxt.binance({
            'apiKey': self.futures_api_key,
            'secret': self.futures_api_secret,
            'enableRateLimit': True,
            'options': {**_common_options, 'defaultType': 'future'}
        })

        if self.use_testnet:
            self.exchange.set_sandbox_mode(True)
            self.futures_exchange.set_sandbox_mode(True)
            logging.info("OrderManager initialized (Mode: TESTNET - SAFE TRADING)")
        else:
            logging.warning("OrderManager initialized (Mode: MAINNET - REAL MONEY!)")

        self._last_time_sync = 0.0
        self._sync_lock = threading.Lock()
        self._sync_clocks(force=True)

    def _sync_clocks(self, force: bool = False) -> None:
        """Refresh CCXT's server-time offset on both Spot and Futures.

        Called once at construction and lazily before each signed request once
        every `_TIME_SYNC_INTERVAL_S` seconds. Prevents -1021 errors when the
        local clock drifts.
        """
        now = time.monotonic()
        if not force and (now - self._last_time_sync) < _TIME_SYNC_INTERVAL_S:
            return
        with self._sync_lock:
            if not force and (now - self._last_time_sync) < _TIME_SYNC_INTERVAL_S:
                return
            for ex, label in ((self.exchange, 'spot'), (self.futures_exchange, 'futures')):
                try:
                    ex.load_time_difference()
                except Exception as e:
                    logging.warning("Clock sync failed (%s): %s", label, e)
            self._last_time_sync = time.monotonic()

    def get_balance(self, asset='USDT'):
        """Returns the free balance of the specified asset."""
        self._sync_clocks()
        try:
            balance = self.exchange.fetch_balance()
            if asset in balance:
                return float(balance[asset]['free'])
            return 0.0
        except ccxt.InvalidNonce as e:
            self._sync_clocks(force=True)
            try:
                balance = self.exchange.fetch_balance()
                if asset in balance:
                    return float(balance[asset]['free'])
                return 0.0
            except Exception as e2:
                logging.error(f"Error getting balance after resync: {e2}")
                return 0.0
        except Exception as e:
            logging.error(f"Error getting balance: {e}")
            return 0.0

    @staticmethod
    def _trade_mode() -> str:
        """Read data/control.json's `trade_mode`. Three values:
           paper   — orders never reach the exchange; routed to paper_book
           testnet — current/legacy default; sends to Binance testnet
           mainnet — real money on Binance live (requires explicit opt-in)
        Defaults to 'testnet' if the field is missing so existing installs
        keep their current behaviour without a config-file migration."""
        try:
            from src.utils.safe_json import read_json
            ctrl = read_json('data/control.json', default={}) or {}
            mode = (ctrl.get('trade_mode') or 'testnet').lower().strip()
            return mode if mode in ('paper', 'testnet', 'mainnet') else 'testnet'
        except Exception:
            return 'testnet'

    def execute_spot_order(self, symbol, side, amount_coin):
        """Sends a market order to the Spot account.
        In `paper` mode we skip the exchange call entirely and book
        internally via src.engine.paper_book — same return shape so
        callers don't need to branch."""
        if self._trade_mode() == 'paper':
            try:
                from src.engine.paper_book import book_market_order
                # Paper booker needs a price — peek at the latest ticker.
                price = 0.0
                try:
                    self._sync_clocks()
                    t = self.exchange.fetch_ticker(symbol)
                    price = float(t.get('last') or t.get('close') or 0)
                except Exception:
                    pass
                return book_market_order(symbol, side, amount_coin, price,
                                         market="spot")
            except Exception as exc:
                logging.error(f"❌ paper SPOT {side} {symbol} failed: {exc}")
                return None
        self._sync_clocks()
        try:
            self.exchange.load_markets()
            amount_coin = self.exchange.amount_to_precision(symbol, amount_coin)

            if side.upper() == 'BUY':
                order = self.exchange.create_market_buy_order(symbol, float(amount_coin))
            else:
                order = self.exchange.create_market_sell_order(symbol, float(amount_coin))
            logging.info(f"✅ SPOT {side.upper()} {amount_coin} {symbol} executed. ID: {order.get('id')}")
            return order
        except Exception as e:
            logging.error(f"❌ Error in Spot order {side} on {symbol}: {e}")
            return None

    @staticmethod
    def to_futures_symbol(symbol: str) -> str:
        """Convert a spot symbol to its perpetual futures equivalent.
        Examples: 'BTC/USDT' -> 'BTC/USDT:USDT', 'BTC/BUSD' -> 'BTC/BUSD:BUSD'
        """
        parts = symbol.split('/')
        if len(parts) != 2:
            raise ValueError(f"Cannot convert symbol to futures format: {symbol!r}")
        base, quote = parts
        return f"{base}/{quote}:{quote}"

    def get_futures_position_amount(self, symbol):
        """Return the absolute contract size of the open futures position on
        Binance, or 0.0 if there's nothing open / the call fails. Used by the
        close path to skip useless reduceOnly attempts when the exchange has
        already closed the position (manual / SL / liquidation). Errors are
        intentionally silent and treated as 0.0 — caller should NOT crash on
        a transient API blip; if the position is actually still open the
        next tick's close attempt will surface a real error."""
        try:
            self.futures_exchange.load_markets()
            futures_symbol = self.to_futures_symbol(symbol)
            if futures_symbol not in self.futures_exchange.markets:
                return 0.0
            self._sync_clocks()
            pos = self.futures_exchange.fetch_position(futures_symbol)
            if not pos:
                return 0.0
            contracts = pos.get('contracts')
            if contracts is None:
                # CCXT older versions / fallback fields
                info = pos.get('info', {}) or {}
                contracts = info.get('positionAmt') or info.get('size') or 0
            try:
                return abs(float(contracts))
            except (TypeError, ValueError):
                return 0.0
        except Exception as exc:
            logging.debug(f"get_futures_position_amount({symbol}): {exc}")
            return 0.0

    def execute_futures_order(self, symbol, side, amount_coin, reduce_only=False):
        """Sends a real market order to the Futures account (LONG / SHORT).

        Returns:
          - the CCXT order dict on success
          - {'reduce_only_rejected': True, 'error_code': -2022} if Binance
            rejected a reduceOnly close because there's no position to
            reduce. Callers MUST treat this as 'already closed' and stop
            retrying — repeating the same call is deterministic-fail.
          - None on any other error (worth retrying).

        In `paper` mode the order is booked internally via paper_book
        instead of hitting the exchange. The dashboard's Live Trading
        switch toggles this via data/control.json's trade_mode field.
        """
        if self._trade_mode() == 'paper':
            try:
                from src.engine.paper_book import book_market_order
                price = 0.0
                try:
                    self._sync_clocks()
                    t = self.futures_exchange.fetch_ticker(self.to_futures_symbol(symbol))
                    price = float(t.get('last') or t.get('close') or 0)
                except Exception:
                    pass
                return book_market_order(symbol, side, amount_coin, price,
                                         market="futures")
            except Exception as exc:
                logging.error(f"❌ paper FUTURES {side} {symbol} failed: {exc}")
                return None
        self._sync_clocks()
        try:
            self.futures_exchange.load_markets()
            futures_symbol = self.to_futures_symbol(symbol)
            if futures_symbol not in self.futures_exchange.markets:
                logging.warning(f"No perpetual futures market for {symbol} ({futures_symbol}) — skipping futures order")
                return None
            amount_coin = self.futures_exchange.amount_to_precision(futures_symbol, amount_coin)
            params = {'reduceOnly': True} if reduce_only else {}

            if side.upper() == 'BUY':
                order = self.futures_exchange.create_market_buy_order(futures_symbol, float(amount_coin), params)
            else:
                order = self.futures_exchange.create_market_sell_order(futures_symbol, float(amount_coin), params)
            logging.info(f"✅ FUTURES {side.upper()} {amount_coin} {symbol} (Reduce: {reduce_only}) executed. ID: {order.get('id')}")
            return order
        except Exception as e:
            err_str = str(e)
            # Binance -2022 = "ReduceOnly Order is rejected" — there's no
            # position to reduce. Distinct from generic API errors so the
            # close-path can short-circuit retries and force-close locally.
            if reduce_only and ('-2022' in err_str or 'ReduceOnly Order is rejected' in err_str):
                logging.warning(
                    f"FUTURES {side.upper()} {symbol} reduceOnly rejected (-2022) — "
                    f"exchange has no open position to close."
                )
                return {'reduce_only_rejected': True, 'error_code': -2022}
            logging.error(f"❌ Error in Futures order {side} on {symbol}: {e}")
            return None

    def execute_limit_futures_order(self, symbol, side, amount_coin, price, reduce_only=False):
        """Sends a LIMIT order to the Futures account (Used for Market Making)"""
        self._sync_clocks()
        try:
            self.futures_exchange.load_markets()
            futures_symbol = self.to_futures_symbol(symbol)
            if futures_symbol not in self.futures_exchange.markets:
                logging.warning(f"No perpetual futures market for {symbol} ({futures_symbol}) — skipping limit futures order")
                return None
            amount_coin = self.futures_exchange.amount_to_precision(futures_symbol, amount_coin)
            price_str = self.futures_exchange.price_to_precision(futures_symbol, price)
            params = {'reduceOnly': True} if reduce_only else {}
            
            order = self.futures_exchange.create_order(
                symbol=futures_symbol, type='limit', side=side.lower(), 
                amount=float(amount_coin), price=float(price_str), params=params
            )
            logging.info(f"📋 FUTURES LIMIT {side.upper()} {amount_coin} @ {price_str} executed. ID: {order.get('id')}")
            return order
        except Exception as e:
            logging.error(f"❌ Error in Limit order {side} on {symbol}: {e}")
            return None
            
    def cancel_all_orders(self, symbol):
        """Cancels all open limit orders for a specific symbol"""
        try:
            self.futures_exchange.load_markets()
            futures_symbol = self.to_futures_symbol(symbol)
            if futures_symbol not in self.futures_exchange.markets:
                return
            self.futures_exchange.cancel_all_orders(futures_symbol)
        except Exception as e:
            logging.error(f"Error canceling orders for {symbol}: {e}")

    # ── Phase 5: institutional circuit breakers ─────────────────────────────

    def circuit_breaker_check(
        self,
        *,
        peak_equity: float,
        current_equity: float,
        api_latency_ms: float,
        last_data_ts_unix: float,
        now_unix: float,
        max_daily_drawdown_pct: float = 0.05,    # 5% hard kill
        max_api_latency_ms:    float = 500.0,
        max_data_staleness_sec: float = 30.0,
    ) -> dict:
        """Return {ok, reason, trigger} per architecture plan §18.

        Triggers immediately HALT all new trades AND signal a flatten when:
          • Daily drawdown breaches `max_daily_drawdown_pct`
          • Round-trip API latency > `max_api_latency_ms`
          • Last market-data tick is older than `max_data_staleness_sec`
        """
        if peak_equity <= 0:
            dd = 0.0
        else:
            dd = max(0.0, (peak_equity - current_equity) / peak_equity)
        triggers = []
        if dd > max_daily_drawdown_pct:
            triggers.append(("max_daily_drawdown",
                             f"{dd*100:.2f}% > {max_daily_drawdown_pct*100:.2f}%"))
        if api_latency_ms > max_api_latency_ms:
            triggers.append(("api_latency", f"{api_latency_ms:.0f}ms > {max_api_latency_ms:.0f}ms"))
        if (now_unix - last_data_ts_unix) > max_data_staleness_sec:
            triggers.append(("data_feed_inconsistency",
                             f"stale {now_unix - last_data_ts_unix:.1f}s"))
        return {
            "ok":       len(triggers) == 0,
            "trigger":  triggers[0][0] if triggers else None,
            "reason":   triggers[0][1] if triggers else None,
            "all_triggers": triggers,
            "drawdown_pct": dd * 100.0,
        }

    # ── Phase 3: alpha-decay exit helper ────────────────────────────────────

    def should_alpha_decay_exit(
        self,
        signal_strength: float,
        time_in_trade: float,
        *,
        decay_rate: float = 0.1,
        exit_threshold: float = 0.2,
    ) -> bool:
        """Return True iff the trade's signal has decayed below `exit_threshold`.

        Per updated_architecture_plan_en.md §12 — replaces hard `max_hold_bars`
        with `signal_strength * exp(-decay_rate * time_in_trade) < threshold`.
        Callers pass the *original* signal strength at entry and how many
        bars the position has been open. Both `decay_rate` and
        `exit_threshold` should be tuned per strategy timeframe (see
        `src/analysis/alpha_decay.py` defaults).
        """
        from src.analysis.alpha_decay import should_exit
        return should_exit(
            signal_strength=signal_strength,
            time_in_trade=time_in_trade,
            decay_rate=decay_rate,
            exit_threshold=exit_threshold,
        )


if __name__ == "__main__":
    manager = OrderManager()
    usdt_bal = manager.get_balance('USDT')
    logging.info(f"Free USDT balance: {usdt_bal}")
    if usdt_bal >= 15:
        amount = 15.0 / 50000  # approximate BTC amount for $15
        manager.execute_spot_order('BTC/USDT', 'BUY', amount)
