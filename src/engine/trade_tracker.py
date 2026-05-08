import os
import uuid
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.safe_json import read_json, write_json


def _detect_trade_mode() -> str:
    """v3.1 step 11 (1D) — best-effort mode detection for trade rows
    that don't have an explicit override. Reads data/control.json
    (operator's mode toggle); falls back to USE_TESTNET env var."""
    try:
        ctrl = read_json('data/control.json', default={}) or {}
        m = (ctrl.get('trade_mode') or '').lower()
        if m in ('paper', 'testnet', 'mainnet'):
            return m
    except Exception:
        pass
    return 'testnet' if os.getenv('USE_TESTNET', 'true').lower() == 'true' else 'mainnet'


class TradeTracker:
    def __init__(self, filepath='data/trades.json'):
        self.filepath = filepath
        self.trades = []
        self.load_trades()

    def load_trades(self):
        loaded = read_json(self.filepath, default=[])
        if isinstance(loaded, list):
            self.trades = loaded
        else:
            self.trades = []

    def save_trades(self):
        write_json(self.filepath, self.trades)

    def open_trade(self, symbol, amount_usdt, current_price, strategy="Base_Elliott",
                   trailing_stop_percent=2.0, market="SPOT", side="LONG",
                   *, mode=None, regime_at_entry=None, model_confidence=None,
                   intended_price=None):
        """Open a trade row.

        v3.1 step 11 (1D) — every NEW row carries the 7 enrichment
        fields (mode, regime_at_entry, model_confidence, mfe_pct,
        mae_pct, slippage_pct, exit_reason) so downstream analytics
        (1L per-market filter, 4A analytical dashboard) can join /
        bucket without ambiguity. mfe_pct / mae_pct start at 0 and
        get updated by update_trailing_stops(); exit_reason stays
        None until the trade closes; slippage is computed at open
        if intended_price is provided.
        """
        slippage_pct = None
        if intended_price and intended_price > 0:
            slippage_pct = (current_price - intended_price) / intended_price * 100.0

        trade = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "market": market,
            "side": side,
            "strategy": strategy,
            "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "buy_price": current_price,
            "highest_price": current_price,
            "lowest_price": current_price,  # for MAE on LONG
            "trailing_stop_percent": trailing_stop_percent,
            "amount_coin": amount_usdt / current_price,
            "invested_usdt": amount_usdt,
            "status": "OPEN",
            "sell_time": None,
            "sell_price": None,
            "pnl_usdt": None,
            "pnl_percent": None,
            # v3.1 step 11 enrichment fields
            "mode": (mode or _detect_trade_mode()),
            "regime_at_entry": regime_at_entry,
            "model_confidence": model_confidence,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "slippage_pct": slippage_pct,
            "exit_reason": None,
        }
        self.trades.append(trade)
        self.save_trades()
        return trade

    def update_trailing_stops(self, symbol, current_price):
        closed_trades = []
        updated = False

        for trade in self.trades:
            if trade["status"] == "OPEN" and trade["symbol"] == symbol:
                # Initialization for old trades (backward compatibility)
                if "highest_price" not in trade:
                    trade["highest_price"] = trade["buy_price"]
                if "trailing_stop_percent" not in trade:
                    trade["trailing_stop_percent"] = 2.0
                if "side" not in trade:
                    trade["side"] = "LONG"
                
                # Update the historical maximum of the trade
                if trade["side"] == "LONG" and current_price > trade["highest_price"]:
                    trade["highest_price"] = current_price
                    updated = True
                elif trade["side"] == "SHORT" and current_price < trade["highest_price"]: # For shorts, the "maximum" is the minimum price
                    trade["highest_price"] = current_price
                    updated = True

                # v3.1 step 11 (1D) — track MFE / MAE so they're
                # available at close time without re-loading 1m bars.
                if "lowest_price" not in trade:
                    trade["lowest_price"] = trade["buy_price"]
                if current_price < trade["lowest_price"]:
                    trade["lowest_price"] = current_price
                    updated = True
                ep = float(trade["buy_price"]) or 1.0
                if trade["side"] == "LONG":
                    mfe_pct = (trade["highest_price"] - ep) / ep * 100.0
                    mae_pct = (trade["lowest_price"]  - ep) / ep * 100.0
                else:
                    mfe_pct = (ep - trade["highest_price"]) / ep * 100.0
                    mae_pct = (ep - trade["lowest_price"])  / ep * 100.0
                trade["mfe_pct"] = max(0.0, float(mfe_pct))
                trade["mae_pct"] = min(0.0, float(mae_pct))
                    
                # Calculate current unrealized PnL for real-time display
                if trade["side"] == "LONG":
                    trade["unrealized_pnl"] = (current_price - trade["buy_price"]) * trade["amount_coin"]
                else: # SHORT
                    trade["unrealized_pnl"] = (trade["buy_price"] - current_price) * trade["amount_coin"]
                    
                trade["unrealized_pnl_percent"] = (trade["unrealized_pnl"] / trade["invested_usdt"]) * 100
                trade["current_price"] = current_price
                updated = True
                
                # Check trailing stop condition (price dropped by a set % from maximum)
                is_stop_hit = False
                if trade["side"] == "LONG":
                    stop_price = trade["highest_price"] * (1 - trade["trailing_stop_percent"] / 100.0)
                    if current_price <= stop_price: is_stop_hit = True
                else: # SHORT trailing stop
                    stop_price = trade["highest_price"] * (1 + trade["trailing_stop_percent"] / 100.0)
                    if current_price >= stop_price: is_stop_hit = True

                if is_stop_hit:
                    # Do not close internally yet. Defer to the main orchestrator to verify Binance execution.
                    closed_trades.append(trade)
        
        if updated:
            self.save_trades()
            
        return closed_trades

    def get_open_trades(self, symbol=None, side=None, market=None):
        open_trades = []
        for trade in self.trades:
            if trade["status"] == "OPEN":
                if symbol and trade["symbol"] != symbol: continue
                if side and trade.get("side", "LONG") != side: continue
                if market and trade.get("market", "SPOT") != market: continue
                open_trades.append(trade)
        return open_trades

    def close_trade_by_id(self, trade_id, real_sell_price, *, exit_reason=None):
        for trade in self.trades:
            if trade["id"] == trade_id and trade["status"] == "OPEN":
                trade["status"] = "CLOSED"
                trade["sell_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade["sell_price"] = real_sell_price

                if trade.get("side", "LONG") == "LONG":
                    trade["pnl_usdt"] = (real_sell_price - trade["buy_price"]) * trade["amount_coin"]
                else:
                    trade["pnl_usdt"] = (trade["buy_price"] - real_sell_price) * trade["amount_coin"]

                trade["pnl_percent"] = (trade["pnl_usdt"] / trade["invested_usdt"]) * 100
                # v3.1 step 11 (1D) — record exit reason. If caller
                # didn't supply one, infer from PnL sign + the
                # trailing-stop logic that fired the close.
                if exit_reason:
                    trade["exit_reason"] = exit_reason
                elif trade.get("exit_reason") is None:
                    trade["exit_reason"] = ('TP' if trade["pnl_usdt"] > 0
                                            else 'SL' if trade["pnl_usdt"] < 0
                                            else 'flat')
                self.save_trades()
                return trade
        return None
