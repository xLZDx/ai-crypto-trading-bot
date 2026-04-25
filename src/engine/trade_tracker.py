import json
import os
from datetime import datetime

class TradeTracker:
    def __init__(self, filepath='data/trades.json'):
        self.filepath = filepath
        self.trades = []
        self.load_trades()

    def load_trades(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.trades = json.load(f)
            except Exception:
                self.trades = []
        else:
            self.trades = []

    def save_trades(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.trades, f, indent=4, ensure_ascii=False)

    def open_trade(self, symbol, amount_usdt, current_price, strategy="Base_Elliott", trailing_stop_percent=2.0, market="SPOT", side="LONG"):
        trade = {
            "id": len(self.trades) + 1,
            "symbol": symbol,
            "market": market,
            "side": side,
            "strategy": strategy,
            "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "buy_price": current_price,
            "highest_price": current_price,
            "trailing_stop_percent": trailing_stop_percent,
            "amount_coin": amount_usdt / current_price,
            "invested_usdt": amount_usdt,
            "status": "OPEN",
            "sell_time": None,
            "sell_price": None,
            "pnl_usdt": None,
            "pnl_percent": None
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
                    trade["status"] = "CLOSED"
                    trade["sell_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trade["sell_price"] = current_price
                    
                    if trade["side"] == "LONG":
                        trade["pnl_usdt"] = (current_price - trade["buy_price"]) * trade["amount_coin"]
                    else:
                        trade["pnl_usdt"] = (trade["buy_price"] - current_price) * trade["amount_coin"]
                        
                    trade["pnl_percent"] = (trade["pnl_usdt"] / trade["invested_usdt"]) * 100
                    closed_trades.append(trade.copy())
                    updated = True
        
        if updated:
            self.save_trades()
            
        return closed_trades

    def close_trades(self, current_price, symbol=None, side=None, market=None):
        closed_trades = []
        for trade in self.trades:
            if trade["status"] == "OPEN":
                if symbol and trade["symbol"] != symbol:
                    continue
                if side and trade.get("side", "LONG") != side:
                    continue
                if market and trade.get("market", "SPOT") != market:
                    continue
                trade["status"] = "CLOSED"
                trade["sell_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade["sell_price"] = current_price
                
                if trade.get("side", "LONG") == "LONG":
                    trade["pnl_usdt"] = (current_price - trade["buy_price"]) * trade["amount_coin"]
                else:
                    trade["pnl_usdt"] = (trade["buy_price"] - current_price) * trade["amount_coin"]
                    
                trade["pnl_percent"] = (trade["pnl_usdt"] / trade["invested_usdt"]) * 100
                closed_trades.append(trade.copy())
        
        if closed_trades:
            self.save_trades()
            
        return closed_trades
