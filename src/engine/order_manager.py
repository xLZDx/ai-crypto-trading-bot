import ccxt
import os
import logging
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        # Initialize Binance exchange (Spot)
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        })
        
        # Initialize Binance exchange (Futures)
        self.futures_exchange = ccxt.binance({
            'apiKey': self.futures_api_key,
            'secret': self.futures_api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        if self.use_testnet:
            self.exchange.set_sandbox_mode(True)
            self.futures_exchange.set_sandbox_mode(True)
            logging.info("OrderManager initialized (Mode: TESTNET - SAFE TRADING)")
        else:
            logging.warning("OrderManager initialized (Mode: MAINNET - REAL MONEY!)")

    def get_balance(self, asset='USDT'):
        """Returns the free balance of the specified asset."""
        # If there are no keys, simulate test balance to avoid spam errors from Binance
        if not self.api_key or self.api_key == 'your_api_key_here':
            if asset == 'USDT': return 10000.0
            return 0.0
            
        try:
            balance = self.exchange.fetch_balance()
            if asset in balance:
                return float(balance[asset]['free'])
            return 0.0
        except Exception as e:
            logging.error(f"Error getting balance: {e}")
            return 0.0

    def execute_spot_order(self, symbol, side, amount_coin):
        """Sends a real market order to the Spot account"""
        if not self.api_key or self.api_key == 'your_api_key_here':
            return True # Simulate success for tests without keys
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

    def execute_futures_order(self, symbol, side, amount_coin, reduce_only=False):
        """Sends a real market order to the Futures account (LONG / SHORT)"""
        if not self.futures_api_key or self.futures_api_key == 'your_api_key_here':
            return True # Simulate success
        try:
            self.futures_exchange.load_markets()
            futures_symbol = f"{symbol.split('/')[0]}/USDT:USDT" # Convert format BTC/USDT -> BTC/USDT:USDT
            amount_coin = self.futures_exchange.amount_to_precision(futures_symbol, amount_coin)
            params = {'reduceOnly': True} if reduce_only else {}
            
            if side.upper() == 'BUY':
                order = self.futures_exchange.create_market_buy_order(futures_symbol, float(amount_coin), params)
            else:
                order = self.futures_exchange.create_market_sell_order(futures_symbol, float(amount_coin), params)
            logging.info(f"✅ FUTURES {side.upper()} {amount_coin} {symbol} (Reduce: {reduce_only}) executed. ID: {order.get('id')}")
            return order
        except Exception as e:
            logging.error(f"❌ Error in Futures order {side} on {symbol}: {e}")
            return None

if __name__ == "__main__":
    # Engine testing: try to buy some Bitcoin for 15 test dollars
    manager = OrderManager()
    
    usdt_bal = manager.get_balance('USDT')
    logging.info(f"Free USDT balance: {usdt_bal}")
    
    # Buy BTC/USDT for $15 (we have $10,000 in testnet, so risk manager will allow it)
    if usdt_bal >= 15:
        manager.buy_market_safe('BTC/USDT', 15.0, max_risk_percent=5.0)
