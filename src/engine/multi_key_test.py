import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

def test_keys(name, api_key, api_secret):
    print(f"\n=========================================")
    print(f"Testing Key Pair: {name}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"=========================================")

    scenarios = [
        {'market': 'spot', 'testnet': True},
        {'market': 'spot', 'testnet': False},
        {'market': 'future', 'testnet': True},
        {'market': 'future', 'testnet': False},
    ]

    for scenario in scenarios:
        market = scenario['market']
        testnet = scenario['testnet']
        
        print(f"\n---> Testing {market.upper()} on {'TESTNET' if testnet else 'MAINNET'}")
        
        try:
            exchange_options = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            }
            
            if market == 'future':
                exchange_options['options'] = {'defaultType': 'future'}
                
            exchange = ccxt.binance(exchange_options)
            
            if testnet:
                exchange.set_sandbox_mode(True)

            balance = exchange.fetch_balance()
            print("     [SUCCESS] Connection established!")
            
            # Print balances for common testnet coins
            for coin in ['USDT', 'BUSD', 'BTC', 'BNB']:
                if coin in balance and float(balance[coin]['free']) > 0:
                    print(f"     [INFO] Free {coin} balance: {balance[coin]['free']}")
            
            if not any(coin in balance and float(balance[coin]['free']) > 0 for coin in ['USDT', 'BUSD', 'BTC', 'BNB']):
                print("     [INFO] Connected, but no balance found for USDT/BUSD/BTC/BNB.")
                
        except ccxt.AuthenticationError as e:
            print(f"     [ERROR] Authentication failed (Invalid key, permissions, or IP).")
        except Exception as e:
            print(f"     [ERROR] Unknown error: {str(e)[:100]}")

if __name__ == "__main__":
    keys = [
        {
            "name": "Binance Spot Testnet Keys",
            "api_key": os.getenv("BINANCE_SPOT_TESTNET_API_KEY", ""),
            "api_secret": os.getenv("BINANCE_SPOT_TESTNET_API_SECRET", "")
        },
        {
            "name": "Binance Futures Testnet Keys",
            "api_key": os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", ""),
            "api_secret": os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")
        }
    ]

    for k in keys:
        if not k["api_key"] or not k["api_secret"]:
            print(f"\n[SKIP] {k['name']}: env vars not set.")
            continue
        test_keys(k["name"], k["api_key"], k["api_secret"])
