import ccxt
import os
from dotenv import load_dotenv

def test_connection():
    # Load variables from .env file
    load_dotenv()
    
    exchange_id = os.getenv('EXCHANGE_ID', 'binance')
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    use_testnet = os.getenv('USE_TESTNET', 'True').lower() in ('true', '1', 't')

    print(f"--- Testing connection to {exchange_id.upper()} SPOT ---")
    print(f"Testnet mode (Paper trading): {use_testnet}")
    
    if not api_key or api_key == 'your_api_key_here':
        print("WARNING: API keys are not configured!")
        return

    try:
        # Dynamically initialize exchange for spot
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        # Enable sandbox mode if specified
        if use_testnet:
            exchange.set_sandbox_mode(True)

        # Try to fetch balance for spot account
        print("\nAttempting to fetch spot account balance...")
        balance = exchange.fetch_balance()
        
        print("\nCONNECTION SUCCESSFUL!")
        
        # Print USDT balance (or fake USDT in testnet)
        if 'USDT' in balance:
            print(f"Free USDT balance in Spot: {balance['USDT']['free']}")
        else:
            print("Balance read successfully, but USDT not found in spot account.")
            
    except ccxt.AuthenticationError as e:
        print(f"\nAUTHENTICATION ERROR: Check your API keys or IP whitelist.\n{e}")
    except Exception as e:
        print(f"\nUNKNOWN ERROR:\n{e}")

if __name__ == "__main__":
    test_connection()
