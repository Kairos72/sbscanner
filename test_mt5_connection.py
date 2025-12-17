"""
Test MT5 Connection
==================
This script tests if MetaTrader 5 is properly installed and accessible.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import MetaTrader5 as mt5
    print("[OK] MetaTrader5 module imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import MetaTrader5: {e}")
    print("\nTo install MetaTrader5, run:")
    print("pip install MetaTrader5")
    sys.exit(1)

# Initialize MT5 connection
print("\nInitializing MT5...")
if mt5.initialize():
    print("[OK] MT5 initialized successfully")

    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"[OK] Terminal: {terminal_info.name}")
        print(f"[OK] Build: {terminal_info.build}")
        print(f"[OK] Company: {terminal_info.company}")
    else:
        print("[WARNING] Could not get terminal info - is MT5 running?")

    # Get account info (if logged in)
    account_info = mt5.account_info()
    if account_info:
        print(f"\n[OK] Account Info:")
        print(f"  - Login: {account_info.login}")
        print(f"  - Server: {account_info.server}")
        print(f"  - Balance: {account_info.balance}")
        print(f"  - Equity: {account_info.equity}")
    else:
        print("\n[WARNING] Not logged into any account")
        print("Please open MT5 and log into your account")
        print("Account: 2428164")
        print("Server: ACGMarkets-Main")

    # Test getting data for EURUSD
    print("\nTesting data retrieval for EURUSD...")
    if mt5.symbol_select("EURUSD", True):
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10)
        if rates is not None and len(rates) > 0:
            print(f"[OK] Successfully retrieved {len(rates)} candle(s) for EURUSD H1")
            print(f"  Latest close: {rates[-1]['close']:.5f}")
        else:
            print("[WARNING] Could not retrieve data for EURUSD")
            print("Make sure EURUSD is enabled in Market Watch")
    else:
        print("[WARNING] Could not select EURUSD symbol")

    # Shutdown MT5
    mt5.shutdown()
    print("\n[OK] MT5 connection closed")

else:
    print("[ERROR] Failed to initialize MT5")
    print("\nTroubleshooting:")
    print("1. Make sure MetaTrader 5 is installed")
    print("2. Make sure MT5 is running")
    print("3. Try running this script as administrator")
    print("\nTo download MT5:")
    print("https://www.metatrader5.com/en/download")