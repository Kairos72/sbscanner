"""
Live MT5 Integration Scanner (Windows Only)
===========================================

This code works on Windows with MetaTrader5 installed.
If you have access to a Windows PC with MT5, use this script.

Installation on Windows:
1. Install Python 3.8+ from python.org
2. Run: pip install MetaTrader5
3. Run this script
"""

# Import the MetaTrader5 library (Windows only)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️  MetaTrader5 library not available on this platform")
    print("This script requires Windows with MT5 installed")
    MT5_AVAILABLE = False
    # We'll create a mock module for testing
    class MockMT5:
        def initialize(self, **kwargs):
            print("🔌 Mock MT5 connection (for testing)")
            return False
        def login(self, **kwargs):
            return False
        def symbol_select(self, *args):
            return False
        def copy_rates_from_pos(self, *args):
            return None
        def shutdown(self):
            pass
        def terminal_info(self):
            return None
        def account_info(self):
            return None
    mt5 = MockMT5()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StaceyBurkeLiveScanner:
    def __init__(self):
        self.account = 2428164
        self.password = "3b6%StTrQj"  # You'll replace this with your credentials
        self.server = "ACGMarkets-Main"

        self.major_pairs = [
            "EURUSD.pro", "GBPUSD.pro", "AUDUSD.pro", "USDJPY.pro",
            "NZDUSD.pro", "USDCAD.pro", "USDCHF.pro"
        ]

    def connect_to_mt5(self):
        """Connect to MetaTrader 5"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                print("❌ MT5 initialize failed")
                return False

            print(f"🔌 MT5 Version: {mt5.version()}")

            # Login to your account
            authorized = mt5.login(
                login=self.account,
                password=self.password,
                server=self.server
            )

            if authorized:
                # Get account info
                account_info = mt5.account_info()
                if account_info:
                    print(f"✅ Connected to account: {account_info.login}")
                    print(f"🏢 Broker: {account_info.server}")
                    print(f"💰 Balance: {account_info.balance} {account_info.currency}")
                    print(f"📊 Equity: {account_info.equity} {account_info.currency}")

                # Enable all symbols
                for symbol in self.major_pairs:
                    mt5.symbol_select(symbol, True)

                return True
            else:
                print(f"❌ Failed to login: {mt5.last_error()}")
                return False

        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def get_rates(self, symbol, timeframe, count=500):
        """Get price data from MT5"""
        try:
            # MT5 timeframe constants
            timeframes = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }

            tf = timeframes.get(timeframe, mt5.TIMEFRAME_H1)

            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

            if rates is not None and len(rates) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(rates)

                # Convert time to datetime
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)

                # Rename columns
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'tick_volume': 'Volume'
                })

                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            else:
                print(f"❌ No data for {symbol}")
                return None

        except Exception as e:
            print(f"❌ Error getting rates for {symbol}: {e}")
            return None

    def analyze_stacey_burke_setup(self, df, symbol):
        """Analyze for Stacey Burke patterns"""
        if df is None or len(df) < 20:
            return None

        # Identify candle types
        df['is_bullish'] = df['Close'] > df['Open']
        df['is_bearish'] = df['Close'] < df['Open']

        # Calculate consecutive runs
        df['consecutive_bulls'] = 0
        df['consecutive_bears'] = 0

        for i in range(1, len(df)):
            if df.iloc[i]['is_bullish']:
                df.loc[df.index[i], 'consecutive_bulls'] = df.iloc[i-1]['consecutive_bulls'] + 1
                df.loc[df.index[i], 'consecutive_bears'] = 0
            elif df.iloc[i]['is_bearish']:
                df.loc[df.index[i], 'consecutive_bears'] = df.iloc[i-1]['consecutive_bears'] + 1
                df.loc[df.index[i], 'consecutive_bulls'] = 0

        # Find recent signals
        recent_signals = []

        # Check last 10 candles for patterns
        for i in range(len(df) - 10, len(df)):
            if i < 5:
                continue

            # Check for FRD
            if df.iloc[i]['is_bearish'] and df.iloc[i-1]['consecutive_bulls'] >= 3:
                recent_signals.append({
                    'date': df.index[i],
                    'type': 'FRD',
                    'price': df.iloc[i]['Close'],
                    'high': df.iloc[i]['High'],
                    'consecutive': df.iloc[i-1]['consecutive_bulls']
                })

            # Check for FGD
            elif df.iloc[i]['is_bullish'] and df.iloc[i-1]['consecutive_bears'] >= 3:
                recent_signals.append({
                    'date': df.index[i],
                    'type': 'FGD',
                    'price': df.iloc[i]['Close'],
                    'low': df.iloc[i]['Low'],
                    'consecutive': df.iloc[i-1]['consecutive_bears']
                })

        # Check for today's setup
        latest_setup = None
        if recent_signals:
            latest_signal = recent_signals[-1]

            # Check if latest signal was very recent (within last 2 candles)
            if latest_signal['date'] >= df.index[-2]:
                if latest_signal['type'] == 'FRD':
                    latest_setup = {
                        'symbol': symbol,
                        'setup': 'FRD_CONFIRMED',
                        'action': 'LOOK_FOR_SHORTS',
                        'entry_zone': latest_signal['high'],
                        'stop_above': latest_signal['high'] * 1.005,
                        'signal_date': latest_signal['date'],
                        'consecutive_days': latest_signal['consecutive']
                    }
                elif latest_signal['type'] == 'FGD':
                    latest_setup = {
                        'symbol': symbol,
                        'setup': 'FGD_CONFIRMED',
                        'action': 'LOOK_FOR_LONGS',
                        'entry_zone': latest_signal['low'],
                        'stop_below': latest_signal['low'] * 0.995,
                        'signal_date': latest_signal['date'],
                        'consecutive_days': latest_signal['consecutive']
                    }

        return {
            'symbol': symbol,
            'latest_price': df.iloc[-1]['Close'],
            'signals_count': len(recent_signals),
            'recent_signals': recent_signals[-2:],  # Last 2 signals
            'setup': latest_setup
        }

    def scan_all_pairs(self):
        """Scan all major pairs"""
        print("\n🔍 Scanning all major pairs for Stacey Burke setups...")
        print("=" * 60)

        results = []

        for symbol in self.major_pairs:
            print(f"\n📍 Analyzing {symbol}...")

            # Get H1 data
            h1_data = self.get_rates(symbol, 'H1', 200)

            if h1_data is not None:
                result = self.analyze_stacey_burke_setup(h1_data, symbol)
                results.append(result)

                if result:
                    print(f"   ✅ Analyzed {len(h1_data)} candles")
                    print(f"   📊 Latest Price: {result['latest_price']:.5f}")
                    print(f"   🔍 Signals Found: {result['signals_count']}")

                    if result['setup']:
                        setup = result['setup']
                        print(f"   🚨 SETUP: {setup['setup']}")
                        print(f"   🎯 ACTION: {setup['action']}")
                        print(f"   📍 Entry Zone: {setup['entry_zone']:.5f}")
            else:
                print(f"   ❌ No data available for {symbol}")

        # Display summary
        self.display_summary(results)

        return results

    def display_summary(self, results):
        """Display summary of all setups found"""
        print("\n" + "=" * 60)
        print("🎯 SETUPS SUMMARY")
        print("=" * 60)

        valid_setups = [r for r in results if r['setup'] is not None]

        if valid_setups:
            print(f"\n✅ Found {len(valid_setups)} valid setup(s):")

            for result in valid_setups:
                setup = result['setup']
                print(f"\n📍 {setup['symbol']}:")
                print(f"   Setup: {setup['setup']}")
                print(f"   Action: {setup['action']}")
                print(f"   Entry: {setup['entry_zone']:.5f}")
                print(f"   Signal: {setup['consecutive_days']} consecutive days")
        else:
            print("\n❌ No valid setups found at this time")
            print("   Wait for FRD or FGD signals on H1 timeframe")

    def run_continuous_scan(self, interval_minutes=60):
        """Run continuous scanning"""
        print(f"\n🔄 Starting continuous scan (every {interval_minutes} minutes)")

        try:
            while True:
                print(f"\n⏰ Scan time: {datetime.now().strftime('%H:%M:%S')}")
                self.scan_all_pairs()

                print(f"\n💤 Next scan in {interval_minutes} minutes...")
                import time
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\n⏹️  Scanning stopped by user")

    def disconnect(self):
        """Disconnect from MT5"""
        if MT5_AVAILABLE:
            mt5.shutdown()
            print("🔌 Disconnected from MT5")

def main():
    """Main function to run the scanner"""
    scanner = StaceyBurkeLiveScanner()

    try:
        # Connect to MT5
        if scanner.connect_to_mt5():
            # Run single scan
            results = scanner.scan_all_pairs()

            # Optionally run continuous scan
            # scanner.run_continuous_scan(interval_minutes=60)

        else:
            print("\n❌ Failed to connect to MT5")
            print("Please check:")
            print("1. MT5 is running")
            print("2. Account credentials are correct")
            print("3. Server name is correct")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        scanner.disconnect()

if __name__ == "__main__":
    main()