"""
MT5 Integration Guide for Stacey Burke Scanner
============================================

Since MetaTrader5 Python package is Windows/Linux only, here are the options:

Option 1: If you have access to a Windows PC with MT5:
----------------------------------------------------
1. Install MT5 on Windows
2. Install Python on Windows
3. Run: pip install MetaTrader5
4. Use the connection code below

Option 2: Use MT5's built-in export features:
-------------------------------------------
1. MT5 can export data to CSV files
2. We can build a scanner that reads these CSV files
3. Schedule regular exports from MT5

Option 3: Use alternative data sources (Alpha Vantage):
------------------------------------------------------
- Use Alpha Vantage as primary data source
- Cross-reference with MT5 manually
- We'll implement this option

For now, let's create a scanner that can work with CSV exports from MT5:
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StaceyBurkeMT5Scanner:
    def __init__(self):
        self.major_pairs = [
            "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
            "NZDUSD", "USDCAD", "USDCHF"
        ]

    def read_mt5_csv_export(self, file_path):
        """
        Read CSV data exported from MT5
        MT5 Export Instructions:
        1. Open MT5 terminal
        2. Open desired pair chart (e.g., EURUSD H1)
        3. Right-click chart → Save As → CSV
        4. Save to the same folder as this script
        """
        try:
            # MT5 CSV format: <TICKER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOLUME>
            df = pd.read_csv(file_path)

            # Parse datetime from separate date/time columns
            df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
            df.set_index('Datetime', inplace=True)

            # Rename columns to standard format
            df = df.rename(columns={
                'OPEN': 'Open',
                'HIGH': 'High',
                'LOW': 'Low',
                'CLOSE': 'Close',
                'VOLUME': 'Volume'
            })

            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            return df.dropna()

        except Exception as e:
            print(f"Error reading MT5 CSV: {e}")
            return None

    def analyze_for_frd_fgd(self, df, symbol):
        """
        Analyze data for Stacey Burke FRD/FGD patterns
        """
        if df is None or len(df) < 10:
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

        # Find recent FRD/FGD signals
        signals = []

        for i in range(5, len(df)):  # Need at least 5 candles for pattern
            # Check for FRD (First Red Day after greens)
            if df.iloc[i]['is_bearish'] and df.iloc[i-1]['consecutive_bulls'] >= 3:
                signals.append({
                    'date': df.index[i],
                    'type': 'FRD',
                    'price': df.iloc[i]['Close'],
                    'high': df.iloc[i]['High'],
                    'consecutive': df.iloc[i-1]['consecutive_bulls']
                })

            # Check for FGD (First Green Day after reds)
            elif df.iloc[i]['is_bullish'] and df.iloc[i-1]['consecutive_bears'] >= 3:
                signals.append({
                    'date': df.index[i],
                    'type': 'FGD',
                    'price': df.iloc[i]['Close'],
                    'low': df.iloc[i]['Low'],
                    'consecutive': df.iloc[i-1]['consecutive_bears']
                })

        # Check for today's setup
        latest_setup = None
        if signals:
            latest_signal = signals[-1]

            # If latest signal was from the most recent completed candle
            if latest_signal['date'] >= df.index[-2]:
                if latest_signal['type'] == 'FRD':
                    latest_setup = {
                        'symbol': symbol,
                        'setup': 'FRD_CONFIRMED',
                        'action': 'LOOK_FOR_SHORTS',
                        'entry_zone': latest_signal['high'],
                        'stop_above': latest_signal['high'] * 1.005,
                        'signal_date': latest_signal['date']
                    }
                elif latest_signal['type'] == 'FGD':
                    latest_setup = {
                        'symbol': symbol,
                        'setup': 'FGD_CONFIRMED',
                        'action': 'LOOK_FOR_LONGS',
                        'entry_zone': latest_signal['low'],
                        'stop_below': latest_signal['low'] * 0.995,
                        'signal_date': latest_signal['date']
                    }

        return {
            'symbol': symbol,
            'signals_found': len(signals),
            'recent_signals': signals[-3:],  # Last 3 signals
            'latest_setup': latest_setup,
            'data_points': len(df)
        }

def create_mt5_export_instructions():
    """
    Create a simple guide for exporting data from MT5
    """
    instructions = """
MT5 DATA EXPORT INSTRUCTIONS
============================

To export data from MetaTrader 5:

1. Open MT5 Terminal
2. Select your ACG Markets-Main demo account (2428164)
3. Open a chart for the pair you want (e.g., EUR/USD)
4. Set timeframe to H1 (for hourly analysis)
5. Adjust the view to show at least 30-50 candles
6. Right-click on the chart
7. Select "Save As"
8. Choose "CSV" format
9. Save as: {SYMBOL}_H1.csv (e.g., EURUSD_H1.csv)
10. Repeat for each pair you want to analyze

Place all CSV files in the same folder as this script.

For Daily (D1) analysis:
- Change timeframe to D1
- Save as: {SYMBOL}_D1.csv (e.g., EURUSD_D1.csv)
"""

    with open('MT5_EXPORT_GUIDE.txt', 'w') as f:
        f.write(instructions)

    print("✅ MT5 Export Guide saved to 'MT5_EXPORT_GUIDE.txt'")
    print("\n" + instructions)

if __name__ == "__main__":
    scanner = StaceyBurkeMT5Scanner()

    print("🔍 STACEY BURKE MT5 SCANNER")
    print("=" * 60)

    # Create export instructions
    create_mt5_export_instructions()

    # Try to scan for exported CSV files
    print("\n📊 Scanning for MT5 CSV exports...")

    results = []

    for pair in scanner.major_pairs:
        h1_file = f"{pair}_H1.csv"
        if os.path.exists(h1_file):
            print(f"\n✅ Found H1 data for {pair}")
            df = scanner.read_mt5_csv_export(h1_file)
            result = scanner.analyze_for_frd_fgd(df, pair)
            results.append(result)
        else:
            print(f"❌ No H1 data found for {pair} (expected: {h1_file})")

    # Display results
    if results:
        print("\n" + "=" * 60)
        print("🎯 ANALYSIS RESULTS")
        print("=" * 60)

        for result in results:
            print(f"\n📍 {result['symbol']}:")
            print(f"   Data Points: {result['data_points']}")
            print(f"   Signals Found: {result['signals_found']}")

            if result['latest_setup']:
                setup = result['latest_setup']
                print(f"   🚨 SETUP: {setup['setup']}")
                print(f"   🎯 ACTION: {setup['action']}")
                print(f"   📍 Entry: {setup['entry_zone']:.5f}")
            else:
                print("   ✅ No valid setup for today")
    else:
        print("\n📋 No data found. Please export data from MT5 first.")
        print("See 'MT5_EXPORT_GUIDE.txt' for instructions.")