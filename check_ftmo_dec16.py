"""
Check FTMO Data for December 16, 2025
=====================================
Comparing FTMO broker data with TradingView
"""

import MetaTrader5 as mt5
import pandas as pd

print("=" * 60)
print("FTMO BROKER DATA - December 16, 2025")
print("=" * 60)

if mt5.initialize():
    # Check terminal info
    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()

    print(f"\nTerminal: {terminal_info.name}")
    print(f"Account: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Company: {terminal_info.company}")

    # Get USDCHF data for Dec 16
    rates = mt5.copy_rates_from_pos('USDCHF', mt5.TIMEFRAME_D1, 0, 20)

    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        print("\nUSDCHF Daily Data (Last 10 days):")
        print("-" * 70)

        for i in range(len(df)-10, len(df)):
            row = df.iloc[i]
            is_green = row['close'] > row['open']
            range_pips = abs(row['close'] - row['open']) * 10000

            print(f"{df.index[i].strftime('%Y-%m-%d %a'):<12} "
                  f"O:{row['open']:.5f} C:{row['close']:.5f} "
                  f"Range:{range_pips:6.0f} "
                  f"{'GREEN' if is_green else 'RED'}")

            if df.index[i].strftime('%m-%d') == '12-16':
                print("\n>>> DECEMBER 16, 2025 DATA <<<")
                print(f"FTMO Broker: {'GREEN' if is_green else 'RED'}")
                print(f"Open: {row['open']:.5f}")
                print(f"Close: {row['close']:.5f}")
                print(f"Range: {range_pips:.1f} pips")

                # Check if this matches TradingView (RED)
                if is_green:
                    print("\n⚠️  ISSUE: FTMO shows GREEN while TradingView shows RED")
                    print("This still creates incorrect FGD/FRD signals!")
                else:
                    print("\n✓ CORRECT: FTMO shows RED, matching TradingView")
                    print("This data feed is more reliable!")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print("If FTMO matches TradingView for Dec 16 USDCHF (RED):")
    print("✓ FTMO has reliable data")
    print("✓ FGD/FRD signals will be accurate")
    print("✓ Trading decisions based on correct data")

    mt5.shutdown()
else:
    print("Failed to connect to MT5")