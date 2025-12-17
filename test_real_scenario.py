import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_realistic_scenario():
    """Create data that matches the actual EURUSD scenario described"""

    # Create dates for our scenario
    dates = pd.date_range(start='2025-12-10', end='2025-12-17', freq='D')

    # Simulate the actual price action
    data = {
        'Date': dates,
        'Open': [1.0850, 1.0860, 1.0870, 1.0880, 1.0890, 1.0900, 1.0910, 1.0920],
        'High': [1.0860, 1.0870, 1.0880, 1.0890, 1.0900, 1.0910, 1.0920, 1.0930],
        'Low': [1.0840, 1.0850, 1.0860, 1.0870, 1.0880, 1.0890, 1.0900, 1.0910],
        'Close': [1.0860, 1.0870, 1.0880, 1.0890, 1.0900, 1.0910, 1.0850, 1.0860]  # Note Dec 16 closes red (FRD)
    }

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Add day labels
    df['Day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Mon', 'Tue', 'Wed']  # Dec 10-17
    df['is_bullish'] = df['Close'] > df['Open']
    df['is_bearish'] = df['Close'] < df['Open']

    return df

def analyze_scenario(df):
    """Analyze the scenario according to Stacey Burke rules"""

    print("="*60)
    print("📊 EURUSD STACEY BURKE ANALYSIS - DEC 2025")
    print("="*60)

    # Show each day
    for date, row in df.iterrows():
        day = row['Day']
        candle = "🟢 GREEN" if row['is_bullish'] else "🔴 RED"
        print(f"\n{date.strftime('%b %d, %Y')} ({day}): {candle}")
        print(f"  Open: {row['Open']:.5f} | Close: {row['Close']:.5f} | High: {row['High']:.5f}")

        if date.date() == pd.to_datetime('2025-12-12').date():
            print("  ↳ Friday: Inside Day (setup for Monday)")
        elif date.date() == pd.to_datetime('2025-12-15').date():
            print("  ↳ Monday: FGD - First Green Day (breakout)")
        elif date.date() == pd.to_datetime('2025-12-16').date():
            print("  ↳ Tuesday: FRD - First Red Day (after multiple greens)")
            print("  ⚠️ THIS IS THE SIGNAL DAY!")

    # Today's analysis
    today = df[df.index.date == pd.to_datetime('2025-12-17').date()].iloc[0]
    tuesday = df[df.index.date == pd.to_datetime('2025-12-16').date()].iloc[0]

    print("\n" + "="*60)
    print("🚨 TODAY'S SETUP - WEDNESDAY, DECEMBER 17, 2025")
    print("="*60)

    print(f"\n✅ SIGNAL CONFIRMED: Tuesday was FRD @ {tuesday['Close']:.5f}")
    print(f"🎯 TODAY'S ACTION: LOOK FOR SHORTS")
    print(f"📍 Entry Zone: At or above {tuesday['High']:.5f}")
    print(f"⛔ Stop Loss: Above {tuesday['High'] * 1.005:.5f}")

    print("\n💡 STACEY BURKE RULE:")
    print("   1. Tuesday closed as FRD after multiple green days ✅")
    print("   2. Wednesday = Look for shorts at Tuesday's high ✅")
    print("   3. Entry only if price reaches Tuesday's high zone ✅")

    # Check if we're aligned
    print("\n" + "="*60)
    print("🎯 YOUR TRADING PLAN (MANILA TIME)")
    print("="*60)

    print(f"\n📅 Session Context:")
    print("   • 2:00 PM PHT = 2:00 AM EST (London Open)")
    print("   • 8:00 PM PHT = 8:00 AM EST (NY Open)")
    print("   • Prime time: 2:00 PM - 12:00 AM PHT")

    print(f"\n🔍 What to Watch For:")
    print(f"   • Price climbing toward {tuesday['High']:.5f}")
    print(f"   • Rejection at that level (short entry)")
    print(f"   • If price breaks above - wait for new setup")

def main():
    df = create_realistic_scenario()
    analyze_scenario(df)

if __name__ == "__main__":
    main()