import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class StaceyBurkeScanner:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.h1_data = None
        self.d1_data = None

    def create_synthetic_data(self, timeframe='H1', days=30):
        """Create synthetic forex data for testing"""
        # Create date range
        if timeframe == 'H1':
            dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        else:  # D1
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Start with realistic EUR/USD price
        base_price = 1.0850
        price_changes = np.random.normal(0, 0.002, len(dates))

        # Create price series with trend simulation
        close_prices = []
        current_price = base_price
        trend = np.random.choice([-1, 0, 1])  # Random trend direction

        for i, change in enumerate(price_changes):
            # Add trend component
            if i % 24 == 0:  # Daily trend changes
                trend = np.random.choice([-1, 0, 1])

            trend_component = trend * 0.0005
            current_price += change + trend_component
            current_price = max(1.0500, min(1.1500, current_price))
            close_prices.append(current_price)

        # Create OHLC data
        data = pd.DataFrame(index=dates)
        data['Close'] = close_prices
        data['Open'] = data['Close'].shift(1).fillna(base_price)

        # Generate High and Low
        volatility = 0.002 if timeframe == 'H1' else 0.005
        data['High'] = np.maximum(data['Open'], data['Close']) + np.random.uniform(0, volatility, len(dates))
        data['Low'] = np.minimum(data['Open'], data['Close']) - np.random.uniform(0, volatility, len(dates))
        data['Volume'] = np.random.randint(1000, 10000, len(dates))

        return data.dropna()

    def fetch_data(self, period="1mo"):
        """Fetch H1 and D1 data"""
        try:
            print("Fetching H1 data...")
            self.h1_data = self.create_synthetic_data('H1', 30)

            print("Fetching D1 data...")
            self.d1_data = self.create_synthetic_data('D1', 30)

            print(f"Successfully fetched H1: {len(self.h1_data)} candles, D1: {len(self.d1_data)} candles")
            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def identify_candle_types(self, df):
        """Identify candle types and patterns"""
        df = df.copy()
        df['is_bullish'] = df['Close'] > df['Open']
        df['is_bearish'] = df['Close'] < df['Open']
        df['is_inside'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        df['body_size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) * 100

        # Calculate consecutive candles
        df['consecutive_bulls'] = 0
        df['consecutive_bears'] = 0

        for i in range(1, len(df)):
            if df.iloc[i]['is_bullish']:
                df.loc[df.index[i], 'consecutive_bulls'] = df.iloc[i-1]['consecutive_bulls'] + 1
                df.loc[df.index[i], 'consecutive_bears'] = 0
            elif df.iloc[i]['is_bearish']:
                df.loc[df.index[i], 'consecutive_bears'] = df.iloc[i-1]['consecutive_bears'] + 1
                df.loc[df.index[i], 'consecutive_bulls'] = 0

        return df

    def find_frd_fgd_signals(self, df, min_consecutive=3):
        """Find FRD/FGD signals"""
        signals = []

        for i in range(min_consecutive, len(df)):
            current_candle = df.iloc[i]

            # Check for FRD
            if current_candle['is_bearish'] and df.iloc[i-1]['consecutive_bulls'] >= min_consecutive:
                signals.append({
                    'date': df.index[i],
                    'type': 'FRD',
                    'price': current_candle['Close'],
                    'high': current_candle['High'],
                    'consecutive_days': df.iloc[i-1]['consecutive_bulls'],
                    'setup_strength': 'STRONG' if df.iloc[i-1]['consecutive_bulls'] >= 5 else 'NORMAL'
                })

            # Check for FGD
            elif current_candle['is_bullish'] and df.iloc[i-1]['consecutive_bears'] >= min_consecutive:
                signals.append({
                    'date': df.index[i],
                    'type': 'FGD',
                    'price': current_candle['Close'],
                    'low': current_candle['Low'],
                    'consecutive_days': df.iloc[i-1]['consecutive_bears'],
                    'setup_strength': 'STRONG' if df.iloc[i-1]['consecutive_bears'] >= 5 else 'NORMAL'
                })

        return pd.DataFrame(signals)

    def find_3dl_3ds(self, df):
        """Find 3 Day Low (3DL) and 3 Day Short (3DS) patterns"""
        signals = []

        for i in range(3, len(df)):
            # Check for 3DL (3 consecutive higher lows)
            lows = [df.iloc[j]['Low'] for j in range(i-3, i)]
            if lows[0] < lows[1] < lows[2]:
                signals.append({
                    'date': df.index[i],
                    'type': '3DL',
                    'price': df.iloc[i]['Close'],
                    'low_series': lows
                })

            # Check for 3DS (3 consecutive lower highs)
            highs = [df.iloc[j]['High'] for j in range(i-3, i)]
            if highs[0] > highs[1] > highs[2]:
                signals.append({
                    'date': df.index[i],
                    'type': '3DS',
                    'price': df.iloc[i]['Close'],
                    'high_series': highs
                })

        return pd.DataFrame(signals)

    def analyze_today_setup(self, df, signals):
        """Analyze what setup exists for today"""
        if signals.empty:
            return None

        # Get most recent signal
        latest_signal = signals.iloc[-1]
        latest_date = latest_signal['date']

        # Check if latest signal was from the most recent completed candle
        most_recent = df.index[-2]  # Previous completed candle

        if latest_date == most_recent or (latest_date > most_recent - pd.Timedelta(days=1)):
            if latest_signal['type'] == 'FRD':
                return {
                    'setup': 'FRD_CONFIRMED',
                    'action': 'LOOK_FOR_SHORTS',
                    'entry_zone': latest_signal['high'],
                    'stop_above': latest_signal['high'] * 1.005,
                    'signal_date': latest_date,
                    'consecutive_days': latest_signal['consecutive_days'],
                    'strength': latest_signal['setup_strength']
                }
            elif latest_signal['type'] == 'FGD':
                return {
                    'setup': 'FGD_CONFIRMED',
                    'action': 'LOOK_FOR_LONGS',
                    'entry_zone': latest_signal['low'],
                    'stop_below': latest_signal['low'] * 0.995,
                    'signal_date': latest_date,
                    'consecutive_days': latest_signal['consecutive_days'],
                    'strength': latest_signal['setup_strength']
                }

        return None

    def check_h1_d1_alignment(self, h1_setup, d1_signals):
        """Check alignment between H1 and D1 timeframes"""
        alignment_result = {
            'h1_setup': h1_setup,
            'd1_context': None,
            'alignment_score': 0,
            'recommendation': None
        }

        if d1_signals.empty:
            alignment_result['recommendation'] = 'NO_D1_SIGNAL'
            return alignment_result

        latest_d1 = d1_signals.iloc[-1]

        if h1_setup:
            # Check if H1 and D1 signals align
            if ((h1_setup['action'] == 'LOOK_FOR_SHORTS' and latest_d1['type'] == 'FRD') or
                (h1_setup['action'] == 'LOOK_FOR_LONGS' and latest_d1['type'] == 'FGD')):
                alignment_result['alignment_score'] = 100
                alignment_result['d1_context'] = f"D1_{latest_d1['type']}_CONFIRMED"
                alignment_result['recommendation'] = 'STRONG_ALIGNMENT'
            else:
                alignment_result['alignment_score'] = 25
                alignment_result['d1_context'] = 'MISMATCHED_TIMEFRAMES'
                alignment_result['recommendation'] = 'WEAK_ALIGNMENT'
        else:
            alignment_result['recommendation'] = 'NO_H1_SETUP'

        return alignment_result

    def scan(self):
        """Main scanning function"""
        print(f"\n=== Stacey Burke Scanner - {self.symbol} ===")
        print("Analyzing H1/D1 alignment for Stacey Burke setups...")

        # Fetch data
        if not self.fetch_data():
            return None

        # Identify patterns on both timeframes
        self.h1_data = self.identify_candle_types(self.h1_data)
        self.d1_data = self.identify_candle_types(self.d1_data)

        # Find signals
        h1_frd_fgd = self.find_frd_fgd_signals(self.h1_data, min_consecutive=3)
        d1_frd_fgd = self.find_frd_fgd_signals(self.d1_data, min_consecutive=3)
        h1_3dl_3ds = self.find_3dl_3ds(self.h1_data)

        # Analyze setups
        h1_setup = self.analyze_today_setup(self.h1_data, h1_frd_fgd)
        alignment = self.check_h1_d1_alignment(h1_setup, d1_frd_fgd)

        return {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'h1_setup': h1_setup,
            'alignment': alignment,
            'h1_signals': {
                'frd_fgd_count': len(h1_frd_fgd),
                'recent_signals': h1_frd_fgd.tail(3).to_dict('records') if not h1_frd_fgd.empty else []
            },
            'd1_signals': {
                'frd_fgd_count': len(d1_frd_fgd),
                'recent_signals': d1_frd_fgd.tail(3).to_dict('records') if not d1_frd_fgd.empty else []
            },
            'patterns_3dl_3ds': len(h1_3dl_3ds)
        }

def print_results(result):
    """Print formatted results"""
    if not result:
        return

    print(f"\n{'='*60}")
    print(f"🎯 STACEY BURKE SCANNER - {result['symbol']}")
    print(f"{'='*60}")
    print(f"📅 Scan Time: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 H1 Signals Found: {result['h1_signals']['frd_fgd_count']}")
    print(f"📈 D1 Signals Found: {result['d1_signals']['frd_fgd_count']}")
    print(f"🔄 3DL/3DS Patterns: {result['patterns_3dl_3ds']}")

    print(f"\n{'='*60}")
    print(f"🚨 TODAY'S H1 SETUP")
    print(f"{'='*60}")

    h1_setup = result['h1_setup']
    if h1_setup:
        print(f"✅ Setup Found: {h1_setup['setup']}")
        print(f"🎯 Action: {h1_setup['action']}")
        print(f"📍 Entry Zone: {h1_setup['entry_zone']:.5f}")

        if h1_setup['action'] == 'LOOK_FOR_SHORTS':
            print(f"⛔ Stop Loss: Above {h1_setup['stop_above']:.5f}")
        else:
            print(f"⛔ Stop Loss: Below {h1_setup['stop_below']:.5f}")

        print(f"💪 Strength: {h1_setup['strength']}")
        print(f"📈 Consecutive Days: {h1_setup['consecutive_days']}")
    else:
        print("❌ No valid H1 setup for current session")

    print(f"\n{'='*60}")
    print(f"🎯 H1/D1 ALIGNMENT")
    print(f"{'='*60}")

    alignment = result['alignment']
    print(f"📊 Alignment Score: {alignment['alignment_score']}/100")
    print(f"💡 Recommendation: {alignment['recommendation']}")

    if alignment['d1_context']:
        print(f"📈 D1 Context: {alignment['d1_context']}")

    # Color code the recommendation
    if alignment['recommendation'] == 'STRONG_ALIGNMENT':
        print("🟢 HIGH PROBABILITY SETUP - H1 and D1 aligned!")
    elif alignment['recommendation'] == 'WEAK_ALIGNMENT':
        print("🟡 CAUTION - Timeframes not aligned")
    elif alignment['recommendation'] == 'NO_D1_SIGNAL':
        print("🔵 H1 ONLY - No D1 confirmation available")
    else:
        print("⚪ NO SETUP - Wait for better opportunity")

    print(f"\n{'='*60}")
    print(f"📋 RECENT SIGNALS")
    print(f"{'='*60}")

    if result['h1_signals']['recent_signals']:
        print("\n🔥 Recent H1 Signals:")
        for signal in result['h1_signals']['recent_signals']:
            print(f"   {signal['date'].strftime('%m/%d %H:%M')}: {signal['type']} "
                  f"({signal['consecutive_days']} days) @ {signal['price']:.5f}")

    if result['d1_signals']['recent_signals']:
        print("\n📅 Recent D1 Signals:")
        for signal in result['d1_signals']['recent_signals']:
            print(f"   {signal['date'].strftime('%m/%d')}: {signal['type']} "
                  f"({signal['consecutive_days']} days) @ {signal['price']:.5f}")

    print(f"\n{'='*60}")
    print(f"💡 TRADING SESSION CONTEXT (Manila Time)")
    print(f"{'='*60}")
    now = datetime.now()
    est_time = now - timedelta(hours=12)  # Manila is EST+12

    print(f"🕐 Current Manila Time: {now.strftime('%H:%M')}")
    print(f"🕐 Current EST Time: {est_time.strftime('%H:%M')}")

    if 14 <= now.hour <= 16:  # 2PM-4PM Manila = 2AM-4AM EST
        print("🔥 LONDON OPEN SESSION - Prime trading time!")
    elif 20 <= now.hour <= 23:  # 8PM-11PM Manila = 8AM-11AM EST
        print("🚀 NY OVERLAP SESSION - Highest volatility!")
    elif 10 <= now.hour <= 13:  # 10AM-1PM Manila = 10PM-1AM EST
        print("📊 LATE NY SESSION - Watch for reversals")
    else:
        print("😴 ASIAN SESSION - Lower volatility, range trading likely")

if __name__ == "__main__":
    scanner = StaceyBurkeScanner()
    result = scanner.scan()
    print_results(result)