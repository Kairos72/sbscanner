import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests

class StaceyBurkeScanner:
    def __init__(self, symbol="EURUSD=X"):
        self.symbol = symbol
        self.data = None

    def fetch_data(self, period="1mo", interval="1h"):
        """Fetch forex data from Yahoo Finance"""
        try:
            # Try different ticker formats for EUR/USD
            tickers_to_try = ["EURUSD=X", "EUR%3FUSD=X", "EURUSD", "EUR/USD=X"]

            for ticker in tickers_to_try:
                try:
                    print(f"Trying ticker: {ticker}")
                    data = yf.download(ticker, period=period, interval=interval, progress=False)

                    if not data.empty:
                        self.data = data
                        print(f"Successfully fetched {len(self.data)} candles using {ticker}")
                        return True
                except Exception as e:
                    print(f"Failed with {ticker}: {e}")
                    continue

            # If all else fails, create synthetic data for testing
            print("Using synthetic data for demonstration...")
            self.data = self.create_synthetic_data()
            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def create_synthetic_data(self):
        """Create synthetic EUR/USD data for testing"""
        # Create date range for last 30 days
        dates = pd.date_range(end=datetime.now(), periods=720, freq='H')  # 30 days of hourly data

        # Start with realistic EUR/USD price
        base_price = 1.0850
        price_changes = np.random.normal(0, 0.002, len(dates))  # Random price changes

        # Create price series
        close_prices = []
        current_price = base_price

        for change in price_changes:
            current_price += change
            # Keep price within realistic bounds
            current_price = max(1.0500, min(1.1500, current_price))
            close_prices.append(current_price)

        # Create OHLC data
        data = pd.DataFrame(index=dates)
        data['Close'] = close_prices

        # Generate Open, High, Low from Close
        data['Open'] = data['Close'].shift(1).fillna(base_price)

        # Add intraday volatility
        high_volatility = np.random.uniform(0.001, 0.003, len(dates))
        low_volatility = np.random.uniform(0.001, 0.003, len(dates))

        data['High'] = np.maximum(data['Open'], data['Close']) + high_volatility
        data['Low'] = np.minimum(data['Open'], data['Close']) - low_volatility
        data['Volume'] = np.random.randint(1000, 10000, len(dates))

        return data.dropna()

    def identify_candle_type(self, df):
        """Identify candle types (bullish/bearish/doji)"""
        df['is_bullish'] = df['Close'] > df['Open']
        df['is_bearish'] = df['Close'] < df['Open']
        df['is_inside'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        df['body_size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) * 100

        return df

    def find_frd_fgd(self, df, min_consecutive_days=3):
        """
        Find First Red Day (FRD) and First Green Day (FGD) patterns
        Returns: DataFrame with FRD/FGD signals
        """
        signals = []

        for i in range(min_consecutive_days, len(df)):
            # Check for FRD (First Red Day)
            if df.iloc[i]['is_bearish']:  # Current day is red
                # Check if previous min_consecutive_days were green
                consecutive_greens = 0
                for j in range(i-min_consecutive_days, i):
                    if df.iloc[j]['is_bullish']:
                        consecutive_greens += 1

                if consecutive_greens >= min_consecutive_days:
                    signals.append({
                        'date': df.index[i],
                        'type': 'FRD',
                        'price': df.iloc[i]['Close'],
                        'consecutive_days': consecutive_greens
                    })

            # Check for FGD (First Green Day)
            if df.iloc[i]['is_bullish']:  # Current day is green
                # Check if previous min_consecutive_days were red
                consecutive_reds = 0
                for j in range(i-min_consecutive_days, i):
                    if df.iloc[j]['is_bearish']:
                        consecutive_reds += 1

                if consecutive_reds >= min_consecutive_days:
                    signals.append({
                        'date': df.index[i],
                        'type': 'FGD',
                        'price': df.iloc[i]['Close'],
                        'consecutive_days': consecutive_reds
                    })

        return pd.DataFrame(signals)

    def analyze_today_setup(self, df, signals_df):
        """
        Analyze what setup exists for today based on yesterday's signals
        Returns: Setup recommendation for today
        """
        if df.empty or signals_df.empty:
            return None

        # Get the most recent signal
        latest_signal = signals_df.iloc[-1]
        latest_signal_date = latest_signal['date']

        # Check if latest signal was from yesterday
        yesterday = df.index[-2]  # Previous day
        today = df.index[-1]      # Current day

        if latest_signal_date == yesterday:
            if latest_signal['type'] == 'FRD':
                return {
                    'setup': 'FRD_CONFIRMED',
                    'action': 'LOOK_FOR_SHORTS',
                    'entry_zone': df.loc[yesterday, 'High'],
                    'stop_above': df.loc[yesterday, 'High'] * 1.005,  # 0.5% above high
                    'signal_date': latest_signal_date,
                    'consecutive_days': latest_signal['consecutive_days']
                }
            elif latest_signal['type'] == 'FGD':
                return {
                    'setup': 'FGD_CONFIRMED',
                    'action': 'LOOK_FOR_LONGS',
                    'entry_zone': df.loc[yesterday, 'Low'],
                    'stop_below': df.loc[yesterday, 'Low'] * 0.995,  # 0.5% below low
                    'signal_date': latest_signal_date,
                    'consecutive_days': latest_signal['consecutive_days']
                }

        return None

    def check_alignment(self, h1_setup, daily_signals, daily_data):
        """
        Check H1 vs D1 alignment
        """
        alignment_result = {
            'h1_setup': h1_setup,
            'daily_context': None,
            'alignment_score': 0
        }

        if not daily_signals.empty and h1_setup:
            # Get latest daily signal
            latest_daily = daily_signals.iloc[-1]

            # Check if H1 and D1 setups align
            if (h1_setup['action'] == 'LOOK_FOR_SHORTS' and
                latest_daily['type'] == 'FRD'):
                alignment_result['alignment_score'] = 100
                alignment_result['daily_context'] = 'BOTH_H1_AND_D1_SHOWING_FRD'

            elif (h1_setup['action'] == 'LOOK_FOR_LONGS' and
                  latest_daily['type'] == 'FGD'):
                alignment_result['alignment_score'] = 100
                alignment_result['daily_context'] = 'BOTH_H1_AND_D1_SHOWING_FGD'
            else:
                alignment_result['alignment_score'] = 50
                alignment_result['daily_context'] = 'MISMATCHED_H1_D1_SETUPS'

        return alignment_result

    def scan(self):
        """Main scanning function"""
        print(f"\n=== Stacey Burke Scanner - {self.symbol} ===")

        # Fetch data
        if not self.fetch_data():
            return None

        # Identify candle patterns
        self.data = self.identify_candle_type(self.data)

        # Find FRD/FGD patterns
        signals = self.find_frd_fgd(self.data)

        # Analyze today's setup
        today_setup = self.analyze_today_setup(self.data, signals)

        return {
            'symbol': self.symbol,
            'last_update': datetime.now(),
            'data_points': len(self.data),
            'signals_found': len(signals),
            'today_setup': today_setup,
            'recent_signals': signals.tail(5).to_dict('records') if not signals.empty else []
        }

def main():
    scanner = StaceyBurkeScanner()
    result = scanner.scan()

    if result:
        print(f"\n📊 SCANNER RESULTS FOR {result['symbol']}")
        print(f"📅 Last Update: {result['last_update']}")
        print(f"📈 Data Points Analyzed: {result['data_points']}")
        print(f"🎯 Total Signals Found: {result['signals_found']}")

        if result['today_setup']:
            setup = result['today_setup']
            print(f"\n🚨 TODAY'S SETUP:")
            print(f"   Setup Type: {setup['setup']}")
            print(f"   Action: {setup['action']}")
            print(f"   Entry Zone: {setup['entry_zone']:.5f}")

            if setup['action'] == 'LOOK_FOR_SHORTS':
                print(f"   Stop Above: {setup['stop_above']:.5f}")
            else:
                print(f"   Stop Below: {setup['stop_below']:.5f}")

            print(f"   Consecutive Days: {setup['consecutive_days']}")
        else:
            print(f"\n✅ No valid Stacey Burke setup for today")

        if result['recent_signals']:
            print(f"\n📋 RECENT SIGNALS:")
            for signal in result['recent_signals']:
                print(f"   {signal['date'].strftime('%Y-%m-%d')}: {signal['type']} "
                      f"({signal['consecutive_days']} consecutive days) @ {signal['price']:.5f}")

if __name__ == "__main__":
    main()