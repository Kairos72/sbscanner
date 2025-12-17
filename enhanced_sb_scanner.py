"""
ENHANCED STACEY BURKE SCANNER
============================

Based on comprehensive research of Stacey Burke's strategy,
this enhanced scanner includes all major patterns and concepts.

Key Features:
- FRD/FGD detection
- Inside Day patterns
- 3DL/3DS identification
- Multi-timeframe analysis
- Proper entry zones
- Risk management calculations
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class EnhancedStaceyBurkeScanner:
    """
    Complete implementation of Stacey Burke's trading strategy
    """

    def __init__(self):
        self.account = 2428164
        self.password = "3b6%StTrQj"
        self.server = "ACGMarkets-Main"

        # Major forex pairs
        self.major_pairs = [
            "EURUSD.pro", "GBPUSD.pro", "AUDUSD.pro",
            "USDJPY.pro", "NZDUSD.pro", "USDCAD.pro", "USDCHF.pro"
        ]

        # Store analysis results
        self.daily_signals = {}
        self.h1_signals = {}
        self.inside_days = {}
        self.weekly_levels = {}

        print("Enhanced Stacey Burke Scanner Initialized")
        print("=" * 50)

    def connect_to_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return False

        # Login to account
        authorized = mt5.login(
            login=self.account,
            password=self.password,
            server=self.server
        )

        if authorized:
            account_info = mt5.account_info()
            print(f"Connected to account: {account_info.login}")
            print(f"Server: {account_info.server}")
            return True
        else:
            print("Login failed")
            return False

    def get_data(self, symbol: str, timeframe, count: int = 200) -> Optional[pd.DataFrame]:
        """Get price data for a symbol"""
        if not mt5.symbol_select(symbol, True):
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def detect_daily_frd_fgd(self, daily_df: pd.DataFrame, min_consecutive: int = 3) -> List[Dict]:
        """
        Detect FRD (First Red DAY) and FGD (First Green DAY) patterns on DAILY timeframe

        CRITICAL RULE: FGD/FRD only trigger AFTER daily candle is fully formed!
        - Do NOT analyze today's candle while it's still forming
        - Only look at COMPLETED daily candles

        Rules:
        - FRD: First RED daily candle after 3+ consecutive GREEN daily candles
        - FGD: First GREEN daily candle after 3+ consecutive RED daily candles
        """
        signals = []

        if len(daily_df) < min_consecutive + 1:
            return signals

        # IMPORTANT: Skip today's candle if it's still forming
        # Only analyze completed candles (exclude the last one if current day)
        analysis_df = daily_df.copy()
        current_time = pd.Timestamp.now()

        # Check if last candle is for today (still forming)
        if not analysis_df.empty:
            last_candle_time = analysis_df.index[-1]
            if last_candle_time.date() == current_time.date():
                # Skip today's forming candle
                analysis_df = analysis_df.iloc[:-1]

        if len(analysis_df) < min_consecutive + 1:
            return signals

        # Calculate consecutive runs on COMPLETED DAILY candles only
        consecutive_greens = 0
        consecutive_reds = 0
        daily_signals = []

        # Track consecutive daily candles
        for i in range(len(analysis_df)):
            if analysis_df.iloc[i]['close'] > analysis_df.iloc[i]['open']:  # Green daily candle
                consecutive_greens += 1
                consecutive_reds = 0

                # Check if this is FGD (after 3+ reds)
                if consecutive_reds >= min_consecutive:
                    daily_signals.append({
                        'type': 'FGD',
                        'time': analysis_df.index[i],
                        'price': analysis_df.iloc[i]['close'],
                        'high': analysis_df.iloc[i]['high'],
                        'low': analysis_df.iloc[i]['low'],
                        'consecutive_reds': consecutive_reds,
                        'entry_zone': analysis_df.iloc[i]['low'],  # Enter below FGD low
                        'action': 'LONG',
                        'stop_level': analysis_df.iloc[i]['low'] * 0.998,  # 0.2% buffer below
                        'target': analysis_df.iloc[i-min_consecutive]['high'],  # Previous high
                        'is_completed': True  # This is a completed signal
                    })
            else:  # Red daily candle
                consecutive_reds += 1
                consecutive_greens = 0

                # Check if this is FRD (after 3+ greens)
                if consecutive_greens >= min_consecutive:
                    daily_signals.append({
                        'type': 'FRD',
                        'time': analysis_df.index[i],
                        'price': analysis_df.iloc[i]['close'],
                        'high': analysis_df.iloc[i]['high'],
                        'low': analysis_df.iloc[i]['low'],
                        'consecutive_greens': consecutive_greens,
                        'entry_zone': analysis_df.iloc[i]['high'],  # Enter above FRD high
                        'action': 'SHORT',
                        'stop_level': analysis_df.iloc[i]['high'] * 1.002,  # 0.2% buffer above
                        'target': analysis_df.iloc[i-min_consecutive]['low'],  # Previous low
                        'is_completed': True  # This is a completed signal
                    })

        return daily_signals

    def detect_inside_day(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Inside Day patterns

        Rules:
        - High < previous day's high
        - Low > previous day's low
        - Indicates consolidation before breakout
        """
        inside_days = []

        for i in range(1, len(df)):
            prev_high = df.iloc[i-1]['high']
            prev_low = df.iloc[i-1]['low']
            curr_high = df.iloc[i]['high']
            curr_low = df.iloc[i]['low']

            if curr_high < prev_high and curr_low > prev_low:
                inside_days.append({
                    'time': df.index[i],
                    'high': curr_high,
                    'low': curr_low,
                    'prev_high': prev_high,
                    'prev_low': prev_low,
                    'range_size': prev_high - prev_low,
                    'action': 'WAIT_FOR_BREAKOUT'
                })

        return inside_days

    def detect_3dl_3ds(self, df: pd.DataFrame, min_days: int = 3) -> List[Dict]:
        """
        Detect 3DL (Three Days of Longs) and 3DS (Three Days of Shorts)

        Rules:
        - 3DL: 3+ consecutive green daily candles (overextended longs)
        - 3DS: 3+ consecutive red daily candles (overextended shorts)
        """
        patterns = []

        # Track consecutive candles
        consecutive_greens = 0
        consecutive_reds = 0

        for i in range(len(df)):
            if df.iloc[i]['close'] > df.iloc[i]['open']:  # Green
                consecutive_greens += 1
                consecutive_reds = 0

                if consecutive_greens >= min_days:
                    patterns.append({
                        'type': '3DL',
                        'time': df.index[i],
                        'consecutive': consecutive_greens,
                        'action': 'LOOK_FOR_SHORTS',
                        'entry_zone': df.iloc[i]['low'],
                        'current_price': df.iloc[i]['close'],
                        'stop_level': df.iloc[i]['high'] * 1.002
                    })
            else:  # Red
                consecutive_reds += 1
                consecutive_greens = 0

                if consecutive_reds >= min_days:
                    patterns.append({
                        'type': '3DS',
                        'time': df.index[i],
                        'consecutive': consecutive_reds,
                        'action': 'LOOK_FOR_LONGS',
                        'entry_zone': df.iloc[i]['high'],
                        'current_price': df.iloc[i]['close'],
                        'stop_level': df.iloc[i]['low'] * 0.998
                    })

        return patterns

    def get_weekly_levels(self, symbol: str) -> Dict:
        """Get key weekly levels for reference"""
        weekly_df = self.get_data(symbol, mt5.TIMEFRAME_W1, 10)
        if weekly_df is None:
            return {}

        latest_week = weekly_df.iloc[-1]

        return {
            'weekly_high': latest_week['high'],
            'weekly_low': latest_week['low'],
            'weekly_open': latest_week['open'],
            'weekly_close': latest_week['close'],
            'weekly_mid': (latest_week['high'] + latest_week['low']) / 2
        }

    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Complete analysis focusing on DAILY timeframe as primary signal generator
        """
        analysis = {
            'symbol': symbol,
            'current_price': 0,
            'daily_analysis': {},
            'h1_analysis': {},
            'weekly_levels': {},
            'recommendations': []
        }

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            analysis['current_price'] = tick.ask

        # Daily timeframe analysis - PRIMARY SOURCE OF SIGNALS
        daily_df = self.get_data(symbol, mt5.TIMEFRAME_D1, 30)
        if daily_df is not None:
            # Get DAILY FRD/FGD signals (the only ones that matter!)
            daily_signals = self.detect_daily_frd_fgd(daily_df)
            daily_3dl_3ds = self.detect_3dl_3ds(daily_df)
            daily_inside = self.detect_inside_day(daily_df)

            analysis['daily_analysis'] = {
                'frd_fgd_signals': daily_signals,  # All daily signals
                'three_day_patterns': daily_3dl_3ds,  # 3DL/3DS patterns
                'inside_days': daily_inside,
                'trend': 'BULLISH' if daily_df.iloc[-1]['close'] > daily_df.iloc[-1]['open'] else 'BEARISH',
                'consecutive_greens': 0,
                'consecutive_reds': 0
            }

            # Count consecutive daily candles
            for i in range(len(daily_df)):
                if daily_df.iloc[i]['close'] > daily_df.iloc[i]['open']:
                    analysis['daily_analysis']['consecutive_greens'] += 1
                    analysis['daily_analysis']['consecutive_reds'] = 0
                else:
                    analysis['daily_analysis']['consecutive_reds'] += 1
                    analysis['daily_analysis']['consecutive_greens'] = 0

        # H1 timeframe analysis - ONLY FOR ENTRY TIMING, not signals
        h1_df = self.get_data(symbol, mt5.TIMEFRAME_H1, 24)
        if h1_df is not None:
            # H1 is used only to time entries after DAILY signals
            h1_green_count = len(h1_df[h1_df['close'] > h1_df['open']])
            h1_red_count = len(h1_df) - h1_green_count

            analysis['h1_analysis'] = {
                'last_24h_trend': 'BULLISH' if h1_df.iloc[-1]['close'] > h1_df.iloc[0]['close'] else 'BEARISH',
                'green_candles_24h': int(h1_green_count),
                'red_candles_24h': int(h1_red_count),
                'current_h1_close': h1_df.iloc[-1]['close'],
                'current_h1_open': h1_df.iloc[-1]['open']
            }

        # Weekly levels for context
        analysis['weekly_levels'] = self.get_weekly_levels(symbol)

        # Generate recommendations based on DAILY signals
        analysis['recommendations'] = self.generate_daily_recommendations(analysis)

        return analysis

    def generate_daily_recommendations(self, analysis: Dict) -> List[Dict]:
        """
        Generate recommendations based ONLY on DAILY signals
        H1 is used only for entry timing, NOT for signal generation
        """
        recommendations = []
        current_price = analysis['current_price']

        # Check DAILY FRD/FGD signals (the only valid signals!)
        daily_signals = analysis.get('daily_analysis', {}).get('frd_fgd_signals', [])
        daily_3dl_3ds = analysis.get('daily_analysis', {}).get('three_day_patterns', [])

        # PRIORITY 1: Check for recent DAILY FRD/FGD signals
        if daily_signals:
            latest_signal = daily_signals[-1]
            signal_date = latest_signal['time']

            # Check if signal is from this week (still valid)
            days_since_signal = (datetime.now() - signal_date).days

            if days_since_signal <= 7:  # Signal still valid this week
                if latest_signal['type'] == 'FRD':
                    if current_price > latest_signal['entry_zone']:
                        recommendations.append({
                            'type': 'SHORT_ENTRY',
                            'reason': f"DAILY FRD activated - price above {latest_signal['entry_zone']:.5f}",
                            'entry': current_price,
                            'stop': latest_signal['stop_level'],
                            'target': latest_signal['target'],
                            'confidence': 'HIGH'
                        })
                    else:
                        recommendations.append({
                            'type': 'WATCH_SHORT',
                            'reason': f"DAILY FRD waiting - price needs above {latest_signal['entry_zone']:.5f}",
                            'entry_zone': latest_signal['entry_zone'],
                            'confidence': 'MEDIUM'
                        })

                elif latest_signal['type'] == 'FGD':
                    if current_price < latest_signal['entry_zone']:
                        recommendations.append({
                            'type': 'LONG_ENTRY',
                            'reason': f"DAILY FGD activated - price below {latest_signal['entry_zone']:.5f}",
                            'entry': current_price,
                            'stop': latest_signal['stop_level'],
                            'target': latest_signal['target'],
                            'confidence': 'HIGH'
                        })
                    else:
                        recommendations.append({
                            'type': 'WATCH_LONG',
                            'reason': f"DAILY FGD waiting - price needs below {latest_signal['entry_zone']:.5f}",
                            'entry_zone': latest_signal['entry_zone'],
                            'confidence': 'MEDIUM'
                        })

        # PRIORITY 2: Check for 3DL/3DS patterns
        if daily_3dl_3ds:
            latest_pattern = daily_3dl_3ds[-1]

            if latest_pattern['type'] == '3DS' and latest_pattern['consecutive'] >= 3:
                recommendations.append({
                    'type': 'CAUTION_BEARISH',
                    'reason': f"{latest_pattern['consecutive']} consecutive red days - oversold, watch for reversal",
                    'current_daily_streak': latest_pattern['consecutive'],
                    'confidence': 'HIGH'
                })
            elif latest_pattern['type'] == '3DL' and latest_pattern['consecutive'] >= 3:
                recommendations.append({
                    'type': 'CAUTION_BULLISH',
                    'reason': f"{latest_pattern['consecutive']} consecutive green days - overbought, watch for reversal",
                    'current_daily_streak': latest_pattern['consecutive'],
                    'confidence': 'HIGH'
                })

        # If no valid signals
        if not recommendations:
            daily_trend = analysis.get('daily_analysis', {}).get('trend', 'UNKNOWN')
            recommendations.append({
                'type': 'NO_SIGNAL',
                'reason': f"No valid DAILY FRD/FGD signals. Daily trend: {daily_trend}",
                'confidence': 'LOW'
            })

        return recommendations

    def scan_all_pairs(self):
        """Scan all major pairs and generate comprehensive report"""
        print("\nSCANNING ALL MAJOR PAIRS")
        print("=" * 60)

        all_results = {}

        for symbol in self.major_pairs:
            print(f"\nAnalyzing {symbol}...")
            result = self.analyze_symbol(symbol)
            all_results[symbol] = result

            # Display summary
            print(f"  Current Price: {result['current_price']:.5f}")

            # Show daily context
            daily = result.get('daily_analysis', {})
            if daily:
                print(f"  Daily Trend: {daily.get('trend', 'N/A')}")
                print(f"  Daily Streak: {daily.get('consecutive_greens', 0)} greens / {daily.get('consecutive_reds', 0)} reds")

            # Show active recommendations
            if result['recommendations']:
                for rec in result['recommendations']:
                    if 'ENTRY' in rec['type']:
                        print(f"  [{rec['type']}] {rec['confidence']}: {rec['reason']}")
                        if rec.get('stop'):
                            print(f"     Stop: {rec['stop']:.5f}")
                        if rec.get('target'):
                            print(f"     Target: {rec['target']:.5f}")
                    elif 'WATCH' in rec['type']:
                        print(f"  [{rec['type']}] {rec['confidence']}: {rec['reason']}")
                        print(f"     Note: Today's candle is still forming - no new signals")
                    else:
                        print(f"  [{rec['type']}] {rec['confidence']}: {rec['reason']}")
                        if rec.get('current_daily_streak'):
                            print(f"     Streak: {rec['current_daily_streak']} days")
            else:
                print("  [INFO] No valid DAILY signals")

        # Display summary table
        print("\n" + "=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)

        print(f"{'Pair':<12} {'Price':<10} {'Daily':<8} {'Daily Streak':<12} {'Recommendation':<20}")
        print("-" * 60)

        for symbol, result in all_results.items():
            daily = result.get('daily_analysis', {})
            daily_trend = daily.get('trend', 'N/A')

            # Get daily streak
            greens = daily.get('consecutive_greens', 0)
            reds = daily.get('consecutive_reds', 0)
            streak = f"{greens}G/{reds}R"

            # Get latest daily signal
            daily_signal = 'NONE'
            daily_signals = daily.get('frd_fgd_signals', [])
            if daily_signals:
                latest = daily_signals[-1]
                days_old = (datetime.now() - latest['time']).days
                if days_old <= 7:
                    daily_signal = f"{latest['type']}({days_old}d)"

            # Get main recommendation
            main_rec = 'HOLD'
            for rec in result.get('recommendations', []):
                if 'ENTRY' in rec['type'] or 'CAUTION' in rec['type']:
                    main_rec = rec['type']
                    break

            print(f"{symbol:<12} {result['current_price']:<10.5f} {daily_trend:<8} {streak:<12} {main_rec:<20}")

        return all_results

    def run(self):
        """Main execution method"""
        try:
            # Connect to MT5
            if not self.connect_to_mt5():
                return

            # Run comprehensive scan
            results = self.scan_all_pairs()

            # Save results to file for reference
            self.save_results(results)

            print("\n" + "=" * 60)
            print("SCAN COMPLETE")
            print("=" * 60)
            print("Results saved to 'scan_results.txt'")
            print("\nKey Remember:")
            print("1. Wait for price to reach entry zones")
            print("2. Always use proper stop losses")
            print("3. Risk only 1% per trade")
            print("4. Higher timeframes control lower timeframes")

        finally:
            # Always disconnect
            mt5.shutdown()
            print("\nMT5 connection closed")

    def save_results(self, results: Dict):
        """Save scan results to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open('scan_results.txt', 'w') as f:
            f.write(f"ENHANCED STACEY BURKE SCANNER RESULTS\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

            for symbol, result in results.items():
                f.write(f"{symbol}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Current Price: {result['current_price']:.5f}\n\n")

                # Daily analysis
                daily = result.get('daily_analysis', {})
                if daily:
                    f.write("DAILY ANALYSIS:\n")
                    f.write(f"  Trend: {daily.get('trend', 'N/A')}\n")

                    if daily.get('frd_fgd_signals'):
                        f.write("  Recent Signals:\n")
                        for sig in daily['frd_fgd_signals']:
                            f.write(f"    - {sig['type']} at {sig['price']:.5f} on {sig['time']}\n")

                    if daily.get('three_day_patterns'):
                        f.write("  3DL/3DS Patterns:\n")
                        for pat in daily['three_day_patterns']:
                            f.write(f"    - {pat['type']} ({pat['consecutive']} days)\n")

                # H1 analysis
                h1 = result.get('h1_analysis', {})
                if h1:
                    f.write("\nH1 ANALYSIS:\n")
                    f.write(f"  24H Trend: {h1.get('last_24h_trend', 'N/A')}\n")

                    if h1.get('frd_fgd_signals'):
                        f.write("  Recent Signals:\n")
                        for sig in h1['frd_fgd_signals']:
                            f.write(f"    - {sig['type']} at {sig['price']:.5f} on {sig['time']}\n")

                # Recommendations
                if result.get('recommendations'):
                    f.write("\nRECOMMENDATIONS:\n")
                    for rec in result['recommendations']:
                        f.write(f"  - {rec['type']}: {rec['reason']}\n")

                f.write("\n" + "=" * 60 + "\n\n")

if __name__ == "__main__":
    scanner = EnhancedStaceyBurkeScanner()
    scanner.run()