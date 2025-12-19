"""
ENHANCED STACEY BURKE SCANNER WITH ASIAN RANGE SWEEP - OANDA VERSION
==================================================================

Modified to connect to OANDA MT5 broker
Now uses proper ACB framework for accurate FGD/FRD detection
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from oanda_symbol_mapper import get_oanda_symbol

# Import proper ACB detectors for accurate pattern recognition
from acb.patterns.frd_fgd import EnhancedFRDFGDDetector
from acb.patterns.enhanced_frd_fgd import AsianRangeEntryDetector

class EnhancedStaceyBurkeScannerWithAsianOANDA:
    """
    Complete implementation with Asian Range Sweep for OANDA
    """

    def __init__(self):
        # OANDA broker - use existing connection
        self.use_oanda = True

        # Major forex pairs
        self.major_pairs = [
            "EURUSD", "GBPUSD", "AUDUSD",
            "USDJPY", "NZDUSD", "USDCAD", "USDCHF"
        ]

        # Initialize proper ACB detectors
        self.fgd_detector = EnhancedFRDFGDDetector()
        self.asian_detector = AsianRangeEntryDetector()

        print("Enhanced Stacey Burke Scanner with Asian Sweep - OANDA Version")
        print("=" * 70)
        print("Now using enhanced ACB framework for accurate pattern detection")
        print("=" * 70)

    def connect_to_mt5(self) -> bool:
        """Connect to existing MT5 connection"""
        if not mt5.initialize():
            print("Failed to initialize MT5")
            print("Please ensure MT5 terminal is running and logged into OANDA")
            return False

        # Get account info from existing connection
        account_info = mt5.account_info()
        if account_info is None:
            print("No active MT5 connection found")
            print("Please ensure MT5 is running and connected to OANDA")
            mt5.shutdown()
            return False

        print(f"Connected to account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Broker: {account_info.company}")
        return True

    def get_data(self, symbol: str, timeframe, count: int) -> Optional[pd.DataFrame]:
        """Get candlestick data"""
        # Convert to OANDA symbol if needed
        oanda_symbol = get_oanda_symbol(symbol)

        rates = mt5.copy_rates_from_pos(oanda_symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def detect_daily_frd_fgd(self, df: pd.DataFrame) -> List[Dict]:
        """Detect FRD/FGD patterns using proper ACB EnhancedFRDFGDDetector"""
        try:
            # Use the proven ACB detector for accurate pattern recognition
            pattern_result = self.fgd_detector.detect_enhanced_frd_fgd(
                df, dmr_levels={}, acb_levels={}, session_analysis=None
            )

            if not pattern_result.get('pattern_detected', False):
                return []

            # Convert to the expected format for the scanner
            signals = []
            current_time = datetime.now()

            if pattern_result.get('signal_type') and pattern_result.get('trigger_day'):
                signal_time = pattern_result['trigger_day']
                if hasattr(signal_time, 'date'):
                    days_ago = (current_time.date() - signal_time.date()).days
                else:
                    days_ago = 0

                # Map signal type to expected format
                signal_type = pattern_result['signal_type'].value if hasattr(pattern_result['signal_type'], 'value') else str(pattern_result['signal_type'])
                direction = 'LONG' if signal_type == 'FGD' else 'SHORT'

                signals.append({
                    'type': signal_type,
                    'time': signal_time,
                    'price': pattern_result.get('current_price', 0),
                    'consecutive_reds': pattern_result.get('consecutive_count', 0),
                    'consecutive_greens': pattern_result.get('consecutive_count', 0),
                    'action': direction,
                    'days_ago': days_ago,
                    'grade': pattern_result.get('signal_grade'),
                    'confidence': pattern_result.get('confidence', 0),
                    'trade_today': pattern_result.get('trade_today', False)
                })

            return signals

        except Exception as e:
            print(f"  [DEBUG] Error in FGD/FRD detection: {e}")
            return []

    def detect_asian_range_sweep(self, symbol: str) -> Dict:
        """Detect Asian Range Sweep pattern"""
        try:
            oanda_symbol = get_oanda_symbol(symbol)

            # Get H1 data
            h1_df = self.get_data(oanda_symbol, mt5.TIMEFRAME_H1, 72)
            if h1_df is None or len(h1_df) == 0:
                return {'found': False, 'reason': 'No H1 data'}

            today = pd.Timestamp.now().date()
            today_candles = h1_df[h1_df.index.date == today]

            if len(today_candles) == 0:
                return {'found': False, 'reason': 'No candles for today'}

            # Asian session candles (00:00-06:00)
            asian_today = today_candles[
                (today_candles.index.hour >= 0) & (today_candles.index.hour <= 6)
            ]

            if len(asian_today) == 0:
                return {'found': False, 'reason': 'No Asian session data'}

            asian_high = asian_today['high'].max()
            asian_low = asian_today['low'].min()
            asian_range = (asian_high - asian_low) * 10000

            # Post-Asian session
            post_asian = today_candles[today_candles.index.hour > 6]

            sweep_info = {
                'found': False,
                'asian_high': asian_high,
                'asian_low': asian_low,
                'asian_range': asian_range,
                'current_price': today_candles.iloc[-1]['close'],
                'swept_low': False,
                'swept_high': False,
                'rejection': None
            }

            if len(post_asian) > 0:
                post_asian_low = post_asian['low'].min()
                post_asian_high = post_asian['high'].max()

                # Check Asian low sweep
                if post_asian_low < asian_low:
                    sweep_info['swept_low'] = True
                    sweep_info['sweep_low_pips'] = (asian_low - post_asian_low) * 10000

                    # Check rejection
                    sweep_candle = post_asian[post_asian['low'] == post_asian_low].iloc[0]
                    if sweep_candle['close'] > sweep_candle['low']:
                        sweep_info['rejection'] = {
                            'type': 'bullish',
                            'entry': sweep_candle['close'],
                            'stop': post_asian_low,
                            'rejection_pips': (sweep_candle['close'] - sweep_candle['low']) * 10000
                        }
                        sweep_info['found'] = True

                # Check Asian high sweep
                if post_asian_high > asian_high:
                    sweep_info['swept_high'] = True
                    sweep_info['sweep_high_pips'] = (post_asian_high - asian_high) * 10000

                    # Check rejection
                    sweep_candle = post_asian[post_asian['high'] == post_asian_high].iloc[0]
                    if sweep_candle['close'] < sweep_candle['high']:
                        if not sweep_info['rejection']:  # Don't overwrite low rejection
                            sweep_info['rejection'] = {
                                'type': 'bearish',
                                'entry': sweep_candle['close'],
                                'stop': post_asian_high,
                                'rejection_pips': (sweep_candle['high'] - sweep_candle['close']) * 10000
                            }
                            sweep_info['found'] = True

            return sweep_info

        except Exception as e:
            return {'found': False, 'reason': f'Error: {e}'}

    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a pair with FRD/FGD and Asian Sweep"""
        print(f"\nAnalyzing {symbol}...")

        analysis = {
            'symbol': symbol,
            'price': 0,
            'fgd_frd': None,
            'asian_sweep': None,
            'action': 'NONE'
        }

        # Get current price
        oanda_symbol = get_oanda_symbol(symbol)
        tick = mt5.symbol_info_tick(oanda_symbol)
        if tick:
            analysis['price'] = tick.ask
            print(f"  Current Price: {tick.ask:.5f}")

        # Daily analysis
        daily_df = self.get_data(symbol, mt5.TIMEFRAME_D1, 30)
        if daily_df is not None:
            daily_signals = self.detect_daily_frd_fgd(daily_df)

            # Daily trend
            current_daily = daily_df.iloc[-1]
            trend = "BULLISH" if current_daily['close'] > current_daily['open'] else "BEARISH"

            # Consecutive count
            consecutive_greens = 0
            consecutive_reds = 0
            for i in range(min(7, len(daily_df))):
                if daily_df.iloc[-(i+1)]['close'] > daily_df.iloc[-(i+1)]['open']:
                    consecutive_greens += 1
                    consecutive_reds = 0
                else:
                    consecutive_reds += 1
                    consecutive_greens = 0

            print(f"  Daily Trend: {trend}")
            print(f"  Daily Streak: {consecutive_greens}G/{consecutive_reds}R")

            # Check FRD/FGD using enhanced detector results
            if daily_signals:
                latest_signal = daily_signals[-1]
                days_ago = latest_signal['days_ago']
                trade_today = latest_signal.get('trade_today', False)
                confidence = latest_signal.get('confidence', 0)

                # Enhanced debugging
                print(f"  [DEBUG] Pattern found: {latest_signal.get('type', 'UNKNOWN')} | Days ago: {days_ago} | Trade today: {trade_today} | Confidence: {confidence}%")

                # Only consider valid signals with trade_today flag or recent signals
                if (trade_today and days_ago <= 1) or (days_ago <= 3 and confidence > 40):
                    analysis['fgd_frd'] = latest_signal
                    if latest_signal['type'] == 'FGD':
                        print(f"  [LOOK_FOR_LONGS] FGD {days_ago} day(s) ago - TODAY look for longs")
                        analysis['action'] = 'LOOK_FOR_TRADE'
                    else:
                        print(f"  [LOOK_FOR_SHORTS] FRD {days_ago} day(s) ago - TODAY look for shorts")
                        analysis['action'] = 'LOOK_FOR_TRADE'
                else:
                    print(f"  [SIGNAL_EXPIRED] {latest_signal['type']} {days_ago} days ago (Trade today: {trade_today})")
                    analysis['action'] = 'EXPIRED'
            else:
                print(f"  [WAIT] No FGD/FRD patterns detected")

            # Caution signals
            if consecutive_greens >= 3:
                print(f"  [CAUTION] {consecutive_greens} consecutive greens - overbought")
                if analysis['action'] == 'NONE':
                    analysis['action'] = 'CAUTION'
            elif consecutive_reds >= 3:
                print(f"  [CAUTION] {consecutive_reds} consecutive reds - oversold")
                if analysis['action'] == 'NONE':
                    analysis['action'] = 'CAUTION'

        # Asian Range Sweep
        asian = self.detect_asian_range_sweep(symbol)
        analysis['asian_sweep'] = asian

        if asian['found']:
            print(f"\n  [ASIAN_SWEEP] Pattern found!")
            print(f"    Asian Range: {asian['asian_low']:.5f} - {asian['asian_high']:.5f} ({asian['asian_range']:.0f} pips)")

            if asian['swept_low']:
                print(f"    Asian Low SWEPT: {asian['sweep_low_pips']:.0f} pips")

            if asian['rejection']:
                r = asian['rejection']
                print(f"    {r['type'].title()} Rejection: {r['rejection_pips']:.0f} pips")
                print(f"    Entry: {r['entry']:.5f}")
                print(f"    Stop: {r['stop']:.5f}")

                if analysis['fgd_frd'] and analysis['fgd_frd']['action'] == r['type'].upper():
                    print(f"\n  [!!!] PERFECT SETUP: FGD/FRD + Asian Sweep!")
                    analysis['action'] = 'STRONG_SETUP'
        else:
            if 'asian_low' in asian:
                print(f"\n  Asian Range: {asian['asian_low']:.5f} - {asian['asian_high']:.5f}")

        # Entry timing reminder
        print(f"\n  Entry Timing: Wait for 2AM EST (London Open)")
        print(f"  Enter on H1 closes at: 3AM, 4AM, 5AM EST")

        return analysis

    def scan_all_pairs(self):
        """Scan all pairs"""
        print("\nSCANNING WITH ASIAN RANGE SWEEP STRATEGY")
        print("=" * 70)

        results = {}
        for pair in self.major_pairs:
            try:
                result = self.analyze_symbol(pair)
                results[pair] = result
            except Exception as e:
                print(f"  [ERROR] {pair}: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Pair':<12} {'Price':<10} {'Signal':<10} {'Asian':<8} {'Action':<15}")
        print("-" * 70)

        for symbol, r in results.items():
            fgd_frd = r.get('fgd_frd')
            signal = fgd_frd.get('type', 'NONE') if fgd_frd else 'NONE'
            asian_sweep = r.get('asian_sweep', {})
            asian = 'YES' if asian_sweep.get('found') else 'NO'
            action = r['action']

            print(f"{symbol:<12} {r['price']:<10.5f} {signal:<10} {asian:<8} {action:<15}")

        # Save results
        with open('asian_sweep_results.txt', 'w') as f:
            f.write(f"Asian Range Sweep Analysis - {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")

            for symbol, r in results.items():
                f.write(f"{symbol}: {r['price']:.5f} - {r['action']}\n")
                if r.get('fgd_frd'):
                    s = r['fgd_frd']
                    f.write(f"  Signal: {s['type']} ({s['days_ago']} days ago)\n")
                if r.get('asian_sweep', {}).get('found'):
                    f.write(f"  Asian Sweep: Active\n")
                f.write("\n")

        print("\nResults saved to 'asian_sweep_results.txt'")

    def run(self):
        """Main execution"""
        try:
            if not self.connect_to_mt5():
                return

            self.scan_all_pairs()

        finally:
            mt5.shutdown()


if __name__ == "__main__":
    scanner = EnhancedStaceyBurkeScannerWithAsianOANDA()
    scanner.run()