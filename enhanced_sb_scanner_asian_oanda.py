"""
ENHANCED STACEY BURKE SCANNER WITH ASIAN RANGE SWEEP - OANDA VERSION
==================================================================

Perfect ACB Strategy Implementation:
✅ FGD/FRD Pattern Detection using Enhanced ACB Framework
✅ Asian Range Sweep Identification with London Session Timing
✅ Pin Bar Momentum Analysis for Entry Confirmation
✅ Session-Aware Trade Execution (2AM-5AM EST Prime Window)
✅ Risk/Reward Calculation with Asian Range Targets

Strategy Requirements:
1. FGD or FRD signal on daily timeframe
2. Asian Range Sweep (yesterday's range reference)
3. Pin Bar rejection during London session (2AM-5AM EST)
4. Proper entry timing on H1 closes (3AM, 4AM, 5AM)
5. Stop below sweep low/high, target at Asian range extreme

Now using proper ACB framework for accurate pattern detection
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
        """
        Detect Asian Range Sweep pattern - Enhanced with London session timing and pin bar detection
        """
        try:
            oanda_symbol = get_oanda_symbol(symbol)

            # Get H1 data - need more data for comprehensive analysis
            h1_df = self.get_data(oanda_symbol, mt5.TIMEFRAME_H1, 120)  # 5 days of data
            if h1_df is None or len(h1_df) == 0:
                return {'found': False, 'reason': 'No H1 data'}

            # Get current time for session analysis
            current_time = datetime.now()
            today = current_time.date()
            current_hour = current_time.hour
            current_minute = current_time.minute

            # Define trading session times (EST)
            NY_SESSION_START = 8  # 8AM EST
            LONDON_START = 2  # 2AM EST (7am UTC)
            LONDON_END = 5  # 5AM EST (10am UTC)

            print(f"    Current Session: {current_hour:02d}{current_minute:02d} EST")

            # Filter today's candles
            today_candles = h1_df[h1_df.index.date == today]
            if len(today_candles) == 0:
                return {'found': False, 'reason': 'No candles for today'}

            # Asian session candles (19:00-23:00 EST = 00:00-05:00 UTC)
            # Yesterday's Asian range (since Asian session starts on previous day in EST)
            yesterday = today - timedelta(days=1)
            asian_session_start = current_time.replace(hour=0, minute=0)  # Start of today
            asian_session_end = current_time.replace(hour=5, minute=0)     # End of Asian session (5am EST)

            # Get yesterday's Asian range (best method for FGD trades)
            yesterday_candles = h1_df[
                (h1_df.index.date == yesterday)
            ]

            if len(yesterday_candles) == 0:
                return {'found': False, 'reason': 'No yesterday data for Asian range'}

            # Yesterday's Asian session: 19:00-23:00 EST
            asian_yesterday = yesterday_candles[
                (yesterday_candles.index.hour >= 19) | (yesterday_candles.index.hour <= 23)
            ]

            if len(asian_yesterday) == 0:
                return {'found': False, 'reason': 'No Asian session data for yesterday'}

            # Calculate yesterday's Asian range
            asian_yesterday_high = asian_yesterday['high'].max()
            asian_yesterday_low = asian_yesterday['low'].min()
            asian_range = (asian_yesterday_high - asian_yesterday_low) * 10000

            # Today's Asian session (00:00-06:00 EST = 05:00-11:00 UTC)
            # Use today's Asian session as reference but also consider yesterday's range
            asian_today = today_candles[
                (today_candles.index.hour >= 5) & (today_candles.index.hour <= 11)
            ]

            if len(asian_today) > 0:
                today_asian_high = asian_today['high'].max()
                today_asian_low = asian_today['low'].min()
                # Combine ranges for better analysis
                asian_high = max(asian_yesterday_high, today_asian_high)
                asian_low = min(asian_yesterday_low, today_asian_low)
                asian_range = (asian_high - asian_low) * 10000
            else:
                asian_high = asian_yesterday_high
                asian_low = asian_yesterday_low

            # Get all post-Asian session data (after 6AM EST)
            post_asian = today_candles[today_candles.index.hour > 11]

            sweep_info = {
                'found': False,
                'asian_high': asian_high,
                'asian_low': asian_low,
                'asian_range': asian_range,
                'yesterday_asian_high': asian_yesterday_high,
                'yesterday_asian_low': asian_yesterday_low,
                'today_asian_high': today_asian_high if len(asian_today) > 0 else None,
                'today_asian_low': today_asian_low if len(asian_today) > 0 else None,
                'current_price': today_candles.iloc[-1]['close'] if len(today_candles) > 0 else h1_df.iloc[-1]['close'],
                'swept_low': False,
                'swept_high': False,
                'rejection': None,
                'pin_bar_detected': False,
                'london_session_active': False,
                'entry_candle': None,
                'entry_time': None
            }

            # Session analysis
            if current_hour >= NY_SESSION_START and current_hour < LONDON_START:
                sweep_info['session_status'] = "PRE_LONDON"
                print(f"    Status: Waiting for London Open (2AM EST)")
            elif current_hour >= LONDON_START and current_hour <= LONDON_END:
                sweep_info['london_session_active'] = True
                sweep_info['session_status'] = "LONDON_ACTIVE"
                print(f"    Status: London session active (2AM-5AM EST) - PRIME ENTRY WINDOW")
            elif current_hour > LONDON_END:
                sweep_info['session_status'] = "POST_LONDON"
                print(f"    Status: After London session - Lower priority")

            # Analyze price action in post-Asian session
            if len(post_asian) > 0:
                post_asian_low = post_asian['low'].min()
                post_asian_high = post_asian['high'].max()

                # Check for Asian low sweep
                if post_asian_low < asian_low:
                    sweep_info['swept_low'] = True
                    sweep_info['sweep_low_pips'] = (asian_low - post_asian_low) * 10000
                    print(f"    Asian Low SWEPT by {sweep_info['sweep_low_pips']:.0f} pips")

                # Check for Asian high sweep
                if post_asian_high > asian_high:
                    sweep_info['swept_high'] = True
                    sweep_info['sweep_high_pips'] = (post_asian_high - asian_high) * 10000
                    print(f"    Asian High SWEPT by {sweep_info['sweep_high_pips']:.0f} pips")

                # Enhanced pin bar detection using mentfx Triple M logic
                for i in range(1, len(post_asian)):  # Start from 1 to compare with previous candle
                    candle = post_asian.iloc[i]
                    prev_candle = post_asian.iloc[i-1]
                    candle_time = post_asian.index[i]

                    # Calculate average volume for comparison
                    if len(post_asian) > 10:
                        avg_volume = post_asian['tick_volume'].mean()
                        volume_threshold = avg_volume * 1.2  # 20% above average
                    else:
                        volume_threshold = 1000  # Default threshold

                    # mentfx Triple M - High Wick Pattern (Bearish Rejection)
                    # Condition: high > prev_high and close < prev_high
                    if (candle['high'] > prev_candle['high'] and
                        candle['close'] < prev_candle['high']):

                        print(f"    [mentfx] High Wick detected: Price tried to go higher but rejected")
                        print(f"    Previous High: {prev_candle['high']:.5f}, Current High: {candle['high']:.5f}, Close: {candle['close']:.5f}")

                        # Only consider if this aligns with Asian sweep and FRD signal
                        if sweep_info['swept_high']:
                            sweep_info['pin_bar_detected'] = True
                            sweep_info['entry_candle'] = candle
                            sweep_info['entry_time'] = candle_time
                            sweep_info['rejection'] = {
                                'type': 'bearish',
                                'entry': candle['close'],
                                'stop': candle['high'],
                                'rejection_pips': (candle['high'] - candle['close']) * 10000,
                                'detection_method': 'mentfx_triple_m_high_wick'
                            }
                            sweep_info['found'] = True
                            print(f"    [ALIGNMENT] High Wick aligns with Asian High sweep!")
                            break

                    # mentfx Triple M - Low Wick Pattern (Bullish Rejection)
                    # Condition: low < prev_low and close > prev_low
                    elif (candle['low'] < prev_candle['low'] and
                          candle['close'] > prev_candle['low']):

                        print(f"    [mentfx] Low Wick detected: Price tried to go lower but rejected")
                        print(f"    Previous Low: {prev_candle['low']:.5f}, Current Low: {candle['low']:.5f}, Close: {candle['close']:.5f}")

                        # Additional volume confirmation
                        if candle['tick_volume'] > volume_threshold:
                            print(f"    [VOLUME] Strong volume confirmation: {candle['tick_volume']:,} > {volume_threshold:,.0f}")
                        else:
                            print(f"    [VOLUME] Volume weak: {candle['tick_volume']:,} < {volume_threshold:,.0f}")

                        # Only consider if this aligns with Asian sweep and FGD signal
                        if sweep_info['swept_low']:
                            sweep_info['pin_bar_detected'] = True
                            sweep_info['entry_candle'] = candle
                            sweep_info['entry_time'] = candle_time
                            sweep_info['rejection'] = {
                                'type': 'bullish',
                                'entry': candle['close'],
                                'stop': candle['low'],
                                'rejection_pips': (candle['close'] - candle['low']) * 10000,
                                'detection_method': 'mentfx_triple_m_low_wick'
                            }
                            sweep_info['found'] = True
                            print(f"    [ALIGNMENT] Low Wick aligns with Asian Low sweep!")
                            break

                # Fallback: Enhanced traditional pin bar detection if no mentfx signals found
                if not sweep_info['pin_bar_detected']:
                    for i in range(len(post_asian)):
                        candle = post_asian.iloc[i]
                        candle_time = post_asian.index[i]

                        # Calculate average volume for comparison
                        if len(post_asian) > 10:
                            avg_volume = post_asian['tick_volume'].mean()
                            volume_threshold = avg_volume * 1.2  # 20% above average
                        else:
                            volume_threshold = 1000  # Default threshold

                        # Bullish pin bar detection (for long entries after FGD)
                        if (candle['close'] > candle['open'] and
                            candle['low'] < post_asian_low * 1.0003 and  # Near sweep low
                            candle['close'] - candle['low'] > 5 and  # Reasonable pin bar range (5+ pips)
                            candle['tick_volume'] > volume_threshold):  # Above average volume

                            sweep_info['pin_bar_detected'] = True
                            sweep_info['entry_candle'] = candle
                            sweep_info['entry_time'] = candle_time
                            sweep_info['rejection'] = {
                                'type': 'bullish',
                                'entry': candle['close'],
                                'stop': candle['low'],
                                'rejection_pips': (candle['close'] - candle['low']) * 10000,
                                'detection_method': 'traditional_pin_bar'
                            }
                            sweep_info['found'] = True
                            print(f"    [FALLBACK] Traditional bullish pin bar detected")
                            break

                        # Bearish pin bar detection (for short entries after FRD)
                        elif (candle['close'] < candle['open'] and
                              candle['high'] > post_asian_high * 0.9997 and  # Near sweep high
                              candle['high'] - candle['close'] > 5 and  # Reasonable pin bar range
                              candle['tick_volume'] > volume_threshold):  # Above average volume

                            if not sweep_info['rejection']:  # Don't overwrite
                                sweep_info['rejection'] = {
                                    'type': 'bearish',
                                    'entry': candle['close'],
                                    'stop': candle['high'],
                                    'rejection_pips': (candle['high'] - candle['close']) * 10000,
                                    'detection_method': 'traditional_pin_bar'
                                }
                                sweep_info['found'] = True
                                print(f"    [FALLBACK] Traditional bearish pin bar detected")
                            sweep_info['entry_candle'] = candle
                            sweep_info['entry_time'] = candle_time
                            break

            # Calculate confidence levels for pattern quality
            confidence_score = 0
            if sweep_info['found']:
                confidence_score += 30  # Asian sweep detected
                if sweep_info['pin_bar_detected']:
                    confidence_score += 40  # Pin bar confirmation
                if sweep_info['london_session_active']:
                    confidence_score += 30  # London session timing

                sweep_info['confidence'] = confidence_score

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
            print(f"\n  [ASIAN_SWEEP] Pattern detected!")
            print(f"    Yesterday's Asian Range: {asian['yesterday_asian_low']:.5f} - {asian['yesterday_asian_high']:.5f} ({asian['asian_range']:.0f} pips)")
            if 'today_asian_high' in asian and asian['today_asian_low']:
                print(f"    Today's Asian Range: {asian['today_asian_low']:.5f} - {asian['today_asian_high']:.5f}")

            if asian['swept_low']:
                print(f"    Asian Low SWEPT: {asian['sweep_low_pips']:.0f} pips")

            if asian['swept_high']:
                print(f"    Asian High SWEPT: {asian['sweep_high_pips']:.0f} pips")

            if asian['pin_bar_detected']:
                entry_candle = asian['entry_candle']
                entry_time = asian['entry_time']
                detection_method = asian.get('rejection', {}).get('detection_method', 'unknown')
                print(f"\n  🔍 PIN BAR DETECTED at {entry_time.strftime('%H:%M EST')}!")
                print(f"    Detection Method: {detection_method.replace('_', ' ').title()}")
                print(f"    Entry Candle: Close: {entry_candle['close']:.5f}")
                print(f"    Candle Range: {entry_candle['low']:.5f} - {entry_candle['high']:.5f}")
                print(f"    Volume: {entry_candle['tick_volume']:,}")

            if asian['rejection']:
                r = asian['rejection']
                print(f"    {r['type'].title()} Rejection: {r['rejection_pips']:.0f} pips")
                print(f"    Entry: {r['entry']:.5f}")
                print(f"    Stop: {r['stop']:.5f}")
                print(f"    Rejection Range: {r['rejection_pips']:.0f} pips")

                # Validate FGD/FRD alignment with Asian sweep
                if analysis['fgd_frd']:
                    fgd_type = analysis['fgd_frd']['type']
                    fgd_direction = analysis['fgd_frd'].get('action', 'LONG' if fgd_type == 'FGD' else 'SHORT')

                    # Perfect setup: FGD + bullish rejection after Asian low sweep
                    if (fgd_type == 'FGD' and fgd_direction == 'LONG' and
                        r['type'] == 'bullish'):
                        print(f"\n  [!!!] PERFECT SETUP: FGD + Bullish Asian Rejection!")
                        analysis['action'] = 'STRONG_SETUP'
                        analysis['trade_direction'] = 'LONG'
                        analysis['entry_price'] = r['entry']
                        analysis['stop_loss'] = r['stop']
                        analysis['target'] = asian['asian_high']  # Target Asian range high

                    # Signal alignment check
                    if analysis['action'] == 'STRONG_SETUP':
                        print(f"\n  [STRATEGY ALIGNMENT] ✓")
                        print(f"    • FGD Signal: {fgd_type} (Days ago: {analysis['fgd_frd']['days_ago']})")
                        print(f"    • Direction: {fgd_direction}")
                        print(f"    • Asian Sweep: {r['type']} rejection")
                        print(f"    • Entry: At {r['entry']:.5f}")
                        print(f"    • Stop: {r['stop']:.5f} (below sweep low)")
                        stop_diff = r['entry'] - r['stop']
                        risk_reward = (r['rejection_pips'] / stop_diff) if stop_diff != 0 else 0
                        print(f"    • Risk/Reward: {risk_reward:.1f}")

                    # London session importance
                    if asian['london_session_active']:
                        print(f"\n  [TIMING] ✓ London session active - PRIME entry window")
                        print(f"    Focus on 3AM, 4AM, 5AM H1 closes")
                    elif asian['session_status'] == 'PRE_LONDON':
                        print(f"\n  [TIMING] Waiting for London Open (2AM EST)")
                    elif asian['session_status'] == 'POST_LONDON':
                        print(f"\n  [WARNING] London session over - Lower priority entries")

                else:
                    # Asian sweep without rejection
                    print(f"    Status: Asian sweep detected but no rejection yet")
                    print(f"    Asian Range: {asian['asian_low']:.5f} - {asian['asian_high']:.5f} ({asian['asian_range']:.0f} pips)")

        else:
            print(f"\n  No Asian sweep pattern detected")

        # Entry timing reminder
        print(f"\n  Entry Timing: Wait for 2AM EST (London Open)")
        print(f"  Entry Criteria: Asian sweep + FGD/FRD + Pin Bar Rejection")
        print(f"  Prime Window: 3AM, 4AM, 5AM H1 closes")

        return analysis

    def scan_all_pairs(self):
        """Scan all pairs with prioritized STRONG setups"""
        print("\nENHANCED ACB + ASIAN RANGE SWEEP SCANNER")
        print("=" * 70)
        print("Searching for FGD/FRD + Asian Sweep Pin Bar Rejections")
        print("=" * 70)

        results = {}
        strong_setups = []
        valid_signals = []

        for pair in self.major_pairs:
            try:
                result = self.analyze_symbol(pair)
                results[pair] = result

                # Categorize results
                if result.get('action') == 'STRONG_SETUP':
                    strong_setups.append(pair)
                elif result.get('action') in ['LOOK_FOR_TRADE', 'STRONG_SETUP']:
                    valid_signals.append(pair)
            except Exception as e:
                print(f"  [ERROR] {pair}: {e}")

        # Display STRONG setups first (highest priority)
        if strong_setups:
            print("\n" + "!" * 70)
            print("!!! STRONG SETUPS DETECTED !!!")
            print("!" * 70)
            print("Perfect FGD/FRD + Asian Sweep + Pin Bar Rejection")
            print("=" * 70)

            for pair in strong_setups:
                r = results[pair]
                fgd_frd = r.get('fgd_frd', {})
                asian = r.get('asian_sweep', {})

                print(f"\n{pair} - STRONG SETUP")
                print(f"  Entry Price: {r.get('entry_price', 'N/A'):.5f}")
                print(f"  Stop Loss: {r.get('stop_loss', 'N/A'):.5f}")
                print(f"  Target: {r.get('target', 'N/A'):.5f}")
                print(f"  FGD Signal: {fgd_frd.get('type', 'UNKNOWN')} ({fgd_frd.get('days_ago', 0)} days ago)")
                print(f"  Asian Range: {asian.get('asian_low', 'N/A'):.5f} - {asian.get('asian_high', 'N/A'):.5f}")
                print(f"  Confidence: {asian.get('confidence', 0)}%")

        # Display valid setups
        if valid_signals and not strong_setups:
            print("\n" + "=" * 70)
            print("VALID SETUPS")
            print("=" * 70)

            for pair in valid_signals:
                r = results[pair]
                fgd_frd = r.get('fgd_frd', {})
                asian = r.get('asian_sweep', {})

                print(f"\n{pair} - {r.get('action')}")
                print(f"  Signal: {fgd_frd.get('type', 'NONE')}")
                print(f"  Asian Sweep: {'Active' if asian.get('found') else 'None'}")

        # Summary table for all pairs
        print("\n" + "=" * 70)
        print("COMPLETE SUMMARY")
        print("=" * 70)
        print(f"{'Pair':<12} {'Price':<10} {'Signal':<10} {'Asian':<8} {'Status':<15}")
        print("-" * 70)

        # Sort: STRONG setups first, then valid signals, then others
        sorted_pairs = []
        for pair in strong_setups:
            sorted_pairs.append(pair)
        for pair in valid_signals:
            if pair not in strong_setups:
                sorted_pairs.append(pair)
        for pair in self.major_pairs:
            if pair not in sorted_pairs:
                sorted_pairs.append(pair)

        for symbol in sorted_pairs:
            r = results.get(symbol, {})
            fgd_frd = r.get('fgd_frd')
            signal = fgd_frd.get('type', 'NONE') if fgd_frd else 'NONE'
            asian_sweep = r.get('asian_sweep', {})
            asian = 'YES' if asian_sweep.get('found') else 'NO'

            # Get action status with priority
            action = r.get('action', 'NONE')
            if action == 'STRONG_SETUP':
                action = f"⭐ {action}"

            print(f"{symbol:<12} {r['price']:<10.5f} {signal:<10} {asian:<8} {action:<15}")

        # Count statistics
        strong_count = len(strong_setups)
        valid_count = len(valid_signals)

        print(f"\n" + "=" * 70)
        print(f"Statistics: {strong_count} STRONG setups, {valid_count} valid signals")

        # Save comprehensive results
        with open('asian_sweep_results.txt', 'w') as f:
            f.write(f"ENHANCED ACB + ASIAN SWEEP ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            # Summary
            f.write("SUMMARY:\n")
            f.write(f"STRONG Setups: {len(strong_setups)}\n")
            f.write(f"Valid Signals: {len(valid_signals)}\n\n")

            # STRONG setups details
            if strong_setups:
                f.write("!!! STRONG SETUPS DETECTED !!!\n\n")
                for pair in strong_setups:
                    r = results[pair]
                    fgd_frd = r.get('fgd_frd', {})
                    asian = r.get('asian_sweep', {})

                    f.write(f"\n{pair}:\n")
                    f.write(f"  Action: STRONG_SETUP\n")
                    f.write(f"  Entry: {r.get('entry_price', 'N/A'):.5f}\n")
                    f.write(f"  Stop: {r.get('stop_loss', 'N/A'):.5f}\n")
                    f.write(f"  Target: {r.get('target', 'N/A'):.5f}\n")
                    f.write(f"  FGD: {fgd_frd.get('type', 'UNKNOWN')} ({fgd_frd.get('days_ago', 0)} days ago)\n")
                    f.write(f"  Asian Range: {asian.get('asian_low', 'N/A'):.5f} - {asian.get('asian_high', 'N/A'):.5f}\n")
                    f.write(f"  Sweep: {'Low sweep' if asian.get('swept_low') else 'High sweep'}\n")
                    f.write(f"  Confidence: {asian.get('confidence', 0)}%\n")

            # All pairs status
            f.write("\n\nALL PAIRS STATUS:\n")
            f.write("-" * 50 + "\n")

            for symbol, r in results.items():
                fgd_frd = r.get('fgd_frd')
                signal = fgd_frd.get('type', 'NONE') if fgd_frd else 'NONE'
                asian_sweep = r.get('asian_sweep', {})
                asian = 'ACTIVE' if asian_sweep.get('found') else 'NONE'
                status = r.get('action', 'NONE')

                f.write(f"{symbol:<10} | Signal: {signal:<8} | Asian: {asian:<7} | Status: {status}\n")

        print("\nDetailed results saved to 'asian_sweep_results.txt'")

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