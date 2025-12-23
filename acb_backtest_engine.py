#!/usr/bin/env python3
"""
ACB + Asian Sweep + mentfx Triple M Backtesting Engine
======================================================

Comprehensive backtesting framework for your exact ACB strategy:
- FGD/FRD daily signals
- Asian range sweep detection
- mentfx Triple M pin bar entries
- London session timing
- Risk management and position sizing

Author: Partner & Claude
Date: December 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
import MetaTrader5 as mt5
from enhanced_sb_scanner_asian_oanda import EnhancedStaceyBurkeScannerWithAsianOANDA, get_oanda_symbol

class ACBBacktestEngine:
    """Advanced backtesting engine for ACB strategy"""

    def __init__(self):
        self.scanner = EnhancedStaceyBurkeScannerWithAsianOANDA()
        self.trades = []
        self.daily_signals = {}
        self.equity_curve = []

        # Backtest parameters
        self.initial_balance = 10000
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.commission_per_lot = 7  # OANDA commission
        self.spread_pips = 2  # Average spread

    def load_historical_data(self, symbol: str, days_back: int = 90) -> Dict:
        """Load comprehensive historical data for backtesting"""
        print(f"Loading {days_back} days of data for {symbol}...")

        oanda_symbol = get_oanda_symbol(symbol)

        # Get different timeframes
        h1_df = self.scanner.get_data(oanda_symbol, mt5.TIMEFRAME_H1, days_back * 24)
        d1_df = self.scanner.get_data(oanda_symbol, mt5.TIMEFRAME_D1, days_back + 30)

        if h1_df is None or d1_df is None or len(h1_df) == 0 or len(d1_df) == 0:
            return {'error': 'Failed to load historical data'}

        return {
            'h1': h1_df,
            'daily': d1_df,
            'symbol': symbol,
            'oanda_symbol': oanda_symbol
        }

    def detect_fgd_frd_signals(self, daily_df: pd.DataFrame, start_date: datetime) -> Dict:
        """Generate FGD/FRD signals for backtesting period using proper ACB framework"""
        signals = {}

        # Use the proper ACB EnhancedFRDFGDDetector
        from acb.patterns.frd_fgd import EnhancedFRDFGDDetector
        detector = EnhancedFRDFGDDetector()

        # We need DMR and ACB levels for the enhanced detector
        # For backtesting, we'll create empty dictionaries and let it work with basic patterns
        dmr_levels = {}
        acb_levels = {}
        session_analysis = {}

        for i in range(4, len(daily_df)):  # Need at least 4 candles for proper analysis
            current_date = daily_df.index[i].date()

            if current_date >= start_date.date():
                # Get daily data up to this point
                historical_data = daily_df.iloc[:i+1].copy()

                # Use the proper ACB framework
                try:
                    patterns = detector.detect_enhanced_frd_fgd(
                        historical_data,
                        dmr_levels,
                        acb_levels,
                        session_analysis
                    )

                    if patterns.get('pattern_detected', False) and patterns.get('signal_type'):
                        signal_type = patterns['signal_type'].value  # Convert enum to string

                        # Map signal types to actions
                        action_map = {
                            'FGD': 'LONG',
                            'FRD': 'SHORT',
                            'THREE_DS': 'LONG',  # 3 Green Days = LONG
                            'THREE_DL': 'SHORT', # 3 Red Days = SHORT
                        }

                        if signal_type in action_map:
                            signals[current_date] = {
                                'type': signal_type,
                                'action': action_map[signal_type],
                                'confidence': patterns.get('confidence', 0),
                                'days_ago': patterns.get('days_since_trigger', 0),
                                'trade_today': patterns.get('trade_today', True),
                                'grade': patterns.get('signal_grade', 'B'),
                                'consecutive_count': patterns.get('consecutive_count', 0)
                            }

                except Exception as e:
                    # If enhanced detector fails, continue
                    continue

        return signals

    def detect_mentfx_signals(self, h1_df: pd.DataFrame, current_time: datetime) -> Dict:
        """Detect mentfx Triple M signals for specific candle"""
        try:
            current_idx = h1_df.index.get_loc(current_time)

            if current_idx < 1:
                return {'signal': 'NO_DATA'}

            current_candle = h1_df.iloc[current_idx]
            prev_candle = h1_df.iloc[current_idx - 1]

            # mentfx Triple M logic
            high_wick = (current_candle['high'] > prev_candle['high'] and
                        current_candle['close'] < prev_candle['high'])

            low_wick = (current_candle['low'] < prev_candle['low'] and
                       current_candle['close'] > prev_candle['low'])

            if low_wick:
                return {
                    'signal': 'GREEN_WICK',
                    'type': 'bullish',
                    'entry': current_candle['close'],
                    'stop': current_candle['low'],
                    'rejection_pips': (current_candle['close'] - current_candle['low']) * 10000,
                    'volume': current_candle['tick_volume'],
                    'candle_time': current_time
                }
            elif high_wick:
                return {
                    'signal': 'RED_WICK',
                    'type': 'bearish',
                    'entry': current_candle['close'],
                    'stop': current_candle['high'],
                    'rejection_pips': (current_candle['high'] - current_candle['close']) * 10000,
                    'volume': current_candle['tick_volume'],
                    'candle_time': current_time
                }
            else:
                return {'signal': 'NONE'}

        except Exception as e:
            return {'signal': 'ERROR', 'error': str(e)}

    def detect_asian_sweep(self, h1_df: pd.DataFrame, current_time: datetime) -> Dict:
        """Detect Asian range sweep for current time

        Asian Session: 19:00-00:00 EST which is 00:00-05:00 UTC the same day
        This creates the Asian range that London session may sweep
        """
        try:
            current_idx = h1_df.index.get_loc(current_time)

            # Asian session definition (UTC): 00:00-05:00 of current trading day
            # This corresponds to 19:00-00:00 EST (same calendar day)
            current_date = current_time.date()

            # Get Asian session candles (00:00-05:00 UTC of current trading day)
            asian_session = h1_df[
                (h1_df.index.date == current_date) &
                (h1_df.index.hour >= 0) & (h1_df.index.hour < 5)
            ]

            # If no Asian session for current day (we're early in the day),
            # try previous day's Asian session
            if len(asian_session) == 0:
                prev_date = current_date - timedelta(days=1)
                asian_session = h1_df[
                    (h1_df.index.date == prev_date) &
                    (h1_df.index.hour >= 0) & (h1_df.index.hour < 5)
                ]

            if len(asian_session) == 0:
                return {'found': False, 'reason': 'No Asian session data'}

            asian_low = asian_session['low'].min()
            asian_high = asian_session['high'].max()

            # Get London session candles to check for sweep (6AM-10AM UTC)
            london_candles = h1_df[
                (h1_df.index.date == current_date) &
                (h1_df.index.hour >= 6) & (h1_df.index.hour <= current_time.hour)
            ]

            if len(london_candles) == 0:
                return {
                    'found': False,
                    'asian_low': asian_low,
                    'asian_high': asian_high,
                    'reason': 'No London session data yet'
                }

            # Check if London session has swept Asian range
            london_low = london_candles['low'].min()
            london_high = london_candles['high'].max()

            swept_low = london_low < asian_low
            swept_high = london_high > asian_high

            # Calculate sweep depth in pips
            sweep_threshold_pips = 2  # Allow touching or slight penetration
            sweep_low_pips = max(0, (asian_low - london_low) * 10000) if swept_low else 0
            sweep_high_pips = max(0, (london_high - asian_high) * 10000) if swept_high else 0

            # Determine if sweep is significant enough
            low_sweep_valid = swept_low and sweep_low_pips >= sweep_threshold_pips
            high_sweep_valid = swept_high and sweep_high_pips >= sweep_threshold_pips

            return {
                'found': low_sweep_valid or high_sweep_valid,
                'asian_low': asian_low,
                'asian_high': asian_high,
                'london_low': london_low,
                'london_high': london_high,
                'swept_low': low_sweep_valid,
                'swept_high': high_sweep_valid,
                'sweep_low_pips': sweep_low_pips,
                'sweep_high_pips': sweep_high_pips,
                'asian_candles': len(asian_session),
                'london_candles': len(london_candles)
            }

        except Exception as e:
            return {'found': False, 'error': str(e)}

    def detect_asian_range_simple(self, h1_df: pd.DataFrame, current_time: datetime) -> Dict:
        """Simple Asian range detection - no sweep requirement

        Asian Session: 19:00-00:00 EST which is 00:00-05:00 UTC the same day
        """
        try:
            # Asian session definition (UTC): 00:00-05:00 of current trading day
            # This corresponds to 19:00-00:00 EST (same calendar day)
            current_date = current_time.date()

            # Get Asian session candles (00:00-05:00 UTC of current trading day)
            asian_session = h1_df[
                (h1_df.index.date == current_date) &
                (h1_df.index.hour >= 0) & (h1_df.index.hour < 5)
            ]

            # If no Asian session for current day (we're early in the day),
            # try previous day's Asian session
            if len(asian_session) == 0:
                prev_date = current_date - timedelta(days=1)
                asian_session = h1_df[
                    (h1_df.index.date == prev_date) &
                    (h1_df.index.hour >= 0) & (h1_df.index.hour < 5)
                ]

            if len(asian_session) == 0:
                return {
                    'found': False,
                    'asian_low': 0,
                    'asian_high': 0,
                    'reason': 'No Asian session data'
                }

            asian_low = asian_session['low'].min()
            asian_high = asian_session['high'].max()

            return {
                'found': True,
                'asian_low': asian_low,
                'asian_high': asian_high,
                'asian_candles': len(asian_session),
                'session_start': asian_session.index[0],
                'session_end': asian_session.index[-1]
            }

        except Exception as e:
            return {'found': False, 'error': str(e)}

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = account_balance * self.risk_per_trade
        position_size = risk_amount / (stop_distance * 100000)  # Convert to lots
        return round(position_size, 2)

    def execute_trade(self, trade_data: Dict, account_balance: float) -> Dict:
        """Execute a trade and calculate P&L"""
        symbol = trade_data['symbol']
        entry_price = trade_data['entry']
        stop_loss = trade_data['stop']
        target_price = trade_data['target']
        direction = trade_data['direction']

        # Calculate position size
        stop_distance = abs(entry_price - stop_loss)
        position_size = self.calculate_position_size(account_balance, stop_distance)

        # Calculate trade costs
        commission = position_size * self.commission_per_lot
        spread_cost = position_size * self.spread_pips

        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'position_size': position_size,
            'entry_time': trade_data['entry_time'],
            'entry_date': trade_data['entry_date'],
            'commission': commission,
            'spread_cost': spread_cost,
            'total_cost': commission + spread_cost,
            'risk_amount': account_balance * self.risk_per_trade,
            'setup_type': trade_data['setup_type']
        }

        return trade

    def close_trade(self, trade: Dict, exit_price: float, exit_time: datetime, exit_reason: str) -> Dict:
        """Close trade and calculate P&L"""
        direction = trade['direction']
        position_size = trade['position_size']
        entry_price = trade['entry_price']

        # Calculate P&L
        if direction == 'LONG':
            pips = (exit_price - entry_price) * 10000
        else:  # SHORT
            pips = (entry_price - exit_price) * 10000

        gross_profit = pips * position_size * 10  # $10 per pip per lot
        net_profit = gross_profit - trade['total_cost']

        # Update trade with exit data
        trade.update({
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pips': pips,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'profit_percent': (net_profit / trade['risk_amount']) * 100,
            'duration_hours': (exit_time - trade['entry_time']).total_seconds() / 3600
        })

        return trade

    def run_backtest(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict:
        """Run comprehensive backtest"""
        print("=" * 80)
        print("ACB + Asian Sweep + mentfx Triple M Backtest")
        print("=" * 80)
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade * 100}%")
        print("=" * 80)

        account_balance = self.initial_balance
        all_trades = []

        for symbol in symbols:
            print(f"\n[PROCESSING] Backtesting {symbol}...")

            # Load data
            data = self.load_historical_data(symbol, days_back=(end_date - start_date).days + 30)
            if 'error' in data:
                print(f"  [ERROR] {data['error']}")
                continue

            h1_df = data['h1']
            daily_df = data['daily']

            # Detect FGD/FRD signals
            daily_signals = self.detect_fgd_frd_signals(daily_df, start_date)
            print(f"  [DEBUG] Found {len(daily_signals)} daily signals for {symbol}")

            # Apply Weekly Trend Filter to daily signals
            weekly_df_sorted = daily_df.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()

            # WEEKLY TREND FILTER DISABLED
            # filtered_signals = {}
            # filtered_count = 0
            #
            # for signal_date, signal_info in daily_signals.items():
            #     signal_type = signal_info['type']
            #
            #     # Find the weekly candle that contains this signal date
            #     signal_week_start = pd.Timestamp(signal_date - timedelta(days=signal_date.weekday()))
            #
            #     # Get completed weekly candles before this week
            #     completed_weeks = weekly_df_sorted[weekly_df_sorted.index < signal_week_start]
            #
            #     if len(completed_weeks) > 0:
            #         latest_week = completed_weeks.iloc[-1]
            #         is_green_week = latest_week['close'] > latest_week['open']
            #
            #         # Apply filter logic
            #         if is_green_week and signal_type == 'FRD':
            #             print(f"    [TREND FILTER] {signal_date}: FRD signal filtered - weekly trend is bullish")
            #             filtered_count += 1
            #             continue
            #         elif not is_green_week and signal_type == 'FGD':
            #             print(f"    [TREND FILTER] {signal_date}: FGD signal filtered - weekly trend is bearish")
            #             filtered_count += 1
            #             continue
            #         else:
            #             print(f"    [TREND FILTER] {signal_date}: {signal_type} -> {signal_info['action']} (aligned with weekly trend)")
            #             filtered_signals[signal_date] = signal_info
            #     else:
            #         # If no completed weeks, allow the signal
            #         print(f"    [TREND FILTER] {signal_date}: {signal_type} -> {signal_info['action']} (no weekly data)")
            #         filtered_signals[signal_date] = signal_info
            #
            # print(f"  [DEBUG] Weekly trend filter removed {filtered_count} signals, {len(filtered_signals)} remain")
            # daily_signals = filtered_signals

            print(f"  [DEBUG] Weekly trend filter DISABLED - using all {len(daily_signals)} signals")

            # Iterate through each hour in the backtest period
            current_time = start_date
            open_trades = []

            print(f"  [DEBUG] Starting hourly loop from {start_date} to {end_date}")

            while current_time <= end_date:
                # Check for new trade setups FIRST
                current_date = current_time.date()

                # Skip if not in trading hours
                if current_time.hour < 2 or current_time.hour > 23:
                    current_time += timedelta(hours=1)
                    continue

                # Add debug every day at 6AM (when London session starts)
                if current_time.hour == 6:
                    print(f"  [DEBUG] Processing {current_time.strftime('%Y-%m-%d %H:00')} - Date: {current_date}")
                    if current_date in daily_signals:
                        print(f"  [DEBUG] >>> SIGNAL FOUND FOR TODAY! {daily_signals[current_date]}")
                    else:
                        print(f"  [DEBUG] >>> No signal for today. Available dates: {list(daily_signals.keys())[:3]}...")

                # Add debug for London session hours when we have signals
                if 6 <= current_time.hour <= 10 and current_date in daily_signals:
                    if current_time.hour == 6:
                        print(f"  [DEBUG] Entering London session for signal day {current_date}")

                try:
                    # Round current_time to nearest hour for data lookup
                    hourly_time = current_time.replace(minute=0, second=0, microsecond=0)

                    # Check if we have data for this time
                    if hourly_time not in h1_df.index:
                        current_time += timedelta(hours=1)
                        continue

                    # Check if we have a daily signal
                    if current_date in daily_signals:
                        daily_signal = daily_signals[current_date]
                        print(f"  [DEBUG] Processing signal for {current_date}: {daily_signal['type']} -> {daily_signal['action']}")

                        # Check for London session (1AM-5AM EST = 6AM-10AM UTC)
                        if 6 <= current_time.hour <= 10:
                            print(f"  [DEBUG] In London session at {current_time.strftime('%Y-%m-%d %H:00')}")
                            # STEP 1: Check if Asian range has been swept/invalidated FIRST
                            signal_action = daily_signal['action'].lower()  # 'long' for FGD, 'short' for FRD

                            # Get Asian range and sweep detection for current time
                            asian_sweep = self.detect_asian_sweep(h1_df, hourly_time)
                            asian_low = asian_sweep.get('asian_low', 0)
                            asian_high = asian_sweep.get('asian_high', 0)

                            # Debug: Check if Asian data found
                            if 'error' in asian_sweep:
                                print(f"  [DEBUG] Asian sweep error: {asian_sweep.get('error', 'Unknown')}")
                            elif not asian_sweep.get('asian_candles', 0):
                                print(f"  [DEBUG] No Asian session data for {hourly_time.strftime('%Y-%m-%d %H:00')}")

                            # Check for sweep requirement based on signal direction
                            sweep_detected = False
                            sweep_details = {}

                            if signal_action == 'long':  # FGD - need Asian LOW sweep
                                print(f"  [DEBUG] Checking for Asian low sweep...")
                                print(f"    Asian Low: {asian_low:.5f}")
                                print(f"    London Low so far: {asian_sweep.get('london_low', 0):.5f}")
                                print(f"    Sweep detected: {asian_sweep.get('swept_low', False)}")
                                if asian_sweep.get('swept_low', False):
                                    sweep_detected = True
                                    sweep_depth = asian_sweep.get('sweep_low_pips', 0)
                                    sweep_details = {
                                        'type': 'asian_low_sweep',
                                        'asian_low': asian_low,
                                        'london_low': asian_sweep.get('london_low', 0),
                                        'sweep_depth': sweep_depth
                                    }
                                    print(f"  [SWEEP DETECTED] FGD: Asian low sweep ({sweep_depth:.0f} pips)")
                                    print(f"    Asian Low: {asian_low:.5f}")
                                    print(f"    London Low: {asian_sweep.get('london_low', 0):.5f}")
                                else:
                                    print(f"  [FILTER] FGD active but NO Asian Low sweep - SKIP (FGD = Longs only)")
                                    # Move to next hour before continuing
                                    current_time += timedelta(hours=1)
                                    continue

                            elif signal_action == 'short':  # FRD - need Asian HIGH sweep
                                print(f"  [DEBUG] Checking for Asian high sweep...")
                                print(f"    Asian High: {asian_high:.5f}")
                                print(f"    London High so far: {asian_sweep.get('london_high', 0):.5f}")
                                print(f"    Sweep detected: {asian_sweep.get('swept_high', False)}")
                                if asian_sweep.get('swept_high', False):
                                    sweep_detected = True
                                    sweep_depth = asian_sweep.get('sweep_high_pips', 0)
                                    sweep_details = {
                                        'type': 'asian_high_sweep',
                                        'asian_high': asian_high,
                                        'london_high': asian_sweep.get('london_high', 0),
                                        'sweep_depth': sweep_depth
                                    }
                                    print(f"  [SWEEP DETECTED] FRD: Asian high sweep ({sweep_depth:.0f} pips)")
                                    print(f"    Asian High: {asian_high:.5f}")
                                    print(f"    London High: {asian_sweep.get('london_high', 0):.5f}")
                                else:
                                    print(f"  [FILTER] FRD active but NO Asian High sweep - SKIP (FRD = Shorts only)")
                                    # Move to next hour before continuing
                                    current_time += timedelta(hours=1)
                                    continue

                            # STEP 2: Only if sweep detected, then look for mentfx signal
                            if sweep_detected:
                                # Detect mentfx signal
                                mentfx_signal = self.detect_mentfx_signals(h1_df, hourly_time)

                                if mentfx_signal['signal'] in ['GREEN_WICK', 'RED_WICK']:
                                    mentfx_type = mentfx_signal['type']

                                    # Fix string comparison - handle 'bearish'/'bullish' vs 'short'/'long'
                                    direction_match = (signal_action == 'long' and mentfx_type == 'bullish') or \
                                                    (signal_action == 'short' and mentfx_type == 'bearish')

                                    # COMPLETE STRATEGY: Sweep detected + mentfx signal during London session
                                    if direction_match:

                                        # Dynamic target calculation for exact 1:2RR
                                        asian_range_pips = (asian_high - asian_low) * 10000
                                        entry_price = mentfx_signal['entry']
                                        stop_price = mentfx_signal['stop']

                                        if signal_action == 'long':  # FGD entry
                                            stop_distance_pips = (entry_price - stop_price) * 10000

                                            # Check if Asian range is wide enough for Asian target
                                            if asian_range_pips >= stop_distance_pips * 3:  # Wide range = room for 1:2RR
                                                target = asian_high
                                                target_type = "Asian High"
                                                rr_ratio = (target - entry_price) * 10000 / stop_distance_pips
                                            else:
                                                # Tight range - use Previous Day High
                                                # Find Previous Day High
                                                current_date_idx = daily_df.index.get_loc(pd.Timestamp(current_date))
                                                if current_date_idx > 0:
                                                    prev_day = daily_df.iloc[current_date_idx - 1]
                                                    pdh = prev_day['high']
                                                else:
                                                    # If no previous day data, use Asian target
                                                    pdh = asian_high

                                                # Calculate required stop for exact 1:2RR
                                                target_pips = (pdh - entry_price) * 10000
                                                required_stop_pips = target_pips / 2  # For 1:2RR
                                                adjusted_stop = entry_price - (required_stop_pips / 10000)

                                                target = pdh
                                                target_type = "Previous Day High"
                                                rr_ratio = 2.0
                                                stop_price = adjusted_stop
                                                print(f"    [ADJUSTED STOP] Tight Asian range - using PDH target with adjusted stop: {required_stop_pips:.0f} pips")

                                        else:  # FRD entry
                                            stop_distance_pips = (stop_price - entry_price) * 10000

                                            # Check if Asian range is wide enough for Asian target
                                            if asian_range_pips >= stop_distance_pips * 3:
                                                target = asian_low
                                                target_type = "Asian Low"
                                                rr_ratio = (entry_price - target) * 10000 / stop_distance_pips
                                            else:
                                                # Tight range - use Previous Day Low
                                                # Find Previous Day Low
                                                current_date_idx = daily_df.index.get_loc(pd.Timestamp(current_date))
                                                if current_date_idx > 0:
                                                    prev_day = daily_df.iloc[current_date_idx - 1]
                                                    pdl = prev_day['low']
                                                else:
                                                    # If no previous day data, use Asian target
                                                    pdl = asian_low

                                                # Calculate required stop for exact 1:2RR
                                                target_pips = (entry_price - pdl) * 10000
                                                required_stop_pips = target_pips / 2  # For 1:2RR
                                                adjusted_stop = entry_price + (required_stop_pips / 10000)

                                                target = pdl
                                                target_type = "Previous Day Low"
                                                rr_ratio = 2.0
                                                stop_price = adjusted_stop
                                                print(f"    [ADJUSTED STOP] Tight Asian range - using PDL target with adjusted stop: {required_stop_pips:.0f} pips")

                                        # Debug logging for complete setup
                                        print(f"  [EXECUTING TRADE] {current_time.strftime('%Y-%m-%d %H:00')}: {signal_action} trade")
                                        print(f"    Setup: {daily_signal['type']} + {sweep_details['type']} + {mentfx_signal['signal']}")
                                        print(f"    Entry: {entry_price:.5f}")
                                        print(f"    Stop: {stop_price:.5f}")
                                        print(f"    Asian Low: {asian_low:.5f}")
                                        print(f"    Asian High: {asian_high:.5f}")
                                        print(f"    Asian Range: {asian_range_pips:.0f} pips")
                                        print(f"    Stop Distance: {stop_distance_pips:.0f} pips")
                                        print(f"    Target: {target:.5f} ({target_type})")
                                        print(f"    Risk/Reward: 1:{rr_ratio:.1f}")
                                        print(f"    Expected direction: {'UP' if signal_action == 'long' else 'DOWN'}")

                                        # Execute trade
                                        trade_data = {
                                            'symbol': symbol,
                                            'direction': 'LONG' if mentfx_signal['type'] == 'bullish' else 'SHORT',
                                            'entry': entry_price,
                                            'stop': stop_price,
                                            'target': target,
                                            'entry_time': current_time,
                                            'entry_date': current_date,
                                            'setup_type': f"{daily_signal['type']}_MENTFX_ASIAN"
                                        }

                                        trade = self.execute_trade(trade_data, account_balance)
                                        open_trades.append(trade)
                    else:
                        # Debug: Check if we should have a signal today
                        if str(current_date) in [str(d) for d in daily_signals.keys()]:
                            print(f"  [DEBUG] Date mismatch! current_date: {current_date}, type: {type(current_date)}")
                            print(f"  [DEBUG] Available dates: {list(daily_signals.keys())}")

                        # Check for London session (1AM-5AM EST = 6AM-10AM UTC)
                        if 6 <= current_time.hour <= 10:
                            print(f"  [DEBUG] In London session at {current_time.strftime('%Y-%m-%d %H:00')} - No daily signal")

                    # Check for trade exits (ONLY on future candles, not entry candle)
                    for trade in open_trades[:]:
                        # Skip exit check if this is the same candle as entry
                        if current_time <= trade['entry_time']:
                            continue

                        # Get current candle
                        current_candle = h1_df.loc[hourly_time]

                        # Check stop loss
                        if trade['direction'] == 'LONG':
                            if current_candle['low'] <= trade['stop_loss']:
                                exit_price = trade['stop_loss']
                                closed_trade = self.close_trade(trade, exit_price, current_time, 'STOP_LOSS')
                                all_trades.append(closed_trade)
                                open_trades.remove(trade)
                                account_balance += closed_trade['net_profit']

                        elif trade['direction'] == 'SHORT':
                            if current_candle['high'] >= trade['stop_loss']:
                                exit_price = trade['stop_loss']
                                closed_trade = self.close_trade(trade, exit_price, current_time, 'STOP_LOSS')
                                all_trades.append(closed_trade)
                                open_trades.remove(trade)
                                account_balance += closed_trade['net_profit']

                        # Check take profit (only if still open)
                        if trade in open_trades:
                            if trade['direction'] == 'LONG':
                                if current_candle['high'] >= trade['target_price']:
                                    exit_price = trade['target_price']
                                    closed_trade = self.close_trade(trade, exit_price, current_time, 'TAKE_PROFIT')
                                    all_trades.append(closed_trade)
                                    open_trades.remove(trade)
                                    account_balance += closed_trade['net_profit']

                            elif trade['direction'] == 'SHORT':
                                if current_candle['low'] <= trade['target_price']:
                                    exit_price = trade['target_price']
                                    closed_trade = self.close_trade(trade, exit_price, current_time, 'TAKE_PROFIT')
                                    all_trades.append(closed_trade)
                                    open_trades.remove(trade)
                                    account_balance += closed_trade['net_profit']

                        # Time-based exit (24 hours) - only if still open
                        if trade in open_trades:
                            if current_time - trade['entry_time'] >= timedelta(hours=24):
                                exit_price = current_candle['close']
                                closed_trade = self.close_trade(trade, exit_price, current_time, 'TIME_EXIT')
                                all_trades.append(closed_trade)
                                open_trades.remove(trade)
                                account_balance += closed_trade['net_profit']

                except Exception as e:
                    print(f"  [WARNING] Error at {current_time}: {e}")

                current_time += timedelta(hours=1)

            # Close any remaining open trades at end of backtest
            for trade in open_trades:
                exit_price = h1_df.iloc[-1]['close']
                closed_trade = self.close_trade(trade, exit_price, end_date, 'END_OF_BACKTEST')
                all_trades.append(closed_trade)
                account_balance += closed_trade['net_profit']

        # Calculate performance metrics
        performance = self.calculate_performance_metrics(all_trades, self.initial_balance, account_balance)

        return {
            'trades': all_trades,
            'performance': performance,
            'initial_balance': self.initial_balance,
            'final_balance': account_balance,
            'total_return': ((account_balance - self.initial_balance) / self.initial_balance) * 100
        }

    def calculate_performance_metrics(self, trades: List[Dict], initial_balance: float, final_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {'error': 'No trades executed'}

        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['net_profit'] > 0]
        losing_trades = [t for t in trades if t['net_profit'] < 0]

        win_rate = (len(winning_trades) / total_trades) * 100
        avg_win = np.mean([t['net_profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_profit'] for t in losing_trades]) if losing_trades else 0

        # Profit metrics
        gross_profit = sum(t['net_profit'] for t in winning_trades)
        gross_loss = sum(t['net_profit'] for t in losing_trades)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        # Risk metrics
        all_profits = [t['net_profit'] for t in trades]
        max_drawdown = self.calculate_max_drawdown(all_profits, initial_balance)
        sharpe_ratio = self.calculate_sharpe_ratio(all_profits)

        # Setup-specific metrics
        setup_performance = {}
        for trade in trades:
            setup = trade['setup_type']
            if setup not in setup_performance:
                setup_performance[setup] = {'wins': 0, 'losses': 0, 'total_pips': 0}

            if trade['net_profit'] > 0:
                setup_performance[setup]['wins'] += 1
            else:
                setup_performance[setup]['losses'] += 1

            setup_performance[setup]['total_pips'] += trade['pips']

        # Monthly performance
        monthly_returns = self.calculate_monthly_returns(trades)

        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_win_loss_ratio': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'net_profit': round(sum(all_profits), 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_percent': round((max_drawdown / initial_balance) * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'avg_trade_duration': round(np.mean([t['duration_hours'] for t in trades]), 1),
            'setup_performance': setup_performance,
            'monthly_returns': monthly_returns,
            'largest_win': round(max(all_profits), 2),
            'largest_loss': round(min(all_profits), 2),
            'total_pips': round(sum([t['pips'] for t in trades]), 1),
            'avg_pips_per_trade': round(np.mean([t['pips'] for t in trades]), 1)
        }

    def calculate_max_drawdown(self, profits: List[float], initial_balance: float) -> float:
        """Calculate maximum drawdown"""
        balance_curve = [initial_balance]
        for profit in profits:
            balance_curve.append(balance_curve[-1] + profit)

        peak = balance_curve[0]
        max_dd = 0

        for balance in balance_curve:
            if balance > peak:
                peak = balance
            drawdown = peak - balance
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if len(profits) < 2:
            return 0

        returns = np.array(profits)
        return np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

    def calculate_monthly_returns(self, trades: List[Dict]) -> Dict:
        """Calculate monthly returns"""
        monthly = {}

        for trade in trades:
            month = trade['entry_time'].strftime('%Y-%m')
            if month not in monthly:
                monthly[month] = 0
            monthly[month] += trade['net_profit']

        return monthly

    def generate_report(self, backtest_results: Dict) -> str:
        """Generate comprehensive backtest report"""
        perf = backtest_results['performance']

        report = f"""
=============================================================
               ACB STRATEGY BACKTEST REPORT
=============================================================

PERFORMANCE OVERVIEW:
-------------------------------------------------------------
Initial Balance: ${backtest_results['initial_balance']:,.2f}
Final Balance:   ${backtest_results['final_balance']:,.2f}
Total Return:    {backtest_results['total_return']:+.2f}%

TRADING STATISTICS:
-------------------------------------------------------------
Total Trades:     {perf['total_trades']}
Win Rate:         {perf['win_rate']:.1f}%
Profit Factor:    {perf['profit_factor']:.2f}
Sharpe Ratio:     {perf['sharpe_ratio']:.3f}

PROFIT METRICS:
-------------------------------------------------------------
Gross Profit:     ${perf['gross_profit']:,.2f}
Gross Loss:       ${perf['gross_loss']:,.2f}
Net Profit:       ${perf['net_profit']:,.2f}
Average Win:      ${perf['avg_win']:,.2f}
Average Loss:     ${perf['avg_loss']:,.2f}
Win/Loss Ratio:   {perf['avg_win_loss_ratio']:.2f}

RISK METRICS:
-------------------------------------------------------------
Max Drawdown:     ${perf['max_drawdown']:,.2f} ({perf['max_drawdown_percent']:.1f}%)
Largest Win:      ${perf['largest_win']:,.2f}
Largest Loss:     ${perf['largest_loss']:,.2f}

TRADE DURATION:
-------------------------------------------------------------
Avg Duration:     {perf['avg_trade_duration']:.1f} hours

SETUP PERFORMANCE:
-------------------------------------------------------------"""

        for setup, stats in perf['setup_performance'].items():
            total = stats['wins'] + stats['losses']
            win_rate = (stats['wins'] / total) * 100 if total > 0 else 0
            avg_pips = stats['total_pips'] / total if total > 0 else 0

            report += f"""
{setup}:
   Trades: {total} | Win Rate: {win_rate:.1f}% | Avg Pips: {avg_pips:.1f}"""

        report += f"""

MONTHLY RETURNS:
-------------------------------------------------------------"""

        for month, profit in perf['monthly_returns'].items():
            report += f"""
{month}: ${profit:,.2f}"""

        report += f"""

SUMMARY:
-------------------------------------------------------------
Total Pips: {perf['total_pips']:.1f}
Avg Pips/Trade: {perf['avg_pips_per_trade']:.1f}

BACKTEST COMPLETE
"""

        return report

    def save_results(self, results: Dict, filename: str = 'acb_backtest_results.json'):
        """Save backtest results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[FILE] Results saved to: {filename}")

def main():
    """Main execution function"""
    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] Failed to initialize MetaTrader5")
        return

    # Check connection
    account_info = mt5.account_info()
    if account_info is None:
        print("[ERROR] No MT5 account connected")
        mt5.shutdown()
        return

    print(f"[OK] Connected to account: {account_info.login}")
    print(f"Server: {account_info.server}")
    print()

    # Create backtest engine
    engine = ACBBacktestEngine()

    # Configure backtest parameters
    symbols = ['AUDUSD', 'EURUSD']  # Testing AUDUSD and EURUSD with weekly trend filter and dynamic targeting
    start_date = datetime.now() - timedelta(days=365)  # 1 year back
    end_date = datetime.now()

    # Run backtest
    results = engine.run_backtest(symbols, start_date, end_date)

    if 'error' in results.get('performance', {}):
        print(f"[ERROR] Backtest error: {results['performance']['error']}")
    else:
        # Generate and display report
        report = engine.generate_report(results)
        print(report)

        # Save results
        engine.save_results(results)

        # Save detailed trade log
        if results['trades']:
            trade_log = pd.DataFrame(results['trades'])
            trade_log.to_csv('acb_backtest_trades.csv', index=False)
            print("[FILE] Trade log saved to: acb_backtest_trades.csv")

    mt5.shutdown()

if __name__ == "__main__":
    main()
