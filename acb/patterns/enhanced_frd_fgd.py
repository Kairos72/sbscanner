"""
Enhanced FRD/FGD Detector with Asian Range Sweep Strategy
=======================================================

Advanced detector combining:
1. Traditional FRD/FGD patterns
2. Asian Range Sweep entry validation
3. Proper UTC time conversion for FTMO broker data

Key Strategy:
- FGD/FRD trigger day (yesterday)
- Asian Range: 19:00-00:00 EST = 00:00-05:00 UTC
- Wait for sweep below/above Asian range
- Entry when candle closes back inside range
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Import from the main module
from acb.patterns.frd_fgd import SignalType


class AsianRangeEntryDetector:
    """
    Detects Asian Range sweep entries with precise timing.
    """

    def __init__(self):
        self.asian_start_hour_utc = 0   # 00:00 UTC = 19:00 EST previous day
        self.asian_end_hour_utc = 5     # 05:00 UTC = 00:00 EST
        self.sweep_threshold_pips = 10   # Minimum 10 pips sweep

    def analyze_asian_range_setup(self, df_h1: pd.DataFrame,
                                  trigger_type: SignalType,
                                  current_time: datetime = None) -> Dict:
        """
        Complete Asian Range analysis for entry day.

        Args:
            df_h1: H1 data (last 100 candles)
            trigger_type: FGD (LONG) or FRD (SHORT)
            current_time: Current UTC time

        Returns:
            Complete analysis with entry signals
        """
        if current_time is None:
            current_time = datetime.utcnow()

        analysis = {
            'asian_range': {},
            'sweep_detected': False,
            'sweep_details': None,
            'entry_signal': False,
            'entry_candle': None,
            'entry_plan': {},
            'current_status': 'WAITING'
        }

        # Step 1: Identify Asian Range (00:00-05:00 UTC)
        asian_range = self._identify_asian_range(df_h1, current_time)
        analysis['asian_range'] = asian_range

        if not asian_range:
            return analysis

        # Step 2: Check for sweep
        sweep_info = self._detect_sweep(df_h1, asian_range, trigger_type, current_time)
        analysis.update(sweep_info)

        # Step 3: Look for entry signal
        if sweep_info['sweep_detected']:
            entry_info = self._validate_entry(df_h1, asian_range, trigger_type, sweep_info)
            analysis.update(entry_info)

        # Step 4: Generate entry plan
        if analysis['entry_signal']:
            entry_plan = self._generate_entry_plan(analysis, trigger_type)
            analysis['entry_plan'] = entry_plan

        return analysis

    def _identify_asian_range(self, df_h1: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        Identify today's Asian range (00:00-05:00 UTC).
        """
        # Get today's date
        today = current_time.date()

        # Asian session: 00:00-05:00 UTC
        asian_start = datetime.combine(today, datetime.min.time())
        asian_end = asian_start + timedelta(hours=5)

        # Get Asian candles
        asian_candles = df_h1[
            (df_h1.index >= asian_start) &
            (df_h1.index < asian_end)
        ]

        if len(asian_candles) == 0:
            return None

        return {
            'start_time': asian_start,
            'end_time': asian_end,
            'high': asian_candles['high'].max(),
            'low': asian_candles['low'].min(),
            'range_pips': (asian_candles['high'].max() - asian_candles['low'].min()) * 10000,
            'candles': asian_candles,
            'candle_count': len(asian_candles)
        }

    def _detect_sweep(self, df_h1: pd.DataFrame,
                     asian_range: Dict,
                     trigger_type: SignalType,
                     current_time: datetime) -> Dict:
        """
        Detect liquidity sweep of Asian range.
        """
        # Get candles after Asian session
        post_asian_candles = df_h1[df_h1.index >= asian_range['end_time']]

        if trigger_type == SignalType.FGD:
            # Look for sweep below Asian low
            return self._detect_low_sweep(post_asian_candles, asian_range)
        else:
            # Look for sweep above Asian high
            return self._detect_high_sweep(post_asian_candles, asian_range)

    def _detect_low_sweep(self, candles: pd.DataFrame, asian_range: Dict) -> Dict:
        """
        Detect sweep below Asian low for long entries.
        """
        sweep_threshold = self.sweep_threshold_pips / 10000
        asian_low = asian_range['low']

        for idx, candle in candles.iterrows():
            if candle['low'] < asian_low - sweep_threshold:
                return {
                    'sweep_detected': True,
                    'sweep_details': {
                        'time': idx,
                        'sweep_type': 'LOW_SWEEP',
                        'sweep_price': candle['low'],
                        'sweep_candle': candle,
                        'asian_low': asian_low,
                        'sweep_distance': (asian_low - candle['low']) * 10000,
                        'sweep_utc': idx.strftime('%H:%M'),
                        'sweep_est': (idx - timedelta(hours=5)).strftime('%H:%M')
                    }
                }

        return {
            'sweep_detected': False,
            'sweep_details': None
        }

    def _detect_high_sweep(self, candles: pd.DataFrame, asian_range: Dict) -> Dict:
        """
        Detect sweep above Asian high for short entries.
        """
        sweep_threshold = self.sweep_threshold_pips / 10000
        asian_high = asian_range['high']

        for idx, candle in candles.iterrows():
            if candle['high'] > asian_high + sweep_threshold:
                return {
                    'sweep_detected': True,
                    'sweep_details': {
                        'time': idx,
                        'sweep_type': 'HIGH_SWEEP',
                        'sweep_price': candle['high'],
                        'sweep_candle': candle,
                        'asian_high': asian_high,
                        'sweep_distance': (candle['high'] - asian_high) * 10000,
                        'sweep_utc': idx.strftime('%H:%M'),
                        'sweep_est': (idx - timedelta(hours=5)).strftime('%H:%M')
                    }
                }

        return {
            'sweep_detected': False,
            'sweep_details': None
        }

    def _validate_entry(self, df_h1: pd.DataFrame,
                       asian_range: Dict,
                       trigger_type: SignalType,
                       sweep_info: Dict) -> Dict:
        """
        Validate entry by looking for candle closing back inside Asian range.
        """
        if not sweep_info['sweep_detected']:
            return {
                'entry_signal': False,
                'entry_candle': None,
                'current_status': 'NO_SWEEP'
            }

        sweep_time = sweep_info['sweep_details']['time']

        # Get candles after sweep
        post_sweep_candles = df_h1[df_h1.index > sweep_time]

        if trigger_type == SignalType.FGD:
            # Look for candle closing above Asian low
            for idx, candle in post_sweep_candles.iterrows():
                if candle['close'] > asian_range['low']:
                    return {
                        'entry_signal': True,
                        'entry_candle': {
                            'time': idx,
                            'close': candle['close'],
                            'close_utc': idx.strftime('%H:%M'),
                            'close_est': (idx - timedelta(hours=5)).strftime('%H:%M'),
                            'close_pht': (idx + timedelta(hours=8)).strftime('%H:%M')
                        },
                        'current_status': 'ENTRY_VALID'
                    }
        else:
            # Look for candle closing below Asian high
            for idx, candle in post_sweep_candles.iterrows():
                if candle['close'] < asian_range['high']:
                    return {
                        'entry_signal': True,
                        'entry_candle': {
                            'time': idx,
                            'close': candle['close'],
                            'close_utc': idx.strftime('%H:%M'),
                            'close_est': (idx - timedelta(hours=5)).strftime('%H:%M'),
                            'close_pht': (idx + timedelta(hours=8)).strftime('%H:%M')
                        },
                        'current_status': 'ENTRY_VALID'
                    }

        return {
            'entry_signal': False,
            'entry_candle': None,
            'current_status': 'WAITING_ENTRY'
        }

    def _generate_entry_plan(self, analysis: Dict, trigger_type: SignalType) -> Dict:
        """
        Generate detailed entry plan.
        """
        if not analysis['entry_signal']:
            return {}

        entry_price = analysis['entry_candle']['close']
        asian_range = analysis['asian_range']

        if trigger_type == SignalType.FGD:
            # Long entry plan
            return {
                'direction': 'LONG',
                'entry_price': entry_price,
                'stop_loss': asian_range['low'] - 0.00020,
                'targets': [
                    {'level': 'ASIAN_HIGH', 'price': asian_range['high']},
                    {'level': 'PREVIOUS_DAY_HIGH', 'price': None}  # To be filled
                ],
                'risk_pips': (entry_price - (asian_range['low'] - 0.00020)) * 10000,
                'entry_reason': 'Asian low sweep rejected - price closed inside range'
            }
        else:
            # Short entry plan
            return {
                'direction': 'SHORT',
                'entry_price': entry_price,
                'stop_loss': asian_range['high'] + 0.00020,
                'targets': [
                    {'level': 'ASIAN_LOW', 'price': asian_range['low']},
                    {'level': 'PREVIOUS_DAY_LOW', 'price': None}  # To be filled
                ],
                'risk_pips': ((asian_range['high'] + 0.00020) - entry_price) * 10000,
                'entry_reason': 'Asian high sweep rejected - price closed inside range'
            }