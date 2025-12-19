"""
Enhanced Pump & Dump Detector
============================

Optimized for forex markets and OANDA data.

Key Improvements:
1. Lower pump threshold (0.15% vs 0.3%)
2. Asian range reference for context
3. Better rejection pattern detection
4. Volume analysis adapted for tick data
5. Wick ratio analysis for candle patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class EnhancedPumpDumpDetector:
    """
    Enhanced pump & dump detector optimized for forex.
    """

    def __init__(self):
        self.thresholds = {
            'pump_threshold': 0.0015,        # 0.15% (15 pips on USDJPY)
            'asian_extension': 0.001,        # 0.1% above Asian high
            'wick_ratio': 0.5,               # Wick must be 50%+ of range
            'volume_multiplier': 1.2,        # Lower threshold for tick volume
            'rejection_candles': 2,          # Need 2 rejection candles
            'lookback_candles': 10           # Look back 10 candles
        }

    def detect_pump_dump(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Detect pump & dump pattern with forex-optimized parameters.

        Args:
            df: DataFrame with price data
            current_price: Current market price

        Returns:
            Dict with pump & dump opportunity if detected
        """
        if len(df) < self.thresholds['lookback_candles']:
            return None

        # Get Asian range for reference
        asian_range = self._calculate_asian_range(df)
        if not asian_range:
            return None

        asian_low = asian_range['low']
        asian_high = asian_range['high']

        # Get recent candles
        recent = df.tail(self.thresholds['lookback_candles'])

        # Check if price is above Asian range (pump condition)
        if current_price > asian_high:
            extension = (current_price - asian_high) / asian_high

            if extension > self.thresholds['asian_extension']:
                # This is a pump above Asian range
                return self._analyze_pump_above_asian(
                    recent, current_price, asian_range, extension
                )

        # Also check for general pump (not Asian range based)
        return self._analyze_general_pump(recent, current_price)

    def _calculate_asian_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Asian session range"""
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour

        # Asian session is 19:00-00:00 EST = 00:00-05:00 UTC
        asian_data = df_copy[
            (df_copy['hour'] >= 0) &
            (df_copy['hour'] < 5)
        ]

        if len(asian_data) > 0:
            return {
                'high': asian_data['high'].max(),
                'low': asian_data['low'].min(),
                'volume': asian_data['tick_volume'].sum()
            }

        return None

    def _analyze_pump_above_asian(self, recent: pd.DataFrame, current_price: float,
                                  asian_range: Dict, extension: float) -> Optional[Dict]:
        """Analyze pump that broke above Asian range"""

        # Find peak candle
        peak_candle = self._find_peak_candle(recent)
        if peak_candle is None:
            return None

        # Check for rejection pattern
        if self._is_rejection_candle(peak_candle) or self._has_recent_rejection(recent):
            peak_price = peak_candle['high']

            # Entry zone: rejection area
            if peak_candle['close'] < peak_candle['open']:  # Rejection confirmed
                entry_min = peak_candle['close']
                entry_max = peak_candle['high'] * 0.9995
            else:
                entry_min = peak_candle['low']
                entry_max = peak_candle['high'] * 0.999

            return {
                'type': 'PUMP_DUMP_SHORT',
                'direction': 'SHORT',
                'entry_zone': {
                    'min': entry_min,
                    'max': entry_max
                },
                'stop_loss': peak_price * 1.001,
                'targets': [
                    asian_range['high'],                           # Target 1: Asian high
                    (asian_range['high'] + asian_range['low']) / 2,  # Target 2: Mid-range
                    asian_range['low']                            # Target 3: Asian low
                ],
                'status': 'CONFIRMED',
                'priority': 'HIGH',
                'confidence': 90,
                'reason': f'Asian range pump to {peak_price:.3f} - rejection confirmed',
                'asian_range': f'{asian_range["low"]:.3f} - {asian_range["high"]:.3f}',
                'extension': f'{extension*100:.1f}%',
                'risk_reward': '2.5:1',
                'pattern_type': 'Asian Range Pump & Dump'
            }

        return None

    def _analyze_general_pump(self, recent: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Analyze general pump pattern (not Asian range based)"""
        if len(recent) < 5:
            return None

        # Calculate price change over 5 candles
        start_price = recent.iloc[0]['close']
        price_change = (current_price - start_price) / start_price

        if price_change > self.thresholds['pump_threshold']:
            # Check volume confirmation
            avg_volume = recent.iloc[:-2]['tick_volume'].mean()
            recent_volume = recent.iloc[-2:]['tick_volume'].mean()

            if recent_volume > avg_volume * self.thresholds['volume_multiplier']:
                # Look for peak and rejection
                peak_candle = self._find_peak_candle(recent)

                if peak_candle and self._is_rejection_candle(peak_candle):
                    return {
                        'type': 'PUMP_DUMP_SHORT',
                        'direction': 'SHORT',
                        'entry_zone': {
                            'min': peak_candle['low'] * 0.999,
                            'max': peak_candle['close'] * 1.001
                        },
                        'stop_loss': peak_candle['high'] * 1.001,
                        'targets': [
                            start_price,
                            (start_price + current_price) / 2,
                            start_price * 0.99
                        ],
                        'status': 'CONFIRMED',
                        'priority': 'MEDIUM',
                        'confidence': 75,
                        'reason': f'General pump detected - {peak_candle["high"]:.3f} peak',
                        'pump_strength': f'{price_change*100:.1f}%',
                        'volume_confirmation': f'{recent_volume/avg_volume:.1f}x',
                        'risk_reward': '2:1'
                    }

        return None

    def _find_peak_candle(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Find the candle with the highest high"""
        if len(df) == 0:
            return None

        peak_high = df['high'].max()
        peak_candles = df[df['high'] == peak_high]

        if len(peak_candles) > 0:
            return peak_candles.iloc[-1]  # Return the most recent peak
        return None

    def _is_rejection_candle(self, candle: pd.Series) -> bool:
        """
        Check if a candle shows rejection pattern.

        Returns True if:
        - Candle is red (close < open) OR
        - Upper wick is >50% of total range
        """
        if candle['close'] < candle['open']:
            return True  # Red candle is rejection

        # Calculate wick ratios
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            return False

        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']

        # Shooting star pattern
        if upper_wick / total_range > self.thresholds['wick_ratio']:
            return True

        return False

    def _has_recent_rejection(self, df: pd.DataFrame) -> bool:
        """Check if there are recent rejection candles"""
        if len(df) < 2:
            return False

        rejection_count = 0
        for i in range(max(0, len(df) - 3), len(df)):
            if self._is_rejection_candle(df.iloc[i]):
                rejection_count += 1

        return rejection_count >= self.thresholds['rejection_candles']