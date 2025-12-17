"""
ACB (Ain't Coming Back) Level Detector
=====================================

Identifies price levels that break and never return.
These are areas where smart money positions and won't be tested again.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class ACBDetector:
    """
    Detects and validates ACB levels based on market manipulation patterns.

    ACB Rule: A level that breaks and price does not return to test within 24 hours
    """

    def __init__(self):
        self.min_hours_to_confirm = 24  # Minimum hours to confirm ACB
        self.max_retest_distance = 0.00020  # Maximum distance for retest (20 pips)
        self.min_break_strength = 0.00030  # Minimum break strength (30 pips)

    def identify_acb_levels(self, df: pd.DataFrame,
                           current_time: datetime = None) -> Dict:
        """
        Identify potential and confirmed ACB levels.

        Args:
            df: DataFrame with OHLC data
            current_time: Current timestamp for validation

        Returns:
            Dictionary with ACB levels and their properties
        """
        if current_time is None:
            current_time = datetime.now()

        acb_levels = {
            'potential': [],
            'confirmed': [],
            'broken': []
        }

        # Find all significant break points
        break_points = self._find_break_points(df)

        # Validate each break point
        for bp in break_points:
            level_info = self._validate_acb_level(df, bp, current_time)

            if level_info['status'] == 'confirmed':
                acb_levels['confirmed'].append(level_info)
            elif level_info['status'] == 'potential':
                acb_levels['potential'].append(level_info)
            else:
                acb_levels['broken'].append(level_info)

        return acb_levels

    def _find_break_points(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find all significant break points in the data.

        Returns:
            List of break points with time and price info
        """
        break_points = []

        # Need at least 50 candles for meaningful analysis
        if len(df) < 50:
            return break_points

        # Calculate rolling highs and lows
        window = 20  # 20-period window for significant levels

        for i in range(window, len(df) - 24):  # Leave room for validation
            # Check for upside break
            if self._is_upside_break(df, i, window):
                break_points.append({
                    'index': i,
                    'time': df.index[i],
                    'price': df.iloc[i]['high'],
                    'type': 'upside',
                    'strength': self._calculate_break_strength(df, i, 'upside', window)
                })

            # Check for downside break
            if self._is_downside_break(df, i, window):
                break_points.append({
                    'index': i,
                    'time': df.index[i],
                    'price': df.iloc[i]['low'],
                    'type': 'downside',
                    'strength': self._calculate_break_strength(df, i, 'downside', window)
                })

        return break_points

    def _is_upside_break(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """
        Check if candle at index breaks above previous significant high.
        """
        current_candle = df.iloc[index]

        # Get the highest high in the window before
        prev_high = df.iloc[index-window:index]['high'].max()

        # Must break above previous high with conviction
        return (current_candle['close'] > prev_high and
                current_candle['high'] > prev_high + self.min_break_strength)

    def _is_downside_break(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """
        Check if candle at index breaks below previous significant low.
        """
        current_candle = df.iloc[index]

        # Get the lowest low in the window before
        prev_low = df.iloc[index-window:index]['low'].min()

        # Must break below previous low with conviction
        return (current_candle['close'] < prev_low and
                current_candle['low'] < prev_low - self.min_break_strength)

    def _calculate_break_strength(self, df: pd.DataFrame, index: int,
                                break_type: str, window: int) -> float:
        """
        Calculate the strength of the break in pips.
        """
        if break_type == 'upside':
            prev_high = df.iloc[index-window:index]['high'].max()
            return (df.iloc[index]['high'] - prev_high) * 10000
        else:
            prev_low = df.iloc[index-window:index]['low'].min()
            return (prev_low - df.iloc[index]['low']) * 10000

    def _validate_acb_level(self, df: pd.DataFrame,
                          break_point: Dict,
                          current_time: datetime) -> Dict:
        """
        Validate if a break point becomes an ACB level.
        """
        bp_index = break_point['index']
        bp_price = break_point['price']
        bp_time = break_point['time']
        bp_type = break_point['type']

        # Calculate hours since break
        hours_elapsed = (current_time - bp_time).total_seconds() / 3600

        # Check if enough time has passed to validate
        if hours_elapsed < self.min_hours_to_confirm:
            return {
                'time': bp_time,
                'price': bp_price,
                'type': bp_type,
                'strength': break_point['strength'],
                'status': 'potential',
                'hours_elapsed': hours_elapsed
            }

        # Check for retests after the break
        retest_info = self._check_for_retest(df, bp_index + 1, bp_price, bp_type)

        if not retest_info['retested']:
            # No retest - confirmed ACB
            return {
                'time': bp_time,
                'price': bp_price,
                'type': bp_type,
                'strength': break_point['strength'],
                'status': 'confirmed',
                'hours_elapsed': hours_elapsed,
                'validation_time': current_time
            }
        else:
            # Retested - not an ACB
            return {
                'time': bp_time,
                'price': bp_price,
                'type': bp_type,
                'strength': break_point['strength'],
                'status': 'broken',
                'hours_elapsed': hours_elapsed,
                'retest_time': retest_info['retest_time'],
                'retest_price': retest_info['retest_price']
            }

    def _check_for_retest(self, df: pd.DataFrame,
                         start_index: int,
                         level_price: float,
                         break_type: str) -> Dict:
        """
        Check if price returned to test the break level.
        """
        retested = False
        retest_time = None
        retest_price = None

        # Only check candles after the break
        for i in range(start_index, min(start_index + 48, len(df))):  # Check next 48 hours
            candle = df.iloc[i]

            if break_type == 'upside':
                # Check if price returned to the level from above
                if (candle['low'] <= level_price + self.max_retest_distance and
                    candle['high'] > level_price + self.max_retest_distance):
                    retested = True
                    retest_time = df.index[i]
                    retest_price = candle['low']
                    break
            else:
                # Check if price returned to the level from below
                if (candle['high'] >= level_price - self.max_retest_distance and
                    candle['low'] < level_price - self.max_retest_distance):
                    retested = True
                    retest_time = df.index[i]
                    retest_price = candle['high']
                    break

        return {
            'retested': retested,
            'retest_time': retest_time,
            'retest_price': retest_price
        }

    def get_nearest_acb_levels(self, current_price: float,
                              acb_levels: Dict,
                              max_distance: float = 0.00500) -> Dict:
        """
        Get nearest ACB levels to current price.

        Args:
            current_price: Current market price
            acb_levels: ACB levels dictionary
            max_distance: Maximum distance to consider (default 50 pips)

        Returns:
            Dictionary with nearest ACB levels above and below
        """
        nearest_above = None
        nearest_below = None

        # Check confirmed ACB levels
        for level in acb_levels['confirmed']:
            distance = level['price'] - current_price

            if abs(distance) <= max_distance:
                if distance > 0:  # Level is above
                    if nearest_above is None or distance < nearest_above['distance']:
                        nearest_above = {
                            'price': level['price'],
                            'distance': distance,
                            'time': level['time'],
                            'type': level['type']
                        }
                else:  # Level is below
                    if nearest_below is None or distance > nearest_below['distance']:
                        nearest_below = {
                            'price': level['price'],
                            'distance': distance,
                            'time': level['time'],
                            'type': level['type']
                        }

        return {
            'above': nearest_above,
            'below': nearest_below,
            'nearest': min([nearest_above, nearest_below],
                          key=lambda x: abs(x['distance']) if x else float('inf'))
        }

    def check_acb_breach(self, df: pd.DataFrame,
                         acb_levels: Dict,
                         lookback_candles: int = 5) -> List[Dict]:
        """
        Check if recent candles have breached any ACB levels.

        Args:
            df: Recent price data
            acb_levels: Dictionary of ACB levels
            lookback_candles: Number of recent candles to check

        Returns:
            List of breached ACB levels
        """
        breaches = []

        if len(df) < lookback_candles:
            return breaches

        # Check recent candles against ACB levels
        for i in range(-lookback_candles, 0):
            candle = df.iloc[i]

            for level in acb_levels['confirmed']:
                if level['type'] == 'upside' and candle['close'] > level['price']:
                    breaches.append({
                        'level': level,
                        'breach_time': df.index[i],
                        'breach_price': candle['close'],
                        'breach_candle': i
                    })
                elif level['type'] == 'downside' and candle['close'] < level['price']:
                    breaches.append({
                        'level': level,
                        'breach_time': df.index[i],
                        'breach_price': candle['close'],
                        'breach_candle': i
                    })

        return breaches