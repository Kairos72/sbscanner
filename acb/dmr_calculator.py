"""
DMR (Daily Market Rotation) Level Calculator
=========================================

Calculates levels that the market must return to test.
Based on the principle that markets rotate between key levels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class DMRLevelCalculator:
    """
    Calculates DMR levels based on various timeframes.

    DMR Rule: Price will almost always return to test key levels
    """

    def __init__(self):
        self.min_pip_distance = 20  # Minimum 20 pips to be significant
        self.lookback_days = 30    # Days to look back for weekly levels
        self.three_day_window = 72  # Hours for 3-day levels
        self.hodlod_pip_threshold = 30  # Minimum 30 pips for HOD/LOD significance

    def calculate_all_dmr_levels(self, df: pd.DataFrame,
                                current_time: datetime = None) -> Dict:
        """
        Calculate all types of DMR levels.

        Args:
            df: DataFrame with OHLC data
            current_time: Current timestamp

        Returns:
            Dictionary containing all DMR levels
        """
        if current_time is None:
            current_time = datetime.now()

        return {
            'daily': self.get_daily_dmr_levels(df),
            'three_day': self.get_three_day_dmr_levels(df),
            'weekly': self.get_weekly_dmr_levels(df, current_time),
            'current': self.get_current_day_levels(df),
            'hodlod': self.get_hodlod_levels(df),  # New: High/Low of Day
            'monthly': self.get_monthly_dmr_levels(df, current_time),  # New: Monthly levels
            'extreme_closes': self.get_extreme_close_levels(df, current_time),  # New: HCOM/LCOM
            'fdtm': self.get_fdtml_levels(df, current_time),  # New: First Day Trading Month
            'active': self.get_active_dmr_levels(df)
        }

    def get_daily_dmr_levels(self, df: pd.DataFrame) -> Dict:
        """
        Get previous day's high and low (PDH/PDL) - most powerful DMR.
        """
        if len(df) < 2:
            return {'high': None, 'low': None}

        # Get previous day's data
        today = df.index[-1].date()
        prev_day_date = None

        # Find the previous trading day
        for i in range(len(df) - 2, -1, -1):
            if df.index[i].date() < today:
                prev_day_date = df.index[i].date()
                break

        if prev_day_date is None:
            return {'high': None, 'low': None}

        # Get all candles for the previous day
        prev_day_candles = df[df.index.date == prev_day_date]

        if len(prev_day_candles) == 0:
            return {'high': None, 'low': None}

        # Get the actual high and low of the entire day
        prev_high = prev_day_candles['high'].max()
        prev_low = prev_day_candles['low'].min()
        high_time = prev_day_candles['high'].idxmax()
        low_time = prev_day_candles['low'].idxmin()

        return {
            'high': {
                'price': prev_high,
                'time': high_time,
                'strength': 'MAXIMUM',
                'type': 'PDH',
                'distance_pips': None
            },
            'low': {
                'price': prev_low,
                'time': low_time,
                'strength': 'MAXIMUM',
                'type': 'PDL',
                'distance_pips': None
            }
        }

    def get_three_day_dmr_levels(self, df: pd.DataFrame) -> Dict:
        """
        Get the highest high and lowest low in the last 3 trading days.
        """
        if len(df) < 72:  # Need at least 3 days of H1 data
            return {'high': None, 'low': None}

        # Get last 72 hours (3 days)
        three_day_df = df.tail(72)

        return {
            'high': {
                'price': three_day_df['high'].max(),
                'time': three_day_df['high'].idxmax(),
                'strength': 'HIGH',
                'type': '3DH',
                'count': len(three_day_df)
            },
            'low': {
                'price': three_day_df['low'].min(),
                'time': three_day_df['low'].idxmin(),
                'strength': 'HIGH',
                'type': '3DL',
                'count': len(three_day_df)
            }
        }

    def get_weekly_dmr_levels(self, df: pd.DataFrame,
                             current_time: datetime) -> Dict:
        """
        Get the current week's high and low.
        """
        # Get week start (Monday)
        week_start = current_time - timedelta(days=current_time.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Filter data for current week
        weekly_df = df[df.index >= week_start]

        if len(weekly_df) == 0:
            return {'high': None, 'low': None}

        max_idx = weekly_df['high'].idxmax()
        min_idx = weekly_df['low'].idxmin()

        return {
            'high': {
                'price': weekly_df['high'].max(),
                'time': max_idx,
                'strength': 'WEEKLY',
                'type': 'WH',
                'day': max_idx.strftime('%A') if hasattr(max_idx, 'strftime') else 'Unknown'
            },
            'low': {
                'price': weekly_df['low'].min(),
                'time': min_idx,
                'strength': 'WEEKLY',
                'type': 'WL',
                'day': min_idx.strftime('%A') if hasattr(min_idx, 'strftime') else 'Unknown'
            }
        }

    def get_current_day_levels(self, df: pd.DataFrame) -> Dict:
        """
        Get today's high and low for intraday analysis.
        """
        today = df.index[-1].date()
        # Use pandas Series dt accessor instead of direct call
        today_df = df[df.index.to_series().dt.date == today]

        if len(today_df) == 0:
            return {'high': None, 'low': None}

        return {
            'high': {
                'price': today_df['high'].max(),
                'time': today_df['high'].idxmax(),
                'strength': 'INTRADAY',
                'type': 'TODAY_H'
            },
            'low': {
                'price': today_df['low'].min(),
                'time': today_df['low'].idxmin(),
                'strength': 'INTRADAY',
                'type': 'TODAY_L'
            }
        }

    def get_active_dmr_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Get all active DMR levels that are significant.
        """
        active_levels = []
        current_price = df.iloc[-1]['close']

        # Get all DMR types
        daily_levels = self.get_daily_dmr_levels(df)
        three_day_levels = self.get_three_day_dmr_levels(df)
        current_levels = self.get_current_day_levels(df)

        # Add significant daily levels
        if daily_levels['high']:
            distance = abs(daily_levels['high']['price'] - current_price) * 10000
            if distance >= self.min_pip_distance:
                active_levels.append(daily_levels['high'])
                active_levels[-1]['distance_pips'] = distance

        if daily_levels['low']:
            distance = abs(daily_levels['low']['price'] - current_price) * 10000
            if distance >= self.min_pip_distance:
                active_levels.append(daily_levels['low'])
                active_levels[-1]['distance_pips'] = distance

        # Add significant three-day levels
        if three_day_levels['high']:
            distance = abs(three_day_levels['high']['price'] - current_price) * 10000
            if distance >= self.min_pip_distance * 1.5:  # Higher threshold for 3-day
                active_levels.append(three_day_levels['high'])
                active_levels[-1]['distance_pips'] = distance

        if three_day_levels['low']:
            distance = abs(three_day_levels['low']['price'] - current_price) * 10000
            if distance >= self.min_pip_distance * 1.5:
                active_levels.append(three_day_levels['low'])
                active_levels[-1]['distance_pips'] = distance

        # Sort by distance (nearest first)
        active_levels.sort(key=lambda x: x.get('distance_pips', float('inf')))

        return active_levels

    def get_nearest_dmr_levels(self, current_price: float,
                              dmr_levels: Dict,
                              max_distance: float = 0.00500) -> Dict:
        """
        Get nearest DMR levels to current price.

        Args:
            current_price: Current market price
            dmr_levels: All DMR levels
            max_distance: Maximum distance to consider (default 50 pips)

        Returns:
            Dictionary with nearest levels above and below
        """
        nearest_above = None
        nearest_below = None
        all_levels = []

        # Collect all levels
        for level_type in ['daily', 'three_day', 'current', 'weekly']:
            for direction in ['high', 'low']:
                level = dmr_levels[level_type].get(direction)
                if level and level['price']:
                    all_levels.append(level)

        # Find nearest above and below
        for level in all_levels:
            distance = level['price'] - current_price

            if abs(distance) <= max_distance:
                if distance > 0:  # Level is above
                    if nearest_above is None or distance < nearest_above['distance']:
                        nearest_above = {
                            'price': level['price'],
                            'distance': distance,
                            'type': level['type'],
                            'strength': level['strength']
                        }
                else:  # Level is below
                    if nearest_below is None or distance > nearest_below['distance']:
                        nearest_below = {
                            'price': level['price'],
                            'distance': distance,
                            'type': level['type'],
                            'strength': level['strength']
                        }

        return {
            'above': nearest_above,
            'below': nearest_below,
            'all_levels': all_levels,
            'nearest': min([nearest_above, nearest_below],
                          key=lambda x: abs(x['distance']) if x else float('inf'))
        }

    def check_rotation_probability(self, current_price: float,
                                  dmr_level: Dict,
                                  time_until_close: float = None) -> Dict:
        """
        Calculate the probability of rotation to a DMR level.

        Args:
            current_price: Current market price
            dmr_level: DMR level to check
            time_until_close: Hours until market close

        Returns:
            Rotation probability and factors
        """
        if not dmr_level:
            return {'probability': 0, 'factors': []}

        distance = abs(dmr_level['price'] - current_price) * 10000
        probability = 90  # Base probability

        factors = []

        # Adjust based on distance
        if distance < 20:
            probability = 95
            factors.append("Very close (< 20 pips)")
        elif distance > 100:
            probability = 70
            factors.append("Far (> 100 pips)")

        # Adjust based on strength
        if dmr_level.get('strength') == 'MAXIMUM':
            probability += 10
            factors.append("Maximum strength (PDH/PDL)")
        elif dmr_level.get('strength') == 'HIGH':
            probability += 5
            factors.append("High strength (3-day)")

        # Adjust based on time
        if time_until_close:
            if time_until_close < 4:  # Less than 4 hours
                probability -= 20
                factors.append("Low time remaining")
            elif time_until_close > 24:  # More than a day
                probability += 5
                factors.append("Ample time for rotation")

        # Ensure probability stays in bounds
        probability = max(10, min(99, probability))

        return {
            'probability': probability,
            'distance_pips': distance,
            'factors': factors,
            'time_critical': time_until_close and time_until_close < 4
        }

    def calculate_rotation_targets(self, entry_price: float,
                                  direction: str,
                                  dmr_levels: Dict) -> List[Dict]:
        """
        Calculate rotation targets based on DMR levels.

        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            dmr_levels: All DMR levels

        Returns:
            List of potential targets
        """
        targets = []

        # Get all levels
        all_levels = []
        for level_type in ['daily', 'three_day', 'weekly']:
            for dir_key in ['high', 'low']:
                level = dmr_levels[level_type].get(dir_key)
                if level and level['price']:
                    all_levels.append(level)

        # Filter and sort based on direction
        if direction.upper() == 'LONG':
            # Long positions - look for levels above
            filtered_levels = [l for l in all_levels if l['price'] > entry_price]
            filtered_levels.sort(key=lambda x: x['price'])
        else:
            # Short positions - look for levels below
            filtered_levels = [l for l in all_levels if l['price'] < entry_price]
            filtered_levels.sort(key=lambda x: x['price'], reverse=True)

        # Create targets
        for level in filtered_levels[:5]:  # Top 5 targets
            targets.append({
                'price': level['price'],
                'type': level['type'],
                'strength': level['strength'],
                'distance_pips': abs(level['price'] - entry_price) * 10000
            })

        return targets

    def get_hodlod_levels(self, df: pd.DataFrame) -> Dict:
        """
        Get today's High of Day (HOD) and Low of Day (LOD) levels.
        These become significant when broken on Monday or after consolidation.
        """
        today = df.index[-1].date()
        today_df = df[df.index.to_series().dt.date == today]

        if len(today_df) == 0:
            return {'high': None, 'low': None}

        return {
            'high': {
                'price': today_df['high'].max(),
                'time': today_df['high'].idxmax(),
                'strength': 'INTRADAY',
                'type': 'HOD',
                'broken': None,  # Will be set if broken
                'break_time': None
            },
            'low': {
                'price': today_df['low'].min(),
                'time': today_df['low'].idxmin(),
                'strength': 'INTRADAY',
                'type': 'LOD',
                'broken': None,
                'break_time': None
            }
        }

    def get_monthly_dmr_levels(self, df: pd.DataFrame,
                               current_time: datetime) -> Dict:
        """
        Get current month's high and low (HOM/LOM) - most powerful levels.
        """
        # Get first day of current month
        month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Get all data for current month
        monthly_df = df[df.index >= month_start]

        if len(monthly_df) == 0:
            return {'high': None, 'low': None}

        max_idx = monthly_df['high'].idxmax()
        min_idx = monthly_df['low'].idxmin()

        return {
            'high': {
                'price': monthly_df['high'].max(),
                'time': max_idx,
                'strength': 'MAXIMUM',
                'type': 'HOM',  # High of Month
                'day': max_idx.strftime('%Y-%m-%d') if hasattr(max_idx, 'strftime') else 'Unknown'
            },
            'low': {
                'price': monthly_df['low'].min(),
                'time': min_idx,
                'strength': 'MAXIMUM',
                'type': 'LOM',  # Low of Month
                'day': min_idx.strftime('%Y-%m-%d') if hasattr(min_idx, 'strftime') else 'Unknown'
            }
        }

    def get_extreme_close_levels(self, df: pd.DataFrame,
                                 current_time: datetime) -> Dict:
        """
        Get Highest Close of Month (HCOM) and Lowest Close of Month (LCOM).
        These are critical ACB validation levels.
        """
        # Get first day of current month
        month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Get daily data (aggregate H1 to daily)
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        # Filter for current month
        monthly_daily = daily_df[daily_df.index >= month_start]

        if len(monthly_daily) == 0:
            return {'high': None, 'low': None}

        max_close_idx = monthly_daily['close'].idxmax()
        min_close_idx = monthly_daily['close'].idxmin()

        return {
            'high': {
                'price': monthly_daily['close'].max(),
                'time': max_close_idx,
                'strength': 'EXTREME',
                'type': 'HCOM',  # Highest Close of Month
                'day': max_close_idx.strftime('%Y-%m-%d') if hasattr(max_close_idx, 'strftime') else 'Unknown'
            },
            'low': {
                'price': monthly_daily['close'].min(),
                'time': min_close_idx,
                'strength': 'EXTREME',
                'type': 'LCOM',  # Lowest Close of Month
                'day': min_close_idx.strftime('%Y-%m-%d') if hasattr(min_close_idx, 'strftime') else 'Unknown'
            }
        }

    def get_fdtml_levels(self, df: pd.DataFrame,
                        current_time: datetime) -> Dict:
        """
        Get First Day of Trading Month (FDTM) levels.
        FDTM establishes the initial O-H-L-C for the new month.
        """
        # Get first day of current month
        month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Find the first trading day of the month
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        # Filter for current month
        monthly_daily = daily_df[daily_df.index >= month_start]

        if len(monthly_daily) == 0:
            return None

        # Get the first trading day
        first_day = monthly_daily.iloc[0]

        return {
            'open': first_day['open'],
            'high': first_day['high'],
            'low': first_day['low'],
            'close': first_day['close'],
            'date': first_day.name,
            'type': 'FDTM',
            'range_size': first_day['high'] - first_day['low']
        }

    def check_monday_breakout(self, df: pd.DataFrame,
                               current_time: datetime) -> Dict:
        """
        Check if Monday broke HOD/LOD levels - special case for DMR.
        According to manual: "When Monday breaks a HOD or LOD level,
        the market may now be 'in play' for the week."
        """
        if current_time.weekday() != 1:  # Not Monday
            return {'monday_breakout': False}

        # Get Monday data
        monday = current_time.date()
        monday_df = df[df.index.to_series().dt.date == monday]

        if len(monday_df) < 20:  # Need enough data
            return {'monday_breakout': False}

        # Get previous day levels (Sunday/Friday)
        prev_day = monday - timedelta(days=1)
        while prev_day.weekday() == 6:  # Skip Saturday
            prev_day -= timedelta(days=1)

        prev_df = df[df.index.to_series().dt.date == prev_day]

        if len(prev_df) == 0:
            return {'monday_breakout': False}

        prev_hod = prev_df['high'].max()
        prev_lod = prev_df['low'].min()
        monday_hod = monday_df['high'].max()
        monday_lod = monday_df['low'].min()

        return {
            'monday_breakout': True,
            'broke_hod': monday_hod > prev_hod + self.hodlod_pip_threshold / 10000,
            'broke_lod': monday_lod < prev_lod - self.hodlod_pip_threshold / 10000,
            'prev_hod': prev_hod,
            'prev_lod': prev_lod,
            'monday_hod': monday_hod,
            'monday_lod': monday_lod
        }