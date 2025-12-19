"""
Enhanced ACB (Ain't Coming Back) Detector
=======================================

Enhanced implementation based on Stacey Burke's methodology:
- Three consecutive higher/lower closes
- Position at extreme closes (HCOM/LCOM)
- EMA coil pattern
- Higher Time Frame alignment
- No return within session

ACB Rule: Strong move with no return during session
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from .dmr_calculator import DMRLevelCalculator


class EnhancedACBDetector:
    """
    Enhanced ACB detector implementing Stacey Burke's complete methodology.
    """

    def __init__(self):
        self.min_hours_to_confirm = 24
        self.max_retest_distance = 0.00020  # 20 pips
        self.min_break_strength = 0.00030  # 30 pips
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_coil_threshold = 0.00010  # 10 pips

    def identify_enhanced_acb_levels(self, df: pd.DataFrame,
                                    dmr_levels: Dict = None,
                                    current_time: datetime = None) -> Dict:
        """
        Identify ACB levels with complete Stacey Burke validation.
        """
        if current_time is None:
            current_time = datetime.now()

        if dmr_levels is None:
            dmr_calc = DMRLevelCalculator()
            dmr_levels = dmr_calc.calculate_all_dmr_levels(df, current_time)

        acb_levels = {
            'potential': [],
            'confirmed': [],
            'extreme': [],  # HCOM/LCOM related ACBs
            'validated': [],  # Full validation passed
            'broken': []
        }

        # Get daily data for analysis
        daily_df = self._get_daily_dataframe(df)

        if len(daily_df) < 10:
            return acb_levels

        # 1. Check for 3 consecutive higher/lower closes
        consecutive_closes = self._check_consecutive_closes(daily_df)

        # 2. Check position at extreme closes
        extreme_position = self._check_extreme_close_position(daily_df, dmr_levels)

        # 3. Check EMA coil pattern
        ema_coil = self._check_ema_coil(daily_df)

        # 4. Check Higher Time Frame alignment
        htf_alignment = self._check_htf_alignment(daily_df, dmr_levels)

        # 5. Find potential break points
        break_points = self._find_enhanced_break_points(df)

        # Validate each break point with complete criteria
        for bp in break_points:
            level_info = self._validate_enhanced_acb(
                df, bp, current_time,
                consecutive_closes,
                extreme_position,
                ema_coil,
                htf_alignment,
                dmr_levels
            )

            if level_info['status'] == 'validated':
                acb_levels['validated'].append(level_info)
            elif level_info['status'] == 'confirmed':
                acb_levels['confirmed'].append(level_info)
            elif level_info['status'] == 'potential':
                acb_levels['potential'].append(level_info)
            elif level_info['status'] == 'extreme':
                acb_levels['extreme'].append(level_info)
            else:
                acb_levels['broken'].append(level_info)

        return acb_levels

    def _get_daily_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert H1 data to daily dataframe."""
        # Check if volume column exists
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'

        return df.resample('D').agg(agg_dict).dropna()

    def _check_consecutive_closes(self, df: pd.DataFrame) -> Dict:
        """
        Check for 3 consecutive higher or lower closes.
        Key criterion for ACB validation.
        """
        if len(df) < 4:
            return {'higher_closes': 0, 'lower_closes': 0, 'consecutive': None}

        # Count consecutive higher closes
        higher_closes = 0
        for i in range(len(df) - 2, -1, -1):
            if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                higher_closes += 1
            else:
                break

        # Count consecutive lower closes
        lower_closes = 0
        for i in range(len(df) - 2, -1, -1):
            if df.iloc[i]['close'] < df.iloc[i-1]['close']:
                lower_closes += 1
            else:
                break

        return {
            'higher_closes': higher_closes,
            'lower_closes': lower_closes,
            'has_three_higher': higher_closes >= 3,
            'has_three_lower': lower_closes >= 3,
            'consecutive': 'higher' if higher_closes >= 3 else 'lower' if lower_closes >= 3 else None
        }

    def _check_extreme_close_position(self, df: pd.DataFrame,
                                     dmr_levels: Dict) -> Dict:
        """
        Check if current position is at HCOM/LCOM or similar extremes.
        """
        if not dmr_levels or not dmr_levels.get('extreme_closes'):
            return {'at_extreme': False}

        current_price = df.iloc[-1]['close']
        hcom = dmr_levels['extreme_closes'].get('high', {}).get('price')
        lcom = dmr_levels['extreme_closes'].get('low', {}).get('price')

        if hcom and lcom:
            distance_to_hcom = abs(hcom - current_price) * 10000
            distance_to_lcom = abs(lcom - current_price) * 10000

            return {
                'at_extreme': True,
                'near_hcom': distance_to_hcom < 50,  # Within 50 pips
                'near_lcom': distance_to_lcom < 50,
                'distance_to_hcom': distance_to_hcom,
                'distance_to_lcom': distance_to_lcom
            }

        return {'at_extreme': False}

    def _check_ema_coil(self, df: pd.DataFrame) -> Dict:
        """
        Check for EMA coil pattern (8 EMA and 21 EMA converging).
        """
        if len(df) < 30:
            return {'coil_detected': False}

        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()

        # Check for coil in last 10 candles
        recent = df.tail(10)
        ema_diff = abs(recent['ema_fast'] - recent['ema_slow'])

        # Coil detected when EMAs are very close
        coil_detected = (ema_diff < self.ema_coil_threshold).all()

        return {
            'coil_detected': coil_detected,
            'ema_fast': recent['ema_fast'].iloc[-1],
            'ema_slow': recent['ema_slow'].iloc[-1],
            'ema_diff': ema_diff.iloc[-1] * 10000,  # in pips
        }

    def _check_htf_alignment(self, df: pd.DataFrame,
                            dmr_levels: Dict) -> Dict:
        """
        Check alignment with Higher Time Frame (Weekly/Monthly).
        """
        alignment = {
            'weekly_aligned': False,
            'monthly_aligned': False,
            'at_weekly_level': False,
            'at_monthly_level': False
        }

        current_price = df.iloc[-1]['close']

        # Check weekly levels
        if dmr_levels.get('weekly'):
            weekly_high = dmr_levels['weekly'].get('high', {}).get('price')
            weekly_low = dmr_levels['weekly'].get('low', {}).get('price')

            if weekly_high and weekly_low:
                alignment['at_weekly_level'] = (
                    abs(current_price - weekly_high) * 10000 < 20 or
                    abs(current_price - weekly_low) * 10000 < 20
                )

        # Check monthly levels
        if dmr_levels.get('monthly'):
            monthly_high = dmr_levels['monthly'].get('high', {}).get('price')
            monthly_low = dmr_levels['monthly'].get('low', {}).get('price')

            if monthly_high and monthly_low:
                alignment['at_monthly_level'] = (
                    abs(current_price - monthly_high) * 10000 < 20 or
                    abs(current_price - monthly_low) * 10000 < 20
                )

        # Determine alignment (price moving towards HTF target)
        if len(df) >= 2:
            prev_close = df.iloc[-2]['close']
            direction = 'up' if current_price > prev_close else 'down'

            if direction == 'up' and monthly_high:
                alignment['monthly_aligned'] = current_price < monthly_high
            elif direction == 'down' and monthly_low:
                alignment['monthly_aligned'] = current_price > monthly_low

        return alignment

    def _find_enhanced_break_points(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find enhanced break points with additional filters.
        """
        break_points = []

        # Need at least 100 candles
        if len(df) < 100:
            return break_points

        window = 20

        for i in range(window, len(df) - 24):
            # Check for upside break
            if self._is_enhanced_upside_break(df, i, window):
                break_points.append({
                    'index': i,
                    'time': df.index[i],
                    'price': df.iloc[i]['high'],
                    'type': 'upside',
                    'strength': self._calculate_break_strength(df, i, 'upside', window)
                })

            # Check for downside break
            elif self._is_enhanced_downside_break(df, i, window):
                break_points.append({
                    'index': i,
                    'time': df.index[i],
                    'price': df.iloc[i]['low'],
                    'type': 'downside',
                    'strength': self._calculate_break_strength(df, i, 'downside', window)
                })

        return break_points

    def _is_enhanced_upside_break(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """
        Enhanced upside break check.
        """
        current_candle = df.iloc[index]
        prev_high = df.iloc[index-window:index]['high'].max()

        return (
            current_candle['close'] > prev_high and
            current_candle['high'] > prev_high + self.min_break_strength
        )

    def _is_enhanced_downside_break(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """
        Enhanced downside break check.
        """
        current_candle = df.iloc[index]
        prev_low = df.iloc[index-window:index]['low'].min()

        return (
            current_candle['close'] < prev_low and
            current_candle['low'] < prev_low - self.min_break_strength
        )

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

    def _validate_enhanced_acb(self, df: pd.DataFrame, break_point: Dict,
                               current_time: datetime,
                               consecutive_closes: Dict,
                               extreme_position: Dict,
                               ema_coil: Dict,
                               htf_alignment: Dict,
                               dmr_levels: Dict) -> Dict:
        """
        Validate ACB with complete Stacey Burke criteria.
        """
        bp_index = break_point['index']
        bp_price = break_point['price']
        bp_type = break_point['type']
        bp_strength = break_point['strength']

        # Calculate hours since break
        hours_elapsed = (current_time - break_point['time']).total_seconds() / 3600

        # Initialize validation score
        validation_score = 0
        validation_criteria = []

        # 1. Time validation (must be at least 24 hours)
        if hours_elapsed < self.min_hours_to_confirm:
            status = 'potential'
            validation_score += 20
            validation_criteria.append('Insufficient time elapsed')
        else:
            validation_score += 40
            validation_criteria.append('Time validation passed')

        # 2. Consecutive closes check (most important!)
        if bp_type == 'upside' and consecutive_closes.get('has_three_higher'):
            validation_score += 30
            validation_criteria.append('Three consecutive higher closes')
        elif bp_type == 'downside' and consecutive_closes.get('has_three_lower'):
            validation_score += 30
            validation_criteria.append('Three consecutive lower closes')
        else:
            validation_criteria.append('No 3 consecutive closes')

        # 3. Extreme position bonus
        if extreme_position.get('at_extreme'):
            validation_score += 20
            validation_criteria.append('At extreme close position')

        # 4. EMA coil bonus
        if ema_coil.get('coil_detected'):
            validation_score += 15
            validation_criteria.append('EMA coil pattern')

        # 5. HTF alignment bonus
        if htf_alignment.get('monthly_aligned') or htf_alignment.get('at_monthly_level'):
            validation_score += 25
            validation_criteria.append('HTF alignment')

        # 6. Break strength bonus
        if bp_strength > 50:  # Strong break (50+ pips)
            validation_score += 10
            validation_criteria.append('Strong break')

        # Check for retests after the break
        retest_info = self._check_for_retest(df, bp_index + 1, bp_price, bp_type)

        if not retest_info['retested']:
            # No retest - good sign for ACB
            if validation_score >= 80:
                status = 'validated'
            elif validation_score >= 60:
                status = 'confirmed'
            else:
                status = 'potential'

            # Check if it's an extreme ACB (at HCOM/LCOM)
            if extreme_position.get('at_extreme') and validation_score >= 70:
                status = 'extreme'
        else:
            # Retested - not an ACB
            status = 'broken'

        return {
            'time': break_point['time'],
            'price': bp_price,
            'type': bp_type,
            'strength': bp_strength,
            'status': status,
            'validation_score': validation_score,
            'hours_elapsed': hours_elapsed,
            'validation_time': current_time,
            'validation_criteria': validation_criteria,
            'consecutive_closes': consecutive_closes,
            'extreme_position': extreme_position,
            'ema_coil': ema_coil,
            'htf_alignment': htf_alignment,
            'retest': retest_info
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

        # Check the 24 hours after break
        end_index = min(start_index + 24, len(df))

        for i in range(start_index, end_index):
            if break_type == 'upside':
                # For upside break, check if low tested the level
                if df.iloc[i]['low'] <= level_price + self.max_retest_distance:
                    retested = True
                    retest_time = df.index[i]
                    retest_price = df.iloc[i]['low']
                    break
            else:
                # For downside break, check if high tested the level
                if df.iloc[i]['high'] >= level_price - self.max_retest_distance:
                    retested = True
                    retest_time = df.index[i]
                    retest_price = df.iloc[i]['high']
                    break

        return {
            'retested': retested,
            'retest_time': retest_time,
            'retest_price': retest_price,
            'hours_to_retest': (retest_time - df.index[start_index]).total_seconds() / 3600 if retested else None
        }