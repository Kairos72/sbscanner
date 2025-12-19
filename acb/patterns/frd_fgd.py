"""
Enhanced FRD/FGD Detector with ACB Context
==========================================

Combines traditional Stacey Burke FRD/FGD patterns with ACB methodology.
Validates signals using DMR levels, session context, and manipulation patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SignalGrade(Enum):
    """Signal quality grading."""
    A_PLUS = "A+"  # 3 consecutive days + ACB validation
    A = "A"        # 2 consecutive days + ACB validation
    B_PLUS = "B+"  # Valid pattern but weak ACB context
    B = "B"        # Basic pattern without ACB validation


class SignalType(Enum):
    """Pattern types."""
    FRD = "FRD"  # First Red Day
    FGD = "FGD"  # First Green Day
    INSIDE = "INSIDE"  # Inside Day
    THREE_DL = "3DL"  # Three Day Lows
    THREE_DS = "3DS"  # Three Day Shorts


class EnhancedFRDFGDDetector:
    """
    Enhanced detector for FRD/FGD patterns with ACB context.
    Integrates traditional Stacey Burke patterns with smart money manipulation analysis.
    """

    def __init__(self):
        self.min_consecutive_days = 2  # Minimum for A setup
        self.max_consecutive_days = 3  # Maximum for A+ setup
        self.dmr_proximity_threshold = 0.00050  # 50 pips proximity to DMR
        self.acb_validation_required = True  # Require ACB validation for high grades

    def detect_enhanced_frd_fgd(self, df: pd.DataFrame,
                               dmr_levels: Dict,
                               acb_levels: Dict,
                               session_analysis: Optional[Dict] = None) -> Dict:
        """
        Detect FRD/FGD patterns with ACB context validation.
        Enhanced to detect post-trigger trading opportunities.

        Args:
            df: Daily timeframe DataFrame with OHLC data
            dmr_levels: Dictionary of DMR levels
            acb_levels: Dictionary of ACB levels
            session_analysis: Session behavior analysis

        Returns:
            Dictionary with enhanced pattern analysis
        """
        # Ensure df is a DataFrame
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])

        if len(df) < 4:
            return {'status': 'insufficient_data'}

        results = {
            'current_pattern': None,
            'signal_type': None,
            'signal_grade': None,
            'trigger_day': None,
            'consecutive_count': 0,
            'days_since_trigger': 0,
            'trade_today': False,
            'trade_direction': None,
            'dmr_validation': {},
            'acb_validation': {},
            'session_context': {},
            'manipulation_phase': None,
            'confidence': 0,
            'entry_plan': {},
            'risk_parameters': {},
            'pattern_detected': False
        }

        # Exclude today's incomplete candle if before daily close (5pm EST / 22:00 UTC)
        df_completed = self._exclude_incomplete_candle(df)

        # Need at least 4 completed candles for analysis
        if len(df_completed) < 4:
            return results

        # Get the last 4 completed daily candles
        recent_4 = df_completed.tail(4).copy()

        # Detect post-trigger trading opportunities
        trigger_info = self._detect_post_trigger_patterns(recent_4)

        if not trigger_info['pattern_detected']:
            results['pattern_detected'] = False
            return results

        # Update results with trigger information
        results.update(trigger_info)
        results['pattern_detected'] = True

        # Get the trigger day candle for validation
        trigger_candle = recent_4.iloc[trigger_info['trigger_position']]
        current_price = recent_4.iloc[-1]['close']

        # Validate with DMR levels (using trigger day)
        dmr_validation = self._validate_with_dmr(
            trigger_candle,
            dmr_levels,
            trigger_info['signal_type']
        )
        results['dmr_validation'] = dmr_validation

        # Validate with ACB levels (using trigger day)
        acb_validation = self._validate_with_acb(
            trigger_candle,
            acb_levels,
            trigger_info['signal_type']
        )
        results['acb_validation'] = acb_validation

        # Session context analysis
        session_context = {}
        if session_analysis:
            session_context = self._analyze_session_context(
                trigger_candle,
                session_analysis,
                trigger_info
            )
            results['session_context'] = session_context

        # Determine manipulation phase
        manipulation_phase = self._identify_manipulation_phase(
            recent_4,
            dmr_levels,
            acb_levels
        )
        results['manipulation_phase'] = manipulation_phase

        # Grade the signal
        signal_grade = self._grade_signal(
            trigger_info,
            dmr_validation,
            acb_validation,
            session_context
        )
        results['signal_grade'] = signal_grade

        # Calculate confidence
        confidence = self._calculate_confidence(results)
        results['confidence'] = confidence

        # Generate entry plan for TODAY (the day after trigger)
        if confidence > 60 and trigger_info['days_since_trigger'] == 1:
            entry_plan = self._generate_entry_plan(results, df)
            results['entry_plan'] = entry_plan

        # Calculate risk parameters
        risk_params = self._calculate_risk_parameters(results, df)
        results['risk_parameters'] = risk_params

        return results

    def _detect_basic_patterns(self, candles: pd.DataFrame) -> Dict:
        """Detect basic FRD/FGD patterns."""
        if len(candles) < 3:
            return {'pattern_found': False}

        # Determine candle colors
        colors = ['GREEN' if c['close'] > c['open'] else 'RED' for _, c in candles.iterrows()]

        # Get last 3 candles for pattern detection
        last_3 = colors[-3:]
        current_candle = candles.iloc[-1]
        current_time = candles.index[-1]

        results = {
            'pattern_found': True,
            'current_time': current_time,
            'current_price': current_candle['close'],
            'candle_color': last_3[-1]
        }

        # Check for FRD (First Red Day after green candles)
        if (len(last_3) >= 3 and
            last_3[-3] == 'GREEN' and
            last_3[-2] == 'GREEN' and
            last_3[-1] == 'RED'):

            results.update({
                'signal_type': SignalType.FRD,
                'trigger_day': current_candle,
                'consecutive_count': 2,
                'pattern_description': "2 Greens followed by Red - FRD Trigger"
            })

        # Check for FGD (First Green Day after red candles)
        elif (len(last_3) >= 3 and
              last_3[-3] == 'RED' and
              last_3[-2] == 'RED' and
              last_3[-1] == 'GREEN'):

            results.update({
                'signal_type': SignalType.FGD,
                'trigger_day': current_candle,
                'consecutive_count': 2,
                'pattern_description': "2 Reds followed by Green - FGD Trigger"
            })

        # Check for 3DL/3DS (3 consecutive same color)
        elif len(last_3) >= 3 and last_3[-3] == last_3[-2] == last_3[-1]:
            if last_3[-1] == 'RED':
                results.update({
                    'signal_type': SignalType.THREE_DL,
                    'trigger_day': current_candle,
                    'consecutive_count': 3,
                    'pattern_description': "3 Red Days - 3DL Trigger (A+ Setup)"
                })
            else:
                results.update({
                    'signal_type': SignalType.THREE_DS,
                    'trigger_day': current_candle,
                    'consecutive_count': 3,
                    'pattern_description': "3 Green Days - 3DS Trigger (A+ Setup)"
                })

        # Check for Inside Day
        elif (len(candles) >= 2 and
              current_candle['high'] < candles.iloc[-2]['high'] and
              current_candle['low'] > candles.iloc[-2]['low']):

            results.update({
                'signal_type': SignalType.INSIDE,
                'trigger_day': current_candle,
                'consecutive_count': 1,
                'pattern_description': "Inside Day - Compression Pattern"
            })

        else:
            results['pattern_found'] = False

        return results

    def _validate_with_dmr(self, candle: pd.Series,
                          dmr_levels: Dict,
                          signal_type: SignalType) -> Dict:
        """Validate pattern with DMR levels."""
        validation = {
            'is_near_dmr': False,
            'nearest_dmr': None,
            'dmr_distance': 0,
            'dmr_type': None,
            'alignment_score': 0
        }

        current_price = candle['close']
        nearest_distance = float('inf')
        nearest_level = None

        # Check all DMR levels
        for level_type in ['daily', 'three_day', 'weekly']:
            for direction in ['high', 'low']:
                level = dmr_levels.get(level_type, {}).get(direction)
                if level and level['price']:
                    distance = abs(level['price'] - current_price)

                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_level = {
                            'price': level['price'],
                            'type': level['type'],
                            'strength': level['strength'],
                            'level_type': level_type
                        }

        if nearest_level and nearest_distance <= self.dmr_proximity_threshold:
            validation.update({
                'is_near_dmr': True,
                'nearest_dmr': nearest_level,
                'dmr_distance': nearest_distance * 10000,  # Convert to pips
                'dmr_type': nearest_level['type']
            })

            # Calculate alignment score
            if signal_type in [SignalType.FGD, SignalType.THREE_DS]:
                # Bullish patterns - look for resistance above
                if nearest_level['price'] > current_price:
                    validation['alignment_score'] = 80
                else:
                    validation['alignment_score'] = 40
            elif signal_type in [SignalType.FRD, SignalType.THREE_DL]:
                # Bearish patterns - look for support below
                if nearest_level['price'] < current_price:
                    validation['alignment_score'] = 80
                else:
                    validation['alignment_score'] = 40

        return validation

    def _validate_with_acb(self, candle: pd.Series,
                          acb_levels: Dict,
                          signal_type: SignalType) -> Dict:
        """Validate pattern with enhanced ACB levels."""
        validation = {
            'has_acb_proximity': False,
            'nearest_acb': None,
            'acb_distance': 0,
            'acb_type': None,
            'acb_break_potential': False,
            'enhanced_acb': False,
            'validation_score': 0,
            'consecutive_closes': None,
            'extreme_position': None,
            'ema_coil': None,
            'htf_alignment': None
        }

        current_price = candle['close']
        nearest_distance = float('inf')
        nearest_level = None

        # Check all ACB levels (potential, confirmed, validated, extreme)
        all_levels = []
        for level_type in ['potential', 'confirmed', 'validated', 'extreme']:
            all_levels.extend(acb_levels.get(level_type, []))

        # Find nearest level
        for level in all_levels:
            distance = abs(level['price'] - current_price)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_level = {
                    'price': level['price'],
                    'type': level['type'],
                    'time': level['time'],
                    'status': level.get('status', 'unknown')
                }

        if nearest_level and nearest_distance <= 0.00150:  # 150 pips (more lenient)
            validation.update({
                'has_acb_proximity': True,
                'nearest_acb': nearest_level,
                'acb_distance': nearest_distance * 10000,
                'acb_type': nearest_level['type']
            })

            # Enhanced validation score based on ACB status
            if nearest_level.get('status') == 'validated':
                validation['validation_score'] += 40
                validation['enhanced_acb'] = True
            elif nearest_level.get('status') == 'confirmed':
                validation['validation_score'] += 30
            elif nearest_level.get('status') == 'extreme':
                validation['validation_score'] += 35  # Extreme ACBs are valuable

            # Store enhanced ACB details
            if 'validation_score' in nearest_level:
                validation['validation_score'] = nearest_level['validation_score']
            if 'consecutive_closes' in nearest_level:
                validation['consecutive_closes'] = nearest_level['consecutive_closes']
            if 'extreme_position' in nearest_level:
                validation['extreme_position'] = nearest_level['extreme_position']
            if 'ema_coil' in nearest_level:
                validation['ema_coil'] = nearest_level['ema_coil']
            if 'htf_alignment' in nearest_level:
                validation['htf_alignment'] = nearest_level['htf_alignment']

            # Check if pattern could break ACB
            if signal_type == SignalType.FGD and nearest_level['type'] == 'upside':
                if current_price < nearest_level['price']:
                    validation['acb_break_potential'] = True
            elif signal_type == SignalType.FRD and nearest_level['type'] == 'downside':
                if current_price > nearest_level['price']:
                    validation['acb_break_potential'] = True

        return validation

    def _analyze_session_context(self, candle: pd.Series,
                                session_analysis: Dict,
                                pattern_info: Dict) -> Dict:
        """Analyze session context for the pattern."""
        context = {
            'session_alignment': False,
            'manipulation_zones_nearby': False,
            'liquidity_hunt_likely': False,
            'optimal_session': False
        }

        # Check if current session is optimal for the pattern
        current_session = session_analysis.get('current_session')
        signal_type = pattern_info.get('signal_type')

        if current_session and signal_type:
            # London session good for breakouts
            if current_session == 'London' and signal_type in [SignalType.FRD, SignalType.FGD]:
                context['optimal_session'] = True
                context['session_alignment'] = True

            # NY session good for continuation
            elif current_session == 'New York' and signal_type in [SignalType.THREE_DL, SignalType.THREE_DS]:
                context['optimal_session'] = True
                context['session_alignment'] = True

        # Check for nearby manipulation zones
        manipulation_zones = session_analysis.get('manipulation_zones', [])
        current_price = candle['close']

        for zone in manipulation_zones:
            if abs(zone['price'] - current_price) < 0.00050:  # 50 pips
                context['manipulation_zones_nearby'] = True
                if zone.get('hunted'):
                    context['liquidity_hunt_likely'] = True

        return context

    def _identify_manipulation_phase(self, candles: pd.DataFrame,
                                   dmr_levels: Dict,
                                   acb_levels: Dict) -> str:
        """Identify current market manipulation phase."""
        current_price = candles.iloc[-1]['close']

        # Check if at DMR level
        for level_type in ['daily', 'three_day']:
            for direction in ['high', 'low']:
                level = dmr_levels.get(level_type, {}).get(direction)
                if level and abs(level['price'] - current_price) < 0.00030:  # 30 pips
                    return "LIQUIDITY_HUNT"

        # Check if approaching ACB level
        for level in acb_levels.get('confirmed', []):
            if abs(level['price'] - current_price) < 0.00050:  # 50 pips
                return "ACB_TEST"

        # Check for range building (Asian session behavior)
        if len(candles) >= 3:
            ranges = [(c['high'] - c['low']) for _, c in candles.tail(3).iterrows()]
            avg_range = sum(ranges) / len(ranges)
            if avg_range < 0.00050:  # Tight range
                return "ACCUMULATION"

        # Default phase
        return "DISTRIBUTION"

    def _grade_signal(self, pattern_info: Dict,
                     dmr_validation: Dict,
                     acb_validation: Dict,
                     session_context: Dict) -> SignalGrade:
        """Grade the signal quality with enhanced Stacey Burke criteria."""
        score = 0

        # Base score for pattern
        if pattern_info.get('consecutive_count') == 3:
            score += 40  # A+ setup
        elif pattern_info.get('consecutive_count') == 2:
            score += 30  # A setup
        else:
            score += 20  # B setup

        # Enhanced DMR validation bonus
        if dmr_validation.get('is_near_dmr'):
            level_type = dmr_validation.get('nearest_dmr', {}).get('type', '')

            # Bonus for Monthly/Weekly/HOM/LOM levels
            if level_type in ['HOM', 'LOM', 'WH', 'WL']:
                score += 35  # Maximum bonus for monthly levels
            elif level_type in ['3DH', '3DL']:
                score += 25  # Good bonus for 3-day levels
            else:
                score += 25  # Standard PDH/PDL bonus

        # Enhanced ACB validation bonus
        if acb_validation.get('has_acb_proximity'):
            acb_score = acb_validation.get('validation_score', 0)

            # Check for enhanced ACB features
            if acb_validation.get('enhanced_acb'):
                score += 35  # Maximum for validated ACB

                # Bonus for consecutive closes
                if acb_validation.get('consecutive_closes', {}).get('has_three_higher') or \
                   acb_validation.get('consecutive_closes', {}).get('has_three_lower'):
                    score += 15

                # Bonus for extreme position
                if acb_validation.get('extreme_position', {}).get('at_extreme'):
                    score += 15

                # Bonus for EMA coil
                if acb_validation.get('ema_coil', {}).get('coil_detected'):
                    score += 10

                # Bonus for HTF alignment
                if acb_validation.get('htf_alignment', {}).get('monthly_aligned') or \
                   acb_validation.get('htf_alignment', {}).get('at_monthly_level'):
                    score += 20
            else:
                score += 20  # Standard ACB proximity

        # Session context bonus (still important)
        if session_context.get('session_alignment'):
            score += 15

        # Additional bonus for Monday breakouts (special DMR case)
        if dmr_validation.get('monday_breakout'):
            score += 10

        # Determine grade with adjusted thresholds for enhanced system
        if score >= 90:
            return SignalGrade.A_PLUS
        elif score >= 75:
            return SignalGrade.A
        elif score >= 60:
            return SignalGrade.B_PLUS
        else:
            return SignalGrade.B

    def _calculate_confidence(self, results: Dict) -> int:
        """Calculate overall confidence score."""
        confidence = 50  # Base confidence

        # Signal grade impact
        grade = results.get('signal_grade')
        if grade == SignalGrade.A_PLUS:
            confidence += 30
        elif grade == SignalGrade.A:
            confidence += 25
        elif grade == SignalGrade.B_PLUS:
            confidence += 15

        # DMR validation impact
        if results.get('dmr_validation', {}).get('is_near_dmr'):
            confidence += 15

        # ACB validation impact
        if results.get('acb_validation', {}).get('has_acb_proximity'):
            confidence += 10

        # Session context impact
        if results.get('session_context', {}).get('optimal_session'):
            confidence += 10

        # Manipulation phase penalty
        phase = results.get('manipulation_phase')
        if phase == "LIQUIDITY_HUNT":
            confidence -= 20

        return max(0, min(100, confidence))

    def _generate_entry_plan(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Generate detailed entry plan."""
        signal_type = results.get('signal_type')
        current_price = results.get('current_price')

        plan = {
            'direction': 'LONG' if signal_type in [SignalType.FGD, SignalType.THREE_DS] else 'SHORT',
            'entry_type': 'LIMIT' if results.get('dmr_validation', {}).get('is_near_dmr') else 'MARKET',
            'entry_zone': None,
            'wait_for_liquidity_hunt': results.get('manipulation_phase') == "LIQUIDITY_HUNT",
            'optimal_entry_time': None,
            'session_preference': 'London' if signal_type in [SignalType.FRD, SignalType.FGD] else 'New York'
        }

        # Define entry zone
        if signal_type in [SignalType.FGD, SignalType.THREE_DS]:
            # Long setups
            plan['entry_zone'] = {
                'high': current_price + 0.00020,
                'low': current_price - 0.00030
            }
        else:
            # Short setups
            plan['entry_zone'] = {
                'high': current_price + 0.00030,
                'low': current_price - 0.00020
            }

        # Optimal entry time
        if plan['session_preference'] == 'London':
            plan['optimal_entry_time'] = '07:00-09:00 UTC'
        else:
            plan['optimal_entry_time'] = '14:00-16:00 UTC'

        return plan

    def _calculate_risk_parameters(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Calculate risk management parameters."""
        signal_type = results.get('signal_type')
        current_price = results.get('current_price')

        # Calculate ATR for dynamic stops
        atr = self._calculate_atr(df.tail(14))

        risk_params = {
            'stop_loss_distance': atr * 2,  # 2x ATR
            'take_profit_distance': atr * 3,  # 3x ATR
            'position_size_percent': 2,  # 2% risk
            'max_drawdown_percent': 5
        }

        # Adjust stop based on ACB/DMR levels
        nearest_acb = results.get('acb_validation', {}).get('nearest_acb')
        if nearest_acb:
            if signal_type in [SignalType.FGD, SignalType.THREE_DS]:
                # Long - place stop below ACB if reasonable
                if current_price - nearest_acb['price'] < atr * 3:
                    risk_params['stop_loss_distance'] = abs(current_price - nearest_acb['price']) + 0.00020
            else:
                # Short - place stop above ACB if reasonable
                if nearest_acb['price'] - current_price < atr * 3:
                    risk_params['stop_loss_distance'] = abs(nearest_acb['price'] - current_price) + 0.00020

        return risk_params

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        # Ensure df is a DataFrame
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1] if not atr.empty else 0.00100

    def _exclude_incomplete_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude today's candle if it hasn't closed yet (before 5pm EST)."""
        # For now, include all candles - we'll handle this in the main logic
        # The issue is we need to detect triggers even if today is incomplete
        return df.copy()

    def _detect_post_trigger_patterns(self, candles: pd.DataFrame) -> Dict:
        """
        Detect FGD/FRD triggers within the last 4 completed daily candles.
        Focuses on post-trigger trading opportunities.
        """
        if len(candles) < 4:
            return {'pattern_detected': False}

        # Convert to list of candles for easier indexing
        candle_list = []
        colors = []

        for _, row in candles.iterrows():
            candle_list.append(row)
            color = "GREEN" if row['close'] > row['open'] else "RED"
            colors.append(color)

        results = {
            'pattern_detected': False,
            'signal_type': None,
            'trigger_date': None,
            'trigger_position': None,
            'consecutive_count': 0,
            'days_since_trigger': 0,
            'trade_today': False,
            'trade_direction': None,
            'pattern_description': None
        }

        # Check for A+ FGD: 3 reds followed by green
        # Pattern: RED-RED-RED-GREEN (trigger on last green)
        if len(colors) >= 4 and (colors[0] == 'RED' and colors[1] == 'RED' and colors[2] == 'RED' and colors[3] == 'GREEN'):
            # If trigger is the last candle in our window, it's either today or yesterday
            days_since = len(candles) - 4  # 0 if trigger is most recent, 1 if we have today's candle too
            results.update({
                'pattern_detected': True,
                'signal_type': SignalType.FGD,
                'trigger_date': candles.index[3],
                'trigger_position': 3,  # Position of trigger in our 4-candle window
                'consecutive_count': 3,
                'days_since_trigger': days_since,
                'trade_today': days_since <= 1,  # Trade if trigger was yesterday (days_since=0) or if we have today's candle (days_since=1 when len>4)
                'trade_direction': 'LONG',
                'pattern_description': "FGD A+ Trigger: 3 Reds followed by Green"
            })

        # Check for A+ FRD: 3 greens followed by red
        elif len(colors) >= 4 and (colors[0] == 'GREEN' and colors[1] == 'GREEN' and colors[2] == 'GREEN' and colors[3] == 'RED'):
            days_since = len(candles) - 4
            results.update({
                'pattern_detected': True,
                'signal_type': SignalType.FRD,
                'trigger_date': candles.index[3],
                'trigger_position': 3,
                'consecutive_count': 3,
                'days_since_trigger': days_since,
                'trade_today': days_since <= 1,
                'trade_direction': 'SHORT',
                'pattern_description': "FRD A+ Trigger: 3 Greens followed by Red"
            })

        # Now check for A FGD: 2 reds followed by green
        # Pattern: RED-RED-GREEN (trigger on 3rd candle)
        elif len(colors) >= 3 and (colors[0] == 'RED' and colors[1] == 'RED' and colors[2] == 'GREEN'):
            results.update({
                'pattern_detected': True,
                'signal_type': SignalType.FGD,
                'trigger_date': candles.index[2],
                'trigger_position': 2,  # Position of trigger in our 4-candle window
                'consecutive_count': 2,
                'days_since_trigger': 1,  # Trigger was yesterday if we have 4 candles
                'trade_today': True,  # Trade TODAY!
                'trade_direction': 'LONG',
                'pattern_description': "FGD A Trigger: 2 Reds followed by Green - TRADE TODAY!"
            })

        # Check for A FRD: 2 greens followed by red
        # Pattern: GREEN-GREEN-RED (trigger on 3rd candle)
        elif len(colors) >= 3 and (colors[0] == 'GREEN' and colors[1] == 'GREEN' and colors[2] == 'RED'):
            results.update({
                'pattern_detected': True,
                'signal_type': SignalType.FRD,
                'trigger_date': candles.index[2],
                'trigger_position': 2,
                'consecutive_count': 2,
                'days_since_trigger': 1,
                'trade_today': True,  # Trade TODAY!
                'trade_direction': 'SHORT',
                'pattern_description': "FRD A Trigger: 2 Greens followed by Red - TRADE TODAY!"
            })

        return results