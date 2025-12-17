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

        Args:
            df: Daily timeframe DataFrame with OHLC data
            dmr_levels: Dictionary of DMR levels
            acb_levels: Dictionary of ACB levels
            session_analysis: Session behavior analysis

        Returns:
            Dictionary with enhanced pattern analysis
        """
        if len(df) < 10:
            return {'status': 'insufficient_data'}

        results = {
            'current_pattern': None,
            'signal_type': None,
            'signal_grade': None,
            'trigger_day': None,
            'consecutive_count': 0,
            'dmr_validation': {},
            'acb_validation': {},
            'session_context': {},
            'manipulation_phase': None,
            'confidence': 0,
            'entry_plan': {},
            'risk_parameters': {}
        }

        # Get the last 5 candles for analysis
        recent_candles = df.tail(5).copy()
        current_candle = recent_candles.iloc[-1]
        previous_candles = recent_candles.iloc[:-1]

        # Detect basic FRD/FGD patterns
        pattern_info = self._detect_basic_patterns(recent_candles)
        if not pattern_info['pattern_found']:
            return results

        results.update(pattern_info)

        # Validate with DMR levels
        dmr_validation = self._validate_with_dmr(
            current_candle,
            dmr_levels,
            pattern_info['signal_type']
        )
        results['dmr_validation'] = dmr_validation

        # Validate with ACB levels
        acb_validation = self._validate_with_acb(
            current_candle,
            acb_levels,
            pattern_info['signal_type']
        )
        results['acb_validation'] = acb_validation

        # Session context analysis
        session_context = {}
        if session_analysis:
            session_context = self._analyze_session_context(
                current_candle,
                session_analysis,
                pattern_info
            )
            results['session_context'] = session_context

        # Determine manipulation phase
        manipulation_phase = self._identify_manipulation_phase(
            recent_candles,
            dmr_levels,
            acb_levels
        )
        results['manipulation_phase'] = manipulation_phase

        # Grade the signal
        signal_grade = self._grade_signal(
            pattern_info,
            dmr_validation,
            acb_validation,
            session_context
        )
        results['signal_grade'] = signal_grade

        # Calculate confidence
        confidence = self._calculate_confidence(results)
        results['confidence'] = confidence

        # Generate entry plan
        if confidence > 60:
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
        """Validate pattern with ACB levels."""
        validation = {
            'has_acb_proximity': False,
            'nearest_acb': None,
            'acb_distance': 0,
            'acb_type': None,
            'acb_break_potential': False
        }

        current_price = candle['close']
        nearest_distance = float('inf')
        nearest_level = None

        # Check confirmed ACB levels
        for level in acb_levels.get('confirmed', []):
            distance = abs(level['price'] - current_price)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_level = {
                    'price': level['price'],
                    'type': level['type'],
                    'time': level['time']
                }

        if nearest_level and nearest_distance <= 0.00100:  # 100 pips
            validation.update({
                'has_acb_proximity': True,
                'nearest_acb': nearest_level,
                'acb_distance': nearest_distance * 10000,
                'acb_type': nearest_level['type']
            })

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
        """Grade the signal quality."""
        score = 0

        # Base score for pattern
        if pattern_info.get('consecutive_count') == 3:
            score += 40  # A+ setup
        elif pattern_info.get('consecutive_count') == 2:
            score += 30  # A setup
        else:
            score += 20  # B setup

        # DMR validation bonus
        if dmr_validation.get('is_near_dmr'):
            score += 25

        # ACB validation bonus
        if acb_validation.get('has_acb_proximity'):
            score += 20

        # Session context bonus
        if session_context.get('session_alignment'):
            score += 15

        # Determine grade
        if score >= 85:
            return SignalGrade.A_PLUS
        elif score >= 70:
            return SignalGrade.A
        elif score >= 55:
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
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1] if not atr.empty else 0.00100