"""
ACB-Aware Signal Generator - Phase 4 Component
=============================================

Generates comprehensive trading signals by combining:
- Traditional FRD/FGD patterns
- ACB level validation
- Pump & Dump pattern recognition
- Session-specific opportunities
- Smart money manipulation context

Confidence scoring: A+, A, B+, B, C+
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..detector import ACBDetector
from ..dmr_calculator import DMRLevelCalculator
from ..session_analyzer import SessionAnalyzer
from ..patterns.frd_fgd import EnhancedFRDFGDDetector, SignalType, SignalGrade
from ..patterns.signal_validator import ValidationLevel
from ..manipulation.market_phase_identifier import MarketPhaseIdentifier, MarketPhase


class SignalConfidence(Enum):
    """Signal confidence levels."""
    A_PLUS = "A+ (Highest Probability)"
    A = "A (High Probability)"
    B_PLUS = "B+ (Good Probability)"
    B = "B (Moderate Probability)"
    C_PLUS = "C+ (Low Probability)"


class SignalType(Enum):
    """Types of trading signals."""
    FGD_BULLISH = "FGD Bullish"
    FRD_BEARISH = "FRD Bearish"
    INSIDE_DAY_BREAK = "Inside Day Breakout"
    THREE_DL_BULLISH = "3-Day Low Bullish"
    THREE_DS_BEARISH = "3-Day High Bearish"
    PUMP_DUMP_LONG = "Pump & Dump Long"
    PUMP_DUMP_SHORT = "Pump & Dump Short"
    ASIAN_RANGE_ENTRY = "Asian Range Entry"
    LIQUIDITY_HUNT_REVERSAL = "Liquidity Hunt Reversal"


class ACBAwareSignalGenerator:
    """
    Generates enhanced trading signals using ACB concepts and smart money analysis.
    """

    def __init__(self):
        self.acb_det = ACBDetector()
        self.dmr_calc = DMRLevelCalculator()
        self.session_analyzer = SessionAnalyzer()
        self.frd_fgd_det = EnhancedFRDFGDDetector()
        self.phase_id = MarketPhaseIdentifier()

        # Signal weighting parameters
        self.weights = {
            'dmr_proximity': 0.25,      # Proximity to DMR levels
            'acb_validation': 0.20,      # ACB level confirmation
            'session_alignment': 0.15,   # Session-specific behavior
            'volume_confirmation': 0.15, # Volume spike confirmation
            'pattern_quality': 0.15,     # Pattern structure quality
            'market_phase': 0.10         # Smart money phase context
        }

    def generate_signals(self, df_h1: pd.DataFrame, df_d1: pd.DataFrame,
                        symbol: str) -> Dict:
        """
        Generate comprehensive trading signals for a symbol.

        Returns:
            Dictionary with all signals, confidence scores, and analysis.
        """
        print(f"\n{'='*60}")
        print(f"GENERATING ACB-AWARE SIGNALS FOR {symbol}")
        print(f"{'='*60}")

        # Get all necessary data
        dmr_levels = self.dmr_calc.calculate_all_dmr_levels(df_h1)
        acb_levels = self.acb_det.identify_acb_levels(df_h1)
        sessions = self.session_analyzer.analyze_session_behavior(df_h1, 72)
        market_phase = self.phase_id.identify_market_phase(df_h1, dmr_levels, acb_levels)

        # Generate all signal types
        signals = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'current_price': df_h1.iloc[-1]['close'],
            'market_phase': market_phase['current_phase'].value,

            # Pattern signals
            'frd_fgd_signals': self._generate_frd_fgd_signals(df_d1, dmr_levels, acb_levels),
            'inside_day_signals': self._generate_inside_day_signals(df_h1, dmr_levels, acb_levels),
            'pump_dump_signals': self._generate_pump_dump_signals(df_h1, dmr_levels, acb_levels),
            'asian_range_signals': self._generate_asian_range_signals(df_h1),

            # Contextual analysis
            'dmr_analysis': self._analyze_dmr_context(df_h1, dmr_levels),
            'acb_analysis': self._analyze_acb_context(df_h1, acb_levels),
            'session_analysis': self._analyze_session_context(sessions),
            'volume_analysis': self._analyze_volume_context(df_h1),

            # Highest probability signals
            'top_signals': [],
            'signal_summary': ''
        }

        # Prioritize and score all signals
        signals['top_signals'] = self._prioritize_signals(signals)
        signals['signal_summary'] = self._generate_signal_summary(signals)

        return signals

    def _generate_frd_fgd_signals(self, df_d1: pd.DataFrame, dmr_levels: Dict,
                                 acb_levels: Dict) -> List[Dict]:
        """Generate FRD/FGD signals with ACB validation."""
        signals = []

        try:
            pattern = self.frd_fgd_det.detect_enhanced_frd_fgd(df_d1, dmr_levels, acb_levels)

            if pattern.get('pattern_detected'):
                # Get additional context
                dmr_context = pattern.get('dmr_context', {})
                validation = pattern.get('validation', {})

                # Calculate confidence score
                confidence = self._calculate_frd_fgd_confidence(pattern, dmr_context, validation)

                signal = {
                    'type': SignalType.FGD_BULLISH if pattern.get('signal_type') == SignalType.FGD else SignalType.FRD_BEARISH,
                    'grade': pattern.get('signal_grade', SignalGrade.B),
                    'confidence': confidence,
                    'entry_zone': self._calculate_entry_zone(pattern, dmr_levels),
                    'stop_loss': self._calculate_stop_loss(pattern, dmr_levels),
                    'targets': self._calculate_targets(pattern, dmr_levels),
                    'reasoning': pattern.get('pattern_description', ''),
                    'dmr_alignment': dmr_context.get('proximity_to_dmr', 'Unknown'),
                    'acb_validation': validation.get('acb_confirmation', 'Unknown')
                }

                signals.append(signal)

        except Exception as e:
            print(f"[ERROR] FRD/FGD signal generation failed: {e}")

        return signals

    def _generate_inside_day_signals(self, df_h1: pd.DataFrame, dmr_levels: Dict,
                                    acb_levels: Dict) -> List[Dict]:
        """Generate inside day breakout signals."""
        signals = []

        try:
            # Look for inside day patterns in recent data
            last_20_days = df_h1.tail(20 * 24)  # Last 20 days of H1 data

            for i in range(len(last_20_days) - 24, len(last_20_days) - 1):  # Check last day
                current_day = last_20_days.iloc[i:i+24]
                prev_day = last_20_days.iloc[i-24:i] if i >= 24 else None

                if prev_day is not None:
                    # Check if current day is inside day
                    curr_high = current_day['high'].max()
                    curr_low = current_day['low'].min()
                    prev_high = prev_day['high'].max()
                    prev_low = prev_day['low'].min()

                    if curr_high < prev_high and curr_low > prev_low:
                        # Inside day detected
                        direction = self._determine_breakout_direction(current_day, dmr_levels)
                        confidence = self._calculate_inside_day_confidence(
                            current_day, prev_day, dmr_levels, acb_levels
                        )

                        if confidence.value in ['A+', 'A', 'B+']:
                            signal = {
                                'type': SignalType.INSIDE_DAY_BREAK,
                                'direction': direction,
                                'confidence': confidence,
                                'entry_zone': {
                                    'high': prev_high,
                                    'low': prev_low,
                                    'range_pips': (prev_high - prev_low) * 10000
                                },
                                'stop_loss': prev_low if direction == 'long' else prev_high,
                                'target': self._find_nearest_dmr(
                                    current_day.iloc[-1]['close'], direction, dmr_levels
                                ),
                                'reasoning': f"Inside day with {direction} bias toward DMR levels",
                                'breakout_confirmation': False
                            }
                            signals.append(signal)

        except Exception as e:
            print(f"[ERROR] Inside day signal generation failed: {e}")

        return signals

    def _generate_pump_dump_signals(self, df_h1: pd.DataFrame, dmr_levels: Dict,
                                   acb_levels: Dict) -> List[Dict]:
        """Generate Pump & Dump pattern signals."""
        signals = []

        try:
            # Look for pump & dump patterns
            last_100_candles = df_h1.tail(100)

            for i in range(20, len(last_100_candles) - 5):  # Need 5 candles for confirmation
                pattern_start = last_100_candles.iloc[i-20:i]
                pattern_candles = last_100_candles.iloc[i:i+5]

                # Detect different P&D patterns
                equilibrium_pd = self._detect_equilibrium_pd(pattern_candles, dmr_levels)
                m_top_w_bottom = self._detect_m_top_w_bottom(pattern_candles)
                lower_range_pd = self._detect_lower_range_pd(pattern_candles)

                # Process detected patterns
                for pattern_type, pattern_data in [
                    ('equilibrium', equilibrium_pd),
                    ('m_top_w_bottom', m_top_w_bottom),
                    ('lower_range', lower_range_pd)
                ]:
                    if pattern_data:
                        confidence = self._calculate_pump_dump_confidence(
                            pattern_type, pattern_data, dmr_levels, acb_levels
                        )

                        if confidence.value in ['A+', 'A', 'B+']:
                            signal = {
                                'type': SignalType.PUMP_DUMP_LONG if pattern_data['direction'] == 'long' else SignalType.PUMP_DUMP_SHORT,
                                'pattern_type': pattern_type,
                                'confidence': confidence,
                                'entry_zone': pattern_data['entry_zone'],
                                'stop_loss': pattern_data['stop_loss'],
                                'target': pattern_data['target'],
                                'reasoning': f"{pattern_type.title()} P&D pattern at DMR level",
                                'manipulation_evidence': pattern_data.get('manipulation_signs', [])
                            }
                            signals.append(signal)

        except Exception as e:
            print(f"[ERROR] Pump & Dump signal generation failed: {e}")

        return signals

    def _generate_asian_range_signals(self, df_h1: pd.DataFrame) -> List[Dict]:
        """Generate Asian range entry signals."""
        signals = []

        try:
            from ..patterns.enhanced_frd_fgd import AsianRangeEntryDetector

            asian_det = AsianRangeEntryDetector()

            # Check both FGD and FRD scenarios
            for signal_type in [SignalType.FGD, SignalType.FRD]:
                analysis = asian_det.analyze_asian_range_setup(df_h1, signal_type, datetime.utcnow())

                if analysis.get('entry_signal'):
                    confidence = self._calculate_asian_range_confidence(analysis)

                    if confidence.value in ['A+', 'A', 'B+']:
                        signal = {
                            'type': SignalType.ASIAN_RANGE_ENTRY,
                            'direction': 'long' if signal_type == SignalType.FGD else 'short',
                            'confidence': confidence,
                            'entry_zone': {
                                'asian_high': analysis['asian_range']['high'],
                                'asian_low': analysis['asian_range']['low'],
                                'current_price': analysis['current_price']
                            },
                            'stop_loss': analysis['asian_range']['low'] if signal_type == SignalType.FGD else analysis['asian_range']['high'],
                            'target': analysis['targets'][0]['price'] if analysis.get('targets') else None,
                            'reasoning': f"Asian range entry after {signal_type.value}",
                            'sweep_detected': analysis['sweep_detected'],
                            'entry_candle_time': analysis.get('entry_candle_time')
                        }
                        signals.append(signal)

        except Exception as e:
            print(f"[ERROR] Asian range signal generation failed: {e}")

        return signals

    def _calculate_frd_fgd_confidence(self, pattern: Dict, dmr_context: Dict,
                                     validation: Dict) -> SignalConfidence:
        """Calculate confidence score for FRD/FGD signals."""
        score = 0

        # Grade base score
        grade_scores = {
            SignalGrade.A_PLUS: 90,
            SignalGrade.A: 80,
            SignalGrade.B_PLUS: 70,
            SignalGrade.B: 60
        }
        score += grade_scores.get(pattern.get('signal_grade'), 50)

        # DMR proximity bonus
        if dmr_context.get('proximity_to_dmr') == 'Excellent':
            score += 10
        elif dmr_context.get('proximity_to_dmr') == 'Good':
            score += 5

        # Validation bonus
        if validation.get('level') == ValidationLevel.CONFIRMED_ACYCLIC:
            score += 10
        elif validation.get('level') == ValidationLevel.STRUCTURED:
            score += 5

        # Convert to confidence
        if score >= 95:
            return SignalConfidence.A_PLUS
        elif score >= 85:
            return SignalConfidence.A
        elif score >= 75:
            return SignalConfidence.B_PLUS
        elif score >= 65:
            return SignalConfidence.B
        else:
            return SignalConfidence.C_PLUS

    def _calculate_inside_day_confidence(self, current_day: pd.DataFrame,
                                        prev_day: pd.DataFrame, dmr_levels: Dict,
                                        acb_levels: Dict) -> SignalConfidence:
        """Calculate confidence for inside day breakout signals."""
        score = 60  # Base score

        # Check proximity to DMR levels
        curr_close = current_day.iloc[-1]['close']
        dmr_distance = self._calculate_dmr_distance(curr_close, dmr_levels)
        if dmr_distance < 20:  # Within 20 pips
            score += 15

        # Check ACB level alignment
        for level in acb_levels.get('confirmed', []):
            if abs(curr_close - level['price']) < 30:
                score += 10
                break

        # Volume confirmation
        avg_volume = current_day['tick_volume'].mean()
        prev_avg_volume = prev_day['tick_volume'].mean()
        if avg_volume > prev_avg_volume * 1.2:
            score += 10

        # Convert to confidence
        if score >= 85:
            return SignalConfidence.A_PLUS
        elif score >= 75:
            return SignalConfidence.A
        elif score >= 65:
            return SignalConfidence.B_PLUS
        else:
            return SignalConfidence.B

    def _calculate_pump_dump_confidence(self, pattern_type: str, pattern_data: Dict,
                                       dmr_levels: Dict, acb_levels: Dict) -> SignalConfidence:
        """Calculate confidence for Pump & Dump patterns."""
        score = 65  # Base score

        # DMR level alignment
        if pattern_data.get('at_dmr_level', False):
            score += 15

        # Volume spike confirmation
        if pattern_data.get('volume_spike', 0) > 2.0:  # 2x average volume
            score += 10

        # Reversal confirmation
        if pattern_data.get('reversal_confirmed', False):
            score += 10

        # Pattern type bonus
        if pattern_type == 'equilibrium':
            score += 5  # Highest probability P&D pattern

        # Convert to confidence
        if score >= 90:
            return SignalConfidence.A_PLUS
        elif score >= 80:
            return SignalConfidence.A
        elif score >= 70:
            return SignalConfidence.B_PLUS
        else:
            return SignalConfidence.B

    def _calculate_asian_range_confidence(self, analysis: Dict) -> SignalConfidence:
        """Calculate confidence for Asian range entries."""
        score = 70  # Base score

        # Sweep detection bonus
        if analysis.get('sweep_detected'):
            score += 15

        # Asian range quality
        if analysis.get('asian_range', {}).get('range_pips', 0) < 30:
            score += 10  # Tight ranges are better

        # Session alignment bonus
        current_hour = datetime.now().hour
        if 2 <= current_hour <= 6:  # Optimal entry window
            score += 10

        # Convert to confidence
        if score >= 90:
            return SignalConfidence.A_PLUS
        elif score >= 80:
            return SignalConfidence.A
        elif score >= 70:
            return SignalConfidence.B_PLUS
        else:
            return SignalConfidence.B

    def _detect_equilibrium_pd(self, candles: pd.DataFrame, dmr_levels: Dict) -> Optional[Dict]:
        """Detect equilibrium pump & dump pattern."""
        try:
            # Look for sharp push to DMR level followed by reversal
            push_candle = candles.iloc[0]
            reversal_candles = candles.iloc[1:]

            # Check if price pushed to DMR level
            dmr_levels_list = []
            for period in ['daily', '3day', 'weekly']:
                if dmr_levels.get(period, {}).get('high'):
                    dmr_levels_list.append(dmr_levels[period]['high']['price'])
                if dmr_levels.get(period, {}).get('low'):
                    dmr_levels_list.append(dmr_levels[period]['low']['price'])

            for dmr_level in dmr_levels_list:
                if abs(push_candle['high'] - dmr_level) < 0.00020:  # Within 20 pips
                    # Check for reversal
                    if reversal_candles['close'].iloc[-1] < push_candle['close'] * 0.997:
                        return {
                            'direction': 'short',
                            'entry_zone': {
                                'entry': push_candle['high'],
                                'stop': push_candle['high'] + 0.0010
                            },
                            'stop_loss': push_candle['high'] + 0.0010,
                            'target': self._find_next_dmr_level(push_candle['high'], 'short', dmr_levels),
                            'at_dmr_level': True,
                            'volume_spike': push_candle['tick_volume'] / candles['tick_volume'].mean(),
                            'reversal_confirmed': True
                        }
        except:
            pass
        return None

    def _detect_m_top_w_bottom(self, candles: pd.DataFrame) -> Optional[Dict]:
        """Detect M-top or W-bottom patterns."""
        # Simplified M/W pattern detection
        try:
            highs = candles['high'].values
            lows = candles['low'].values

            # Look for M-top (double top)
            if len(highs) >= 4:
                first_top = max(highs[:2])
                second_top = max(highs[2:4])

                if abs(first_top - second_top) < 0.00015:  # Within 15 pips
                    if candles['close'].iloc[-1] < min(lows[:4]):
                        return {
                            'direction': 'short',
                            'entry_zone': {'entry': second_top, 'stop': second_top + 0.0010},
                            'stop_loss': second_top + 0.0010,
                            'target': candles['low'].iloc[-1] - 0.0020,
                            'at_dmr_level': False,
                            'volume_spike': 1.0,
                            'reversal_confirmed': True
                        }
        except:
            pass
        return None

    def _detect_lower_range_pd(self, candles: pd.DataFrame) -> Optional[Dict]:
        """Detect extreme sell-off followed by reversal."""
        try:
            # Look for big down candle followed by reversal
            down_candle = candles.iloc[0]

            if down_candle['close'] < down_candle['open'] * 0.997:  # At least 30 pips down
                # Check next candles for reversal
                for i in range(1, min(4, len(candles))):
                    if candles['close'].iloc[i] > down_candle['close'] * 1.001:
                        return {
                            'direction': 'long',
                            'entry_zone': {'entry': candles['close'].iloc[i], 'stop': down_candle['low']},
                            'stop_loss': down_candle['low'],
                            'target': down_candle['high'],
                            'at_dmr_level': False,
                            'volume_spike': down_candle['tick_volume'] / candles['tick_volume'].mean(),
                            'reversal_confirmed': True
                        }
        except:
            pass
        return None

    def _prioritize_signals(self, signals: Dict) -> List[Dict]:
        """Prioritize all signals by confidence and setup quality."""
        all_signals = []

        # Collect all signals with their types
        signal_groups = [
            ('FRD/FGD', signals['frd_fgd_signals']),
            ('Inside Day', signals['inside_day_signals']),
            ('Pump & Dump', signals['pump_dump_signals']),
            ('Asian Range', signals['asian_range_signals'])
        ]

        for group_name, group_signals in signal_groups:
            for signal in group_signals:
                signal['group'] = group_name
                all_signals.append(signal)

        # Sort by confidence score
        confidence_order = {
            SignalConfidence.A_PLUS: 4,
            SignalConfidence.A: 3,
            SignalConfidence.B_PLUS: 2,
            SignalConfidence.B: 1,
            SignalConfidence.C_PLUS: 0
        }

        all_signals.sort(key=lambda x: (
            confidence_order.get(x['confidence'], 0),
            x.get('grade', SignalGrade.B).value if 'grade' in x else 0
        ), reverse=True)

        return all_signals[:5]  # Return top 5 signals

    def _generate_signal_summary(self, signals: Dict) -> str:
        """Generate human-readable summary of top signals."""
        summary = f"\n{'='*60}\n"
        summary += f"SIGNAL SUMMARY - {signals['symbol']}\n"
        summary += f"{'='*60}\n"
        summary += f"Current Price: {signals['current_price']:.5f}\n"
        summary += f"Market Phase: {signals['market_phase']}\n\n"

        if signals['top_signals']:
            summary += "TOP SIGNALS:\n\n"

            for i, signal in enumerate(signals['top_signals'][:3], 1):
                summary += f"{i}. {signal['type'].value}\n"
                summary += f"   Confidence: {signal['confidence'].value}\n"
                summary += f"   Direction: {signal.get('direction', 'N/A')}\n"
                summary += f"   Entry: {signal.get('entry_zone', {}).get('entry', signal.get('entry_zone', {}).get('high', 'N/A')):.5f}\n"
                summary += f"   Stop: {signal.get('stop_loss', 'N/A')}\n"
                summary += f"   Target: {signal.get('target', 'N/A')}\n"
                summary += f"   Reason: {signal.get('reasoning', 'N/A')}\n\n"
        else:
            summary += "No high-confidence signals detected at this time.\n"

        return summary

    # Helper methods
    def _calculate_dmr_distance(self, price: float, dmr_levels: Dict) -> float:
        """Calculate distance to nearest DMR level in pips."""
        distances = []

        for period in ['daily', '3day', 'weekly']:
            if dmr_levels.get(period, {}).get('high'):
                distances.append(abs(price - dmr_levels[period]['high']['price']) * 10000)
            if dmr_levels.get(period, {}).get('low'):
                distances.append(abs(price - dmr_levels[period]['low']['price']) * 10000)

        return min(distances) if distances else 999

    def _find_nearest_dmr(self, price: float, direction: str, dmr_levels: Dict) -> float:
        """Find nearest DMR level in the direction of trade."""
        targets = []

        for period in ['daily', '3day', 'weekly']:
            if direction == 'long':
                if dmr_levels.get(period, {}).get('high'):
                    targets.append(dmr_levels[period]['high']['price'])
            else:
                if dmr_levels.get(period, {}).get('low'):
                    targets.append(dmr_levels[period]['low']['price'])

        # Return nearest target
        if targets:
            return min(targets) if direction == 'long' else max(targets)
        return price

    def _find_next_dmr_level(self, price: float, direction: str, dmr_levels: Dict) -> float:
        """Find the next DMR level beyond current price."""
        levels = []

        for period in ['daily', '3day', 'weekly']:
            if dmr_levels.get(period, {}).get('high'):
                levels.append(dmr_levels[period]['high']['price'])
            if dmr_levels.get(period, {}).get('low'):
                levels.append(dmr_levels[period]['low']['price'])

        # Filter levels in the direction of trade
        if direction == 'short':
            levels = [l for l in levels if l > price]
        else:
            levels = [l for l in levels if l < price]

        return min(levels) if levels else price

    def _determine_breakout_direction(self, current_day: pd.DataFrame, dmr_levels: Dict) -> str:
        """Determine likely breakout direction from inside day."""
        current_price = current_day.iloc[-1]['close']

        # Check proximity to DMR levels
        nearest_high = self._find_nearest_dmr(current_price, 'long', dmr_levels)
        nearest_low = self._find_nearest_dmr(current_price, 'short', dmr_levels)

        dist_to_high = abs(nearest_high - current_price) * 10000
        dist_to_low = abs(current_price - nearest_low) * 10000

        return 'long' if dist_to_high < dist_to_low else 'short'

    def _analyze_dmr_context(self, df_h1: pd.DataFrame, dmr_levels: Dict) -> Dict:
        """Analyze current position relative to DMR levels."""
        current_price = df_h1.iloc[-1]['close']

        return {
            'nearest_dmr_high': self._find_nearest_dmr(current_price, 'long', dmr_levels),
            'nearest_dmr_low': self._find_nearest_dmr(current_price, 'short', dmr_levels),
            'distance_to_high': self._calculate_dmr_distance(current_price, dmr_levels),
            'dmr_breached': self._check_dmr_breach(df_h1, dmr_levels),
            'rotation_status': self._determine_rotation_status(current_price, dmr_levels)
        }

    def _analyze_acb_context(self, df_h1: pd.DataFrame, acb_levels: Dict) -> Dict:
        """Analyze ACB levels and their relevance."""
        current_price = df_h1.iloc[-1]['close']

        return {
            'nearest_acb': self._find_nearest_acb(current_price, acb_levels),
            'acb_support_resistance': self._check_acb_support_resistance(current_price, acb_levels),
            'recent_acb_breach': self._check_recent_acb_breach(df_h1, acb_levels)
        }

    def _analyze_session_context(self, sessions: Dict) -> Dict:
        """Analyze current session behavior."""
        return {
            'current_session': self._get_current_session(),
            'session_behavior': sessions.get('sessions', {}),
            'manipulation_detected': sessions.get('manipulation_detected', False)
        }

    def _analyze_volume_context(self, df_h1: pd.DataFrame) -> Dict:
        """Analyze volume patterns."""
        recent_volume = df_h1.tail(10)['tick_volume']
        historical_volume = df_h1['tick_volume']

        return {
            'current_volume': recent_volume.iloc[-1],
            'avg_volume_10': recent_volume.mean(),
            'avg_volume_historical': historical_volume.mean(),
            'volume_ratio': recent_volume.iloc[-1] / historical_volume.mean(),
            'volume_spike_detected': recent_volume.iloc[-1] > historical_volume.mean() * 2
        }

    # Additional helper methods
    def _check_dmr_breach(self, df_h1: pd.DataFrame, dmr_levels: Dict) -> bool:
        """Check if any DMR levels have been breached today."""
        today = df_h1.index.date[-1]
        today_candles = df_h1[df_h1.index.date == today]

        for period in ['daily', '3day', 'weekly']:
            if dmr_levels.get(period, {}).get('high'):
                if (today_candles['high'] > dmr_levels[period]['high']['price']).any():
                    return True
        return False

    def _determine_rotation_status(self, price: float, dmr_levels: Dict) -> str:
        """Determine current rotation status relative to DMR."""
        if self._calculate_dmr_distance(price, dmr_levels) < 10:
            return "At DMR"
        elif self._calculate_dmr_distance(price, dmr_levels) < 50:
            return "Rotating to DMR"
        else:
            return "Far from DMR"

    def _find_nearest_acb(self, price: float, acb_levels: Dict) -> Optional[float]:
        """Find nearest ACB level."""
        all_levels = []

        for level in acb_levels.get('confirmed', []):
            all_levels.append(level['price'])
        for level in acb_levels.get('potential', []):
            all_levels.append(level['price'])

        if not all_levels:
            return None

        return min(all_levels, key=lambda x: abs(price - x))

    def _check_acb_support_resistance(self, price: float, acb_levels: Dict) -> Dict:
        """Check if price is at ACB support/resistance."""
        nearest_acb = self._find_nearest_acb(price, acb_levels)

        if not nearest_acb:
            return {'support': None, 'resistance': None}

        return {
            'support': nearest_acb if nearest_acb < price else None,
            'resistance': nearest_acb if nearest_acb > price else None,
            'distance_pips': abs(price - nearest_acb) * 10000
        }

    def _check_recent_acb_breach(self, df_h1: pd.DataFrame, acb_levels: Dict) -> bool:
        """Check if any ACB levels have been recently breached."""
        recent_candles = df_h1.tail(24)  # Last day

        for level in acb_levels.get('confirmed', []):
            if (recent_candles['high'] > level['price']).any() or (recent_candles['low'] < level['price']).any():
                return True
        return False

    def _get_current_session(self) -> str:
        """Get current trading session."""
        hour = datetime.now().hour

        if 0 <= hour < 6:
            return "Asian"
        elif 6 <= hour < 14:
            return "London"
        elif 13 <= hour < 22:
            return "New York"
        else:
            return "Overlap"