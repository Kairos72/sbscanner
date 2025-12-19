"""
5-Star Setup Prioritizer - Phase 4 Component
==========================================

Prioritizes high-probability trading setups based on:
1. Daily Equilibrium P&D (5 stars)
2. Inside Day B/L (4.5 stars)
3. Daily M-Top/W-Bottom P&D (4 stars)
4. 3-Day Market Cycle (FRD/FGD) (3.5 stars)
5. Coil/Spring compression (3 stars)

Ranking criteria:
- Pattern quality and completion
- DMR level proximity and alignment
- Session timing and alignment
- Market structure context
- Volume confirmation
- Smart money manipulation evidence
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .signal_generator import ACBAwareSignalGenerator, SignalType, SignalConfidence
from ..dmr_calculator import DMRLevelCalculator
from ..session_analyzer import SessionAnalyzer
from ..manipulation.market_phase_identifier import MarketPhaseIdentifier, MarketPhase


class SetupType(Enum):
    """Types of trading setups."""
    DAILY_EQUILIBRIUM_PD = "Daily Equilibrium Pump & Dump"
    INSIDE_DAY_BREAKOUT = "Inside Day Breakout"
    M_TOP_W_BOTTOM_PD = "M-Top/W-Bottom Pump & Dump"
    THREE_DAY_CYCLE = "3-Day Market Cycle (FRD/FGD)"
    COIL_SPRING = "Coil/Spring Compression"


class SetupRating(Enum):
    """Setup quality ratings."""
    FIVE_STAR = "5 Stars (Excellent)"
    FOUR_HALF_STAR = "4.5 Stars (Very Good)"
    FOUR_STAR = "4 Stars (Good)"
    THREE_HALF_STAR = "3.5 Stars (Above Average)"
    THREE_STAR = "3 Stars (Average)"
    TWO_STAR = "2 Stars (Below Average)"


class FiveStarSetupPrioritizer:
    """
    Identifies and prioritizes the highest probability trading setups.
    """

    def __init__(self):
        self.signal_gen = ACBAwareSignalGenerator()
        self.dmr_calc = DMRLevelCalculator()
        self.session_analyzer = SessionAnalyzer()
        self.phase_id = MarketPhaseIdentifier()

        # Rating criteria weights
        self.criteria_weights = {
            'pattern_quality': 0.30,      # How well-formed is the pattern
            'dmr_alignment': 0.25,        # Alignment with DMR levels
            'session_timing': 0.15,       # Optimal session timing
            'market_structure': 0.15,     # Context within market structure
            'volume_confirmation': 0.10,  # Volume spike confirmation
            'manipulation_evidence': 0.05 # Smart money footprints
        }

    def prioritize_setups(self, df_h1: pd.DataFrame, df_d1: pd.DataFrame,
                         symbol: str) -> Dict:
        """
        Identify and prioritize all trading setups for a symbol.

        Returns:
            Dictionary with ranked setups and detailed analysis.
        """
        print(f"\n{'='*60}")
        print(f"PRIORITIZING 5-STAR SETUPS FOR {symbol}")
        print(f"{'='*60}")

        # Get base analysis
        dmr_levels = self.dmr_calc.calculate_all_dmr_levels(df_h1)
        sessions = self.session_analyzer.analyze_session_behavior(df_h1, 72)
        market_phase = self.phase_id.identify_market_phase(df_h1, dmr_levels, {})

        # Detect all setup types
        setups = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'current_price': df_h1.iloc[-1]['close'],
            'market_phase': market_phase['current_phase'].value,

            # Setup types
            'daily_equilibrium_pd': self._detect_daily_equilibrium_pd(df_h1, dmr_levels),
            'inside_day_breakouts': self._detect_inside_day_breakouts(df_h1, dmr_levels),
            'm_top_w_bottom_pd': self._detect_m_top_w_bottom_pd(df_h1, dmr_levels),
            'three_day_cycles': self._detect_three_day_cycles(df_d1, dmr_levels),
            'coil_spring_patterns': self._detect_coil_spring_patterns(df_h1),

            # Market context
            'dmr_context': self._analyze_dmr_context(df_h1, dmr_levels),
            'session_context': self._analyze_session_context(sessions),
            'manipulation_context': self._analyze_manipulation_context(df_h1, market_phase),

            # Ranked results
            'ranked_setups': [],
            'top_setup': None,
            'setup_summary': ''
        }

        # Rate and rank all setups
        setups['ranked_setups'] = self._rate_and_rank_setups(setups)
        setups['top_setup'] = setups['ranked_setups'][0] if setups['ranked_setups'] else None
        setups['setup_summary'] = self._generate_setup_summary(setups)

        return setups

    def _detect_daily_equilibrium_pd(self, df_h1: pd.DataFrame, dmr_levels: Dict) -> List[Dict]:
        """Detect Daily Equilibrium Pump & Dump patterns."""
        setups = []

        try:
            # Look for sharp moves to DMR levels with immediate rejection
            last_48_candles = df_h1.tail(48)

            for i in range(5, len(last_48_candles) - 5):
                window = last_48_candles.iloc[i-5:i+10]  # 5 candles before, 10 after

                # Check for pump to DMR level
                for period in ['daily', 'three_day', 'weekly']:
                    if dmr_levels.get(period, {}).get('high'):
                        dmr_high = dmr_levels[period]['high']['price']
                        pump_candle = window.iloc[5]  # Middle candle

                        # Check if price touched DMR high
                        if abs(pump_candle['high'] - dmr_high) < 0.00020:  # Within 20 pips
                            # Check for dump (reversal)
                            next_candles = window.iloc[6:10]
                            if all(next_candles['close'] < pump_candle['close'] * 0.998):
                                # Calculate rating
                                rating = self._rate_equilibrium_pd(pump_candle, next_candles, dmr_high, period)

                                if rating['stars'] == SetupRating.FIVE_STAR or rating['stars'] == SetupRating.FOUR_HALF_STAR or rating['stars'] == SetupRating.FOUR_STAR:
                                    setup = {
                                        'type': SetupType.DAILY_EQUILIBRIUM_PD,
                                        'direction': 'short',
                                        'rating': rating,
                                        'entry_zone': {
                                            'entry': pump_candle['high'],
                                            'stop': dmr_high + 0.0010
                                        },
                                        'dmr_level': dmr_high,
                                        'dmr_period': period,
                                        'pump_candle': pump_candle,
                                        'volume_spike': pump_candle['tick_volume'] / window['tick_volume'].mean(),
                                        'manipulation_signs': self._detect_manipulation_signs(window),
                                        'probability': rating['probability']
                                    }
                                    setups.append(setup)

                    if dmr_levels.get(period, {}).get('low'):
                        dmr_low = dmr_levels[period]['low']['price']
                        pump_candle = window.iloc[5]

                        # Check for dump to DMR low
                        if abs(pump_candle['low'] - dmr_low) < 0.00020:
                            # Check for pump (reversal up)
                            next_candles = window.iloc[6:10]
                            if all(next_candles['close'] > pump_candle['close'] * 1.002):
                                rating = self._rate_equilibrium_pd(pump_candle, next_candles, dmr_low, period, 'long')

                                if rating['stars'] == SetupRating.FIVE_STAR or rating['stars'] == SetupRating.FOUR_HALF_STAR or rating['stars'] == SetupRating.FOUR_STAR:
                                    setup = {
                                        'type': SetupType.DAILY_EQUILIBRIUM_PD,
                                        'direction': 'long',
                                        'rating': rating,
                                        'entry_zone': {
                                            'entry': pump_candle['low'],
                                            'stop': dmr_low - 0.0010
                                        },
                                        'dmr_level': dmr_low,
                                        'dmr_period': period,
                                        'pump_candle': pump_candle,
                                        'volume_spike': pump_candle['tick_volume'] / window['tick_volume'].mean(),
                                        'manipulation_signs': self._detect_manipulation_signs(window),
                                        'probability': rating['probability']
                                    }
                                    setups.append(setup)

        except Exception as e:
            print(f"[ERROR] Daily equilibrium PD detection failed: {e}")

        return setups

    def _detect_inside_day_breakouts(self, df_h1: pd.DataFrame, dmr_levels: Dict) -> List[Dict]:
        """Detect Inside Day Breakout patterns."""
        setups = []

        try:
            # Check last 5 days for inside day patterns
            last_120_candles = df_h1.tail(120)  # 5 days of H1

            for day_start in range(0, len(last_120_candles) - 48, 24):
                prev_day = last_120_candles.iloc[day_start:day_start+24]
                curr_day = last_120_candles.iloc[day_start+24:day_start+48]

                # Check if current day is inside previous day
                prev_high = prev_day['high'].max()
                prev_low = prev_day['low'].min()
                curr_high = curr_day['high'].max()
                curr_low = curr_day['low'].min()

                if float(curr_high) < float(prev_high) and float(curr_low) > float(prev_low):
                    # Inside day detected
                    direction = self._determine_breakout_direction(curr_day, dmr_levels)
                    rating = self._rate_inside_day(prev_day, curr_day, dmr_levels, direction)

                    if rating['stars'] == SetupRating.FIVE_STAR or rating['stars'] == SetupRating.FOUR_HALF_STAR or rating['stars'] == SetupRating.FOUR_STAR or rating['stars'] == SetupRating.THREE_HALF_STAR:
                        setup = {
                            'type': SetupType.INSIDE_DAY_BREAKOUT,
                            'direction': direction,
                            'rating': rating,
                            'entry_zone': {
                                'high': prev_high,
                                'low': prev_low,
                                'range_pips': (prev_high - prev_low) * 10000
                            },
                            'breakout_level': prev_high if direction == 'long' else prev_low,
                            'dmr_target': self._find_dmr_target(curr_day.iloc[-1]['close'], direction, dmr_levels),
                            'compression_level': (prev_high - prev_low) / prev_high,
                            'volume_buildup': curr_day['tick_volume'].mean() / prev_day['tick_volume'].mean(),
                            'probability': rating['probability']
                        }
                        setups.append(setup)

        except Exception as e:
            print(f"[ERROR] Inside day breakout detection failed: {e}")

        return setups

    def _detect_m_top_w_bottom_pd(self, df_h1: pd.DataFrame, dmr_levels: Dict) -> List[Dict]:
        """Detect M-Top or W-Bottom Pump & Dump patterns."""
        setups = []

        try:
            # Look for M/W patterns near DMR levels
            last_72_candles = df_h1.tail(72)

            for i in range(12, len(last_72_candles) - 12):
                pattern_window = last_72_candles.iloc[i-12:i+12]

                # Detect M-top pattern
                m_top = self._identify_m_top(pattern_window)
                if m_top and self._is_near_dmr_level(m_top['peak'], dmr_levels):
                    rating = self._rate_m_top_w_bottom(m_top, dmr_levels, 'top')

                    if rating['stars'] == SetupRating.FIVE_STAR or rating['stars'] == SetupRating.FOUR_HALF_STAR or rating['stars'] == SetupRating.FOUR_STAR:
                        setup = {
                            'type': SetupType.M_TOP_W_BOTTOM_PD,
                            'direction': 'short',
                            'rating': rating,
                            'entry_zone': {
                                'entry': m_top['valley'],
                                'stop': m_top['peak'] + 0.0010
                            },
                            'pattern_levels': {
                                'left_peak': m_top['left_peak'],
                                'right_peak': m_top['right_peak'],
                                'valley': m_top['valley']
                            },
                            'dmr_resistance': self._find_nearest_dmr(m_top['peak'], 'high', dmr_levels),
                            'symmetry_score': m_top['symmetry'],
                            'volume_pattern': m_top['volume_pattern'],
                            'probability': rating['probability']
                        }
                        setups.append(setup)

                # Detect W-bottom pattern
                w_bottom = self._identify_w_bottom(pattern_window)
                if w_bottom and self._is_near_dmr_level(w_bottom['valley'], dmr_levels):
                    rating = self._rate_m_top_w_bottom(w_bottom, dmr_levels, 'bottom')

                    if rating['stars'] == SetupRating.FIVE_STAR or rating['stars'] == SetupRating.FOUR_HALF_STAR or rating['stars'] == SetupRating.FOUR_STAR:
                        setup = {
                            'type': SetupType.M_TOP_W_BOTTOM_PD,
                            'direction': 'long',
                            'rating': rating,
                            'entry_zone': {
                                'entry': w_bottom['peak'],
                                'stop': w_bottom['valley'] - 0.0010
                            },
                            'pattern_levels': {
                                'left_valley': w_bottom['left_valley'],
                                'right_valley': w_bottom['right_valley'],
                                'peak': w_bottom['peak']
                            },
                            'dmr_support': self._find_nearest_dmr(w_bottom['valley'], 'low', dmr_levels),
                            'symmetry_score': w_bottom['symmetry'],
                            'volume_pattern': w_bottom['volume_pattern'],
                            'probability': rating['probability']
                        }
                        setups.append(setup)

        except Exception as e:
            # Non-critical error, continue without setups
            print(f"[ERROR] M-Top/W-Bottom detection failed: {e}")
            # Uncomment to debug: import traceback; traceback.print_exc()

        return setups

    def _detect_three_day_cycles(self, df_d1: pd.DataFrame, dmr_levels: Dict) -> List[Dict]:
        """Detect 3-Day Market Cycle (FRD/FGD) patterns."""
        setups = []

        try:
            # Get FRD/FGD signals
            signals = self.signal_gen._generate_frd_fgd_signals(df_d1, dmr_levels, {})

            for signal in signals:
                if signal['confidence'].value in ['A+', 'A']:
                    # Calculate rating for 3-day cycle
                    rating = self._rate_three_day_cycle(signal, dmr_levels)

                    if rating['stars'] == SetupRating.FIVE_STAR or rating['stars'] == SetupRating.FOUR_HALF_STAR or rating['stars'] == SetupRating.FOUR_STAR or rating['stars'] == SetupRating.THREE_HALF_STAR:
                        setup = {
                            'type': SetupType.THREE_DAY_CYCLE,
                            'direction': 'long' if signal['type'] == SignalType.FGD_BULLISH else 'short',
                            'rating': rating,
                            'signal': signal,
                            'cycle_day': 1,  # Assuming this is day 1 of the cycle
                            'dmr_alignment': signal['dmr_alignment'],
                            'expected_rotation': self._calculate_expected_rotation(signal, dmr_levels),
                            'cycle_strength': self._calculate_cycle_strength(df_d1),
                            'probability': rating['probability']
                        }
                        setups.append(setup)

        except Exception as e:
            # Non-critical error, continue without setups
            print(f"[ERROR] 3-day cycle detection failed: {e}")
            # Uncomment to debug: import traceback; traceback.print_exc()

        return setups

    def _detect_coil_spring_patterns(self, df_h1: pd.DataFrame) -> List[Dict]:
        """Detect Coil/Spring compression patterns."""
        setups = []

        try:
            # Look for compression patterns
            last_96_candles = df_h1.tail(96)  # 4 days

            for i in range(24, len(last_96_candles) - 24):
                compression_window = last_96_candles.iloc[i-24:i]

                # Calculate compression metrics
                high_trend = self._calculate_trend(compression_window['high'])
                low_trend = self._calculate_trend(compression_window['low'])
                range_contraction = self._calculate_range_contraction(compression_window)

                # Detect coil (both trends converging)
                if abs(high_trend - low_trend) < 0.0001 and range_contraction > 0.5:
                    direction = self._predict_breakout_direction(compression_window)
                    rating = self._rate_coil_spring(compression_window, range_contraction)

                    # Convert rating to numeric value for comparison
                    rating_value = list(SetupRating).index(rating['stars'])
                    threshold_value = list(SetupRating).index(SetupRating.THREE_STAR)

                    if rating_value >= threshold_value:
                        setup = {
                            'type': SetupType.COIL_SPRING,
                            'direction': direction,
                            'rating': rating,
                            'entry_zone': {
                                'high': compression_window['high'].max(),
                                'low': compression_window['low'].min(),
                                'compression_pips': (compression_window['high'].max() - compression_window['low'].min()) * 10000
                            },
                            'compression_ratio': range_contraction,
                            'time_in_compression': len(compression_window),
                            'volume_decline': compression_window['tick_volume'].iloc[-10:].mean() / compression_window['tick_volume'].mean(),
                            'breakout_momentum': self._estimate_breakout_momentum(compression_window),
                            'probability': rating['probability']
                        }
                        setups.append(setup)

        except Exception as e:
            # Non-critical error, continue without setups
            print(f"[ERROR] Coil/Spring detection failed: {e}")
            # Uncomment to debug: import traceback; traceback.print_exc()

        return setups

    def _rate_and_rank_setups(self, setups: Dict) -> List[Dict]:
        """Rate all setups and rank by quality."""
        all_setups = []

        # Collect all setups
        setup_categories = [
            ('Daily Equilibrium P&D', setups['daily_equilibrium_pd']),
            ('Inside Day Breakouts', setups['inside_day_breakouts']),
            ('M-Top/W-Bottom P&D', setups['m_top_w_bottom_pd']),
            ('3-Day Cycles', setups['three_day_cycles']),
            ('Coil/Spring Patterns', setups['coil_spring_patterns'])
        ]

        for category, category_setups in setup_categories:
            for setup in category_setups:
                setup['category'] = category
                all_setups.append(setup)

        # Sort by star rating and probability
        star_order = {
            SetupRating.FIVE_STAR: 5,
            SetupRating.FOUR_HALF_STAR: 4.5,
            SetupRating.FOUR_STAR: 4,
            SetupRating.THREE_HALF_STAR: 3.5,
            SetupRating.THREE_STAR: 3,
            SetupRating.TWO_STAR: 2
        }

        all_setups.sort(key=lambda x: (
            star_order.get(x['rating']['stars'], 0),
            x.get('probability', 0)
        ), reverse=True)

        return all_setups[:10]  # Return top 10 setups

    def _rate_equilibrium_pd(self, pump_candle: pd.Series, reversal_candles: pd.DataFrame,
                            dmr_level: float, period: str, direction: str = 'short') -> Dict:
        """Rate Daily Equilibrium Pump & Dump setup."""
        score = 0

        # Pattern quality
        if abs(pump_candle['high'] - dmr_level) < 0.00010:
            score += 30  # Perfect touch
        elif abs(pump_candle['high'] - dmr_level) < 0.00020:
            score += 25  # Close touch

        # DMR alignment (already at DMR)
        score += 25

        # Volume confirmation
        volume_ratio = pump_candle['tick_volume'] / 1000  # Normalize
        if volume_ratio > 2:
            score += 10
        elif volume_ratio > 1.5:
            score += 7

        # Immediate reversal
        immediate_reversal = all(c['close'] < pump_candle['close'] * 0.998 for c in reversal_candles[:3])
        if immediate_reversal:
            score += 15

        # DMR period importance
        if period == 'daily':
            score += 10
        elif period == '3day':
            score += 5

        # Manipulation evidence
        if pump_candle['high'] - pump_candle['close'] > (pump_candle['open'] - pump_candle['low']) * 1.5:
            score += 5  # Long wick indicates manipulation

        # Convert to stars
        if score >= 90:
            stars = SetupRating.FIVE_STAR
            probability = 0.85
        elif score >= 80:
            stars = SetupRating.FOUR_HALF_STAR
            probability = 0.75
        elif score >= 70:
            stars = SetupRating.FOUR_STAR
            probability = 0.65
        else:
            stars = SetupRating.THREE_HALF_STAR
            probability = 0.55

        return {
            'stars': stars,
            'score': score,
            'probability': probability,
            'strengths': self._get_equilibrium_strengths(score),
            'weaknesses': self._get_equilibrium_weaknesses(score)
        }

    def _rate_inside_day(self, prev_day: pd.DataFrame, curr_day: pd.DataFrame,
                        dmr_levels: Dict, direction: str) -> Dict:
        """Rate Inside Day Breakout setup."""
        score = 0

        # Pattern quality (tight compression)
        range_pips = (prev_day['high'].max() - prev_day['low'].min()) * 10000
        if range_pips < 50:
            score += 30
        elif range_pips < 75:
            score += 25
        elif range_pips < 100:
            score += 20

        # DMR alignment
        curr_price = curr_day.iloc[-1]['close']
        dmr_distance = self._calculate_dmr_distance(curr_price, dmr_levels)
        if dmr_distance < 20:
            score += 25
        elif dmr_distance < 40:
            score += 20

        # Session timing
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10:  # London session optimal
            score += 15
        elif 13 <= current_hour <= 16:  # NY session good
            score += 10

        # Volume buildup
        volume_ratio = curr_day['tick_volume'].mean() / prev_day['tick_volume'].mean()
        if volume_ratio > 1.2:
            score += 10
        elif volume_ratio > 1.1:
            score += 7

        # Market structure context
        curr_day_open = curr_day.iloc[0]['open']
        if direction == 'long' and curr_price > curr_day_open:
            score += 10
        elif direction == 'short' and curr_price < curr_day_open:
            score += 10

        # Convert to stars
        if score >= 85:
            stars = SetupRating.FOUR_HALF_STAR
            probability = 0.75
        elif score >= 75:
            stars = SetupRating.FOUR_STAR
            probability = 0.65
        elif score >= 65:
            stars = SetupRating.THREE_HALF_STAR
            probability = 0.55
        else:
            stars = SetupRating.THREE_STAR
            probability = 0.45

        return {
            'stars': stars,
            'score': score,
            'probability': probability,
            'strengths': self._get_inside_day_strengths(score),
            'weaknesses': self._get_inside_day_weaknesses(score)
        }

    def _rate_m_top_w_bottom(self, pattern: Dict, dmr_levels: Dict, pattern_type: str) -> Dict:
        """Rate M-Top or W-Bottom pattern."""
        score = 0

        # Pattern quality (symmetry)
        if pattern['symmetry'] > 0.9:
            score += 30
        elif pattern['symmetry'] > 0.8:
            score += 25
        elif pattern['symmetry'] > 0.7:
            score += 20

        # DMR alignment
        key_level = pattern['peak'] if pattern_type == 'top' else pattern['valley']
        dmr_distance = self._calculate_level_dmr_distance(key_level, dmr_levels)
        if dmr_distance < 10:
            score += 25
        elif dmr_distance < 20:
            score += 20

        # Volume pattern
        if pattern['volume_pattern'] == 'declining':
            score += 10  # Shows exhaustion

        # Time between peaks/valleys
        if pattern.get('time_between', 0) >= 8 and pattern.get('time_between', 0) <= 16:
            score += 10  # Ideal timing

        # Convert to stars
        if score >= 80:
            stars = SetupRating.FOUR_HALF_STAR
            probability = 0.70
        elif score >= 70:
            stars = SetupRating.FOUR_STAR
            probability = 0.60
        elif score >= 60:
            stars = SetupRating.THREE_HALF_STAR
            probability = 0.50
        else:
            stars = SetupRating.THREE_STAR
            probability = 0.40

        return {
            'stars': stars,
            'score': score,
            'probability': probability,
            'strengths': self._get_mw_strengths(score),
            'weaknesses': self._get_mw_weaknesses(score)
        }

    def _rate_three_day_cycle(self, signal: Dict, dmr_levels: Dict) -> Dict:
        """Rate 3-Day Market Cycle setup."""
        score = 0

        # Signal quality
        if signal['grade'].value == 'A+':
            score += 30
        elif signal['grade'].value == 'A':
            score += 25
        elif signal['grade'].value == 'B+':
            score += 20

        # DMR alignment
        if signal['dmr_alignment'] == 'Excellent':
            score += 25
        elif signal['dmr_alignment'] == 'Good':
            score += 20

        # Market phase
        if signal.get('market_phase') in ['DISTRIBUTION', 'ACCUMULATION']:
            score += 15

        # Volume confirmation
        if signal.get('volume_confirmation'):
            score += 10

        # Historical success rate (would need backtest data)
        score += 10  # Assumed base success

        # Convert to stars
        if score >= 75:
            stars = SetupRating.FOUR_STAR
            probability = 0.65
        elif score >= 65:
            stars = SetupRating.THREE_HALF_STAR
            probability = 0.55
        else:
            stars = SetupRating.THREE_STAR
            probability = 0.45

        return {
            'stars': stars,
            'score': score,
            'probability': probability,
            'strengths': self._get_cycle_strengths(score),
            'weaknesses': self._get_cycle_weaknesses(score)
        }

    def _rate_coil_spring(self, window: pd.DataFrame, compression_ratio: float) -> Dict:
        """Rate Coil/Spring compression pattern."""
        score = 0

        # Compression quality
        if compression_ratio > 0.7:
            score += 30
        elif compression_ratio > 0.5:
            score += 25
        elif compression_ratio > 0.3:
            score += 20

        # Time in compression
        if len(window) >= 16:
            score += 15  # Longer compression = stronger breakout

        # Volume decline
        recent_vol = window['tick_volume'].iloc[-5:].mean()
        avg_vol = window['tick_volume'].mean()
        if recent_vol < avg_vol * 0.8:
            score += 10

        # Price position in range
        current_price = window.iloc[-1]['close']
        range_mid = (window['high'].max() + window['low'].min()) / 2
        if abs(current_price - range_mid) < (window['high'].max() - window['low'].min()) * 0.1:
            score += 10

        # Convert to stars
        if score >= 70:
            stars = SetupRating.THREE_HALF_STAR
            probability = 0.55
        elif score >= 60:
            stars = SetupRating.THREE_STAR
            probability = 0.45
        else:
            stars = SetupRating.TWO_STAR
            probability = 0.35

        return {
            'stars': stars,
            'score': score,
            'probability': probability,
            'strengths': self._get_coil_strengths(score),
            'weaknesses': self._get_coil_weaknesses(score)
        }

    def _generate_setup_summary(self, setups: Dict) -> str:
        """Generate human-readable summary of top setups."""
        summary = f"\n{'='*60}\n"
        summary += f"5-STAR SETUP PRIORITY - {setups['symbol']}\n"
        summary += f"{'='*60}\n"
        summary += f"Current Price: {setups['current_price']:.5f}\n"
        summary += f"Market Phase: {setups['market_phase']}\n\n"

        if setups['top_setup']:
            top = setups['top_setup']
            summary += "TOP RATED SETUP:\n\n"
            summary += f"Setup: {top['type'].value}\n"
            summary += f"Rating: {top['rating']['stars'].value}\n"
            summary += f"Direction: {top['direction']}\n"
            summary += f"Probability: {top['probability']:.0%}\n\n"

            if 'entry_zone' in top:
                ez = top['entry_zone']
                if 'high' in ez and 'low' in ez:
                    summary += f"Entry Zone: {ez['low']:.5f} - {ez['high']:.5f}\n"
                    summary += f"Range: {ez.get('range_pips', 0):.1f} pips\n\n"

            summary += f"Key Strengths:\n"
            for strength in top['rating']['strengths'][:3]:
                summary += f"  • {strength}\n"

            summary += f"\nNext Best Setups:\n"
            for i, setup in enumerate(setups['ranked_setups'][1:4], 2):
                summary += f"{i}. {setup['type'].value} - {setup['rating']['stars'].value}\n"
        else:
            summary += "No high-probability setups detected at this time.\n"

        return summary

    # Helper methods
    def _detect_manipulation_signs(self, window: pd.DataFrame) -> List[str]:
        """Detect signs of smart money manipulation."""
        signs = []

        try:
            # Check for volume spike without price continuation
            max_vol_idx = window['tick_volume'].idxmax()
            max_vol_candle = window.loc[max_vol_idx]

            # Get the position in the DataFrame
            position = window.index.get_loc(max_vol_idx)

            if max_vol_candle['tick_volume'] > window['tick_volume'].mean() * 2:
                # Check if there's a next candle
                if position + 1 < len(window):
                    next_candle = window.iloc[position + 1]
                    if (max_vol_candle['close'] - max_vol_candle['open']) * (next_candle['close'] - next_candle['open']) < 0:
                        signs.append("Volume spike with immediate reversal")

            # Check for long wicks
            for _, candle in window.iterrows():
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                body = abs(candle['close'] - candle['open'])

                if upper_wick > body * 2:
                    signs.append("Long upper wick (rejection)")
                if lower_wick > body * 2:
                    signs.append("Long lower wick (support)")

        except Exception as e:
            # If we fail to detect manipulation signs, just return empty list
            print(f"[WARNING] Could not detect manipulation signs: {e}")

        return signs

    def _identify_m_top(self, window: pd.DataFrame) -> Optional[Dict]:
        """Identify M-top pattern in window."""
        # Simplified M-top detection
        highs = window['high'].values
        lows = window['low'].values
        volumes = window['tick_volume'].values

        # Find two peaks
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
               highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))

        if len(peaks) >= 2:
            # Find valley between peaks
            first_peak = peaks[0]
            second_peak = peaks[1]
            valley_idx = window.iloc[first_peak[0]+1:second_peak[0]]['low'].idxmin()
            valley = window.loc[valley_idx, 'low']

            # Check symmetry
            price_diff = abs(first_peak[1] - second_peak[1])
            symmetry = 1 - (price_diff / first_peak[1])

            return {
                'left_peak': first_peak[1],
                'right_peak': second_peak[1],
                'peak': max(first_peak[1], second_peak[1]),
                'valley': valley,
                'symmetry': symmetry,
                'volume_pattern': 'declining' if volumes[second_peak[0]] < volumes[first_peak[0]] else 'increasing'
            }

        return None

    def _identify_w_bottom(self, window: pd.DataFrame) -> Optional[Dict]:
        """Identify W-bottom pattern in window."""
        # Simplified W-bottom detection (inverse of M-top)
        lows = window['low'].values
        highs = window['high'].values
        volumes = window['tick_volume'].values

        # Find two valleys
        valleys = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and \
               lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                valleys.append((i, lows[i]))

        if len(valleys) >= 2:
            # Find peak between valleys
            first_valley = valleys[0]
            second_valley = valleys[1]
            peak_idx = window.iloc[first_valley[0]+1:second_valley[0]]['high'].idxmax()
            peak = window.loc[peak_idx, 'high']

            # Check symmetry
            price_diff = abs(first_valley[1] - second_valley[1])
            symmetry = 1 - (price_diff / first_valley[1])

            return {
                'left_valley': first_valley[1],
                'right_valley': second_valley[1],
                'valley': min(first_valley[1], second_valley[1]),
                'peak': peak,
                'symmetry': symmetry,
                'volume_pattern': 'declining' if volumes[second_valley[0]] < volumes[first_valley[0]] else 'increasing'
            }

        return None

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend of price series."""
        x = np.arange(len(series))
        y = series.values
        return np.polyfit(x, y, 1)[0]

    def _calculate_range_contraction(self, window: pd.DataFrame) -> float:
        """Calculate how much the range has contracted."""
        first_half_range = window.iloc[:len(window)//2]['high'].max() - window.iloc[:len(window)//2]['low'].min()
        second_half_range = window.iloc[len(window)//2:]['high'].max() - window.iloc[len(window)//2:]['low'].min()

        if first_half_range > 0:
            return (first_half_range - second_half_range) / first_half_range
        return 0

    def _predict_breakout_direction(self, window: pd.DataFrame) -> str:
        """Predict likely breakout direction from compression."""
        # Simple momentum-based prediction
        recent_closes = window['close'].tail(5)
        earlier_closes = window['close'].head(5)

        if recent_closes.mean() > earlier_closes.mean():
            return 'long'
        else:
            return 'short'

    def _estimate_breakout_momentum(self, window: pd.DataFrame) -> float:
        """Estimate potential breakout momentum."""
        # Based on compression duration and volume decline
        compression_duration = len(window)
        volume_ratio = window['tick_volume'].tail(5).mean() / window['tick_volume'].head(5).mean()

        return compression_duration * (1 - volume_ratio)

    def _is_near_dmr_level(self, price: float, dmr_levels: Dict, threshold_pips: int = 20) -> bool:
        """Check if price is near a DMR level."""
        for period in ['daily', '3day', 'weekly']:
            if dmr_levels.get(period, {}).get('high'):
                if abs(price - dmr_levels[period]['high']['price']) < threshold_pips / 10000:
                    return True
            if dmr_levels.get(period, {}).get('low'):
                if abs(price - dmr_levels[period]['low']['price']) < threshold_pips / 10000:
                    return True
        return False

    def _find_nearest_dmr(self, price: float, level_type: str, dmr_levels: Dict) -> Optional[float]:
        """Find nearest DMR level of specific type."""
        levels = []

        for period in ['daily', '3day', 'weekly']:
            if dmr_levels.get(period, {}).get(level_type):
                levels.append(dmr_levels[period][level_type]['price'])

        return min(levels, key=lambda x: abs(price - x)) if levels else None

    def _calculate_dmr_distance(self, price: float, dmr_levels: Dict) -> float:
        """Calculate distance to nearest DMR level in pips."""
        distances = []

        for period in ['daily', '3day', 'weekly']:
            if dmr_levels.get(period, {}).get('high'):
                distances.append(abs(price - dmr_levels[period]['high']['price']) * 10000)
            if dmr_levels.get(period, {}).get('low'):
                distances.append(abs(price - dmr_levels[period]['low']['price']) * 10000)

        return min(distances) if distances else 999

    def _calculate_level_dmr_distance(self, level: float, dmr_levels: Dict) -> float:
        """Calculate distance from a specific level to nearest DMR."""
        return self._calculate_dmr_distance(level, dmr_levels)

    def _find_dmr_target(self, price: float, direction: str, dmr_levels: Dict) -> Optional[float]:
        """Find appropriate DMR target based on direction."""
        if direction == 'long':
            targets = []
            for period in ['daily', '3day', 'weekly']:
                if dmr_levels.get(period, {}).get('high'):
                    targets.append(dmr_levels[period]['high']['price'])
            return min(targets) if targets else None
        else:
            targets = []
            for period in ['daily', '3day', 'weekly']:
                if dmr_levels.get(period, {}).get('low'):
                    targets.append(dmr_levels[period]['low']['price'])
            return max(targets) if targets else None

    def _determine_breakout_direction(self, day_data: pd.DataFrame, dmr_levels: Dict) -> str:
        """Determine likely breakout direction from inside day."""
        current_price = day_data.iloc[-1]['close']
        nearest_high = self._find_dmr_target(current_price, 'long', dmr_levels)
        nearest_low = self._find_dmr_target(current_price, 'short', dmr_levels)

        if not nearest_high or not nearest_low:
            return 'long' if day_data.iloc[-1]['close'] > day_data.iloc[0]['open'] else 'short'

        dist_to_high = abs(nearest_high - current_price)
        dist_to_low = abs(current_price - nearest_low)

        return 'long' if dist_to_high < dist_to_low else 'short'

    def _calculate_expected_rotation(self, signal: Dict, dmr_levels: Dict) -> Dict:
        """Calculate expected rotation for 3-day cycle."""
        direction = 'long' if signal['type'] == SignalType.FGD_BULLISH else 'short'

        return {
            'target_level': self._find_dmr_target(signal.get('entry_zone', {}).get('entry', 0), direction, dmr_levels),
            'estimated_time': '1-3 days',
            'confidence': 'High' if signal['confidence'].value in ['A+', 'A'] else 'Medium'
        }

    def _calculate_cycle_strength(self, df_d1: pd.DataFrame) -> float:
        """Calculate strength of 3-day cycle based on recent performance."""
        # Simplified calculation (would need historical data in reality)
        return 0.65  # Assumed base success rate

    def _analyze_dmr_context(self, df_h1: pd.DataFrame, dmr_levels: Dict) -> Dict:
        """Analyze current DMR context."""
        current_price = df_h1.iloc[-1]['close']

        return {
            'nearest_high': self._find_dmr_target(current_price, 'long', dmr_levels),
            'nearest_low': self._find_dmr_target(current_price, 'short', dmr_levels),
            'distance_to_high_pips': self._calculate_dmr_distance(current_price, dmr_levels),
            'rotation_status': 'In Progress' if self._calculate_dmr_distance(current_price, dmr_levels) < 50 else 'Pending'
        }

    def _analyze_session_context(self, sessions: Dict) -> Dict:
        """Analyze current session context."""
        current_hour = datetime.now().hour

        return {
            'current_session': 'Asian' if 0 <= current_hour < 6 else 'London' if 6 <= current_hour < 14 else 'New York',
            'optimal_for_entries': 6 <= current_hour <= 10 or 13 <= current_hour <= 16,
            'session_behavior': sessions.get('sessions', {})
        }

    def _analyze_manipulation_context(self, df_h1: pd.DataFrame, market_phase: Dict) -> Dict:
        """Analyze smart money manipulation context."""
        return {
            'current_phase': market_phase['current_phase'].value,
            'manipulation_likely': market_phase['current_phase'] in [MarketPhase.MANIPULATION, MarketPhase.DISTRIBUTION],
            'phase_description': market_phase.get('phase_description', ''),
            'trading_implications': market_phase.get('trading_implications', [])
        }

    # Strength and weakness methods
    def _get_equilibrium_strengths(self, score: int) -> List[str]:
        """Get strengths for equilibrium PD setup."""
        strengths = []
        if score >= 90:
            strengths.extend(["Perfect DMR touch", "Strong volume spike", "Immediate reversal"])
        elif score >= 75:
            strengths.extend(["Near DMR level", "Volume confirmation", "Clear rejection"])
        return strengths

    def _get_equilibrium_weaknesses(self, score: int) -> List[str]:
        """Get weaknesses for equilibrium PD setup."""
        weaknesses = []
        if score < 80:
            weaknesses.extend(["Not at exact DMR level", "Weak volume", "Slow reversal"])
        return weaknesses

    def _get_inside_day_strengths(self, score: int) -> List[str]:
        """Get strengths for inside day setup."""
        strengths = []
        if score >= 80:
            strengths.extend(["Tight compression", "Near DMR target", "Optimal session timing"])
        elif score >= 70:
            strengths.extend(["Good compression", "Reasonable DMR distance"])
        return strengths

    def _get_inside_day_weaknesses(self, score: int) -> List[str]:
        """Get weaknesses for inside day setup."""
        weaknesses = []
        if score < 75:
            weaknesses.extend(["Wide range", "Far from DMR levels", "Poor timing"])
        return weaknesses

    def _get_mw_strengths(self, score: int) -> List[str]:
        """Get strengths for M-Top/W-Bottom setup."""
        strengths = []
        if score >= 80:
            strengths.extend(["High symmetry", "At DMR resistance/support", "Volume divergence"])
        elif score >= 70:
            strengths.extend(["Good symmetry", "Near key level"])
        return strengths

    def _get_mw_weaknesses(self, score: int) -> List[str]:
        """Get weaknesses for M-Top/W-Bottom setup."""
        weaknesses = []
        if score < 75:
            weaknesses.extend(["Poor symmetry", "Far from DMR", "Weak volume pattern"])
        return weaknesses

    def _get_cycle_strengths(self, score: int) -> List[str]:
        """Get strengths for 3-day cycle setup."""
        strengths = []
        if score >= 75:
            strengths.extend(["Strong signal grade", "Excellent DMR alignment", "Favorable market phase"])
        elif score >= 65:
            strengths.extend(["Good signal quality", "Decent DMR proximity"])
        return strengths

    def _get_cycle_weaknesses(self, score: int) -> List[str]:
        """Get weaknesses for 3-day cycle setup."""
        weaknesses = []
        if score < 70:
            weaknesses.extend(["Weak signal grade", "Poor DMR alignment", "Unfavorable conditions"])
        return weaknesses

    def _get_coil_strengths(self, score: int) -> List[str]:
        """Get strengths for coil/spring setup."""
        strengths = []
        if score >= 70:
            strengths.extend(["High compression", "Long consolidation", "Volume drying up"])
        elif score >= 60:
            strengths.extend(["Good compression", "Decent consolidation"])
        return strengths

    def _get_coil_weaknesses(self, score: int) -> List[str]:
        """Get weaknesses for coil/spring setup."""
        weaknesses = []
        if score < 65:
            weaknesses.extend(["Low compression", "Short consolidation period"])
        return weaknesses