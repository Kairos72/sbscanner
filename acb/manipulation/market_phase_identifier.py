"""
Market Phase Identifier - Phase 3 Component
==========================================

Identifies the current market phase based on smart money activity:
- ACCUMULATION: Smart money building positions
- MANIPULATION: Stop hunting, false breakouts
- DISTRIBUTION: Smart money exiting positions
- ROTATION: Return to DMR levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketPhase(Enum):
    """Market phases based on smart money activity."""
    ACCUMULATION = "Accumulation"      # Smart money buying/accumulating
    MANIPULATION = "Manipulation"      # Stop hunts, false breakouts
    DISTRIBUTION = "Distribution"      # Smart money selling/distributing
    ROTATION = "Rotation"             # Moving between DMR levels
    UNCERTAIN = "Uncertain"           # No clear phase identified


class PhaseConfidence(Enum):
    """Confidence level in phase identification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class MarketPhaseIdentifier:
    """
    Identifies market phases by analyzing price action patterns,
    volume behavior, and interaction with key levels.

    Key indicators:
    - Volume patterns (spikes, sustained)
    - Price action at key levels
    - Wicking and reversal patterns
    - Range expansion/contraction
    """

    def __init__(self):
        self.lookback_period = 50        # Candles for analysis
        self.volume_window = 20          # Window for volume average
        self.volatility_window = 14      # Window for volatility
        self.atr_multiplier = 2.0        # ATR multiplier for significant moves

    def identify_market_phase(self,
                            df: pd.DataFrame,
                            dmr_levels: Optional[Dict] = None,
                            acb_levels: Optional[Dict] = None) -> Dict:
        """
        Identify the current market phase and provide detailed analysis.

        Args:
            df: DataFrame with OHLCV data
            dmr_levels: DMR levels from calculator
            acb_levels: ACB levels from detector

        Returns:
            Dictionary with phase information and analysis
        """
        # Calculate indicators
        indicators = self._calculate_indicators(df)

        # Analyze recent price action
        recent_candles = df.tail(self.lookback_period)
        analysis = self._analyze_price_action(recent_candles, dmr_levels, acb_levels)

        # Determine phase based on all evidence
        phase_result = self._determine_phase(indicators, analysis, dmr_levels)

        # Generate detailed report
        return {
            'current_phase': phase_result['phase'],
            'confidence': phase_result['confidence'],
            'evidence': phase_result['evidence'],
            'key_indicators': indicators,
            'price_analysis': analysis,
            'phase_description': self._get_phase_description(phase_result['phase']),
            'expected_behavior': self._get_expected_behavior(phase_result['phase']),
            'trading_implications': self._get_trading_implications(phase_result['phase'])
        }

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate various indicators for phase identification.
        """
        # Volume indicators
        volume_sma = df['tick_volume'].rolling(window=self.volume_window).mean()
        volume_spike_threshold = volume_sma * 1.5
        recent_volume_ratio = df['tick_volume'].iloc[-1] / volume_sma.iloc[-1]

        # Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.volatility_window).mean()

        # Price range analysis
        recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
        historical_range = df['high'].tail(self.lookback_period).max() - df['low'].tail(self.lookback_period).min()
        range_expansion = recent_range / historical_range

        # Trend strength
        price_change = df['close'].diff()
        trend_strength = abs(price_change.rolling(window=14).mean())

        # Wick analysis
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        total_wick = upper_wick + lower_wick
        avg_wick_ratio = (total_wick / (df['high'] - df['low'])).tail(20).mean()

        return {
            'volume_ratio': recent_volume_ratio,
            'atr': atr.iloc[-1],
            'range_expansion': range_expansion,
            'trend_strength': trend_strength.iloc[-1],
            'avg_wick_ratio': avg_wick_ratio,
            'price_position': (df['close'].iloc[-1] - df['close'].tail(self.lookback_period).min()) / \
                           (df['close'].tail(self.lookback_period).max() - df['close'].tail(self.lookback_period).min())
        }

    def _analyze_price_action(self,
                            df: pd.DataFrame,
                            dmr_levels: Optional[Dict],
                            acb_levels: Optional[Dict]) -> Dict:
        """
        Analyze recent price action patterns.
        """
        patterns = []

        # Check for accumulation patterns
        if self._detect_accumulation_pattern(df):
            patterns.append('accumulation')

        # Check for distribution patterns
        if self._detect_distribution_pattern(df):
            patterns.append('distribution')

        # Check for manipulation patterns
        if self._detect_manipulation_pattern(df, dmr_levels):
            patterns.append('manipulation')

        # Check for rotation patterns
        if self._detect_rotation_pattern(df, dmr_levels):
            patterns.append('rotation')

        # Analyze level interactions
        level_interactions = self._analyze_level_interactions(df, dmr_levels, acb_levels)

        # Check for reversals at key levels
        reversals = self._detect_level_reversals(df, dmr_levels, acb_levels)

        return {
            'patterns': patterns,
            'level_interactions': level_interactions,
            'reversals_at_levels': reversals,
            'range_tightening': self._check_range_tightening(df),
            'volume_profile': self._analyze_volume_profile(df)
        }

    def _detect_accumulation_pattern(self, df: pd.DataFrame) -> bool:
        """
        Detect signs of accumulation (smart money buying).
        """
        # Price is holding above support with buying on dips
        recent_low = df['low'].tail(10).min()
        current_price = df['close'].iloc[-1]

        # Check for higher lows
        lows = df['low'].tail(20)
        higher_lows = all(lows.iloc[i] >= lows.iloc[i-5] for i in range(5, len(lows)))

        # Volume on up days heavier than down days
        up_candles = df[df['close'] > df['open']]
        down_candles = df[df['close'] < df['open']]

        if len(up_candles) > 0 and len(down_candles) > 0:
            up_volume = up_candles['tick_volume'].tail(10).mean()
            down_volume = down_candles['tick_volume'].tail(10).mean()
            volume_bias = up_volume > down_volume * 1.2
        else:
            volume_bias = False

        # Price is not making new lows
        above_recent_low = current_price > recent_low

        return higher_lows and volume_bias and above_recent_low

    def _detect_distribution_pattern(self, df: pd.DataFrame) -> bool:
        """
        Detect signs of distribution (smart money selling).
        """
        # Price is struggling at resistance with selling on rallies
        recent_high = df['high'].tail(10).max()
        current_price = df['close'].iloc[-1]

        # Check for lower highs
        highs = df['high'].tail(20)
        lower_highs = all(highs.iloc[i] <= highs.iloc[i-5] for i in range(5, len(highs)))

        # Volume on down days heavier than up days
        up_candles = df[df['close'] > df['open']]
        down_candles = df[df['close'] < df['open']]

        if len(up_candles) > 0 and len(down_candles) > 0:
            up_volume = up_candles['tick_volume'].tail(10).mean()
            down_volume = down_candles['tick_volume'].tail(10).mean()
            volume_bias = down_volume > up_volume * 1.2
        else:
            volume_bias = False

        # Price is not making new highs
        below_recent_high = current_price < recent_high

        return lower_highs and volume_bias and below_recent_high

    def _detect_manipulation_pattern(self, df: pd.DataFrame, dmr_levels: Optional[Dict]) -> bool:
        """
        Detect signs of manipulation (stop hunts, false breakouts).
        """
        # Look for quick reversals after breaking levels
        if not dmr_levels:
            return False

        # Check recent candles for false breakouts
        for i in range(5, 0, -1):
            candle = df.iloc[-i]
            prev_candle = df.iloc[-i-1]

            # Check if candle broke a level but quickly reversed
            for level_type, levels in dmr_levels.items():
                if isinstance(levels, dict):
                    for level_name, level_info in levels.items():
                        if level_info and isinstance(level_info, dict) and 'price' in level_info:
                            level_price = level_info['price']
                        elif isinstance(level_info, (int, float)):
                            level_price = level_info

                            # False breakout above
                            if (candle['high'] > level_price and
                                prev_candle['close'] < level_price and
                                candle['close'] < level_price):
                                return True

                            # False breakout below
                            if (candle['low'] < level_price and
                                prev_candle['close'] > level_price and
                                candle['close'] > level_price):
                                return True

        # Check for large wicks with quick reversals
        wick_ratio = ((df['high'] - df['low']) - abs(df['close'] - df['open'])) / (df['high'] - df['low'])
        large_wicks = (wick_ratio.tail(10) > 0.6).sum()

        return large_wicks >= 3

    def _detect_rotation_pattern(self, df: pd.DataFrame, dmr_levels: Optional[Dict]) -> bool:
        """
        Detect rotation between DMR levels.
        """
        if not dmr_levels:
            return False

        # Check if price is moving between DMR levels
        current_price = df['close'].iloc[-1]
        touched_levels = []

        # Check recent candles touching DMR levels
        for i in range(20):
            candle = df.iloc[-i-1]
            for level_type, levels in dmr_levels.items():
                if isinstance(levels, dict):
                    for level_name, level_info in levels.items():
                        if level_info and isinstance(level_info, dict) and 'price' in level_info:
                            level_price = level_info['price']
                        elif isinstance(level_info, (int, float)):
                            level_price = level_info
                            if (candle['low'] <= level_price <= candle['high'] and
                                level_price not in touched_levels):
                                touched_levels.append(level_price)

        # Rotation if multiple DMR levels were touched
        return len(touched_levels) >= 2

    def _analyze_level_interactions(self,
                                  df: pd.DataFrame,
                                  dmr_levels: Optional[Dict],
                                  acb_levels: Optional[Dict]) -> Dict:
        """
        Analyze how price is interacting with key levels.
        """
        interactions = {
            'dmr_respects': 0,
            'dmr_breakthroughs': 0,
            'acb_respects': 0,
            'acb_breakthroughs': 0
        }

        # Count level interactions
        for i in range(10, len(df)):
            candle = df.iloc[i]

            # DMR interactions
            if dmr_levels:
                for level_type, levels in dmr_levels.items():
                    if isinstance(levels, dict):
                        for level_name, level_info in levels.items():
                            if level_info and isinstance(level_info, dict) and 'price' in level_info:
                                level_price = level_info['price']
                            elif isinstance(level_info, (int, float)):
                                level_price = level_info
                            else:
                                continue

                                # Check if level acted as support/resistance
                                if (candle['low'] <= level_price <= candle['high'] and
                                    abs(candle['close'] - level_price) > 0.0002):
                                    if (candle['close'] > level_price and candle['open'] > level_price) or \
                                       (candle['close'] < level_price and candle['open'] < level_price):
                                        interactions['dmr_respects'] += 1

            # ACB interactions
            if acb_levels:
                for level_type in acb_levels:
                    if level_type in ['confirmed', 'potential']:
                        for level_info in acb_levels[level_type]:
                            level_price = level_info['price']
                            if abs(candle['low'] - level_price) < 0.0001:
                                interactions['acb_respects'] += 1

        return interactions

    def _detect_level_reversals(self,
                              df: pd.DataFrame,
                              dmr_levels: Optional[Dict],
                              acb_levels: Optional[Dict]) -> List[Dict]:
        """
        Detect reversals at key levels.
        """
        reversals = []

        for i in range(5, len(df)):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            next_candle = df.iloc[i+1] if i+1 < len(df) else None

            # Check for reversal at DMR
            if dmr_levels:
                for level_type, levels in dmr_levels.items():
                    if isinstance(levels, dict):
                        for level_name, level_info in levels.items():
                            if level_info and isinstance(level_info, dict) and 'price' in level_info:
                                level_price = level_info['price']
                            elif isinstance(level_info, (int, float)):
                                level_price = level_info
                            else:
                                continue

                                # Bullish reversal at support
                                if (candle['low'] <= level_price and
                                    candle['close'] > level_price and
                                    prev_candle['close'] < prev_candle['open']):
                                    reversals.append({
                                        'type': 'bullish',
                                        'level': level_price,
                                        'level_type': f"{level_type}_{level_name}",
                                        'time': df.index[i]
                                    })

                                # Bearish reversal at resistance
                                if (candle['high'] >= level_price and
                                    candle['close'] < level_price and
                                    prev_candle['close'] > prev_candle['open']):
                                    reversals.append({
                                        'type': 'bearish',
                                        'level': level_price,
                                        'level_type': f"{level_type}_{level_name}",
                                        'time': df.index[i]
                                    })

        return reversals[-5:] if reversals else []  # Return last 5 reversals

    def _check_range_tightening(self, df: pd.DataFrame) -> bool:
        """
        Check if the range is tightening (compression).
        """
        recent_range = df['high'].tail(10).max() - df['low'].tail(10).min()
        older_range = df['high'].tail(30).max() - df['low'].tail(30).min()

        return recent_range < older_range * 0.6

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Simple volume profile analysis.
        """
        # Find high volume nodes
        volume_profile = {}

        for i in range(len(df)):
            candle = df.iloc[i]
            price_level = round(candle['close'], 4)

            if price_level not in volume_profile:
                volume_profile[price_level] = 0
            volume_profile[price_level] += candle['tick_volume']

        # Find highest volume price
        if volume_profile:
            poc = max(volume_profile.items(), key=lambda x: x[1])
            return {
                'point_of_control': poc[0],
                'poc_volume': poc[1],
                'value_area_high': max(volume_profile.keys()),
                'value_area_low': min(volume_profile.keys())
            }

        return {}

    def _determine_phase(self,
                        indicators: Dict,
                        analysis: Dict,
                        dmr_levels: Optional[Dict]) -> Dict:
        """
        Determine the most likely market phase based on all evidence.
        """
        evidence = []
        phase_scores = {
            MarketPhase.ACCUMULATION: 0,
            MarketPhase.DISTRIBUTION: 0,
            MarketPhase.MANIPULATION: 0,
            MarketPhase.ROTATION: 0
        }

        # Volume evidence
        if indicators['volume_ratio'] > 1.5:
            if 'accumulation' in analysis['patterns']:
                phase_scores[MarketPhase.ACCUMULATION] += 2
                evidence.append("High volume with accumulation patterns")
            elif 'distribution' in analysis['patterns']:
                phase_scores[MarketPhase.DISTRIBUTION] += 2
                evidence.append("High volume with distribution patterns")
            elif 'manipulation' in analysis['patterns']:
                phase_scores[MarketPhase.MANIPULATION] += 3
                evidence.append("High volume with manipulation patterns")

        # Range evidence
        if indicators['range_expansion'] > 1.5:
            phase_scores[MarketPhase.MANIPULATION] += 2
            evidence.append("Range expansion suggests manipulation")
        elif analysis['range_tightening']:
            phase_scores[MarketPhase.ROTATION] += 1
            evidence.append("Range tightening suggests rotation/accumulation")

        # Price position evidence
        if indicators['price_position'] > 0.8:  # Near top of range
            if 'distribution' in analysis['patterns']:
                phase_scores[MarketPhase.DISTRIBUTION] += 2
                evidence.append("Price at range top with distribution signs")
        elif indicators['price_position'] < 0.2:  # Near bottom of range
            if 'accumulation' in analysis['patterns']:
                phase_scores[MarketPhase.ACCUMULATION] += 2
                evidence.append("Price at range bottom with accumulation signs")

        # Pattern evidence
        if 'manipulation' in analysis['patterns']:
            phase_scores[MarketPhase.MANIPULATION] += 3
            evidence.append("Clear manipulation patterns detected")

        if 'rotation' in analysis['patterns']:
            phase_scores[MarketPhase.ROTATION] += 3
            evidence.append("Rotation between DMR levels detected")

        # Determine winning phase
        if max(phase_scores.values()) == 0:
            return {
                'phase': MarketPhase.UNCERTAIN,
                'confidence': PhaseConfidence.LOW,
                'evidence': ["No clear phase indicators"]
            }

        winning_phase = max(phase_scores, key=phase_scores.get)
        score = phase_scores[winning_phase]

        # Determine confidence
        if score >= 6:
            confidence = PhaseConfidence.VERY_HIGH
        elif score >= 4:
            confidence = PhaseConfidence.HIGH
        elif score >= 2:
            confidence = PhaseConfidence.MEDIUM
        else:
            confidence = PhaseConfidence.LOW

        return {
            'phase': winning_phase,
            'confidence': confidence,
            'evidence': evidence
        }

    def _get_phase_description(self, phase: MarketPhase) -> str:
        """
        Get description of the market phase.
        """
        descriptions = {
            MarketPhase.ACCUMULATION: "Smart money is actively buying at support levels. "
                                     "Look for higher lows and volume on up moves.",
            MarketPhase.DISTRIBUTION: "Smart money is distributing (selling) into strength. "
                                     "Look for lower highs and volume on down moves.",
            MarketPhase.MANIPULATION: "Smart money is hunting stops and creating false breakouts. "
                                     "High volatility with deceptive price action.",
            MarketPhase.ROTATION: "Price is rotating between DMR levels as smart money "
                                "positions for the next move. Range-bound behavior.",
            MarketPhase.UNCERTAIN: "No clear market phase identified. Wait for more "
                                  "price development before taking action."
        }

        return descriptions.get(phase, "Unknown phase")

    def _get_expected_behavior(self, phase: MarketPhase) -> str:
        """
        Get expected market behavior for the phase.
        """
        behaviors = {
            MarketPhase.ACCUMULATION: "Expect price to hold support and gradually rise. "
                                     "Look for breakouts above accumulation range.",
            MarketPhase.DISTRIBUTION: "Expect price to fail at resistance and gradually fall. "
                                     "Look for breakdown below distribution range.",
            MarketPhase.MANIPULATION: "Expect volatile, deceptive moves with false breakouts. "
                                     "Be cautious of stop hunts.",
            MarketPhase.ROTATION: "Expect range-bound movement between DMR levels. "
                                "Wait for clear breakout direction.",
            MarketPhase.UNCERTAIN: "Market direction unclear. Monitor key levels for clarity."
        }

        return behaviors.get(phase, "Unknown behavior")

    def _get_trading_implications(self, phase: MarketPhase) -> Dict:
        """
        Get trading implications for the phase.
        """
        implications = {
            MarketPhase.ACCUMULATION: {
                'bias': 'Bullish',
                'entry': 'Buy dips to support',
                'stop_loss': 'Below accumulation low',
                'target': 'Breakout above range',
                'caution': 'False breakdowns possible'
            },
            MarketPhase.DISTRIBUTION: {
                'bias': 'Bearish',
                'entry': 'Sell rallies to resistance',
                'stop_loss': 'Above distribution high',
                'target': 'Breakdown below range',
                'caution': 'False breakouts possible'
            },
            MarketPhase.MANIPULATION: {
                'bias': 'Neutral/Cautious',
                'entry': 'Wait for confirmation',
                'stop_loss': 'Wider stops required',
                'target': 'Quick targets only',
                'caution': 'High risk of stop hunts'
            },
            MarketPhase.ROTATION: {
                'bias': 'Neutral',
                'entry': 'Buy low, sell high in range',
                'stop_loss': 'Outside rotation zone',
                'target': 'Opposite side of range',
                'caution': 'Wait for clear breakout'
            },
            MarketPhase.UNCERTAIN: {
                'bias': 'Stay flat',
                'entry': 'No entry recommended',
                'stop_loss': 'N/A',
                'target': 'N/A',
                'caution': 'Wait for clarity'
            }
        }

        return implications.get(phase, {})