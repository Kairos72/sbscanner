"""
Liquidity Hunt Detector - Phase 3 Component
===========================================

Identifies smart money manipulation patterns:
- Stop hunting below/above key levels
- Asian session liquidity grabs
- Volume spike confirmation
- Wicking patterns analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class LiquidityHuntType(Enum):
    """Types of liquidity hunts detected."""
    ASIAN_LOW_HUNT = "Asian Low Hunt"
    ASIAN_HIGH_HUNT = "Asian High Hunt"
    DMR_STOP_RUN = "DMR Stop Run"
    ACB_LEVEL_HUNT = "ACB Level Hunt"
    ROUND_NUMBER_HUNT = "Round Number Hunt"
    PREVIOUS_SWING_HUNT = "Previous Swing Hunt"


class HuntStrength(Enum):
    """Strength of liquidity hunt."""
    WEAK = "Weak"      # Small wick, low volume
    MODERATE = "Moderate"  # Decent wick, moderate volume
    STRONG = "Strong"     # Large wick, high volume
    EXTREME = "Extreme"   # Massive wick, extreme volume


class LiquidityHuntDetector:
    """
    Detects liquidity hunting patterns by smart money.

    Key concepts:
    - Smart money needs liquidity to fill large orders
    - They hunt stops below key levels to create panic selling
    - Volume spikes confirm manipulation
    - Wicking patterns show failed attempts
    """

    def __init__(self):
        self.asian_session_start = 0      # 00:00 UTC
        self.asian_session_end = 5        # 05:00 UTC
        self.volume_threshold = 1.5       # 1.5x average volume
        self.wick_threshold = 0.0005      # 5 pips minimum wick
        self.round_numbers = [0.57000, 0.57500, 0.58000, 0.58500, 0.59000]  # For NZDUSD

    def detect_liquidity_hunts(self,
                              df: pd.DataFrame,
                              dmr_levels: Optional[Dict] = None,
                              acb_levels: Optional[Dict] = None,
                              asian_range: Optional[Dict] = None) -> Dict:
        """
        Scan for all types of liquidity hunts in the data.

        Args:
            df: DataFrame with OHLCV data
            dmr_levels: DMR levels from calculator
            acb_levels: ACB levels from detector
            asian_range: Asian session range if available

        Returns:
            Dictionary with all detected hunts and analysis
        """
        hunts = []

        # Calculate average volume for context
        avg_volume = df['tick_volume'].rolling(window=20).mean()

        # Check each candle for hunt patterns
        for i in range(10, len(df)):  # Start from 10 to have some history
            candle = df.iloc[i]
            candle_time = df.index[i]
            prev_candles = df.iloc[i-10:i]

            # Detect different types of hunts
            hunt = self._analyze_candle_for_hunt(
                candle, candle_time, prev_candles, avg_volume.iloc[i],
                dmr_levels, acb_levels, asian_range
            )

            if hunt:
                hunts.append(hunt)

        # Analyze hunt patterns
        analysis = {
            'total_hunts': len(hunts),
            'hunts': hunts,
            'hunt_frequency': len(hunts) / len(df) * 100,
            'common_targets': self._analyze_hunt_targets(hunts),
            'success_rate': self._calculate_hunt_success_rate(hunts, df),
            'manipulation_zones': self._identify_manipulation_zones(hunts)
        }

        return analysis

    def _analyze_candle_for_hunt(self,
                                candle: pd.Series,
                                candle_time: datetime,
                                prev_candles: pd.DataFrame,
                                avg_volume: float,
                                dmr_levels: Optional[Dict],
                                acb_levels: Optional[Dict],
                                asian_range: Optional[Dict]) -> Optional[Dict]:
        """
        Analyze a single candle for liquidity hunt patterns.
        """
        hunts = []

        # Check for Asian range hunt
        if asian_range and self._is_asian_session(candle_time):
            hunt = self._check_asian_hunt(candle, asian_range, avg_volume)
            if hunt:
                hunts.append(hunt)

        # Check for DMR stop run
        if dmr_levels:
            hunt = self._check_dmr_hunt(candle, dmr_levels, avg_volume)
            if hunt:
                hunts.append(hunt)

        # Check for ACB level hunt
        if acb_levels:
            hunt = self._check_acb_hunt(candle, acb_levels, avg_volume)
            if hunt:
                hunts.append(hunt)

        # Check for round number hunt
        hunt = self._check_round_number_hunt(candle, avg_volume)
        if hunt:
            hunts.append(hunt)

        # Return the strongest hunt if multiple found
        if hunts:
            return max(hunts, key=lambda x: self._strength_value(x['strength']))

        return None

    def _check_asian_hunt(self, candle: pd.Series, asian_range: Dict, avg_volume: float) -> Optional[Dict]:
        """
        Check for hunt of Asian session range.
        """
        low_hunt = None
        high_hunt = None

        # Check for low hunt
        if candle['low'] < asian_range['low'] - self.wick_threshold:
            wick_size = asian_range['low'] - candle['low']
            volume_ratio = candle['tick_volume'] / avg_volume

            strength = self._calculate_hunt_strength(wick_size, volume_ratio)

            low_hunt = {
                'type': LiquidityHuntType.ASIAN_LOW_HUNT,
                'time': candle.name,
                'level_hunted': asian_range['low'],
                'hunt_price': candle['low'],
                'wick_size': wick_size,
                'close_price': candle['close'],
                'tick_volume_ratio': volume_ratio,
                'strength': strength,
                'recovered': candle['close'] > asian_range['low']
            }

        # Check for high hunt
        if candle['high'] > asian_range['high'] + self.wick_threshold:
            wick_size = candle['high'] - asian_range['high']
            volume_ratio = candle['tick_volume'] / avg_volume

            strength = self._calculate_hunt_strength(wick_size, volume_ratio)

            high_hunt = {
                'type': LiquidityHuntType.ASIAN_HIGH_HUNT,
                'time': candle.name,
                'level_hunted': asian_range['high'],
                'hunt_price': candle['high'],
                'wick_size': wick_size,
                'close_price': candle['close'],
                'tick_volume_ratio': volume_ratio,
                'strength': strength,
                'recovered': candle['close'] < asian_range['high']
            }

        # Return the stronger hunt
        if low_hunt and high_hunt:
            return max([low_hunt, high_hunt], key=lambda x: self._strength_value(x['strength']))
        elif low_hunt:
            return low_hunt
        elif high_hunt:
            return high_hunt

        return None

    def _check_dmr_hunt(self, candle: pd.Series, dmr_levels: Dict, avg_volume: float) -> Optional[Dict]:
        """
        Check for hunt of DMR levels.
        """
        hunts = []

        # Check each DMR level
        for level_type, levels in dmr_levels.items():
            if isinstance(levels, dict):
                for level_name, level_info in levels.items():
                    if level_info and isinstance(level_info, dict) and 'price' in level_info:
                        level_price = level_info['price']
                    elif isinstance(level_info, (int, float)):
                        level_price = level_info

                        # Check low hunt
                        if candle['low'] < level_price - self.wick_threshold:
                            wick_size = level_price - candle['low']
                            volume_ratio = candle['tick_volume'] / avg_volume

                            hunts.append({
                                'type': LiquidityHuntType.DMR_STOP_RUN,
                                'time': candle.name,
                                'level_hunted': level_price,
                                'level_type': f"{level_type}_{level_name}",
                                'hunt_price': candle['low'],
                                'wick_size': wick_size,
                                'close_price': candle['close'],
                                'tick_volume_ratio': volume_ratio,
                                'strength': self._calculate_hunt_strength(wick_size, volume_ratio),
                                'recovered': candle['close'] > level_price
                            })

                        # Check high hunt
                        if candle['high'] > level_price + self.wick_threshold:
                            wick_size = candle['high'] - level_price
                            volume_ratio = candle['tick_volume'] / avg_volume

                            hunts.append({
                                'type': LiquidityHuntType.DMR_STOP_RUN,
                                'time': candle.name,
                                'level_hunted': level_price,
                                'level_type': f"{level_type}_{level_name}",
                                'hunt_price': candle['high'],
                                'wick_size': wick_size,
                                'close_price': candle['close'],
                                'tick_volume_ratio': volume_ratio,
                                'strength': self._calculate_hunt_strength(wick_size, volume_ratio),
                                'recovered': candle['close'] < level_price
                            })

        # Return strongest hunt if any
        if hunts:
            return max(hunts, key=lambda x: self._strength_value(x['strength']))

        return None

    def _check_acb_hunt(self, candle: pd.Series, acb_levels: Dict, avg_volume: float) -> Optional[Dict]:
        """
        Check for hunt of ACB levels (less common but significant).
        """
        hunts = []

        for level_type in ['confirmed', 'potential']:
            if level_type in acb_levels:
                for level_info in acb_levels[level_type]:
                    level_price = level_info['price']

                    # Check for hunt
                    if abs(candle['low'] - level_price) < self.wick_threshold:
                        wick_size = level_price - candle['low']
                        volume_ratio = candle['tick_volume'] / avg_volume

                        hunts.append({
                            'type': LiquidityHuntType.ACB_LEVEL_HUNT,
                            'time': candle.name,
                            'level_hunted': level_price,
                            'level_type': level_type,
                            'hunt_price': candle['low'],
                            'wick_size': wick_size,
                            'close_price': candle['close'],
                            'tick_volume_ratio': volume_ratio,
                            'strength': self._calculate_hunt_strength(wick_size, volume_ratio),
                            'recovered': candle['close'] > level_price
                        })

        if hunts:
            return max(hunts, key=lambda x: self._strength_value(x['strength']))

        return None

    def _check_round_number_hunt(self, candle: pd.Series, avg_volume: float) -> Optional[Dict]:
        """
        Check for hunt of round numbers (psychological levels).
        """
        hunts = []

        for round_num in self.round_numbers:
            # Check low hunt
            if abs(candle['low'] - round_num) < self.wick_threshold:
                wick_size = round_num - candle['low']
                volume_ratio = candle['tick_volume'] / avg_volume

                hunts.append({
                    'type': LiquidityHuntType.ROUND_NUMBER_HUNT,
                    'time': candle.name,
                    'level_hunted': round_num,
                    'hunt_price': candle['low'],
                    'wick_size': wick_size,
                    'close_price': candle['close'],
                    'tick_volume_ratio': volume_ratio,
                    'strength': self._calculate_hunt_strength(wick_size, volume_ratio),
                    'recovered': candle['close'] > round_num
                })

            # Check high hunt
            if abs(candle['high'] - round_num) < self.wick_threshold:
                wick_size = candle['high'] - round_num
                volume_ratio = candle['tick_volume'] / avg_volume

                hunts.append({
                    'type': LiquidityHuntType.ROUND_NUMBER_HUNT,
                    'time': candle.name,
                    'level_hunted': round_num,
                    'hunt_price': candle['high'],
                    'wick_size': wick_size,
                    'close_price': candle['close'],
                    'tick_volume_ratio': volume_ratio,
                    'strength': self._calculate_hunt_strength(wick_size, volume_ratio),
                    'recovered': candle['close'] < round_num
                })

        if hunts:
            return max(hunts, key=lambda x: self._strength_value(x['strength']))

        return None

    def _calculate_hunt_strength(self, wick_size: float, volume_ratio: float) -> HuntStrength:
        """
        Calculate the strength of a liquidity hunt.
        """
        # Convert wick to pips
        wick_pips = wick_size * 10000

        # Determine strength based on wick size and volume
        if wick_pips > 20 and volume_ratio > 2.5:
            return HuntStrength.EXTREME
        elif wick_pips > 15 and volume_ratio > 2.0:
            return HuntStrength.STRONG
        elif wick_pips > 10 and volume_ratio > 1.5:
            return HuntStrength.MODERATE
        else:
            return HuntStrength.WEAK

    def _is_asian_session(self, time: datetime) -> bool:
        """
        Check if time is during Asian session.
        """
        return self.asian_session_start <= time.hour < self.asian_session_end

    def _strength_value(self, strength: HuntStrength) -> int:
        """
        Convert strength enum to numeric value for comparison.
        """
        strength_map = {
            HuntStrength.WEAK: 1,
            HuntStrength.MODERATE: 2,
            HuntStrength.STRONG: 3,
            HuntStrength.EXTREME: 4
        }
        return strength_map.get(strength, 0)

    def _analyze_hunt_targets(self, hunts: List[Dict]) -> Dict:
        """
        Analyze which levels are most commonly hunted.
        """
        target_counts = {}

        for hunt in hunts:
            hunt_type = hunt['type'].value
            target_counts[hunt_type] = target_counts.get(hunt_type, 0) + 1

        return target_counts

    def _calculate_hunt_success_rate(self, hunts: List[Dict], df: pd.DataFrame) -> float:
        """
        Calculate what percentage of hunts lead to immediate reversals.
        """
        if not hunts:
            return 0.0

        successful_hunts = sum(1 for hunt in hunts if hunt.get('recovered', False))
        return (successful_hunts / len(hunts)) * 100

    def _identify_manipulation_zones(self, hunts: List[Dict]) -> List[Dict]:
        """
        Identify price zones where multiple hunts occur.
        """
        # Group hunts by price level
        zones = {}

        for hunt in hunts:
            level = hunt['level_hunted']
            if level not in zones:
                zones[level] = []
            zones[level].append(hunt)

        # Identify zones with multiple hunts
        manipulation_zones = []
        for level, zone_hunts in zones.items():
            if len(zone_hunts) > 1:
                manipulation_zones.append({
                    'price_level': level,
                    'hunt_count': len(zone_hunts),
                    'hunt_types': list(set(h['type'].value for h in zone_hunts)),
                    'avg_strength': np.mean([self._strength_value(h['strength']) for h in zone_hunts]),
                    'times': [h['time'] for h in zone_hunts]
                })

        return sorted(manipulation_zones, key=lambda x: x['hunt_count'], reverse=True)

    def get_manipulation_summary(self, analysis: Dict) -> str:
        """
        Generate human-readable summary of liquidity hunt analysis.
        """
        if analysis['total_hunts'] == 0:
            return "No liquidity hunts detected in the analyzed period."

        summary = f"\n=== LIQUIDITY HUNT ANALYSIS ===\n"
        summary += f"Total hunts detected: {analysis['total_hunts']}\n"
        summary += f"Hunt frequency: {analysis['hunt_frequency']:.1f}% of candles\n"
        summary += f"Hunt success rate: {analysis['success_rate']:.1f}%\n\n"

        summary += "Most targeted levels:\n"
        for target, count in sorted(analysis['common_targets'].items(),
                                   key=lambda x: x[1], reverse=True)[:3]:
            summary += f"  - {target}: {count} hunts\n"

        if analysis['manipulation_zones']:
            summary += "\nKey manipulation zones:\n"
            for zone in analysis['manipulation_zones'][:3]:
                summary += f"  - {zone['price_level']:.5f}: {zone['hunt_count']} hunts\n"

        return summary