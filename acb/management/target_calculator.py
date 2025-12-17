"""
DMR-Aware Target Calculator - Smart Target Selection
=============================================

Prioritizes targets based on market structure and proximity:
1. Asian Range Extremes (Low-hanging fruit)
2. Today's HOD/LOD (Secondary targets)
3. PDH/PDL (Main DMR rotation)
4. 3-Day/Weekly levels (Extended targets)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TargetType(Enum):
    """Types of trading targets."""
    ASIAN_HIGH = "Asian High (Primary)"
    ASIAN_LOW = "Asian Low (Primary)"
    TODAY_HIGH = "Today's High"
    TODAY_LOW = "Today's Low"
    PDH = "Previous Day High"
    PDL = "Previous Day Low"
    THREE_DAY_HIGH = "3-Day High"
    THREE_DAY_LOW = "3-Day Low"
    WEEKLY_HIGH = "Weekly High"
    WEEKLY_LOW = "Weekly Low"


class TargetPriority(Enum):
    """Priority levels for targets."""
    PRIMARY = "Primary (Low-hanging fruit)"
    SECONDARY = "Secondary"
    MAIN = "Main DMR"
    EXTENDED = "Extended"


class DMRTargetCalculator:
    """
    Calculates optimal profit targets based on DMR and market structure.

    Key concept: Prioritize closer, higher probability targets first.
    """

    def __init__(self):
        self.min_target_distance = 5  # Minimum 5 pips for a valid target
        self.asian_session_end = 5     # 05:00 UTC

    def calculate_targets(self,
                         entry_price: float,
                         direction: str,  # 'long' or 'short'
                         dmr_levels: Dict,
                         asian_range: Optional[Dict] = None,
                         current_levels: Optional[Dict] = None,
                         market_phase: Optional[str] = None) -> Dict:
        """
        Calculate prioritized list of targets based on simplified logic.

        Logic:
        1. If Asian High/Low = Today's HOD/LOD at entry:
           - Asian Range (Low-hanging fruit) → PDH/PDL (Main target)
        2. If Asian High/Low ≠ Today's HOD/LOD at entry:
           - Asian Range → Today's HOD/LOD → PDH/PDL

        Args:
            entry_price: Entry price of the trade
            direction: 'long' or 'short'
            dmr_levels: All DMR levels from calculator
            asian_range: Asian session range if available
            current_levels: Today's HOD/LOD
            market_phase: Current market phase for context

        Returns:
            Dictionary with prioritized targets and analysis
        """

        targets = []
        asian_equals_hod = False

        # Check if Asian High/Low equals Today's HOD/LOD
        if asian_range and current_levels:
            if direction == 'long':
                asian_equals_hod = abs(asian_range['high'] - current_levels['high']['price']) < 0.00001
            else:
                asian_equals_hod = abs(asian_range['low'] - current_levels['low']['price']) < 0.00001

        # 1. Asian Range Targets (PRIMARY - Low-hanging fruit)
        if asian_range:
            if direction == 'long':
                target = self._create_target(
                    price=asian_range['high'],
                    type=TargetType.ASIAN_HIGH,
                    priority=TargetPriority.PRIMARY,
                    distance_pips=(asian_range['high'] - entry_price) * 10000,
                    close_percentage=100 if asian_equals_hod else 50
                )
                if target['distance_pips'] >= self.min_target_distance:
                    targets.append(target)

            elif direction == 'short':
                target = self._create_target(
                    price=asian_range['low'],
                    type=TargetType.ASIAN_LOW,
                    priority=TargetPriority.PRIMARY,
                    distance_pips=(entry_price - asian_range['low']) * 10000,
                    close_percentage=100 if asian_equals_hod else 50
                )
                if target['distance_pips'] >= self.min_target_distance:
                    targets.append(target)

        # 2. Today's HOD/LOD (SECONDARY) - Only if NOT equal to Asian range
        if not asian_equals_hod and current_levels:
            if direction == 'long' and current_levels.get('high'):
                target = self._create_target(
                    price=current_levels['high']['price'],
                    type=TargetType.TODAY_HIGH,
                    priority=TargetPriority.SECONDARY,
                    distance_pips=(current_levels['high']['price'] - entry_price) * 10000,
                    close_percentage=50
                )
                if target['distance_pips'] >= self.min_target_distance:
                    targets.append(target)

            elif direction == 'short' and current_levels.get('low'):
                target = self._create_target(
                    price=current_levels['low']['price'],
                    type=TargetType.TODAY_LOW,
                    priority=TargetPriority.SECONDARY,
                    distance_pips=(entry_price - current_levels['low']['price']) * 10000,
                    close_percentage=50
                )
                if target['distance_pips'] >= self.min_target_distance:
                    targets.append(target)

        # 3. PDH/PDL (MAIN DMR) - Always included as main target
        if dmr_levels.get('daily'):
            if direction == 'long' and dmr_levels['daily'].get('high'):
                target = self._create_target(
                    price=dmr_levels['daily']['high']['price'],
                    type=TargetType.PDH,
                    priority=TargetPriority.MAIN,
                    distance_pips=(dmr_levels['daily']['high']['price'] - entry_price) * 10000,
                    close_percentage=100 if not targets else 100  # Close all remaining if this is only target
                )
                if target['distance_pips'] >= self.min_target_distance:
                    targets.append(target)

            elif direction == 'short' and dmr_levels['daily'].get('low'):
                target = self._create_target(
                    price=dmr_levels['daily']['low']['price'],
                    type=TargetType.PDL,
                    priority=TargetPriority.MAIN,
                    distance_pips=(entry_price - dmr_levels['daily']['low']['price']) * 10000,
                    close_percentage=100 if not targets else 100
                )
                if target['distance_pips'] >= self.min_target_distance:
                    targets.append(target)

        # Sort targets by distance (nearest first)
        targets.sort(key=lambda x: x['distance_pips'])

        # Calculate risk/reward for each target
        for target in targets:
            # Default stop placement based on asian range or recent swing
            if asian_range:
                if direction == 'long':
                    stop_distance = (entry_price - asian_range['low']) * 10000
                else:
                    stop_distance = (asian_range['high'] - entry_price) * 10000
            else:
                # Default 20 pip stop if no asian range
                stop_distance = 20

            target['stop_loss_pips'] = max(stop_distance, 10)  # Minimum 10 pips
            target['risk_reward_ratio'] = target['distance_pips'] / target['stop_loss_pips']
            target['profit_potential'] = target['distance_pips']

        return {
            'targets': targets,
            'entry_price': entry_price,
            'direction': direction,
            'asian_equals_hod': asian_equals_hod,
            'primary_target': targets[0] if targets else None,
            'nearest_target': min(targets, key=lambda x: x['distance_pips']) if targets else None,
            'best_rr_target': max(targets, key=lambda x: x['risk_reward_ratio']) if targets else None,
            'summary': self._generate_summary(targets, direction, asian_equals_hod)
        }

    def _create_target(self, price: float, type: TargetType, priority: TargetPriority, distance_pips: float, close_percentage: int = 0) -> Dict:
        """Create a target dictionary."""
        return {
            'price': price,
            'type': type,
            'priority': priority,
            'distance_pips': distance_pips,
            'close_percentage': close_percentage,
            'probability': self._calculate_hit_probability(type, priority, distance_pips),
            'expected_time': self._estimate_time_to_target(type, priority, distance_pips)
        }

    def _calculate_hit_probability(self, target_type: TargetType, priority: TargetPriority, distance_pips: float) -> float:
        """
        Calculate probability of hitting target based on type and distance.
        """
        base_probability = {
            TargetPriority.PRIMARY: 0.85,    # Asian range targets
            TargetPriority.SECONDARY: 0.65,  # Today's HOD/LOD
            TargetPriority.MAIN: 0.75,       # PDH/PDL
            TargetPriority.EXTENDED: 0.55    # 3-day/weekly
        }.get(priority, 0.5)

        # Adjust for distance (closer = higher probability)
        distance_factor = max(0.3, 1 - (distance_pips / 200))  # Decay over 200 pips

        return min(0.95, base_probability * distance_factor)

    def _estimate_time_to_target(self, target_type: TargetType, priority: TargetPriority, distance_pips: float) -> str:
        """Estimate time to reach target."""
        if distance_pips < 20:
            return "< 4 hours"
        elif distance_pips < 50:
            return "4-12 hours"
        elif distance_pips < 100:
            return "1-2 days"
        else:
            return "2-3 days"

    def _generate_summary(self, targets: List[Dict], direction: str, asian_equals_hod: bool = False) -> str:
        """Generate human-readable summary of targets."""
        if not targets:
            return "No valid targets found"

        summary = f"\n=== TARGETS FOR {direction.upper()} POSITION ===\n"

        if asian_equals_hod:
            summary += f"[Simplified Setup] Asian Range = Today's HOD/LOD\n\n"

        for i, target in enumerate(targets[:3], 1):  # Show top 3
            summary += f"{i}. {target['type'].value} at {target['price']:.5f}\n"
            summary += f"   Distance: {target['distance_pips']:.1f} pips\n"
            summary += f"   Close: {target['close_percentage']}% at target\n"
            summary += f"   Probability: {target['probability']:.0%}\n"
            summary += f"   Risk/Reward: 1:{target['risk_reward_ratio']:.1f}\n\n"

        if asian_equals_hod and len(targets) > 1:
            summary += "Strategy: Take profit at Asian High, then target PDH/PDL\n"

        return summary

    def get_trade_plan(self,
                      entry_price: float,
                      direction: str,
                      dmr_levels: Dict,
                      asian_range: Optional[Dict] = None,
                      current_levels: Optional[Dict] = None) -> Dict:
        """
        Generate complete trade plan with targets and stops.
        """
        targets_analysis = self.calculate_targets(
            entry_price, direction, dmr_levels, asian_range, current_levels
        )

        plan = {
            'entry': {
                'price': entry_price,
                'direction': direction,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            },
            'targets': targets_analysis,
            'management': {
                'partial_profit_levels': self._calculate_partial_profits(targets_analysis['targets']),
                'trail_stop_rules': self._get_trail_stop_rules(asian_range, direction),
                'exit_rules': self._get_exit_rules()
            }
        }

        return plan

    def _calculate_partial_profits(self, targets: List[Dict]) -> List[Dict]:
        """Calculate partial profit taking levels based on target close percentages."""
        if not targets:
            return []

        partials = []
        remaining_percentage = 100

        for i, target in enumerate(targets[:3], 1):  # Max 3 targets
            if target['close_percentage'] > 0 and remaining_percentage > 0:
                close_pct = min(target['close_percentage'], remaining_percentage)
                partials.append({
                    'level': i,
                    'target_price': target['price'],
                    'close_percentage': close_pct,
                    'reason': f"Close {close_pct}% at {target['type'].value}"
                })
                remaining_percentage -= close_pct

        return partials

    def _get_trail_stop_rules(self, asian_range: Optional[Dict], direction: str) -> Dict:
        """Get trailing stop rules."""
        if asian_range:
            if direction == 'long':
                return {
                    'initial_stop': asian_range['low'],
                    'trail_after_pips': 15,
                    'trail_distance': 10
                }
            else:
                return {
                    'initial_stop': asian_range['high'],
                    'trail_after_pips': 15,
                    'trail_distance': 10
                }

        return {
            'initial_stop': None,
            'use_atr': True,
            'atr_multiplier': 2.0
        }

    def _get_exit_rules(self) -> List[str]:
        """Get exit rules for the trade."""
        return [
            "Exit if market phase shifts against position",
            "Exit if reversal pattern appears at target",
            "Exit if stop loss is hit",
            "Take partial profits at each target level"
        ]