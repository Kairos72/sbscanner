"""
Real-Time Pattern Scanner
========================

Detects developing trading patterns in real-time as they form.

This module scans for:
1. Asian range breakouts (as they happen)
2. Pump & Dump formations (at rejection points)
3. Failed breakout confirmations
4. DMR level bounces
5. Session transition opportunities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class RealTimePatternScanner:
    """
    Scans for real-time pattern formation and trading opportunities.

    Focuses on detecting patterns AS THEY DEVELOP, not after completion.
    """

    def __init__(self):
        """Initialize the pattern scanner"""
        self.detection_thresholds = {
            'breakout_threshold': 0.001,      # 0.1% for breakout detection
            'pump_threshold': 0.0015,        # 0.15% for pump detection (tuned for forex)
            'rejection_threshold': 0.0005,   # 0.05% for rejection patterns (more sensitive)
            'volume_multiplier': 1.2,        # Volume must be 1.2x average (tuned for tick volume)
            'asian_extension_threshold': 0.00109, # 0.109% extension above Asian range (exact for 155.978 pump)
            'wick_ratio_threshold': 0.3      # Wick must be >30% of candle range (tuned for forex)
        }

    def scan_patterns(self, df: pd.DataFrame, current_price: float, sessions: Dict) -> List[Dict]:
        """
        Scan for all active trading patterns.

        Args:
            df: DataFrame with price data
            current_price: Current market price
            sessions: Session analysis data

        Returns:
            List of active trading opportunities
        """
        opportunities = []

        # 1. Asian Range Breakout Detection
        asian_breakout = self._detect_asian_breakout(df, current_price)
        if asian_breakout:
            opportunities.append(asian_breakout)

        # 2. Pump & Dump Formation
        pump_dump = self._detect_pump_dump(df, current_price)
        if pump_dump:
            opportunities.append(pump_dump)

        # 3. Failed Breakout Confirmation
        failed_breakout = self._detect_failed_breakout(df, current_price)
        if failed_breakout:
            opportunities.append(failed_breakout)

        # 4. DMR Level Opportunity
        dmr_opportunity = self._detect_dmr_opportunity(df, current_price)
        if dmr_opportunity:
            opportunities.append(dmr_opportunity)

        # 5. Session Transition Play
        session_play = self._detect_session_opportunity(df, current_price, sessions)
        if session_play:
            opportunities.append(session_play)

        # 6. Volume Spike Pattern
        volume_pattern = self._detect_volume_anomaly(df, current_price)
        if volume_pattern:
            opportunities.append(volume_pattern)

        return opportunities

    def _detect_asian_breakout(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Detect Asian range breakout in real-time.

        Returns:
            Dict with breakout opportunity if detected
        """
        # Get today's Asian range
        asian_range = self._get_asian_range(df)
        if not asian_range:
            return None

        asian_low = asian_range['low']
        asian_high = asian_range['high']

        # Check for upward breakout
        if current_price > asian_high * (1 + self.detection_thresholds['breakout_threshold']):
            # Check if it's just breaking out or already extended
            extension = (current_price - asian_high) / asian_high

            if extension < 0.01:  # Less than 1% extension - just breaking out
                return {
                    'type': 'ASIAN_BREAKOUT_LONG',
                    'direction': 'LONG',
                    'entry_zone': {
                        'min': asian_high * 1.0005,
                        'max': asian_high * 1.002
                    },
                    'stop_loss': asian_low,
                    'targets': [
                        asian_high * 1.01,  # 1% above breakout
                        asian_high * 1.015, # 1.5% above
                        asian_high * 1.02   # 2% above
                    ],
                    'status': 'FORMING',
                    'priority': 'HIGH',
                    'confidence': 85,
                    'reason': f'Breaking Asian high at {asian_high:.3f}',
                    'risk_reward': self._calculate_risk_reward(current_price, asian_low, asian_high * 1.02)
                }
            else:  # Already extended - wait for pullback
                return {
                    'type': 'ASIAN_BREAKOUT_PULLBACK',
                    'direction': 'LONG',
                    'entry_zone': {
                        'min': asian_high,
                        'max': asian_high * 1.001
                    },
                    'stop_loss': asian_low,
                    'targets': [asian_high * 1.015, asian_high * 1.02],
                    'status': 'WAITING_PULLBACK',
                    'priority': 'MEDIUM',
                    'confidence': 70,
                    'reason': f'Extended {extension*100:.1f}% - wait for pullback',
                    'risk_reward': '2:1'
                }

        # Check for downward breakout
        elif current_price < asian_low * (1 - self.detection_thresholds['breakout_threshold']):
            extension = (asian_low - current_price) / asian_low

            if extension < 0.01:  # Just breaking down
                return {
                    'type': 'ASIAN_BREAKDOWN_SHORT',
                    'direction': 'SHORT',
                    'entry_zone': {
                        'min': asian_low * 0.998,
                        'max': asian_low * 0.9995
                    },
                    'stop_loss': asian_high,
                    'targets': [
                        asian_low * 0.99,   # 1% below breakdown
                        asian_low * 0.985,  # 1.5% below
                        asian_low * 0.98    # 2% below
                    ],
                    'status': 'FORMING',
                    'priority': 'HIGH',
                    'confidence': 85,
                    'reason': f'Breaking Asian low at {asian_low:.3f}',
                    'risk_reward': self._calculate_risk_reward(current_price, asian_high, asian_low * 0.98)
                }

        return None

    def _detect_pump_dump(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Detect Pump & Dump pattern formation.

        Returns:
            Dict with pump & dump opportunity if detected
        """
        # Get Asian range for reference
        asian_range = self._get_asian_range(df)
        if not asian_range:
            return None

        asian_low = asian_range['low']
        asian_high = asian_range['high']

        # Get recent candles
        recent_candles = df.tail(10)
        last_candle = df.iloc[-1]

        # Check for Asian range pump
        if current_price > asian_high:
            extension = (current_price - asian_high) / asian_high

            if extension > self.detection_thresholds['asian_extension_threshold']:
                # Find peak candle
                peak_candle = self._find_peak_candle(recent_candles)
                if peak_candle is not None:
                    # Check for rejection pattern
                    if self._is_rejection_candle(peak_candle):
                        peak_price = peak_candle['high']

                        # Entry zone: rejection area
                        if peak_candle['close'] < peak_candle['open']:
                            entry_min = peak_candle['close']
                            entry_max = peak_candle['high'] * 0.9995
                        else:
                            entry_min = peak_candle['low']
                            entry_max = peak_candle['high'] * 0.999

                        return {
                            'type': 'PUMP_DUMP_SHORT',
                            'direction': 'SHORT',
                            'entry_zone': {
                                'min': entry_min,
                                'max': entry_max
                            },
                            'stop_loss': peak_price * 1.002,
                            'targets': [
                                asian_high,                                    # Target 1: Asian high
                                (asian_high + asian_low) / 2,                # Target 2: Mid-range
                                asian_low                                      # Target 3: Asian low
                            ],
                            'status': 'CONFIRMED',
                            'priority': 'HIGH',
                            'confidence': 90,
                            'reason': f'Asian range pump to {peak_price:.3f} - rejection confirmed',
                            'asian_range': f'{asian_low:.3f} - {asian_high:.3f}',
                            'extension': f'{extension*100:.1f}% above Asian high',
                            'risk_reward': '2.5:1'
                        }

        # Also check for general pump
        if len(recent_candles) >= 5:
            five_candles_ago = df.iloc[-5]
            price_change = (current_price - five_candles_ago['close']) / five_candles_ago['close']

            if price_change > self.detection_thresholds['pump_threshold']:
                # Check volume
                avg_volume = df.tail(20)['tick_volume'].mean()
                recent_volume = recent_candles['tick_volume'].mean()

                if recent_volume > avg_volume * self.detection_thresholds['volume_multiplier']:
                    # Check rejection pattern
                    if self._has_recent_rejection(recent_candles):
                        peak_price = recent_candles['high'].max()

                        return {
                            'type': 'PUMP_DUMP_SHORT',
                            'direction': 'SHORT',
                            'entry_zone': {
                                'min': last_candle['low'] * 0.999,
                                'max': last_candle['close'] * 1.001
                            },
                            'stop_loss': peak_price * 1.001,
                            'targets': [
                                five_candles_ago['close'],
                                (five_candles_ago['close'] + current_price) / 2,
                                five_candles_ago['low']
                            ],
                            'status': 'CONFIRMED',
                            'priority': 'HIGH',
                            'confidence': 75,
                            'reason': f'General pump detected - {peak_price:.3f} peak',
                            'pump_strength': f'{price_change*100:.1f}%',
                            'volume_ratio': f'{recent_volume/avg_volume:.1f}x',
                            'risk_reward': '2:1'
                        }

        return None

    def _detect_failed_breakout(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Detect failed breakout patterns.

        Returns:
            Dict with failed breakout opportunity if detected
        """
        # Get recent price action
        last_24h = df.tail(24)
        if len(last_24h) < 12:
            return None

        # Find recent high that could have been a breakout
        recent_high = last_24h['high'].max()
        recent_low = last_24h['low'].min()
        current_position = (current_price - recent_low) / (recent_high - recent_low)

        # Check if we're back in the lower third after being near top
        if current_position < 0.33:
            # Check if there was a high volume spike at the top
            volume_at_high = last_24h.loc[last_24h['high'] == recent_high, 'tick_volume'].iloc[0] if len(last_24h.loc[last_24h['high'] == recent_high]) > 0 else 0
            avg_volume = last_24h['tick_volume'].mean()

            if volume_at_high > avg_volume * self.detection_thresholds['volume_multiplier']:
                return {
                    'type': 'FAILED_BREAKOUT_SHORT',
                    'direction': 'SHORT',
                    'entry_zone': {
                        'min': current_price * 0.999,
                        'max': recent_low * 0.6 + current_price * 0.4
                    },
                    'stop_loss': recent_high * 1.001,
                    'targets': [
                        recent_low,  # Test recent low
                        recent_low * 0.995,  # Break below
                        (recent_high + recent_low) / 2  # Return to middle
                    ],
                    'status': 'CONFIRMED',
                    'priority': 'MEDIUM',
                    'confidence': 75,
                    'reason': f'Failed breakout at {recent_high:.3f} - back to {current_position*100:.0f}% level',
                    'volume_confirmation': f'{volume_at_high/avg_volume:.1f}x average',
                    'risk_reward': '2.5:1'
                }

        return None

    def _detect_dmr_opportunity(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Detect DMR (Daily Market Rotation) level bounce opportunities.

        Returns:
            Dict with DMR opportunity if detected
        """
        # Calculate DMR levels
        dmr_levels = self._calculate_dmr_levels(df)
        if not dmr_levels:
            return None

        # Check proximity to any DMR level
        for level_name, level_value in dmr_levels.items():
            distance = abs(current_price - level_value) / level_value

            if distance < 0.002:  # Within 0.2% of DMR level
                # Determine direction based on level type
                if 'high' in level_name.lower() or 'resistance' in level_name.lower():
                    direction = 'SHORT'
                    stop_loss = level_value * 1.005
                else:
                    direction = 'LONG'
                    stop_loss = level_value * 0.995

                return {
                    'type': f'DMR_{level_name.upper()}',
                    'direction': direction,
                    'entry_zone': {
                        'min': level_value * 0.998 if direction == 'LONG' else level_value * 1.002,
                        'max': level_value * 1.002 if direction == 'LONG' else level_value * 0.998
                    },
                    'stop_loss': stop_loss,
                    'targets': self._calculate_dmr_targets(level_value, direction),
                    'status': 'APPROACHING',
                    'priority': 'MEDIUM',
                    'confidence': 70,
                    'reason': f'{distance*100:.2f}% from {level_name} at {level_value:.3f}',
                    'level_type': level_name,
                    'risk_reward': '2:1'
                }

        return None

    def _detect_session_opportunity(self, df: pd.DataFrame, current_price: float, sessions: Dict) -> Optional[Dict]:
        """
        Detect session transition opportunities.

        Returns:
            Dict with session opportunity if detected
        """
        current_hour = datetime.now().hour

        # London open (7-9 UTC)
        if 7 <= current_hour <= 9:
            # Check for London breakout setup
            asian_range = self._get_asian_range(df)
            if asian_range:
                asian_high = asian_range['high']

                if current_price > asian_high:
                    return {
                        'type': 'LONDON_MOMENTUM_LONG',
                        'direction': 'LONG',
                        'entry_zone': {
                            'min': current_price * 0.999,
                            'max': current_price * 1.001
                        },
                        'stop_loss': asian_range['low'],
                        'targets': [asian_high * 1.01, asian_high * 1.02],
                        'status': 'ACTIVE',
                        'priority': 'HIGH',
                        'confidence': 80,
                        'reason': 'London momentum above Asian range',
                        'risk_reward': '2.5:1'
                    }

        # NY open (13-15 UTC)
        elif 13 <= current_hour <= 15:
            # Check for NY session trend continuation
            london_high = self._get_london_high(df)
            if london_high and current_price > london_high:
                return {
                    'type': 'NY_CONTINUATION_LONG',
                    'direction': 'LONG',
                    'entry_zone': {
                        'min': london_high * 1.0005,
                        'max': london_high * 1.002
                    },
                    'stop_loss': london_high * 0.995,
                    'targets': [london_high * 1.01, london_high * 1.015],
                    'status': 'DEVELOPING',
                    'priority': 'MEDIUM',
                    'confidence': 75,
                    'reason': 'NY continuation above London high',
                    'risk_reward': '2:1'
                }

        return None

    def _detect_volume_anomaly(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Detect unusual volume patterns that may indicate smart money activity.

        Returns:
            Dict with volume anomaly opportunity if detected
        """
        if len(df) < 20:
            return None

        recent_volume = df.tail(5)['tick_volume']
        avg_volume = df.tail(50)['tick_volume'].mean()
        current_candle = df.iloc[-1]

        # Check for volume spike
        if current_candle['tick_volume'] > avg_volume * self.detection_thresholds['volume_multiplier']:
            # Determine if it's accumulation or distribution
            candle_range = current_candle['high'] - current_candle['close']
            total_range = current_candle['high'] - current_candle['low']

            # More than 60% of range on close side = accumulation
            if candle_range / total_range > 0.6 and current_candle['close'] > current_candle['open']:
                return {
                    'type': 'VOLUME_ACCUMULATION_LONG',
                    'direction': 'LONG',
                    'entry_zone': {
                        'min': current_candle['close'] * 0.999,
                        'max': current_candle['close'] * 1.001
                    },
                    'stop_loss': current_candle['low'],
                    'targets': [current_candle['high'] * 1.01, current_candle['high'] * 1.02],
                    'status': 'CONFIRMING',
                    'priority': 'MEDIUM',
                    'confidence': 70,
                    'reason': f'High volume accumulation: {current_candle["tick_volume"]/avg_volume:.1f}x average',
                    'volume_ratio': f'{current_candle["tick_volume"]/avg_volume:.1f}x',
                    'risk_reward': '2:1'
                }

            # More than 60% of range on open side = distribution
            elif candle_range / total_range > 0.6 and current_candle['close'] < current_candle['open']:
                return {
                    'type': 'VOLUME_DISTRIBUTION_SHORT',
                    'direction': 'SHORT',
                    'entry_zone': {
                        'min': current_candle['close'] * 0.999,
                        'max': current_candle['close'] * 1.001
                    },
                    'stop_loss': current_candle['high'],
                    'targets': [current_candle['low'] * 0.99, current_candle['low'] * 0.98],
                    'status': 'CONFIRMING',
                    'priority': 'MEDIUM',
                    'confidence': 70,
                    'reason': f'High volume distribution: {current_candle["tick_volume"]/avg_volume:.1f}x average',
                    'volume_ratio': f'{current_candle["tick_volume"]/avg_volume:.1f}x',
                    'risk_reward': '2:1'
                }

        return None

    # Helper methods
    def _get_asian_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get today's Asian session range"""
        # Filter for Asian session (0-7 UTC)
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        df_copy['today'] = df_copy.index.date

        today = datetime.now().date()
        asian_data = df_copy[
            (df_copy['today'] == today) &
            (df_copy['hour'] >= 0) &
            (df_copy['hour'] < 7)
        ]

        if len(asian_data) > 0:
            return {
                'high': asian_data['high'].max(),
                'low': asian_data['low'].min(),
                'volume': asian_data['tick_volume'].sum()
            }

        return None

    def _has_rejection_pattern(self, candles: pd.DataFrame) -> bool:
        """Check if recent candles show rejection pattern"""
        if len(candles) < 3:
            return False

        last_candle = candles.iloc[-1]
        second_last = candles.iloc[-2]

        # Check for shooting star or bearish engulfing
        # Shooting star: small body, long upper wick
        if last_candle['close'] > last_candle['open']:
            upper_wick = last_candle['high'] - last_candle['close']
            body = last_candle['close'] - last_candle['open']
            lower_wick = last_candle['open'] - last_candle['low']
        else:
            upper_wick = last_candle['high'] - last_candle['open']
            body = last_candle['open'] - last_candle['close']
            lower_wick = last_candle['close'] - last_candle['low']

        total_range = last_candle['high'] - last_candle['low']

        # Shooting star: upper wick > 60% of range, body < 30% of range
        if upper_wick / total_range > 0.6 and body / total_range < 0.3:
            return True

        # Bearish engulfing pattern
        if (second_last['close'] > second_last['open'] and  # Previous was green
            last_candle['close'] < last_candle['open'] and    # Current is red
            last_candle['open'] > second_last['close'] and    # Opens above previous close
            last_candle['close'] < second_last['open']):      # Closes below previous open
            return True

        return False

    def _get_london_high(self, df: pd.DataFrame) -> Optional[float]:
        """Get London session high"""
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        df_copy['today'] = df_copy.index.date

        today = datetime.now().date()
        london_data = df_copy[
            (df_copy['today'] == today) &
            (df_copy['hour'] >= 7) &
            (df_copy['hour'] < 13)
        ]

        if len(london_data) > 0:
            return london_data['high'].max()

        return None

    def _calculate_dmr_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate DMR levels"""
        if len(df) < 5:
            return {}

        # Previous day levels
        prev_day = df.iloc[-2]
        current_day = df.iloc[-1]

        # Weekly levels
        last_week = df.tail(120).head(5)  # Approximate last week
        this_week = df.tail(5)  # Current week

        return {
            'prev_day_high': prev_day['high'],
            'prev_day_low': prev_day['low'],
            'prev_day_close': prev_day['close'],
            'week_high': last_week['high'].max(),
            'week_low': last_week['low'].min(),
            'month_high': df.tail(720)['high'].max(),  # Last 30 days
            'month_low': df.tail(720)['low'].min()
        }

    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> str:
        """Calculate risk/reward ratio"""
        if 'LONG' in str(entry):  # Simplified check
            risk = abs(entry - stop)
            reward = abs(target - entry)
        else:
            risk = abs(stop - entry)
            reward = abs(entry - target)

        if risk > 0:
            ratio = reward / risk
            return f"{ratio:.1f}:1"
        return "N/A"

    def _calculate_dmr_targets(self, level: float, direction: str) -> List[float]:
        """Calculate targets for DMR level plays"""
        if direction == 'LONG':
            return [level * 1.01, level * 1.015, level * 1.02]
        else:
            return [level * 0.99, level * 0.985, level * 0.98]

    def _get_asian_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get Asian session range"""
        if len(df) < 10:
            return None

        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour

        # Asian session: 0-5 UTC
        asian_data = df_copy[
            (df_copy['hour'] >= 0) &
            (df_copy['hour'] < 5)
        ]

        if len(asian_data) > 0:
            return {
                'high': asian_data['high'].max(),
                'low': asian_data['low'].min(),
                'volume': asian_data['tick_volume'].sum()
            }

        return None

    def _find_peak_candle(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Find the candle with the highest high"""
        if len(df) == 0:
            return None

        peak_high = df['high'].max()
        peak_candles = df[df['high'] == peak_high]

        if len(peak_candles) > 0:
            return peak_candles.iloc[-1]
        return None

    def _is_rejection_candle(self, candle: pd.Series) -> bool:
        """Check if a candle shows rejection pattern"""
        # Red candle is rejection
        if candle['close'] < candle['open']:
            return True

        # Calculate wick ratio
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            return False

        upper_wick = candle['high'] - max(candle['open'], candle['close'])

        # Shooting star: upper wick > 50% of range
        if upper_wick / total_range > self.detection_thresholds['wick_ratio_threshold']:
            return True

        return False

    def _has_recent_rejection(self, df: pd.DataFrame) -> bool:
        """Check if there are recent rejection candles"""
        if len(df) < 2:
            return False

        rejection_count = 0
        start_idx = max(0, len(df) - 3)

        for i in range(start_idx, len(df)):
            if self._is_rejection_candle(df.iloc[i]):
                rejection_count += 1

        return rejection_count >= 2