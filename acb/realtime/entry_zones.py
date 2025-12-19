"""
Entry Zone Calculator
====================

Calculates precise entry zones, stop losses, and profit targets
for different market conditions and patterns.

Provides:
- Dynamic entry zones based on volatility
- ATR-based stop losses
- Multiple profit target levels
- Risk/reward calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class EntryZoneCalculator:
    """
    Calculates entry zones with precise price levels for trading setups.

    Uses multiple factors:
    - Current market volatility (ATR)
    - Support/resistance levels
    - Pattern-specific calculations
    - Risk management rules
    """

    def __init__(self):
        """Initialize the entry zone calculator"""
        self.atr_periods = {
            'fast': 5,   # 5-period ATR for short-term
            'normal': 14, # 14-period ATR standard
            'slow': 28    # 28-period ATR for major moves
        }

        self.risk_percentages = {
            'conservative': 0.01,  # 1% risk
            'standard': 0.02,      # 2% risk
            'aggressive': 0.03     # 3% risk
        }

    def get_zones(self, df: pd.DataFrame, current_price: float, sessions: Dict) -> Dict:
        """
        Get all active entry zones for current market conditions.

        Args:
            df: DataFrame with price data
            current_price: Current market price
            sessions: Session analysis data

        Returns:
            Dictionary with all active entry zones
        """
        # Calculate ATR for volatility
        atr_values = self._calculate_atr(df)

        # Get key market levels
        key_levels = self._identify_key_levels(df, sessions)

        # Calculate zones for different scenarios
        zones = {
            'asian_range_zones': self._calculate_asian_zones(df, current_price, atr_values),
            'breakout_zones': self._calculate_breakout_zones(df, current_price, key_levels, atr_values),
            'pullback_zones': self._calculate_pullback_zones(df, current_price, key_levels, atr_values),
            'reversal_zones': self._calculate_reversal_zones(df, current_price, key_levels, atr_values),
            'range_trading_zones': self._calculate_range_zones(df, current_price, atr_values),
            'session_specific_zones': self._calculate_session_zones(df, current_price, sessions, atr_values)
        }

        # Add risk management info
        zones['risk_management'] = self._get_risk_management(df, current_price, atr_values)

        return zones

    def _calculate_asian_zones(self, df: pd.DataFrame, current_price: float, atr: Dict) -> Dict:
        """Calculate entry zones related to Asian range"""
        asian_range = self._get_asian_range(df)
        if not asian_range:
            return {'status': 'No Asian Range Data'}

        asian_low = asian_range['low']
        asian_high = asian_range['high']
        atr_normal = atr['normal']

        zones = {
            'long_entries': [],
            'short_entries': []
        }

        # Long entry zones
        if current_price > asian_low:
            # Zone 1: Just above Asian low
            zones['long_entries'].append({
                'zone_name': 'Asian Low Support',
                'entry_min': asian_low * 1.0002,
                'entry_max': asian_low * 1.001,
                'stop_loss': asian_low * 0.9995,
                'target_1': asian_high * 0.998,
                'target_2': asian_high,
                'target_3': asian_high * 1.01,
                'confidence': 85,
                'reason': 'Bounce from Asian session low'
            })

            # Zone 2: Asian range midpoint
            midpoint = (asian_low + asian_high) / 2
            if current_price < midpoint:
                zones['long_entries'].append({
                    'zone_name': 'Asian Midpoint',
                    'entry_min': midpoint * 0.9995,
                    'entry_max': midpoint * 1.0005,
                    'stop_loss': asian_low * 0.999,
                    'target_1': asian_high,
                    'target_2': asian_high * 1.01,
                    'target_3': asian_high * 1.015,
                    'confidence': 75,
                    'reason': 'Rebound toward Asian high'
                })

        # Short entry zones
        if current_price < asian_high:
            # Zone 1: Just below Asian high
            zones['short_entries'].append({
                'zone_name': 'Asian High Resistance',
                'entry_min': asian_high * 0.999,
                'entry_max': asian_high * 0.998,
                'stop_loss': asian_high * 1.001,
                'target_1': midpoint,
                'target_2': asian_low,
                'target_3': asian_low * 0.99,
                'confidence': 85,
                'reason': 'Rejection from Asian session high'
            })

            # Zone 2: Asian range breakout short
            if current_price > asian_high:
                zones['short_entries'].append({
                    'zone_name': 'Failed Asian Breakout',
                    'entry_min': asian_high * 1.0015,
                    'entry_max': asian_high * 1.003,
                    'stop_loss': asian_high * 1.005,
                    'target_1': asian_high,
                    'target_2': midpoint,
                    'target_3': asian_low,
                    'confidence': 90,
                    'reason': 'Short the failed breakout'
                })

        return zones

    def _calculate_breakout_zones(self, df: pd.DataFrame, current_price: float,
                                 key_levels: Dict, atr: Dict) -> Dict:
        """Calculate entry zones for breakout trades"""
        zones = {
            'breakout_long': [],
            'breakout_short': []
        }

        # Recent high/low for breakout context
        recent_period = 24  # Last 24 hours
        if len(df) >= recent_period:
            recent_data = df.tail(recent_period)
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()

            atr_fast = atr['fast']

            # Long breakout zones
            if current_price < recent_high:
                # Entry zone on breakout
                zones['breakout_long'].append({
                    'zone_name': '24h High Breakout',
                    'entry_min': recent_high * 1.0001,
                    'entry_max': recent_high * (1 + atr_fast * 0.5),
                    'stop_loss': recent_high * (1 - atr_fast),
                    'target_1': recent_high * (1 + atr_fast * 2),
                    'target_2': recent_high * (1 + atr_fast * 3),
                    'target_3': recent_high * (1 + atr_fast * 5),
                    'confidence': 80,
                    'trigger_condition': 'Close above recent high',
                    'reason': 'Breakout of 24-hour range'
                })

            # Short breakout zones
            if current_price > recent_low:
                zones['breakout_short'].append({
                    'zone_name': '24h Low Breakdown',
                    'entry_min': recent_low * (1 - atr_fast * 0.5),
                    'entry_max': recent_low * 0.9999,
                    'stop_loss': recent_low * (1 + atr_fast),
                    'target_1': recent_low * (1 - atr_fast * 2),
                    'target_2': recent_low * (1 - atr_fast * 3),
                    'target_3': recent_low * (1 - atr_fast * 5),
                    'confidence': 80,
                    'trigger_condition': 'Close below recent low',
                    'reason': 'Breakdown of 24-hour range'
                })

        return zones

    def _calculate_pullback_zones(self, df: pd.DataFrame, current_price: float,
                                 key_levels: Dict, atr: Dict) -> Dict:
        """Calculate entry zones for pullback trades"""
        zones = {
            'pullback_long': [],
            'pullback_short': []
        }

        # Look for recent strong moves
        if len(df) >= 10:
            recent_candles = df.tail(10)

            # Identify strong uptrend
            if self._is_strong_uptrend(recent_candles):
                high_point = recent_candles['high'].max()
                current_low = recent_candles['low'].min()
                pullback_level = (high_point + current_low) / 2

                if current_price <= pullback_level:
                    zones['pullback_long'].append({
                        'zone_name': 'Uptrend Pullback',
                        'entry_min': pullback_level * 0.9995,
                        'entry_max': pullback_level * 1.0005,
                        'stop_loss': current_low * 0.998,
                        'target_1': high_point,
                        'target_2': high_point * 1.01,
                        'target_3': high_point * 1.02,
                        'confidence': 85,
                        'reason': 'Buy the dip in uptrend'
                    })

            # Identify strong downtrend
            elif self._is_strong_downtrend(recent_candles):
                low_point = recent_candles['low'].min()
                current_high = recent_candles['high'].max()
                pullback_level = (low_point + current_high) / 2

                if current_price >= pullback_level:
                    zones['pullback_short'].append({
                        'zone_name': 'Downtrend Pullback',
                        'entry_min': pullback_level * 0.9995,
                        'entry_max': pullback_level * 1.0005,
                        'stop_loss': current_high * 1.002,
                        'target_1': low_point,
                        'target_2': low_point * 0.99,
                        'target_3': low_point * 0.98,
                        'confidence': 85,
                        'reason': 'Sell the rally in downtrend'
                    })

        return zones

    def _calculate_reversal_zones(self, df: pd.DataFrame, current_price: float,
                                 key_levels: Dict, atr: Dict) -> Dict:
        """Calculate entry zones for reversal trades"""
        zones = {
            'reversal_long': [],
            'reversal_short': []
        }

        # Look for exhaustion patterns
        if len(df) >= 20:
            recent_data = df.tail(20)
            last_candle = df.iloc[-1]

            # Check for bearish exhaustion at highs
            if self._is_bearish_exhaustion(recent_data):
                resistance = recent_data['high'].max()
                zones['reversal_short'].append({
                    'zone_name': 'Bearish Exhaustion',
                    'entry_min': last_candle['close'] * 0.999,
                    'entry_max': last_candle['close'] * 1.001,
                    'stop_loss': resistance * 1.002,
                    'target_1': (resistance + recent_data['low'].min()) / 2,
                    'target_2': recent_data['low'].min(),
                    'target_3': recent_data['low'].min() * 0.99,
                    'confidence': 75,
                    'reason': 'Potential bearish reversal at exhaustion'
                })

            # Check for bullish exhaustion at lows
            elif self._is_bullish_exhaustion(recent_data):
                support = recent_data['low'].min()
                zones['reversal_long'].append({
                    'zone_name': 'Bullish Exhaustion',
                    'entry_min': last_candle['close'] * 0.999,
                    'entry_max': last_candle['close'] * 1.001,
                    'stop_loss': support * 0.998,
                    'target_1': (support + recent_data['high'].max()) / 2,
                    'target_2': recent_data['high'].max(),
                    'target_3': recent_data['high'].max() * 1.01,
                    'confidence': 75,
                    'reason': 'Potential bullish reversal at exhaustion'
                })

        return zones

    def _calculate_range_zones(self, df: pd.DataFrame, current_price: float, atr: Dict) -> Dict:
        """Calculate entry zones for range-bound trading"""
        zones = {
            'range_long': [],
            'range_short': []
        }

        # Identify range boundaries
        if len(df) >= 48:  # Last 2 days
            range_data = df.tail(48)
            range_high = range_data['high'].max()
            range_low = range_data['low'].min()
            range_mid = (range_high + range_low) / 2

            # Check if actually in range (no major breakouts)
            if (current_price > range_low * 1.02 and current_price < range_high * 0.98):
                atr_slow = atr['slow']
                range_width = range_high - range_low

                # Only trade if range is wide enough
                if range_width > atr_slow * 3:
                    # Long entry at range bottom
                    if current_price <= range_low + (range_width * 0.2):
                        zones['range_long'].append({
                            'zone_name': 'Range Bottom Support',
                            'entry_min': range_low * 1.0005,
                            'entry_max': range_low * 1.002,
                            'stop_loss': range_low * 0.998,
                            'target_1': range_mid,
                            'target_2': range_high * 0.99,
                            'confidence': 70,
                            'reason': 'Range trading at bottom'
                        })

                    # Short entry at range top
                    elif current_price >= range_high - (range_width * 0.2):
                        zones['range_short'].append({
                            'zone_name': 'Range Top Resistance',
                            'entry_min': range_high * 0.998,
                            'entry_max': range_high * 0.9995,
                            'stop_loss': range_high * 1.002,
                            'target_1': range_mid,
                            'target_2': range_low * 1.01,
                            'confidence': 70,
                            'reason': 'Range trading at top'
                        })

        return zones

    def _calculate_session_zones(self, df: pd.DataFrame, current_price: float,
                               sessions: Dict, atr: Dict) -> Dict:
        """Calculate session-specific entry zones"""
        zones = {}

        current_hour = datetime.now().hour

        # Session transition zones
        if 6 <= current_hour <= 8:  # Approaching London open
            asian_range = self._get_asian_range(df)
            if asian_range:
                zones['london_open_prep'] = {
                    'scenario': 'Breakout setup',
                    'long_entry': asian_range['high'] * 1.001,
                    'short_entry': asian_range['low'] * 0.999,
                    'stop_long': asian_range['low'],
                    'stop_short': asian_range['high']
                }

        elif 12 <= current_hour <= 14:  # Approaching NY open
            london_data = self._get_london_session(df)
            if london_data:
                zones['ny_open_prep'] = {
                    'scenario': 'Momentum continuation',
                    'long_entry': london_data['high'] * 1.0005,
                    'short_entry': london_data['low'] * 0.9995,
                    'stop_long': london_data['low'],
                    'stop_short': london_data['high']
                }

        elif 21 <= current_hour or current_hour <= 1:  # Late NY/Early Asian
            # End of day positions
            zones['day_end'] = {
                'scenario': 'Position squaring',
                'observation': 'Watch for reversal patterns near key levels'
            }

        return zones

    def _get_risk_management(self, df: pd.DataFrame, current_price: float, atr: Dict) -> Dict:
        """Calculate risk management parameters"""
        return {
            'current_atr': {
                'fast': atr['fast'],
                'normal': atr['normal'],
                'slow': atr['slow']
            },
            'position_sizing': {
                '1_percent_risk': self._calculate_position_size(current_price, atr['normal'], 0.01),
                '2_percent_risk': self._calculate_position_size(current_price, atr['normal'], 0.02),
                '3_percent_risk': self._calculate_position_size(current_price, atr['normal'], 0.03)
            },
            'volatility_adjustment': self._get_volatility_adjustment(atr['normal']),
            'recommended_stop_distance': atr['normal'] * 2,  # 2x ATR
            'profit_taking_levels': {
                'partial_1': atr['normal'] * 2,
                'partial_2': atr['normal'] * 3,
                'full_exit': atr['normal'] * 5
            }
        }

    # Helper methods
    def _calculate_atr(self, df: pd.DataFrame) -> Dict:
        """Calculate multiple ATR values"""
        atr_values = {}

        for period_name, period in self.atr_periods.items():
            if len(df) > period:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift(1))
                low_close = np.abs(df['low'] - df['close'].shift(1))

                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_values[period_name] = true_range.rolling(period).mean().iloc[-1]
            else:
                atr_values[period_name] = 0

        return atr_values

    def _identify_key_levels(self, df: pd.DataFrame, sessions: Dict) -> Dict:
        """Identify key support/resistance levels"""
        if len(df) < 50:
            return {}

        return {
            'prev_day_high': df.iloc[-2]['high'],
            'prev_day_low': df.iloc[-2]['low'],
            'prev_week_high': df.tail(120).head(5)['high'].max(),
            'prev_week_low': df.tail(120).head(5)['low'].min(),
            'monthly_high': df.tail(720)['high'].max(),
            'monthly_low': df.tail(720)['low'].min(),
            'volume_profile_high': df.tail(100).loc[df.tail(100)['tick_volume'].idxmax(), 'high'],
            'volume_profile_low': df.tail(100).loc[df.tail(100)['tick_volume'].idxmax(), 'low']
        }

    def _get_asian_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get Asian session range"""
        if len(df) < 10:
            return None

        # Filter for Asian session hours (approximate)
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour

        asian_data = df_copy[df_copy['hour'].isin([0, 1, 2, 3, 4, 5, 6])]

        if len(asian_data) > 0:
            return {
                'high': asian_data['high'].max(),
                'low': asian_data['low'].min(),
                'volume': asian_data['tick_volume'].sum()
            }

        return None

    def _get_london_session(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get London session data"""
        if len(df) < 10:
            return None

        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour

        london_data = df_copy[df_copy['hour'].isin([7, 8, 9, 10, 11, 12])]

        if len(london_data) > 0:
            return {
                'high': london_data['high'].max(),
                'low': london_data['low'].min(),
                'volume': london_data['tick_volume'].sum()
            }

        return None

    def _is_strong_uptrend(self, df: pd.DataFrame) -> bool:
        """Check if recent candles show strong uptrend"""
        if len(df) < 5:
            return False

        # Higher highs and higher lows
        highs = df['high'].values
        lows = df['low'].values

        higher_highs = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
        higher_lows = sum(lows[i] > lows[i-1] for i in range(1, len(lows)))

        return (higher_highs > len(highs) * 0.6 and
                higher_lows > len(lows) * 0.6 and
                df['close'].iloc[-1] > df['close'].iloc[0])

    def _is_strong_downtrend(self, df: pd.DataFrame) -> bool:
        """Check if recent candles show strong downtrend"""
        if len(df) < 5:
            return False

        # Lower highs and lower lows
        highs = df['high'].values
        lows = df['low'].values

        lower_highs = sum(highs[i] < highs[i-1] for i in range(1, len(highs)))
        lower_lows = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))

        return (lower_highs > len(highs) * 0.6 and
                lower_lows > len(lows) * 0.6 and
                df['close'].iloc[-1] < df['close'].iloc[0])

    def _is_bearish_exhaustion(self, df: pd.DataFrame) -> bool:
        """Check for bearish exhaustion pattern"""
        if len(df) < 5:
            return False

        # Recent high with rejection
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        recent_highs = df['high'].tail(5)
        max_high = recent_highs.max()

        return (last_candle['high'] == max_high and
                last_candle['close'] < last_candle['open'] and
                (last_candle['high'] - last_candle['close']) > (last_candle['close'] - last_candle['low']) * 2)

    def _is_bullish_exhaustion(self, df: pd.DataFrame) -> bool:
        """Check for bullish exhaustion pattern"""
        if len(df) < 5:
            return False

        last_candle = df.iloc[-1]
        recent_lows = df['low'].tail(5)
        min_low = recent_lows.min()

        return (last_candle['low'] == min_low and
                last_candle['close'] > last_candle['open'] and
                (last_candle['close'] - last_candle['low']) > (last_candle['high'] - last_candle['close']) * 2)

    def _calculate_position_size(self, price: float, atr: float, risk_percent: float) -> Dict:
        """Calculate position size for given risk percentage"""
        # Simplified calculation - would need account size in real implementation
        stop_distance = atr * 2
        risk_per_unit = stop_distance / price

        return {
            'stop_distance': stop_distance,
            'risk_per_unit': f'{risk_per_unit*100:.2f}%',
            'recommended_units': f'{risk_percent / risk_per_unit:.1f} units'
        }

    def _get_volatility_adjustment(self, atr: float) -> Dict:
        """Get volatility adjustment recommendations"""
        # Compare current ATR to historical average
        if atr > 0:
            return {
                'current_volatility': 'High' if atr > 0.01 else 'Normal',
                'recommendation': 'Reduce position size' if atr > 0.01 else 'Normal sizing',
                'stop_multiplier': 2.5 if atr > 0.01 else 2.0
            }
        return {'recommendation': 'Insufficient data'}