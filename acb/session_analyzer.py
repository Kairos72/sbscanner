"""
Session Analyzer for Market Structure
====================================

Analyzes price behavior across different trading sessions.
Critical for understanding smart money manipulation patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Session(Enum):
    """Trading sessions with UTC times"""
    ASIAN = (0, 6, "Asian")
    LONDON = (6, 14, "London")
    NY = (13, 22, "New York")
    OVERLAP = (14, 16, "London/NY")
    QUIET = (22, 24, "Quiet")


class SessionAnalyzer:
    """
    Analyzes market behavior across trading sessions.
    Identifies manipulation patterns and smart money activity.
    """

    def __init__(self):
        self.sessions = self._create_session_map()
        self.min_range_pips = 10  # Minimum range for significant move
        self.manipulation_threshold = 1.5  # 1.5x ATR for manipulation

    def _create_session_map(self) -> Dict:
        """Create hour to session mapping."""
        session_map = {}

        for hour in range(24):
            if 0 <= hour < 6:
                session_map[hour] = Session.ASIAN
            elif 6 <= hour < 13:
                session_map[hour] = Session.LONDON
            elif 13 <= hour < 14:
                session_map[hour] = Session.NY
            elif 14 <= hour < 16:
                session_map[hour] = Session.OVERLAP
            elif 16 <= hour < 22:
                session_map[hour] = Session.NY
            else:
                session_map[hour] = Session.QUIET

        return session_map

    def identify_session(self, timestamp: datetime) -> Session:
        """
        Identify which session the timestamp falls into.
        """
        return self.sessions.get(timestamp.hour, Session.QUIET)

    def analyze_session_behavior(self, df: pd.DataFrame,
                                lookback_hours: int = 72) -> Dict:
        """
        Analyze behavior across different sessions.

        Args:
            df: DataFrame with OHLC data
            lookback_hours: Hours to analyze

        Returns:
            Dictionary with session analysis results
        """
        if len(df) < lookback_hours:
            return {}

        analysis = {
            'sessions': {},
            'patterns': {},
            'manipulation_zones': [],
            'range_stats': {}
        }

        # Group candles by session
        session_candles = self._group_by_session(df.tail(lookback_hours))

        # Analyze each session
        for session_name, candles in session_candles.items():
            if len(candles) > 0:
                analysis['sessions'][session_name] = self._analyze_session_stats(candles, session_name)

        # Identify patterns
        analysis['patterns'] = self._identify_session_patterns(session_candles)

        # Find manipulation zones
        analysis['manipulation_zones'] = self._find_manipulation_zones(df)

        # Calculate range statistics
        analysis['range_stats'] = self._calculate_range_stats(session_candles)

        return analysis

    def _group_by_session(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group candles by trading session."""
        session_groups = {}

        for idx, row in df.iterrows():
            session = self.identify_session(idx)
            session_name = session.value[2]

            if session_name not in session_groups:
                session_groups[session_name] = []

            session_groups[session_name].append(row)

        # Convert to DataFrames
        for session_name in session_groups:
            if session_groups[session_name]:
                session_groups[session_name] = pd.DataFrame(session_groups[session_name])

        return session_groups

    def _analyze_session_stats(self, candles: pd.DataFrame, session_name: str) -> Dict:
        """
        Analyze statistics for a specific session.
        """
        if len(candles) == 0:
            return {}

        # Calculate basic stats
        stats = {
            'candle_count': len(candles),
            'avg_range': (candles['high'] - candles['low']).mean() * 10000,
            'max_range': (candles['high'] - candles['low']).max() * 10000,
            'min_range': (candles['high'] - candles['low']).min() * 10000,
            'avg_body': abs(candles['close'] - candles['open']).mean() * 10000,
            'direction': self._calculate_session_direction(candles),
            'volatility': self._calculate_volatility(candles),
            'session_role': self._determine_session_role(session_name)
        }

        # Add OHLC
        stats['session_high'] = candles['high'].max()
        stats['session_low'] = candles['low'].min()
        stats['session_open'] = candles.iloc[0]['open']
        stats['session_close'] = candles.iloc[-1]['close']

        # Check for significant moves
        if stats['avg_range'] > self.min_range_pips:
            stats['characteristic'] = 'ACTIVE'
        else:
            stats['characteristic'] = 'QUIET'

        return stats

    def _calculate_session_direction(self, candles: pd.DataFrame) -> str:
        """
        Calculate the dominant direction of the session.
        """
        net_change = candles.iloc[-1]['close'] - candles.iloc[0]['open']
        total_range = candles['high'].max() - candles['low'].min()

        if total_range == 0:
            return 'NEUTRAL'

        # Calculate directional strength
        directional_strength = abs(net_change) / total_range

        if directional_strength < 0.3:
            return 'NEUTRAL'
        elif net_change > 0:
            return 'BULLISH'
        else:
            return 'BEARISH'

    def _calculate_volatility(self, candles: pd.DataFrame) -> float:
        """
        Calculate volatility as range percentage.
        """
        if len(candles) < 2:
            return 0

        avg_price = candles['close'].mean()
        avg_range = (candles['high'] - candles['low']).mean()

        return (avg_range / avg_price) * 100

    def _determine_session_role(self, session_name: str) -> str:
        """
        Determine the typical role of the session in market structure.
        """
        roles = {
            'Asian': 'Range Building',
            'London': 'Manipulation/Breakout',
            'New York': 'Confirmation/Distribution',
            'London/NY': 'High Volatility',
            'Quiet': 'Positioning'
        }

        return roles.get(session_name, 'Unknown')

    def _identify_session_patterns(self, session_candles: Dict) -> Dict:
        """
        Identify recurring patterns across sessions.
        """
        patterns = {
            'asian_range_builder': False,
            'london_breakout': False,
            'london_fakeout': False,
            'ny_trend': False,
            'ny_reversal': False
        }

        # Check Asian range building
        if 'Asian' in session_candles:
            asian_stats = self._analyze_session_stats(session_candles['Asian'], 'Asian')
            if asian_stats and asian_stats['avg_range'] < 15:
                patterns['asian_range_builder'] = True

        # Check London behavior
        if 'London' in session_candles and 'Asian' in session_candles:
            asian_low = session_candles['Asian']['low'].min()
            asian_high = session_candles['Asian']['high'].max()
            london_low = session_candles['London']['low'].min()
            london_high = session_candles['London']['high'].max()

            # Check for breakout
            if london_low < asian_low - 0.00010 or london_high > asian_high + 0.00010:
                patterns['london_breakout'] = True

            # Check for fakeout (break then reverse)
            london_open = session_candles['London'].iloc[0]['open']
            london_close = session_candles['London'].iloc[-1]['close']

            if (london_low < asian_low and london_close > london_open) or \
               (london_high > asian_high and london_close < london_open):
                patterns['london_fakeout'] = True

        # Check NY behavior
        if 'New York' in session_candles:
            ny_candles = session_candles['New York']
            ny_direction = self._calculate_session_direction(ny_candles)

            if ny_direction in ['BULLISH', 'BEARISH']:
                patterns['ny_trend'] = True

            # Check for reversal
            if 'London' in session_candles and 'New York' in session_candles:
                london_direction = self._calculate_session_direction(session_candles['London'])
                if london_direction != 'NEUTRAL' and london_direction != ny_direction:
                    patterns['ny_reversal'] = True

        return patterns

    def _find_manipulation_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify potential manipulation zones.
        """
        manipulation_zones = []

        if len(df) < 24:
            return manipulation_zones

        # Get current date
        current_date = df.index[-1].date()

        # Find Asian range (0-6 UTC)
        asian_candles = df[
            (df.index.to_series().dt.date == current_date) &
            (df.index.to_series().dt.hour >= 0) &
            (df.index.to_series().dt.hour < 6)
        ]

        if len(asian_candles) > 0:
            asian_high = asian_candles['high'].max()
            asian_low = asian_candles['low'].min()

            # Check if London/NA hunted these levels
            manipulation_zones.append({
                'level': 'ASIAN_HIGH',
                'price': asian_high,
                'hunted': self._check_hunt(df, asian_high, 'high'),
                'hunt_time': self._get_hunt_time(df, asian_high, 'high')
            })

            manipulation_zones.append({
                'level': 'ASIAN_LOW',
                'price': asian_low,
                'hunted': self._check_hunt(df, asian_low, 'low'),
                'hunt_time': self._get_hunt_time(df, asian_low, 'low')
            })

        return manipulation_zones

    def _check_hunt(self, df: pd.DataFrame, target_price: float, direction: str) -> bool:
        """
        Check if price hunted a specific level.
        """
        # Get current date
        current_date = df.index[-1].date()

        # Look at candles after Asian session (after 6 UTC)
        london_ny_candles = df[
            (df.index.to_series().dt.date == current_date) &
            (df.index.to_series().dt.hour >= 6)
        ]

        for _, candle in london_ny_candles.iterrows():
            if direction == 'high' and candle['high'] > target_price:
                return True
            elif direction == 'low' and candle['low'] < target_price:
                return True

        return False

    def _get_hunt_time(self, df: pd.DataFrame, target_price: float, direction: str) -> Optional[datetime]:
        """
        Get the time when a level was hunted.
        """
        # Get current date
        current_date = df.index[-1].date()

        # Look at candles after Asian session
        london_ny_candles = df[
            (df.index.to_series().dt.date == current_date) &
            (df.index.to_series().dt.hour >= 6)
        ]

        for timestamp, candle in london_ny_candles.iterrows():
            if direction == 'high' and candle['high'] > target_price:
                return timestamp
            elif direction == 'low' and candle['low'] < target_price:
                return timestamp

        return None

    def _calculate_range_stats(self, session_candles: Dict) -> Dict:
        """
        Calculate range statistics across sessions.
        """
        stats = {}

        for session_name, candles in session_candles.items():
            if len(candles) > 0:
                range_pips = (candles['high'].max() - candles['low'].min()) * 10000
                stats[session_name] = {
                    'range_pips': range_pips,
                    'attempts': self._count_range_attempts(candles)
                }

        return stats

    def _count_range_attempts(self, candles: pd.DataFrame) -> int:
        """
        Count how many times price attempted to break the range.
        """
        if len(candles) < 3:
            return 0

        session_high = candles['high'].max()
        session_low = candles['low'].min()
        attempts = 0

        for _, candle in candles.iterrows():
            # Check for break attempts
            if (candle['high'] > session_high * 0.9995 or
                candle['low'] < session_low * 1.0005):
                attempts += 1

        return attempts

    def get_session_entry_signal(self, current_session: Session,
                               current_price: float,
                               session_analysis: Dict) -> Dict:
        """
        Generate session-specific entry signals.
        """
        signal = {
            'session': current_session.value[2],
            'bias': 'NEUTRAL',
            'entry_zone': None,
            'manipulation_risk': 'LOW',
            'confidence': 0
        }

        if current_session == Session.ASIAN:
            signal['bias'] = 'RANGE_BOUND'
            signal['entry_zone'] = 'Trade range extremes'
            signal['confidence'] = 60

        elif current_session == Session.LONDON:
            # Check if breaking Asian range
            asian_stats = session_analysis.get('sessions', {}).get('Asian', {})
            if asian_stats:
                asian_range = asian_stats['session_high'] - asian_stats['session_low']
                if current_price > asian_stats['session_high'] + asian_range * 0.1:
                    signal['bias'] = 'BULLISH_BREAKOUT'
                    signal['confidence'] = 70
                elif current_price < asian_stats['session_low'] - asian_range * 0.1:
                    signal['bias'] = 'BEARISH_BREAKOUT'
                    signal['confidence'] = 70

            signal['manipulation_risk'] = 'HIGH'

        elif current_session == Session.NY:
            # Check for London session reversal
            london_stats = session_analysis.get('sessions', {}).get('London', {})
            if london_stats and london_stats.get('direction') != 'NEUTRAL':
                signal['bias'] = 'CONTINUATION'
                signal['confidence'] = 65
            else:
                signal['bias'] = 'UNCERTAIN'
                signal['confidence'] = 40

        return signal

    def predict_next_session_behavior(self, current_session: Session,
                                     session_analysis: Dict) -> Dict:
        """
        Predict behavior in the next session.
        """
        prediction = {
            'next_session': None,
            'expected_activity': 'UNKNOWN',
            'key_levels': []
        }

        # Determine next session
        if current_session == Session.ASIAN:
            next_session = Session.LONDON
            expected_activity = 'Breakout potential'
        elif current_session == Session.LONDON:
            next_session = Session.OVERLAP
            expected_activity = 'High volatility'
        elif current_session == Session.OVERLAP:
            next_session = Session.NY
            expected_activity = 'Trend confirmation'
        else:
            next_session = Session.QUIET
            expected_activity = 'Positioning'

        prediction['next_session'] = next_session.value[2]
        prediction['expected_activity'] = expected_activity

        # Add key levels
        if 'Asian' in session_analysis.get('sessions', {}):
            asian_stats = session_analysis['sessions']['Asian']
            prediction['key_levels'] = [
                {'level': asian_stats['session_high'], 'type': 'resistance'},
                {'level': asian_stats['session_low'], 'type': 'support'}
            ]

        return prediction