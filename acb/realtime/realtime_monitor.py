"""
ACB Real-Time Monitor - Main Wrapper
===================================

Wraps the existing ACB signal generator with real-time pattern detection
and trading opportunity identification.

This is the main class that coordinates all real-time monitoring components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..signal.signal_generator import ACBAwareSignalGenerator
from ..session_analyzer import SessionAnalyzer
from .pattern_scanner import RealTimePatternScanner
from .entry_zones import EntryZoneCalculator
from .alert_manager import AlertManager
from .trader_display import TraderDisplay


class ACBRealtimeMonitor:
    """
    Real-time wrapper that enhances existing signal generator with live pattern detection.

    Architecture:
    - Base Generator: Existing signal analysis (unchanged)
    - Pattern Scanner: Detects developing patterns in real-time
    - Entry Calculator: Provides precise entry/stop/target levels
    - Alert Manager: Manages and triggers alerts
    """

    def __init__(self, base_generator: Optional[ACBAwareSignalGenerator] = None):
        """
        Initialize the real-time monitor.

        Args:
            base_generator: Optional existing signal generator instance
        """
        # Core components
        self.base_generator = base_generator or ACBAwareSignalGenerator()
        self.session_analyzer = SessionAnalyzer()

        # Real-time components
        self.scanner = RealTimePatternScanner()
        self.zone_calc = EntryZoneCalculator()
        self.alert_manager = AlertManager()
        self.display = TraderDisplay()  # Clean trader-focused display

        # State tracking
        self.active_patterns = {}
        self.last_update = None
        self.last_price = None

        print("ACB Real-Time Monitor initialized")
        print("=" * 50)

    def get_enhanced_signals(self,
                           df_h1: pd.DataFrame,
                           df_d1: pd.DataFrame,
                           current_price: float,
                           symbol: str = "USDJPY") -> Dict:
        """
        Get comprehensive analysis including base signals and real-time opportunities.

        Args:
            df_h1: Hourly DataFrame
            df_d1: Daily DataFrame
            current_price: Current market price
            symbol: Trading symbol

        Returns:
            Dictionary with base analysis + real-time opportunities
        """
        timestamp = datetime.now()
        self.last_update = timestamp
        self.last_price = current_price

        print(f"\n{'='*60}")
        print(f"ACB REAL-TIME MONITOR - {symbol}")
        print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Price: {current_price:.3f}")
        print(f"{'='*60}")

        # 1. Get existing analysis (unchanged)
        print("\n[BASE ANALYSIS]")
        print("-" * 30)
        base_signals = self.base_generator.generate_signals(df_h1, df_d1, symbol)

        # 2. Get session analysis
        print("\n[SESSION ANALYSIS]")
        print("-" * 30)
        sessions = self.session_analyzer.analyze_session_behavior(df_h1)
        current_session = self._get_current_session(sessions)

        # 3. Scan for real-time patterns
        print("\n[REAL-TIME PATTERN SCAN]")
        print("-" * 30)
        realtime_opportunities = self.scanner.scan_patterns(df_h1, current_price, sessions)

        # 4. Calculate entry zones
        print("\n[ENTRY ZONE CALCULATION]")
        print("-" * 30)
        active_zones = self.zone_calc.get_zones(df_h1, current_price, sessions)

        # 5. Manage alerts
        print("\n[ALERT MANAGER]")
        print("-" * 30)
        alerts = self.alert_manager.update_alerts(realtime_opportunities, current_price)

        # 6. Compile comprehensive results
        enhanced_signals = {
            # Metadata
            'symbol': symbol,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'current_price': current_price,

            # Base analysis (unchanged)
            'base_analysis': base_signals,

            # Session information
            'current_session': current_session,

            # Real-time enhancements
            'realtime_opportunities': realtime_opportunities,
            'active_entry_zones': active_zones,
            'active_alerts': alerts,

            # Summary
            'summary': self._create_summary(realtime_opportunities, current_session, current_price)
        }

        # Display clean trader report
        self.display.display_trader_report(enhanced_signals)

        return enhanced_signals

    def _get_current_session(self, sessions: Dict) -> Dict:
        """Determine current trading session based on user's trading hours"""
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        utc_hour = utc_now.hour
        utc_day = utc_now.weekday()  # 0=Monday, 6=Sunday

        # Check if weekend
        if utc_day >= 5:  # Saturday (5) or Sunday (6)
            return {'session': 'CLOSED', 'status': 'Weekend - Market Closed', 'hours_remaining': 0}

        # User's specific trading hours
        # London Session: 7AM-10AM UTC (2AM-5AM EST)
        if 7 <= utc_hour < 10:
            return {
                'session': 'LONDON SESSION',
                'status': 'YOUR TRADING TIME - ACTIVE',
                'hours_remaining': 10 - utc_hour,
                'note': 'Prime trading window'
            }
        # New York Session: 12PM-15PM UTC (7AM-10AM EST)
        elif 12 <= utc_hour < 15:
            return {
                'session': 'NEW YORK SESSION',
                'status': 'YOUR TRADING TIME - ACTIVE',
                'hours_remaining': 15 - utc_hour,
                'note': 'Prime trading window'
            }
        # Asian Session (user doesn't trade) - 19:00-00:00 EST = 0:00-5:00 UTC
        elif 0 <= utc_hour < 5:
            return {
                'session': 'ASIAN SESSION',
                'status': 'NO TRADING - Asian Session',
                'hours_remaining': 5 - utc_hour,
                'note': f'Asian closes in {5 - utc_hour} hours, London opens in {7 - utc_hour} hours'
            }
        # Early Morning/Late Asian (5-7 UTC) - No Trading
        elif 5 <= utc_hour < 7:
            return {
                'session': 'EARLY MORNING',
                'status': 'NO TRADING - Outside Window',
                'hours_remaining': 7 - utc_hour,
                'note': f'London opens in {7 - utc_hour} hours'
            }
        # Late NY/Australian (user doesn't trade)
        elif 15 <= utc_hour < 24:
            return {
                'session': 'LATE NY/AUSSIE',
                'status': 'NO TRADING - Outside Window',
                'hours_remaining': 24 - utc_hour + 7,  # Until London open
                'note': f'London opens in {24 - utc_hour + 7} hours'
            }
        # Mid-Day Asian (10-12 UTC)
        else:  # 10 <= utc_hour < 12
            return {
                'session': 'MID-DAY ASIAN',
                'status': 'NO TRADING - Outside Window',
                'hours_remaining': 12 - utc_hour,  # Until NY open
                'note': f'New York opens in {12 - utc_hour} hours'
            }

    def _create_summary(self, opportunities: List, session: Dict, price: float) -> Dict:
        """Create a summary of current opportunities"""
        return {
            'total_opportunities': len(opportunities),
            'high_priority': sum(1 for o in opportunities if o.get('priority', 'MEDIUM') == 'HIGH'),
            'session_bias': self._calculate_session_bias(opportunities, session),
            'key_level': self._find_key_level(opportunities, price),
            'action_required': len([o for o in opportunities if 'Entry' in o.get('status', '')])
        }

    def _calculate_session_bias(self, opportunities: List, session: Dict) -> str:
        """Calculate overall bias from opportunities"""
        bullish = sum(1 for o in opportunities if 'LONG' in o.get('type', '').upper())
        bearish = sum(1 for o in opportunities if 'SHORT' in o.get('type', '').upper())

        if bullish > bearish:
            return "BULLISH"
        elif bearish > bullish:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _find_key_level(self, opportunities: List, price: float) -> Optional[float]:
        """Find the most important level near current price"""
        if not opportunities:
            return None

        closest = None
        min_distance = float('inf')

        for opp in opportunities:
            entry = opp.get('entry', 0)
            distance = abs(entry - price)
            if distance < min_distance:
                min_distance = distance
                closest = entry

        return closest

    def _print_enhanced_summary(self, signals: Dict):
        """Print formatted summary of enhanced signals"""
        print(f"\n{'='*60}")
        print("ENHANCED SIGNALS SUMMARY")
        print(f"{'='*60}")

        # Current status
        print(f"\nCurrent Price: {signals['current_price']:.3f}")
        print(f"Session: {signals['current_session']['session']} ({signals['current_session']['status']})")

        # Opportunities
        opps = signals['realtime_opportunities']
        print(f"\nActive Opportunities: {len(opps)}")
        if opps:
            for i, opp in enumerate(opps, 1):
                print(f"  {i}. {opp['type']} - {opp.get('status', 'Unknown')}")
                print(f"     Entry: {opp.get('entry', 'N/A')} | Stop: {opp.get('stop', 'N/A')}")

        # Entry zones
        zones = signals['active_entry_zones']
        print(f"\nEntry Zones Active:")
        for zone_type, zone in zones.items():
            print(f"  - {zone_type}: {zone}")

        # Summary
        summary = signals['summary']
        print(f"\nBias: {summary['session_bias']}")
        print(f"Key Level: {summary.get('key_level', 'None')}")
        print(f"High Priority Setups: {summary['high_priority']}")

        print(f"\n{'='*60}")
        print("MONITORING FOR REAL-TIME OPPORTUNITIES...")
        print(f"{'='*60}\n")

    def check_pattern_updates(self, current_price: float) -> List[Dict]:
        """
        Quick update check for pattern status changes.

        Args:
            current_price: Current market price

        Returns:
            List of pattern status changes
        """
        if not self.active_patterns:
            return []

        updates = []
        for pattern_id, pattern in self.active_patterns.items():
            # Check if pattern triggered
            if self._check_pattern_trigger(pattern, current_price):
                updates.append({
                    'pattern_id': pattern_id,
                    'status': 'TRIGGERED',
                    'action': 'EXECUTE_ENTRY',
                    'price': current_price
                })

            # Check if pattern invalidated
            elif self._check_pattern_invalidation(pattern, current_price):
                updates.append({
                    'pattern_id': pattern_id,
                    'status': 'INVALIDATED',
                    'action': 'CANCEL',
                    'price': current_price
                })

        return updates

    def _check_pattern_trigger(self, pattern: Dict, price: float) -> bool:
        """Check if pattern entry criteria are met"""
        entry_zone = pattern.get('entry_zone', {})
        if not entry_zone:
            return False

        entry_min = entry_zone.get('min', 0)
        entry_max = entry_zone.get('max', float('inf'))

        return entry_min <= price <= entry_max

    def _check_pattern_invalidation(self, pattern: Dict, price: float) -> bool:
        """Check if pattern is invalidated"""
        stop_loss = pattern.get('stop_loss')
        if not stop_loss:
            return False

        # For long patterns
        if 'LONG' in pattern.get('type', ''):
            return price < stop_loss
        # For short patterns
        elif 'SHORT' in pattern.get('type', ''):
            return price > stop_loss

        return False