"""
ACB Real-Time Monitoring Module
===============================

Enhances the existing signal generator with real-time pattern detection
and trading opportunity identification.

Components:
- ACBRealtimeMonitor: Main wrapper class
- RealTimePatternScanner: Detects developing patterns
- EntryZoneCalculator: Calculates entry/stop/target levels
- AlertManager: Manages real-time alerts
"""

from .realtime_monitor import ACBRealtimeMonitor
from .pattern_scanner import RealTimePatternScanner
from .entry_zones import EntryZoneCalculator
from .alert_manager import AlertManager

__all__ = [
    'ACBRealtimeMonitor',
    'RealTimePatternScanner',
    'EntryZoneCalculator',
    'AlertManager'
]