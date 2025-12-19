"""
Alert Manager
=============

Manages real-time trading alerts and notifications.

Features:
- Pattern trigger alerts
- Entry zone activation
- Stop loss and target notifications
- Alert history and tracking
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum


class AlertType(Enum):
    """Types of trading alerts"""
    PATTERN_FORMING = "Pattern Forming"
    ENTRY_TRIGGERED = "Entry Triggered"
    STOP_LOSS = "Stop Loss Alert"
    PROFIT_TARGET = "Profit Target"
    PATTERN_INVALIDATED = "Pattern Invalidated"
    VOLUME_SPIKE = "Volume Spike"
    BREAKOUT_CONFIRMED = "Breakout Confirmed"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AlertManager:
    """
    Manages all real-time trading alerts and notifications.

    Tracks patterns, entry zones, and provides timely alerts
    for trading opportunities.
    """

    def __init__(self):
        """Initialize the alert manager"""
        self.active_alerts = []
        self.alert_history = []
        self.alert_counter = 0
        self.max_history = 1000

        # Alert settings
        self.alert_settings = {
            'pattern_alerts': True,
            'entry_zone_alerts': True,
            'volume_alerts': True,
            'breakout_alerts': True,
            'minimum_priority': AlertPriority.MEDIUM
        }

        print("Alert Manager initialized")

    def update_alerts(self, opportunities: List[Dict], current_price: float) -> List[Dict]:
        """
        Update alerts based on current opportunities and price.

        Args:
            opportunities: List of current trading opportunities
            current_price: Current market price

        Returns:
            List of active alerts
        """
        # Clear old alerts
        self._clear_old_alerts()

        # Check for new alerts
        new_alerts = self._check_opportunity_alerts(opportunities, current_price)
        self.active_alerts.extend(new_alerts)

        # Check price-based alerts
        price_alerts = self._check_price_alerts(current_price)
        self.active_alerts.extend(price_alerts)

        # Update alert history
        self._update_history()

        return self.active_alerts

    def get_active_alerts(self, priority: Optional[AlertPriority] = None) -> List[Dict]:
        """
        Get currently active alerts.

        Args:
            priority: Filter by priority level (optional)

        Returns:
            List of active alerts
        """
        if priority:
            return [alert for alert in self.active_alerts
                   if alert['priority'] == priority]
        return self.active_alerts

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """
        Get alert history for specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history
               if alert['timestamp'] > cutoff_time]

    def clear_alert(self, alert_id: int) -> bool:
        """
        Clear a specific alert.

        Args:
            alert_id: ID of alert to clear

        Returns:
            True if alert was cleared
        """
        for i, alert in enumerate(self.active_alerts):
            if alert['id'] == alert_id:
                self.active_alerts.pop(i)
                return True
        return False

    def _check_opportunity_alerts(self, opportunities: List[Dict], current_price: float) -> List[Dict]:
        """Check for alerts based on trading opportunities"""
        alerts = []

        for opp in opportunities:
            # High priority pattern forming
            if opp.get('priority') == 'HIGH':
                alerts.append(self._create_alert(
                    alert_type=AlertType.PATTERN_FORMING,
                    priority=AlertPriority.HIGH,
                    message=f"{opp['type']} - {opp.get('status', 'Unknown')}",
                    details=opp,
                    entry_zone=opp.get('entry_zone')
                ))

            # Entry zone triggered
            entry_zone = opp.get('entry_zone')
            if entry_zone:
                entry_min = entry_zone.get('min', 0)
                entry_max = entry_zone.get('max', float('inf'))

                if entry_min <= current_price <= entry_max:
                    alerts.append(self._create_alert(
                        alert_type=AlertType.ENTRY_TRIGGERED,
                        priority=AlertPriority.CRITICAL,
                        message=f"ENTRY ZONE ACTIVE: {opp['type']}",
                        details=opp,
                        actionable=True
                    ))

            # Pump & Dump special alert
            if 'PUMP_DUMP' in opp['type']:
                alerts.append(self._create_alert(
                    alert_type=AlertType.PATTERN_FORMING,
                    priority=AlertPriority.HIGH,
                    message=f"PUMP & DETECTED - {opp.get('reason', '')}",
                    details=opp,
                    urgent=True
                ))

        return alerts

    def _check_price_alerts(self, current_price: float) -> List[Dict]:
        """Check for price-based alerts"""
        alerts = []

        # Check proximity to active entry zones
        for alert in self.active_alerts:
            if 'entry_zone' in alert and alert['status'] == 'ACTIVE':
                zone = alert['entry_zone']
                if zone:
                    distance = abs(current_price - (zone['min'] + zone['max']) / 2)
                    if distance < 0.001:  # Very close
                        alerts.append(self._create_alert(
                            alert_type=AlertType.ENTRY_TRIGGERED,
                            priority=AlertPriority.CRITICAL,
                            message=f"PRICE IN ENTRY ZONE: {alert['message']}",
                            details={'current_price': current_price, 'distance': distance}
                        ))

        return alerts

    def _create_alert(self, alert_type: AlertType, priority: AlertPriority,
                     message: str, details: Optional[Dict] = None,
                     actionable: bool = False, urgent: bool = False,
                     entry_zone: Optional[Dict] = None) -> Dict:
        """Create a new alert"""
        self.alert_counter += 1

        alert = {
            'id': self.alert_counter,
            'type': alert_type,
            'priority': priority,
            'message': message,
            'timestamp': datetime.now(),
            'status': 'ACTIVE',
            'details': details or {},
            'actionable': actionable,
            'urgent': urgent,
            'entry_zone': entry_zone
        }

        return alert

    def _clear_old_alerts(self):
        """Clear old inactive alerts"""
        # Remove alerts older than 1 hour unless they're critical
        cutoff_time = datetime.now() - timedelta(hours=1)

        self.active_alerts = [
            alert for alert in self.active_alerts
            if (alert['timestamp'] > cutoff_time or
                alert['priority'] == AlertPriority.CRITICAL)
        ]

        # Remove duplicates
        seen_messages = set()
        unique_alerts = []

        for alert in self.active_alerts:
            message_key = (alert['type'], alert['message'])
            if message_key not in seen_messages:
                unique_alerts.append(alert)
                seen_messages.add(message_key)

        self.active_alerts = unique_alerts

    def _update_history(self):
        """Update alert history"""
        # Move inactive alerts to history
        for alert in self.active_alerts:
            if alert['status'] != 'ACTIVE' and alert not in self.alert_history:
                self.alert_history.append(alert)

        # Limit history size
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

    def print_alerts(self):
        """Print current alerts in formatted way"""
        if not self.active_alerts:
            print("\n[NO ACTIVE ALERTS]")
            return

        print(f"\n[ACTIVE ALERTS: {len(self.active_alerts)}]")
        print("-" * 50)

        # Sort by priority
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3
        }

        sorted_alerts = sorted(
            self.active_alerts,
            key=lambda x: priority_order.get(x['priority'], 4)
        )

        for alert in sorted_alerts:
            priority_indicator = {
                AlertPriority.CRITICAL: "🔴",
                AlertPriority.HIGH: "🟠",
                AlertPriority.MEDIUM: "🟡",
                AlertPriority.LOW: "🔵"
            }.get(alert['priority'], "⚪")

            time_str = alert['timestamp'].strftime('%H:%M:%S')

            print(f"{priority_indicator} [{time_str}] {alert['message']}")

            if alert.get('actionable'):
                print(f"   ▶ ACTIONABLE - Entry zone active")

            if alert.get('urgent'):
                print(f"   ⚠ URGENT - Immediate attention required")

            if 'entry_zone' in alert and alert['entry_zone']:
                zone = alert['entry_zone']
                print(f"   📍 Entry: {zone.get('min', 'N/A')} - {zone.get('max', 'N/A')}")

            print()

    def get_alert_summary(self) -> Dict:
        """Get summary of current alerts"""
        priority_counts = {}
        for priority in AlertPriority:
            priority_counts[priority.value] = sum(
                1 for alert in self.active_alerts
                if alert['priority'] == priority
            )

        return {
            'total_alerts': len(self.active_alerts),
            'by_priority': priority_counts,
            'actionable_count': sum(1 for alert in self.active_alerts if alert.get('actionable')),
            'urgent_count': sum(1 for alert in self.active_alerts if alert.get('urgent')),
            'most_recent': max([alert['timestamp'] for alert in self.active_alerts]) if self.active_alerts else None
        }