"""
Trade Management Module - Phase 5 Component
===========================================

This module provides tools for managing trades:
- DMR-Aware Target Calculator
- Dynamic Stop Loss Calculator
- Risk Management Tools
"""

from .target_calculator import DMRTargetCalculator, TargetType, TargetPriority

__all__ = [
    'DMRTargetCalculator',
    'TargetType',
    'TargetPriority'
]