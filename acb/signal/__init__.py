"""
Signal Generation Module - Phase 4 Component
============================================

This module provides tools for generating and prioritizing trading signals:
- ACB-Aware Signal Generator
- 5-Star Setup Prioritizer
- Enhanced Signal Validation
"""

from .signal_generator import (
    ACBAwareSignalGenerator,
    SignalType,
    SignalConfidence
)
from .setup_prioritizer import (
    FiveStarSetupPrioritizer,
    SetupType,
    SetupRating
)

__all__ = [
    'ACBAwareSignalGenerator',
    'SignalType',
    'SignalConfidence',
    'FiveStarSetupPrioritizer',
    'SetupType',
    'SetupRating'
]