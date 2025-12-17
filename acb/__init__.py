"""
ACB (Ain't Coming Back) Enhanced Scanner Modules
===============================================

Core modules for implementing Stacey Burke's ACB methodology:
- ACB Level Detection
- DMR Level Calculation
- Session Analysis
- Enhanced FRD/FGD Patterns
- Signal Validation
- Market Structure Understanding
- Smart Money Manipulation Analysis (Phase 3)
"""

from .detector import ACBDetector
from .dmr_calculator import DMRLevelCalculator
from .session_analyzer import SessionAnalyzer
from .patterns.frd_fgd import EnhancedFRDFGDDetector, SignalType, SignalGrade
from .patterns.enhanced_frd_fgd import AsianRangeEntryDetector
from .patterns.signal_validator import SignalValidator, ValidationLevel
from .manipulation.liquidity_hunt_detector import LiquidityHuntDetector, LiquidityHuntType, HuntStrength
from .manipulation.market_phase_identifier import MarketPhaseIdentifier, MarketPhase, PhaseConfidence

__all__ = [
    'ACBDetector',
    'DMRLevelCalculator',
    'SessionAnalyzer',
    'EnhancedFRDFGDDetector',
    'AsianRangeEntryDetector',
    'SignalValidator',
    'SignalType',
    'SignalGrade',
    'ValidationLevel',
    # Phase 3 - Smart Money Manipulation Analysis
    'LiquidityHuntDetector',
    'LiquidityHuntType',
    'HuntStrength',
    'MarketPhaseIdentifier',
    'MarketPhase',
    'PhaseConfidence'
]