"""
Signal Validator for ACB-Enhanced Patterns
==========================================

Validates trading signals using multiple layers of analysis:
1. Technical validation (price action, levels)
2. Session context validation
3. Manipulation pattern validation
4. Risk/Reward validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation confidence levels."""
    HIGH = "HIGH"          # Strong validation, proceed
    MEDIUM = "MEDIUM"      # Moderate validation, caution
    LOW = "LOW"           # Weak validation, avoid
    REJECT = "REJECT"      # Invalid, do not trade


class ValidationReason(Enum):
    """Specific validation reasons."""
    # Valid reasons
    STRONG_DMR_ALIGNMENT = "Price at key DMR level"
    ACB_BREAKOUT_CONFIRMED = "Confirmed ACB breakout"
    OPTIMAL_SESSION_CONTEXT = "Optimal session for pattern"
    LIQUIDITY_HUNT_COMPLETE = "Liquidity hunt completed"
    STRONG_CONFLUENCE = "Multiple factors aligned"

    # Caution reasons
    PARTIAL_DMR_ALIGNMENT = "Near but not at DMR"
    WEAK_SESSION_CONTEXT = "Suboptimal session timing"
    UNCLEAR_MANIPULATION = "Manipulation phase unclear"

    # Rejection reasons
    AGAINST_TREND = "Signal against dominant trend"
    NO_DMR_ALIGNMENT = "No DMR level proximity"
    POOR_RISK_REWARD = "Risk/Reward ratio too low"
    VOLATILE_MARKET = "Market too volatile"
    NEWS_EVENT_RISK = "High-impact news pending"


@dataclass
class ValidationResult:
    """Result of signal validation."""
    level: ValidationLevel
    reason: ValidationReason
    confidence: int
    details: Dict
    recommendations: List[str]
    risk_factors: List[str]


class SignalValidator:
    """
    Validates trading signals using ACB methodology.
    Ensures signals meet strict criteria before execution.
    """

    def __init__(self):
        self.min_confidence_threshold = 70
        self.min_risk_reward = 2.0
        self.max_distance_from_dmr = 0.00080  # 80 pips
        self.required_confluence_factors = 2

    def validate_signal(self, signal: Dict,
                       df: pd.DataFrame,
                       dmr_levels: Dict,
                       acb_levels: Dict,
                       session_analysis: Optional[Dict] = None,
                       market_context: Optional[Dict] = None) -> ValidationResult:
        """
        Comprehensive signal validation.

        Args:
            signal: Signal information from detector
            df: Price data
            dmr_levels: DMR level information
            acb_levels: ACB level information
            session_analysis: Session behavior data
            market_context: Additional market context

        Returns:
            ValidationResult with recommendation
        """
        validations = []
        risk_factors = []
        recommendations = []

        # 1. Technical validation
        tech_validation = self._validate_technical_setup(
            signal, df, dmr_levels, acb_levels
        )
        validations.append(tech_validation)

        # 2. Session validation
        if session_analysis:
            session_validation = self._validate_session_context(
                signal, session_analysis
            )
            validations.append(session_validation)

        # 3. Pattern validation
        pattern_validation = self._validate_pattern_quality(signal, df)
        validations.append(pattern_validation)

        # 4. Risk/Reward validation
        risk_validation = self._validate_risk_reward(
            signal, df, dmr_levels, acb_levels
        )
        validations.append(risk_validation)

        # 5. Market context validation
        if market_context:
            context_validation = self._validate_market_context(
                signal, market_context
            )
            validations.append(context_validation)

        # Aggregate validations
        final_validation = self._aggregate_validations(validations)

        # Compile risk factors
        for validation in validations:
            risk_factors.extend(validation.risk_factors)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            final_validation, validations
        )

        return ValidationResult(
            level=final_validation.level,
            reason=final_validation.reason,
            confidence=final_validation.confidence,
            details={
                'signal_type': signal.get('signal_type'),
                'signal_grade': signal.get('signal_grade'),
                'confluence_score': len([v for v in validations if v.level == ValidationLevel.HIGH]),
                'validation_summary': [v.level.value for v in validations]
            },
            recommendations=recommendations,
            risk_factors=risk_factors
        )

    def _validate_technical_setup(self, signal: Dict,
                                  df: pd.DataFrame,
                                  dmr_levels: Dict,
                                  acb_levels: Dict) -> ValidationResult:
        """Validate technical aspects of the setup."""
        current_price = signal.get('current_price')
        signal_type = signal.get('signal_type')

        # Check DMR alignment
        dmr_validation = signal.get('dmr_validation', {})
        is_near_dmr = dmr_validation.get('is_near_dmr', False)
        dmr_distance = dmr_validation.get('dmr_distance', float('inf'))

        # Check ACB alignment
        acb_validation = signal.get('acb_validation', {})
        has_acb_proximity = acb_validation.get('has_acb_proximity', False)

        # Initialize with neutral validation
        validation = ValidationResult(
            level=ValidationLevel.MEDIUM,
            reason=ValidationReason.PARTIAL_DMR_ALIGNMENT,
            confidence=50,
            details={},
            recommendations=[],
            risk_factors=[]
        )

        # Strong DMR alignment
        if is_near_dmr and dmr_distance < 30:  # Less than 30 pips
            validation.level = ValidationLevel.HIGH
            validation.reason = ValidationReason.STRONG_DMR_ALIGNMENT
            validation.confidence = 85

        # ACB breakout potential
        elif has_acb_proximity and acb_validation.get('acb_break_potential'):
            validation.level = ValidationLevel.HIGH
            validation.reason = ValidationReason.ACB_BREAKOUT_CONFIRMED
            validation.confidence = 80

        # No alignment
        elif not is_near_dmr and not has_acb_proximity:
            validation.level = ValidationLevel.LOW
            validation.reason = ValidationReason.NO_DMR_ALIGNMENT
            validation.confidence = 30
            validation.risk_factors.append("No key level alignment")

        return validation

    def _validate_session_context(self, signal: Dict,
                                 session_analysis: Dict) -> ValidationResult:
        """Validate session context for the signal."""
        session_context = signal.get('session_context', {})
        signal_type = signal.get('signal_type')

        validation = ValidationResult(
            level=ValidationLevel.MEDIUM,
            reason=ValidationReason.WEAK_SESSION_CONTEXT,
            confidence=60,
            details={},
            recommendations=[],
            risk_factors=[]
        )

        # Check session alignment
        if session_context.get('session_alignment'):
            validation.level = ValidationLevel.HIGH
            validation.reason = ValidationReason.OPTIMAL_SESSION_CONTEXT
            validation.confidence = 80
        else:
            validation.risk_factors.append("Suboptimal session timing")
            validation.recommendations.append("Wait for optimal session")

        # Check for liquidity hunt
        if session_context.get('liquidity_hunt_likely'):
            validation.level = ValidationLevel.HIGH
            validation.reason = ValidationReason.LIQUIDITY_HUNT_COMPLETE
            validation.confidence = 90
            validation.recommendations.append("Wait for hunt completion before entry")

        return validation

    def _validate_pattern_quality(self, signal: Dict,
                                 df: pd.DataFrame) -> ValidationResult:
        """Validate the quality of the pattern."""
        signal_grade = signal.get('signal_grade')
        consecutive_count = signal.get('consecutive_count', 0)
        confidence = signal.get('confidence', 0)

        validation = ValidationResult(
            level=ValidationLevel.MEDIUM,
            reason=ValidationReason.STRONG_CONFLUENCE,
            confidence=confidence,
            details={},
            recommendations=[],
            risk_factors=[]
        )

        # Adjust based on pattern grade
        if signal_grade in ['A+', 'A']:
            validation.level = ValidationLevel.HIGH
            validation.confidence = min(95, confidence + 10)
        elif signal_grade == 'B+':
            validation.confidence = max(50, confidence - 10)
        else:
            validation.level = ValidationLevel.LOW
            validation.confidence = max(30, confidence - 20)
            validation.risk_factors.append("Weak pattern grade")

        # Check pattern structure
        if consecutive_count == 3:
            validation.reason = ValidationReason.STRONG_CONFLUENCE
            validation.confidence = min(95, validation.confidence + 5)

        return validation

    def _validate_risk_reward(self, signal: Dict,
                             df: pd.DataFrame,
                             dmr_levels: Dict,
                             acb_levels: Dict) -> ValidationResult:
        """Validate risk/reward ratio."""
        risk_params = signal.get('risk_parameters', {})
        stop_distance = risk_params.get('stop_loss_distance', 0)
        profit_distance = risk_params.get('take_profit_distance', 0)

        if stop_distance > 0 and profit_distance > 0:
            risk_reward = profit_distance / stop_distance
        else:
            risk_reward = 1.5  # Default assumption

        validation = ValidationResult(
            level=ValidationLevel.MEDIUM,
            reason=ValidationReason.STRONG_CONFLUENCE,
            confidence=70,
            details={'risk_reward_ratio': risk_reward},
            recommendations=[],
            risk_factors=[]
        )

        if risk_reward >= self.min_risk_reward:
            if risk_reward >= 3.0:
                validation.level = ValidationLevel.HIGH
                validation.confidence = 90
            else:
                validation.confidence = 75
        else:
            validation.level = ValidationLevel.REJECT
            validation.reason = ValidationReason.POOR_RISK_REWARD
            validation.confidence = 20
            validation.risk_factors.append(f"Risk/Reward ratio {risk_reward:.1f} below threshold")

        return validation

    def _validate_market_context(self, signal: Dict,
                                market_context: Dict) -> ValidationResult:
        """Validate broader market context."""
        validation = ValidationResult(
            level=ValidationLevel.MEDIUM,
            reason=ValidationReason.STRONG_CONFLUENCE,
            confidence=70,
            details={},
            recommendations=[],
            risk_factors=[]
        )

        # Check for high-impact news
        if market_context.get('high_impact_news'):
            validation.level = ValidationLevel.REJECT
            validation.reason = ValidationReason.NEWS_EVENT_RISK
            validation.confidence = 10
            validation.risk_factors.append("High-impact news event pending")

        # Check market volatility
        volatility = market_context.get('volatility', 0)
        if volatility > 0.00200:  # Very high volatility
            validation.level = ValidationLevel.LOW
            validation.reason = ValidationReason.VOLATILE_MARKET
            validation.confidence = 40
            validation.risk_factors.append("Market too volatile")

        # Check trend alignment
        trend = market_context.get('trend')
        signal_type = signal.get('signal_type')
        if trend and signal_type:
            if (trend == 'BULLISH' and signal_type in ['FGD', '3DS']) or \
               (trend == 'BEARISH' and signal_type in ['FRD', '3DL']):
                validation.level = ValidationLevel.HIGH
                validation.confidence = 85
            elif (trend == 'BULLISH' and signal_type in ['FRD', '3DL']) or \
                 (trend == 'BEARISH' and signal_type in ['FGD', '3DS']):
                validation.level = ValidationLevel.LOW
                validation.reason = ValidationReason.AGAINST_TREND
                validation.confidence = 35
                validation.risk_factors.append("Signal against dominant trend")

        return validation

    def _aggregate_validations(self,
                              validations: List[ValidationResult]) -> ValidationResult:
        """Aggregate multiple validations into final decision."""
        high_count = sum(1 for v in validations if v.level == ValidationLevel.HIGH)
        medium_count = sum(1 for v in validations if v.level == ValidationLevel.MEDIUM)
        reject_count = sum(1 for v in validations if v.level == ValidationLevel.REJECT)

        # Any rejection means reject
        if reject_count > 0:
            return ValidationResult(
                level=ValidationLevel.REJECT,
                reason=ValidationReason.NO_DMR_ALIGNMENT,
                confidence=10,
                details={'reject_count': reject_count},
                recommendations=["Do not trade this setup"],
                risk_factors=[]
            )

        # High confidence if multiple validations agree
        if high_count >= self.required_confluence_factors:
            avg_confidence = sum(v.confidence for v in validations) / len(validations)
            return ValidationResult(
                level=ValidationLevel.HIGH,
                reason=ValidationReason.STRONG_CONFLUENCE,
                confidence=int(avg_confidence),
                details={'high_validations': high_count},
                recommendations=["Consider this setup"],
                risk_factors=[]
            )

        # Medium confidence
        elif medium_count >= 2:
            avg_confidence = sum(v.confidence for v in validations) / len(validations)
            return ValidationResult(
                level=ValidationLevel.MEDIUM,
                reason=ValidationReason.PARTIAL_DMR_ALIGNMENT,
                confidence=int(avg_confidence),
                details={'medium_validations': medium_count},
                recommendations=["Proceed with caution"],
                risk_factors=["Mixed validation results"]
            )

        # Low confidence
        else:
            return ValidationResult(
                level=ValidationLevel.LOW,
                reason=ValidationReason.WEAK_SESSION_CONTEXT,
                confidence=30,
                details={},
                recommendations=["Wait for better setup"],
                risk_factors=["Insufficient confluence"]
            )

    def _generate_recommendations(self,
                                 final_validation: ValidationResult,
                                 all_validations: List[ValidationResult]) -> List[str]:
        """Generate trading recommendations based on validation."""
        recommendations = []

        if final_validation.level == ValidationLevel.HIGH:
            recommendations.extend([
                "Setup meets all criteria",
                "Consider entering on next session",
                "Use proper risk management"
            ])
        elif final_validation.level == ValidationLevel.MEDIUM:
            recommendations.extend([
                "Setup has some merits",
                "Wait for additional confirmation",
                "Reduce position size if trading"
            ])
        elif final_validation.level == ValidationLevel.LOW:
            recommendations.extend([
                "Weak setup, better to wait",
                "Look for stronger opportunities",
                "Monitor for improvement"
            ])
        else:  # REJECT
            recommendations.extend([
                "Do not trade this setup",
                "Risk factors outweigh benefits",
                "Wait for next opportunity"
            ])

        # Add specific recommendations from validations
        for validation in all_validations:
            recommendations.extend(validation.recommendations)

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations