"""
Enhanced ACB Integration Module
================================

Integrates the enhanced DMR and ACB detectors with FGD/FRD patterns.
Provides complete Stacey Burke methodology implementation.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .enhanced_acb_detector import EnhancedACBDetector
from .dmr_calculator import DMRLevelCalculator
from .patterns.frd_fgd import EnhancedFRDFGDDetector


class EnhancedACBIntegration:
    """
    Enhanced integration of all ACB components.
    """

    def __init__(self):
        self.dmr_calculator = DMRLevelCalculator()
        self.enhanced_acb_detector = EnhancedACBDetector()
        self.fgd_detector = EnhancedFRDFGDDetector()

    def analyze_market_complete(self, df: pd.DataFrame,
                              current_time: datetime = None) -> Dict:
        """
        Complete market analysis with all enhanced components.
        """
        if current_time is None:
            current_time = datetime.now()

        # 1. Calculate all DMR levels (enhanced)
        dmr_levels = self.dmr_calculator.calculate_all_dmr_levels(df, current_time)

        # 2. Identify all ACB levels (enhanced)
        acb_levels = self.enhanced_acb_detector.identify_enhanced_acb_levels(
            df, dmr_levels, current_time
        )

        # 3. Convert H1 to daily for FGD/FRD detection
        df_daily = self._convert_to_daily(df)

        # 4. Detect FGD/FRD patterns
        pattern_result = self.fgd_detector.detect_enhanced_frd_fgd(
            df_daily, dmr_levels, acb_levels
        )

        # 5. Check for Monday breakout (special DMR case)
        monday_breakout = self.dmr_calculator.check_monday_breakout(df, current_time)

        # 6. Analyze market structure
        market_structure = self._analyze_market_structure(df, dmr_levels, acb_levels)

        # 7. Generate comprehensive report
        return {
            'timestamp': current_time,
            'dmr_levels': dmr_levels,
            'acb_levels': acb_levels,
            'fgd_pattern': pattern_result,
            'monday_breakout': monday_breakout,
            'market_structure': market_structure,
            'trading_opportunities': self._identify_trading_opportunities(
                pattern_result, dmr_levels, acb_levels
            ),
            'confidence_analysis': self._analyze_confidence(
                pattern_result, dmr_levels, acb_levels
            )
        }

    def _analyze_market_structure(self, df: pd.DataFrame,
                                  dmr_levels: Dict,
                                  acb_levels: Dict) -> Dict:
        """Analyze overall market structure."""
        current_price = df.iloc[-1]['close']

        structure = {
            'current_price': current_price,
            'nearest_dmr_above': None,
            'nearest_dmr_below': None,
            'nearest_acb': None,
            'extreme_levels': [],
            'breakout_candidates': []
        }

        # Find nearest DMR levels
        all_dmr = []
        for level_type in ['daily', 'three_day', 'weekly', 'monthly']:
            for direction in ['high', 'low']:
                level = dmr_levels.get(level_type, {}).get(direction)
                if level and level.get('price'):
                    all_dmr.append(level)

        # Sort by distance
        all_dmr.sort(key=lambda x: abs(x['price'] - current_price))

        for level in all_dmr[:5]:  # Top 5 nearest
            if level['price'] > current_price:
                structure['nearest_dmr_above'] = level
            else:
                structure['nearest_dmr_below'] = level

        # Check extreme levels
        if dmr_levels.get('extreme_closes'):
            if dmr_levels['extreme_closes'].get('high'):
                structure['extreme_levels'].append(dmr_levels['extreme_closes']['high'])
            if dmr_levels['extreme_closes'].get('low'):
                structure['extreme_levels'].append(dmr_levels['extreme_closes']['low'])

        # Check for potential breakouts
        structure['breakout_candidates'] = self._find_breakout_candidates(
            df, dmr_levels
        )

        return structure

    def _find_breakout_candidates(self, df: pd.DataFrame,
                                 dmr_levels: Dict) -> List[Dict]:
        """Find instruments near DMR levels that might breakout."""
        candidates = []
        current_price = df.iloc[-1]['close']

        # Check proximity to key DMR levels
        for level_type in ['daily', 'three_day', 'weekly', 'monthly']:
            for direction in ['high', 'low']:
                level = dmr_levels.get(level_type, {}).get(direction)
                if level and level.get('price'):
                    distance = abs(level['price'] - current_price) * 10000

                    # Within 20 pips of DMR level
                    if distance < 20:
                        candidates.append({
                            'level': level,
                            'distance_pips': distance,
                            'type': f"{level_type.upper()}_{direction.upper()}",
                            'breakout_type': direction
                        })

        return candidates

    def _identify_trading_opportunities(self, pattern_result: Dict,
                                        dmr_levels: Dict,
                                        acb_levels: Dict) -> List[Dict]:
        """Identify high-probability trading opportunities."""
        opportunities = []

        # FGD/FRD opportunity
        if pattern_result.get('pattern_detected'):
            opp = {
                'type': 'FGD/FRD',
                'signal_type': pattern_result.get('signal_type'),
                'pattern': pattern_result.get('pattern_description'),
                'confidence': pattern_result.get('confidence'),
                'direction': pattern_result.get('trade_direction'),
                'trade_today': pattern_result.get('trade_today')
            }

            # Add DMR context
            if pattern_result.get('dmr_validation', {}).get('is_near_dmr'):
                opp['dmr_aligned'] = True
                opp['dmr_level'] = pattern_result['dmr_validation']['nearest_dmr']

            # Add ACB context
            if pattern_result.get('acb_validation', {}).get('has_acb_proximity'):
                opp['acb_aligned'] = True
                opp['acb_level'] = pattern_result['acb_validation']['nearest_acb']

            opportunities.append(opp)

        # ACB breakouts
        for acb in acb_levels.get('validated', []):
            opp = {
                'type': 'ACB Breakout',
                'signal_type': 'LONG' if acb['type'] == 'upside' else 'SHORT',
                'level': acb['price'],
                'time': acb['time'],
                'validation_score': acb.get('validation_score', 0),
                'confidence': min(90, 50 + acb.get('validation_score', 0) // 2)
            }

            # Add context
            if acb.get('consecutive_closes'):
                opp['consecutive_closes'] = acb['consecutive_closes']
            if acb.get('extreme_position', {}).get('at_extreme'):
                opp['extreme_acb'] = True
            if acb.get('ema_coil', {}).get('coil_detected'):
                opp['ema_coil'] = True

            opportunities.append(opp)

        return opportunities

    def _analyze_confidence(self, pattern_result: Dict,
                           dmr_levels: Dict,
                           acb_levels: Dict) -> Dict:
        """Analyze overall confidence based on all validations."""
        confidence_factors = {
            'base_confidence': 50,  # Base confidence
            'pattern_confidence': 0,
            'dmr_confidence': 0,
            'acb_confidence': 0,
            'structure_confidence': 0
        }

        # Pattern confidence
        if pattern_result.get('pattern_detected'):
            if pattern_result.get('consecutive_count') == 3:
                confidence_factors['pattern_confidence'] = 30  # A+ pattern
            elif pattern_result.get('consecutive_count') == 2:
                confidence_factors['pattern_confidence'] = 20  # A pattern

            # Add validation confidences
            if pattern_result.get('dmr_validation', {}).get('is_near_dmr'):
                level_type = pattern_result['dmr_validation']['nearest_dmr'].get('type', '')
                if level_type in ['HOM', 'LOM']:
                    confidence_factors['dmr_confidence'] = 25
                else:
                    confidence_factors['dmr_confidence'] = 15

            if pattern_result.get('acb_validation', {}).get('enhanced_acb'):
                acb_score = pattern_result['acb_validation'].get('validation_score', 0)
                confidence_factors['acb_confidence'] = min(30, acb_score // 3)

        # Calculate total confidence
        total_confidence = (
            confidence_factors['base_confidence'] +
            confidence_factors['pattern_confidence'] +
            confidence_factors['dmr_confidence'] +
            confidence_factors['acb_confidence']
        )

        # Cap at 95%
        total_confidence = min(95, total_confidence)

        return {
            'confidence_factors': confidence_factors,
            'total_confidence': total_confidence,
            'grade': self._calculate_grade_from_confidence(total_confidence)
        }

    def _convert_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert H1 dataframe to daily dataframe."""
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'

        return df.resample('D').agg(agg_dict).dropna()

    def _calculate_grade_from_confidence(self, confidence: int) -> str:
        """Calculate signal grade from confidence score."""
        if confidence >= 90:
            return "A+"
        elif confidence >= 75:
            return "A"
        elif confidence >= 60:
            return "B+"
        else:
            return "B"