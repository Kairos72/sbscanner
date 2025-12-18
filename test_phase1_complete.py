"""
ACB Enhanced Scanner - Comprehensive Test Suite
===============================================

This test suite validates all implemented phases of the ACB Enhanced Scanner:
Phase 1: Core ACB Detection ✓
Phase 2: Enhanced FRD/FGD with ACB Context ✓
Phase 3: Smart Money Manipulation Analysis ✓
Phase 4: Enhanced Signal Generation ✓

Key Features Tested:
- ACB Level Detection and Validation
- DMR (Daily Market Rotation) Levels
- Session Analysis and Manipulation Detection
- Enhanced FRD/FGD Pattern Recognition
- Asian Range Sweep Entry Strategy
- Signal Validation and Risk Management
- Liquidity Hunt Detection (Phase 3)
- Market Phase Identification (Phase 3)
- ACB-Aware Signal Generator (Phase 4)
- 5-Star Setup Prioritizer (Phase 4)

Usage:
  python test_phase1_complete.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import all ACB modules
from acb import (
    ACBDetector,
    DMRLevelCalculator,
    SessionAnalyzer,
    EnhancedFRDFGDDetector,
    AsianRangeEntryDetector,
    SignalValidator,
    SignalType,
    SignalGrade,
    ValidationLevel,
    # Phase 3 - Smart Money Manipulation Analysis
    LiquidityHuntDetector,
    MarketPhaseIdentifier,
    LiquidityHuntType,
    MarketPhase,
    # Phase 4 - Enhanced Signal Generation
    ACBAwareSignalGenerator,
    SignalConfidence,
    FiveStarSetupPrioritizer,
    SetupType,
    SetupRating,
    # Phase 5 - Trade Management
    DMRTargetCalculator,
    TargetType,
    TargetPriority
)


class TestResults:
    """Track and report test results."""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_details = []
        self.performance_metrics = {}
        self.start_time = None

    def start_test(self, test_name: str):
        """Start a new test."""
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")

    def record_result(self, test_name: str, passed: bool, details: str = "", performance: float = None):
        """Record test result."""
        self.total_tests += 1

        if passed:
            self.passed_tests += 1
            status = "[PASS]"
        else:
            self.failed_tests += 1
            status = "[FAIL]"

        self.test_details.append({
            'test': test_name,
            'status': status,
            'details': details,
            'performance': performance
        })

        if performance:
            self.performance_metrics[test_name] = performance

        print(f"{status} {test_name}")
        if details:
            print(f"    Details: {details}")
        if performance:
            print(f"    Performance: {performance:.3f}s")

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        if self.start_time:
            total_time = time.time() - self.start_time
        else:
            total_time = 0

        report = f"\n{'='*70}\n"
        report += f"TEST SUITE SUMMARY\n"
        report += f"{'='*70}\n"
        report += f"Total Tests: {self.total_tests}\n"
        report += f"Passed: {self.passed_tests}\n"
        report += f"Failed: {self.failed_tests}\n"
        report += f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%\n"
        report += f"Total Time: {total_time:.2f}s\n"

        if self.performance_metrics:
            report += f"\nPerformance Metrics:\n"
            for test, perf in self.performance_metrics.items():
                report += f"  {test}: {perf:.3f}s\n"

        if self.failed_tests > 0:
            report += f"\nFailed Tests:\n"
            for detail in self.test_details:
                if "[FAIL]" in detail['status']:
                    report += f"  - {detail['test']}: {detail['details']}\n"

        report += f"{'='*70}\n"

        return report


def validate_mt5_connection() -> bool:
    """Validate MT5 connection and basic functionality."""
    if not mt5.initialize():
        return False

    # Check account info
    account_info = mt5.account_info()
    if account_info is None:
        mt5.shutdown()
        return False

    return True


def test_acb_detector(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test ACB Level Detector."""
    if results:
        results.start_test("ACB Level Detector")

    detector = ACBDetector()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test ACB detection
    try:
        start_time = time.time()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        if rates is None or len(rates) == 0:
            if results:
                results.record_result("ACB Detection", False, f"No data for {symbol}")
            return False

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        acb_levels = detector.identify_acb_levels(df)
        performance = time.time() - start_time

        if results:
            results.record_result(
                "ACB Detection",
                True,
                f"Found {len(acb_levels['confirmed'])} confirmed, {len(acb_levels['potential'])} potential",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("ACB Detection", False, f"Error: {str(e)}")
        return False

    return True


def test_dmr_calculator(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test DMR Level Calculator."""
    if results:
        results.start_test("DMR Level Calculator")

    calculator = DMRLevelCalculator()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test DMR calculation
    try:
        start_time = time.time()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        dmr_levels = calculator.calculate_all_dmr_levels(df)
        performance = time.time() - start_time

        if results:
            pdh = dmr_levels['daily']['high']['price'] if dmr_levels['daily']['high'] else None
            pdl = dmr_levels['daily']['low']['price'] if dmr_levels['daily']['low'] else None
            pdh_str = f"{pdh:.5f}" if pdh is not None else "None"
            pdl_str = f"{pdl:.5f}" if pdl is not None else "None"
            results.record_result(
                "DMR Calculation",
                True,
                f"PDH: {pdh_str}, PDL: {pdl_str}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("DMR Calculation", False, f"Error: {str(e)}")
        return False

    return True


def test_session_analyzer(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Session Analyzer."""
    if results:
        results.start_test("Session Analyzer")

    analyzer = SessionAnalyzer()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test session analysis
    try:
        start_time = time.time()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        session_analysis = analyzer.analyze_session_behavior(df, 72)
        performance = time.time() - start_time

        if results:
            sessions_count = len(session_analysis.get('sessions', {}))
            results.record_result(
                "Session Analysis",
                True,
                f"Analyzed {sessions_count} sessions",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Session Analysis", False, f"Error: {str(e)}")
        return False

    return True


def test_phase2_enhanced_frd_fgd(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Enhanced FRD/FGD with ACB context."""
    if results:
        results.start_test("Phase 2: Enhanced FRD/FGD")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    detector = EnhancedFRDFGDDetector()

    # Test enhanced FRD/FGD detection
    try:
        start_time = time.time()

        # Get daily data
        daily_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 30)
        df_daily = pd.DataFrame(daily_rates)
        df_daily['time'] = pd.to_datetime(daily_rates['time'], unit='s')
        df_daily.set_index('time', inplace=True)

        # Get H1 data for ACB/DMR
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Calculate levels
        dmr_calc = DMRLevelCalculator()
        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)

        acb_det = ACBDetector()
        acb_levels = acb_det.identify_acb_levels(df_h1)

        # Detect patterns
        pattern_result = detector.detect_enhanced_frd_fgd(
            df_daily,
            dmr_levels,
            acb_levels
        )

        performance = time.time() - start_time

        if results:
            pattern_desc = pattern_result.get('pattern_description', 'No pattern')
            signal_grade = pattern_result.get('signal_grade', 'N/A')
            results.record_result(
                "Enhanced FRD/FGD",
                True,
                f"Pattern: {pattern_desc}, Grade: {signal_grade}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Enhanced FRD/FGD", False, f"Error: {str(e)}")
        return False

    return True


def test_asian_range_strategy(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Asian Range Sweep Strategy."""
    if results:
        results.start_test("Asian Range Strategy")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    detector = AsianRangeEntryDetector()

    # Test Asian range analysis
    try:
        start_time = time.time()

        # Get H1 data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Analyze Asian range (assuming FGD setup)
        analysis = detector.analyze_asian_range_setup(
            df_h1,
            SignalType.FGD,
            datetime.utcnow()
        )

        performance = time.time() - start_time

        if results:
            asian_range = analysis.get('asian_range', {})
            if asian_range:
                results.record_result(
                    "Asian Range Detection",
                    True,
                    f"Range: {asian_range['range_pips']:.1f} pips, {asian_range['candle_count']} candles",
                    performance
                )
            else:
                results.record_result(
                    "Asian Range Detection",
                    True,
                    "No Asian range yet (normal)"
                )

    except Exception as e:
        if results:
            results.record_result("Asian Range Detection", False, f"Error: {str(e)}")
        return False

    return True


def test_signal_validation(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Signal Validator."""
    if results:
        results.start_test("Signal Validation")

    validator = SignalValidator()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test signal validation
    try:
        start_time = time.time()

        # Get data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Create dummy signal for testing
        test_signal = {
            'signal_type': SignalType.FGD,
            'confidence': 75,
            'current_price': df_h1.iloc[-1]['close'],
            'dmr_validation': {'is_near_dmr': False},
            'acb_validation': {'has_acb_proximity': False}
        }

        validation = validator.validate_signal(
            test_signal,
            df_h1,
            {},
            {}
        )

        performance = time.time() - start_time

        if results:
            results.record_result(
                "Signal Validation",
                True,
                f"Level: {validation.level.value}, Confidence: {validation.confidence}%",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Signal Validation", False, f"Error: {str(e)}")
        return False

    return True


def test_integration(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test all modules working together."""
    if results:
        results.start_test("Integration Test")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test full integration
    try:
        start_time = time.time()

        # Initialize all modules
        acb_det = ACBDetector()
        dmr_calc = DMRLevelCalculator()
        session_an = SessionAnalyzer()
        fgd_det = EnhancedFRDFGDDetector()
        asian_det = AsianRangeEntryDetector()

        # Get data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Run all analyses
        acb_levels = acb_det.identify_acb_levels(df_h1)
        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)
        session_analysis = session_an.analyze_session_behavior(df_h1, 72)
        asian_analysis = asian_det.analyze_asian_range_setup(
            df_h1,
            SignalType.FGD,
            datetime.utcnow()
        )

        performance = time.time() - start_time

        if results:
            results.record_result(
                "Integration Test",
                True,
                f"All modules working, {len(acb_levels['confirmed'])} ACB levels",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Integration Test", False, f"Error: {str(e)}")
        return False

    return True


def test_multiple_symbols(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test with multiple symbols."""
    if results:
        results.start_test("Multiple Symbols Test")

    symbols = test_config['symbols'] if test_config else ['EURUSD', 'GBPUSD', 'USDJPY']
    success_count = 0

    for symbol in symbols:
        try:
            start_time = time.time()

            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None or len(rates) == 0:
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(rates['time'], unit='s')
            df.set_index('time', inplace=True)

            # Quick ACB check
            detector = ACBDetector()
            acb_levels = detector.identify_acb_levels(df)

            # Quick DMR check
            calculator = DMRLevelCalculator()
            daily_dmr = calculator.get_daily_dmr_levels(df)

            performance = time.time() - start_time
            success_count += 1

            if results:
                pdh_str = f"{daily_dmr['high']['price']:.5f}" if daily_dmr['high'] else "None"
                results.record_result(
                    f"Symbol {symbol}",
                    True,
                    f"ACB: {len(acb_levels['confirmed'])} confirmed, PDH: {pdh_str}",
                    performance
                )

        except Exception as e:
            if results:
                results.record_result(f"Symbol {symbol}", False, f"Error: {str(e)}")

    if results:
        results.record_result(
            "Multiple Symbols Summary",
            success_count == len(symbols),
            f"Successfully tested {success_count}/{len(symbols)} symbols"
        )

    return success_count == len(symbols)


def test_liquidity_hunt_detector(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Liquidity Hunt Detector (Phase 3)."""
    if results:
        results.start_test("Liquidity Hunt Detector")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    detector = LiquidityHuntDetector()

    # Test liquidity hunt detection
    try:
        start_time = time.time()

        # Get data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Calculate DMR and ACB levels for context
        dmr_calc = DMRLevelCalculator()
        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)

        acb_det = ACBDetector()
        acb_levels = acb_det.identify_acb_levels(df_h1)

        # Detect liquidity hunts
        hunt_analysis = detector.detect_liquidity_hunts(df_h1, dmr_levels, acb_levels)

        performance = time.time() - start_time

        if results:
            results.record_result(
                "Liquidity Hunt Detection",
                True,
                f"Found {hunt_analysis['total_hunts']} hunts, {hunt_analysis['hunt_frequency']:.1f}% frequency",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Liquidity Hunt Detection", False, f"Error: {str(e)}")
        return False

    return True


def test_market_phase_identifier(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Market Phase Identifier (Phase 3)."""
    if results:
        results.start_test("Market Phase Identifier")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    identifier = MarketPhaseIdentifier()

    # Test market phase identification
    try:
        start_time = time.time()

        # Get data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Calculate levels for context
        dmr_calc = DMRLevelCalculator()
        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)

        acb_det = ACBDetector()
        acb_levels = acb_det.identify_acb_levels(df_h1)

        # Identify market phase
        phase_analysis = identifier.identify_market_phase(df_h1, dmr_levels, acb_levels)

        performance = time.time() - start_time

        if results:
            phase_name = phase_analysis['current_phase'].value
            confidence = phase_analysis['confidence'].value
            results.record_result(
                "Market Phase Identification",
                True,
                f"Phase: {phase_name}, Confidence: {confidence}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Market Phase Identification", False, f"Error: {str(e)}")
        return False

    return True


def test_phase3_integration(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Phase 3 integration with existing modules."""
    if results:
        results.start_test("Phase 3 Integration")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test full Phase 3 integration
    try:
        start_time = time.time()

        # Initialize all modules
        acb_det = ACBDetector()
        dmr_calc = DMRLevelCalculator()
        hunt_det = LiquidityHuntDetector()
        phase_id = MarketPhaseIdentifier()

        # Get data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Run all analyses
        acb_levels = acb_det.identify_acb_levels(df_h1)
        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)
        hunt_analysis = hunt_det.detect_liquidity_hunts(df_h1, dmr_levels, acb_levels)
        phase_analysis = phase_id.identify_market_phase(df_h1, dmr_levels, acb_levels)

        performance = time.time() - start_time

        if results:
            results.record_result(
                "Phase 3 Integration",
                True,
                f"All Phase 3 modules working, {hunt_analysis['total_hunts']} hunts detected",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Phase 3 Integration", False, f"Error: {str(e)}")
        return False

    return True


def test_acb_aware_signal_generator(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test ACB-Aware Signal Generator (Phase 4)."""
    if results:
        results.start_test("ACB-Aware Signal Generator")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    signal_gen = ACBAwareSignalGenerator()

    # Test signal generation
    try:
        start_time = time.time()

        # Get H1 data for detailed analysis
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
        if h1_rates is None or len(h1_rates) == 0:
            if results:
                results.record_result("Signal Generation", False, f"No data for {symbol}")
            return False

        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Get D1 data for FRD/FGD analysis
        d1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 30)
        df_d1 = pd.DataFrame(d1_rates)
        df_d1['time'] = pd.to_datetime(d1_rates['time'], unit='s')
        df_d1.set_index('time', inplace=True)

        # Generate signals
        signals = signal_gen.generate_signals(df_h1, df_d1, symbol)
        performance = time.time() - start_time

        if results:
            top_signals_count = len(signals['top_signals'])
            current_price = signals['current_price']
            market_phase = signals['market_phase']

            results.record_result(
                "Signal Generation",
                True,
                f"Current: {current_price:.5f}, Phase: {market_phase}, Top signals: {top_signals_count}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Signal Generation", False, f"Error: {str(e)}")
        return False

    return True


def test_five_star_setup_prioritizer(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test 5-Star Setup Prioritizer (Phase 4)."""
    if results:
        results.start_test("5-Star Setup Prioritizer")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    prioritizer = FiveStarSetupPrioritizer()

    # Test setup prioritization
    try:
        start_time = time.time()

        # Get H1 data for detailed analysis
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
        if h1_rates is None or len(h1_rates) == 0:
            if results:
                results.record_result("Setup Prioritization", False, f"No data for {symbol}")
            return False

        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Get D1 data for FRD/FGD analysis
        d1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 30)
        df_d1 = pd.DataFrame(d1_rates)
        df_d1['time'] = pd.to_datetime(d1_rates['time'], unit='s')
        df_d1.set_index('time', inplace=True)

        # Prioritize setups
        setups = prioritizer.prioritize_setups(df_h1, df_d1, symbol)
        performance = time.time() - start_time

        if results:
            ranked_count = len(setups['ranked_setups'])
            current_price = setups['current_price']
            market_phase = setups['market_phase']

            if setups['top_setup']:
                top_type = setups['top_setup']['type'].value
                top_rating = setups['top_setup']['rating']['stars'].value
                top_details = f"Top: {top_type}, {top_rating}"
            else:
                top_details = "No high-probability setups"

            results.record_result(
                "Setup Prioritization",
                True,
                f"Current: {current_price:.5f}, Phase: {market_phase}, {top_details}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Setup Prioritization", False, f"Error: {str(e)}")
        return False

    return True


def test_target_calculator(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test DMR Target Calculator (Phase 5)."""
    if results:
        results.start_test("DMR Target Calculator")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'
    target_calc = DMRTargetCalculator()

    # Test target calculation
    try:
        start_time = time.time()

        # Get H1 data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        # Calculate DMR levels
        dmr_calc = DMRLevelCalculator()
        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)

        # Test long target calculation
        entry_price = df_h1.iloc[-1]['close']
        direction = 'long'

        targets = target_calc.calculate_targets(
            entry_price=entry_price,
            direction=direction,
            dmr_levels=dmr_levels
        )

        performance = time.time() - start_time

        if results:
            target_count = len(targets['targets'])
            if targets['primary_target']:
                primary = targets['primary_target']
                primary_dist = primary['distance_pips']
                primary_rr = primary['risk_reward_ratio']
                details = f"Targets: {target_count}, Primary: {primary_dist:.1f} pips, RR: 1:{primary_rr:.1f}"
            else:
                details = f"Targets: {target_count}, No primary target"

            results.record_result(
                "Target Calculation",
                True,
                details,
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Target Calculation", False, f"Error: {str(e)}")
        return False

    return True


def test_phase4_integration(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Phase 4 integration with all modules."""
    if results:
        results.start_test("Phase 4 Integration")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test full Phase 4 integration
    try:
        start_time = time.time()

        # Initialize all modules
        signal_gen = ACBAwareSignalGenerator()
        prioritizer = FiveStarSetupPrioritizer()
        target_calc = DMRTargetCalculator()

        # Get data
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)

        d1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 30)
        df_d1 = pd.DataFrame(d1_rates)
        df_d1['time'] = pd.to_datetime(d1_rates['time'], unit='s')
        df_d1.set_index('time', inplace=True)

        # Run all Phase 4 analyses
        signals = signal_gen.generate_signals(df_h1, df_d1, symbol)
        setups = prioritizer.prioritize_setups(df_h1, df_d1, symbol)

        # Calculate targets if we have a signal
        if signals['top_signals']:
            entry_price = df_h1.iloc[-1]['close']
            direction = signals['top_signals'][0].get('direction', 'long')

            dmr_calc = DMRLevelCalculator()
            dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)

            targets = target_calc.calculate_targets(
                entry_price=entry_price,
                direction=direction,
                dmr_levels=dmr_levels
            )
        else:
            targets = {'targets': []}

        performance = time.time() - start_time

        if results:
            results.record_result(
                "Phase 4 Integration",
                True,
                f"All Phase 4 modules working, {len(signals['top_signals'])} signals, {len(setups['ranked_setups'])} setups",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Phase 4 Integration", False, f"Error: {str(e)}")
        return False

    return True


def run_phase4_tests(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> int:
    """Run Phase 4 tests."""
    if results is None:
        results = TestResults()

    print("\n[PHASE 4] Enhanced Signal Generation")
    print("-" * 50)

    tests_passed = 0
    total_tests = 4

    # Run Phase 4 tests
    if test_acb_aware_signal_generator(test_config, results):
        tests_passed += 1

    if test_five_star_setup_prioritizer(test_config, results):
        tests_passed += 1

    if test_target_calculator(test_config, results):
        tests_passed += 1

    if test_phase4_integration(test_config, results):
        tests_passed += 1

    return tests_passed


def run_phase1_tests(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> int:
    """Run Phase 1 tests."""
    if results is None:
        results = TestResults()

    print("\n[PHASE 1] Core ACB Detection")
    print("-" * 50)

    tests_passed = 0
    total_tests = 4

    # Run Phase 1 tests
    if test_acb_detector(test_config, results):
        tests_passed += 1

    if test_dmr_calculator(test_config, results):
        tests_passed += 1

    if test_session_analyzer(test_config, results):
        tests_passed += 1

    if test_integration(test_config, results):
        tests_passed += 1

    return tests_passed


def run_phase2_tests(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> int:
    """Run Phase 2 tests."""
    if results is None:
        results = TestResults()

    print("\n[PHASE 2] Enhanced FRD/FGD with ACB")
    print("-" * 50)

    tests_passed = 0
    total_tests = 4

    # Run Phase 2 tests
    if test_phase2_enhanced_frd_fgd(test_config, results):
        tests_passed += 1

    if test_asian_range_strategy(test_config, results):
        tests_passed += 1

    if test_signal_validation(test_config, results):
        tests_passed += 1

    if test_multiple_symbols(test_config, results):
        tests_passed += 1

    return tests_passed


def run_phase3_tests(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> int:
    """Run Phase 3 tests."""
    if results is None:
        results = TestResults()

    print("\n[PHASE 3] Smart Money Manipulation Analysis")
    print("-" * 50)

    tests_passed = 0
    total_tests = 3

    # Run Phase 3 tests
    if test_liquidity_hunt_detector(test_config, results):
        tests_passed += 1

    if test_market_phase_identifier(test_config, results):
        tests_passed += 1

    if test_phase3_integration(test_config, results):
        tests_passed += 1

    return tests_passed


def main():
    """Run comprehensive test suite."""
    print("ACB ENHANCED SCANNER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("Testing all implemented phases...")
    print("=" * 70)

    # Initialize test results tracker
    results = TestResults()

    # Initialize MT5
    if not validate_mt5_connection():
        print("[FATAL] Failed to initialize MT5")
        return

    print("\n[INFO] MT5 Connection Established")

    # Test configuration
    test_config = {
        'symbols': ['NZDUSD', 'EURUSD', 'GBPUSD'],
        'timeframes': ['H1', 'H4', 'D1'],
        'verbose': True
    }

    try:
        # Run Phase 1 tests
        phase1_passed = run_phase1_tests(test_config, results)

        # Run Phase 2 tests
        phase2_passed = run_phase2_tests(test_config, results)

        # Run Phase 3 tests
        phase3_passed = run_phase3_tests(test_config, results)

        # Run Phase 4 tests
        phase4_passed = run_phase4_tests(test_config, results)

        # Generate final report
        print(results.generate_report())

        total_tests = phase1_passed + phase2_passed + phase3_passed + phase4_passed
        max_tests = 15

        if total_tests == max_tests:
            print(f"\n[SUCCESS] ALL PHASES 1-4 TESTS PASSED! ({total_tests}/{max_tests})")
            print("[OK] Core ACB detection and enhanced FRD/FGD systems working correctly!")
            print("[OK] Asian Range Sweep strategy implemented!")
            print("[OK] Smart Money Manipulation Analysis working!")
            print("[OK] Enhanced Signal Generation and Setup Prioritization working!")
            print("\nNext: Phase 6 - Visualization & Alerts")
        else:
            print(f"\n[PARTIAL] Some tests failed ({total_tests}/{max_tests})")
            print("[WARNING] Please review and fix before proceeding")

    except Exception as e:
        print(f"\n[FATAL] Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()