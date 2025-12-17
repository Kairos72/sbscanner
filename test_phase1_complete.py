"""
ACB Enhanced Scanner - Comprehensive Test Suite
===============================================

This test suite will grow with each implementation phase:
Phase 1: Core ACB Detection ✓
Phase 2: Enhanced FRD/FGD with ACB
Phase 3: Smart Money Manipulation Analysis
Phase 4: Enhanced Signal Generation
Phase 5: Advanced Trade Management
Phase 6: Visualization & Alerts
Phase 7: Market Structure Analysis
Phase 8: Integration & Optimization

Usage:
  - Run tests after implementing each phase
  - Use as reference for ACB concepts
  - Debug issues by running specific tests
  - Validate with different symbols/timeframes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from acb import ACBDetector, DMRLevelCalculator, SessionAnalyzer


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
    """Test ACB Level Detector thoroughly."""
    if results:
        results.start_test("ACB Level Detector")

    detector = ACBDetector()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test 1: Data retrieval
    try:
        start_time = time.time()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        if rates is None or len(rates) == 0:
            if results:
                results.record_result("ACB Data Retrieval", False, f"No data for {symbol}")
            return False

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        performance = time.time() - start_time
        if results:
            results.record_result("ACB Data Retrieval", True, f"Retrieved {len(df)} candles", performance)

    except Exception as e:
        if results:
            results.record_result("ACB Data Retrieval", False, f"Error: {str(e)}")
        return False

    # Test 2: ACB Level Detection
    try:
        start_time = time.time()
        acb_levels = detector.identify_acb_levels(df)
        performance = time.time() - start_time

        if results:
            results.record_result(
                "ACB Level Detection",
                True,
                f"Found {len(acb_levels['confirmed'])} confirmed, {len(acb_levels['potential'])} potential",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("ACB Level Detection", False, f"Error: {str(e)}")
        return False

    # Test 3: Nearest ACB Detection
    try:
        start_time = time.time()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            if results:
                results.record_result("ACB Nearest Level", False, "No price data")
            return False

        current_price = tick.bid
        nearest = detector.get_nearest_acb_levels(current_price, acb_levels)
        performance = time.time() - start_time

        distance_str = f"{abs(nearest['nearest']['distance'])*10000:.1f} pips" if nearest['nearest'] else "No nearby levels"

        if results:
            results.record_result(
                "ACB Nearest Level",
                True,
                f"Current: {current_price:.5f}, Nearest: {distance_str}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("ACB Nearest Level", False, f"Error: {str(e)}")
        return False

    return True


def test_dmr_calculator(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test DMR Level Calculator thoroughly."""
    if results:
        results.start_test("DMR Level Calculator")

    calculator = DMRLevelCalculator()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test 1: Daily DMR (PDH/PDL)
    try:
        start_time = time.time()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        daily_dmr = calculator.get_daily_dmr_levels(df)
        performance = time.time() - start_time

        pdh_str = f"{daily_dmr['high']['price']:.5f}" if daily_dmr['high'] else "None"
        pdl_str = f"{daily_dmr['low']['price']:.5f}" if daily_dmr['low'] else "None"

        if results:
            results.record_result(
                "DMR Daily Levels",
                True,
                f"PDH: {pdh_str}, PDL: {pdl_str}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("DMR Daily Levels", False, f"Error: {str(e)}")
        return False

    # Test 2: 3-Day DMR
    try:
        start_time = time.time()
        three_day_dmr = calculator.get_three_day_dmr_levels(df)
        performance = time.time() - start_time

        if results:
            three_dh_str = f"{three_day_dmr['high']['price']:.5f}" if three_day_dmr['high'] else "None"
            three_dl_str = f"{three_day_dmr['low']['price']:.5f}" if three_day_dmr['low'] else "None"
            results.record_result(
                "DMR 3-Day Levels",
                True,
                f"3DH: {three_dh_str}, 3DL: {three_dl_str}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("DMR 3-Day Levels", False, f"Error: {str(e)}")
        return False

    # Test 3: Active DMR Levels
    try:
        start_time = time.time()
        active_dmr = calculator.get_active_dmr_levels(df)
        performance = time.time() - start_time

        if results:
            results.record_result(
                "DMR Active Levels",
                True,
                f"Found {len(active_dmr)} active levels",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("DMR Active Levels", False, f"Error: {str(e)}")
        return False

    # Test 4: Rotation Targets
    try:
        start_time = time.time()
        tick = mt5.symbol_info_tick(symbol)
        if tick and active_dmr:
            targets = calculator.calculate_rotation_targets(tick.bid, 'LONG', calculator.calculate_all_dmr_levels(df))
            performance = time.time() - start_time

            if results:
                results.record_result(
                    "DMR Rotation Targets",
                    True,
                    f"Generated {len(targets)} targets",
                    performance
                )

    except Exception as e:
        if results:
            results.record_result("DMR Rotation Targets", False, f"Error: {str(e)}")
        return False

    return True


def test_session_analyzer(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test Session Analyzer thoroughly."""
    if results:
        results.start_test("Session Analyzer")

    analyzer = SessionAnalyzer()
    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test 1: Session Identification
    try:
        start_time = time.time()
        current_time = datetime.now()
        current_session = analyzer.identify_session(current_time)
        performance = time.time() - start_time

        if results:
            results.record_result(
                "Session Identification",
                True,
                f"Current session: {current_session.value[2]}",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Session Identification", False, f"Error: {str(e)}")
        return False

    # Test 2: Session Analysis
    try:
        start_time = time.time()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        session_analysis = analyzer.analyze_session_behavior(df, 72)
        performance = time.time() - start_time

        sessions_count = len(session_analysis.get('sessions', {}))
        manipulation_count = len(session_analysis.get('manipulation_zones', []))

        if results:
            results.record_result(
                "Session Analysis",
                True,
                f"Analyzed {sessions_count} sessions, {manipulation_count} manipulation zones",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Session Analysis", False, f"Error: {str(e)}")
        return False

    # Test 3: Entry Signal Generation
    try:
        start_time = time.time()
        current_session = analyzer.identify_session(datetime.now())
        tick = mt5.symbol_info_tick(symbol)

        if tick and session_analysis:
            signal = analyzer.get_session_entry_signal(current_session, tick.bid, session_analysis)
            performance = time.time() - start_time

            if results:
                results.record_result(
                    "Session Entry Signal",
                    True,
                    f"Bias: {signal['bias']}, Confidence: {signal['confidence']}%",
                    performance
                )

    except Exception as e:
        if results:
            results.record_result("Session Entry Signal", False, f"Error: {str(e)}")
        return False

    return True


def test_integration(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test all modules working together."""
    if results:
        results.start_test("Integration Test")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Test 1: Combined Analysis
    try:
        start_time = time.time()

        # Get data once
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        # Initialize all modules
        detector = ACBDetector()
        calculator = DMRLevelCalculator()
        analyzer = SessionAnalyzer()

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            if results:
                results.record_result("Integration Analysis", False, "No price data")
            return False

        current_price = tick.bid

        # Run all analyses
        acb_levels = detector.identify_acb_levels(df)
        all_dmr = calculator.calculate_all_dmr_levels(df)
        session_analysis = analyzer.analyze_session_behavior(df, 72)

        # Generate comprehensive analysis
        analysis = {
            'current_price': current_price,
            'nearest_acb': detector.get_nearest_acb_levels(current_price, acb_levels),
            'nearest_dmr': calculator.get_nearest_dmr_levels(current_price, all_dmr),
            'rotation_probability': None,
            'session_signal': None
        }

        if analysis['nearest_dmr']['nearest']:
            rotation = calculator.check_rotation_probability(
                current_price,
                analysis['nearest_dmr']['nearest']
            )
            analysis['rotation_probability'] = rotation['probability']

        if session_analysis:
            current_session = analyzer.identify_session(datetime.now())
            analysis['session_signal'] = analyzer.get_session_entry_signal(
                current_session,
                current_price,
                session_analysis
            )

        performance = time.time() - start_time

        if results:
            nearest_acb_str = f"{analysis['nearest_acb']['nearest']['price']:.5f}" if analysis['nearest_acb']['nearest'] else "None"
            nearest_dmr_str = f"{analysis['nearest_dmr']['nearest']['price']:.5f}" if analysis['nearest_dmr']['nearest'] else "None"

            results.record_result(
                "Integration Analysis",
                True,
                f"Price: {current_price:.5f}, ACB: {nearest_acb_str}, DMR: {nearest_dmr_str}, "
                f"Rotation: {analysis['rotation_probability']}%",
                performance
            )

    except Exception as e:
        if results:
            results.record_result("Integration Analysis", False, f"Error: {str(e)}")
        return False

    return True


def test_multiple_symbols(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test modules with different symbols."""
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


def test_performance_benchmarks(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
    """Test performance benchmarks."""
    if results:
        results.start_test("Performance Benchmarks")

    symbol = test_config['symbols'][0] if test_config else 'NZDUSD'

    # Benchmark data retrieval
    try:
        iterations = 10
        start_time = time.time()

        for _ in range(iterations):
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)

        avg_time = (time.time() - start_time) / iterations

        if results:
            results.record_result(
                "Data Retrieval Performance",
                avg_time < 0.5,
                f"Average: {avg_time:.3f}s per request",
                avg_time
            )

    except Exception as e:
        if results:
            results.record_result("Data Retrieval Performance", False, f"Error: {str(e)}")

    # Benchmark ACB detection
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(rates['time'], unit='s')
        df.set_index('time', inplace=True)

        detector = ACBDetector()
        iterations = 5
        start_time = time.time()

        for _ in range(iterations):
            detector.identify_acb_levels(df)

        avg_time = (time.time() - start_time) / iterations

        if results:
            results.record_result(
                "ACB Detection Performance",
                avg_time < 1.0,
                f"Average: {avg_time:.3f}s per detection",
                avg_time
            )

    except Exception as e:
        if results:
            results.record_result("ACB Detection Performance", False, f"Error: {str(e)}")

    return True


# Phase 2 Test Functions (to be implemented)
# def test_phase2_enhanced_frd_fgd(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test FRD/FGD with ACB context"""
#     pass
#
# def test_phase3_manipulation_analysis(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test smart money manipulation detection"""
#     pass
#
# def test_phase4_signal_generation(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test ACB-aware signal generation"""
#     pass
#
# def test_phase5_trade_management(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test dynamic trade management"""
#     pass
#
# def test_phase6_visualization(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test visualization and alerts"""
#     pass
#
# def test_phase7_market_structure(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test market structure analysis"""
#     pass
#
# def test_phase8_integration(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> bool:
#     """Test full system integration"""
#     pass


def run_phase1_tests(test_config: Optional[Dict] = None, results: Optional[TestResults] = None) -> int:
    """Run all Phase 1 tests."""
    if results is None:
        results = TestResults()

    print("\n[PHASE 1] Core ACB Detection")
    print("-" * 50)

    tests_passed = 0
    total_tests = 6

    # Run all Phase 1 tests
    if test_acb_detector(test_config, results):
        tests_passed += 1

    if test_dmr_calculator(test_config, results):
        tests_passed += 1

    if test_session_analyzer(test_config, results):
        tests_passed += 1

    if test_integration(test_config, results):
        tests_passed += 1

    if test_multiple_symbols(test_config, results):
        tests_passed += 1

    if test_performance_benchmarks(test_config, results):
        tests_passed += 1

    return tests_passed


def main():
    """Run comprehensive ACB test suite."""
    print("ACB ENHANCED SCANNER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("Testing all implemented modules...")
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

        # Generate final report
        print(results.generate_report())

        if phase1_passed == 6:
            print(f"[SUCCESS] ALL PHASE 1 TESTS PASSED! ({phase1_passed}/6)")
            print("[OK] Core ACB detection system is working correctly!")
            print("[OK] Ready for Phase 2 implementation")
            print("\nNext: Phase 2 - Enhanced FRD/FGD with ACB context")
        else:
            print(f"[PARTIAL] Some tests failed ({phase1_passed}/6)")
            print("[WARNING] Please review and fix before proceeding to Phase 2")

    except Exception as e:
        print(f"[FATAL] Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()