"""
Real-time Major Pairs Monitoring Service with Full ACB Integration
==================================================================

Continuous background service that monitors ALL major currency pairs with
COMPLETE ACB analysis integration using the unified master script.

Features:
- Monitors EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD
- Uses unified_master_script.py for complete 14-feature analysis
- Real-time alerts for all ACB patterns
- Rotating full analysis every 5 minutes
- Quick checks every 60 seconds

To run: python realtime_major_pairs_service.py
"""

import sys
import os
import time
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd
from oanda_symbol_mapper import get_oanda_symbol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('major_pairs_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MajorPairs_Monitor')


class RealtimeMajorPairsService:
    """
    Continuous background service for monitoring all major currency pairs
    with complete ACB analysis integration
    """

    def __init__(self):
        """Initialize the monitoring service"""
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD'
        ]
        self.symbols = {pair: get_oanda_symbol(pair) for pair in self.major_pairs}
        self.running = False
        self.check_interval = 60  # Check every 60 seconds
        self.full_analysis_interval = 300  # Run complete analysis every 5 minutes

        # Statistics
        self.checks_completed = 0
        self.full_analysis_count = 0
        self.alerts_generated = 0
        self.start_time = datetime.now()
        self.last_full_analysis = None

        # Per-pair tracking
        self.pair_data = {}
        self.last_alerts = {}  # To avoid duplicate alerts

        logger.info(f"Real-time Major Pairs Service with ACB integration initialized")
        logger.info(f"Monitoring {len(self.major_pairs)} pairs: {', '.join(self.major_pairs)}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        logger.info(f"Using unified_master_script.py for complete 14-feature analysis")

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False

            # Verify all symbols are available
            for pair, symbol in self.symbols.items():
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"Symbol {symbol} ({pair}) not found")
                    return False

            logger.info(f"MT5 initialized successfully for all {len(self.major_pairs)} pairs")
            return True

        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            return False

    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Fetch the latest market data for a symbol"""
        try:
            # Get recent H1 data (last 100 candles)
            h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if h1_rates is None or len(h1_rates) == 0:
                return None

            df_h1 = pd.DataFrame(h1_rates)
            df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
            df_h1.set_index('time', inplace=True)

            # Get recent D1 data (last 30 days)
            d1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 30)
            if d1_rates is None or len(d1_rates) == 0:
                return None

            df_d1 = pd.DataFrame(d1_rates)
            df_d1['time'] = pd.to_datetime(d1_rates['time'], unit='s')
            df_d1.set_index('time', inplace=True)

            current_price = df_h1.iloc[-1]['close']

            return {
                'symbol': symbol,
                'df_h1': df_h1,
                'df_d1': df_d1,
                'current_price': current_price,
                'timestamp': df_h1.index[-1]
            }

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def run_complete_acb_analysis(self, pair: str) -> bool:
        """Run the complete unified master ACB analysis for a pair"""
        try:
            print(f"\n{'='*120}")
            print(f"RUNNING COMPLETE ACB ANALYSIS FOR {pair}")
            print(f"{'='*120}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S UTC')}")
            print(f"Using unified_master_script.py with 14 ACB features...")
            print(f"{'='*120}")

            # Read the unified master script
            script_path = "unified_master_script.py"
            with open(script_path, 'r') as f:
                script_content = f.read()

            # Substitute the symbol
            script_content = script_content.replace('symbol_base = "AUDUSD"', f'symbol_base = "{pair}"')
            script_content = script_content.replace('run_ultimate_usdjpy_analysis', 'run_unified_master_analysis')

            # Create temporary script
            temp_file = f"temp_acb_{pair.lower()}.py"
            with open(temp_file, 'w') as f:
                f.write(script_content)

            # Run the analysis
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                # Print the complete output
                print(result.stdout)

                # Parse for key signals
                self._parse_acb_output(pair, result.stdout)
            else:
                print(f"[ERROR] Analysis failed for {pair}")
                print(result.stderr[:1000])

            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Analysis for {pair} took too long")
            return False
        except Exception as e:
            logger.error(f"Error running ACB analysis for {pair}: {e}")
            return False

    def _parse_acb_output(self, pair: str, output: str) -> None:
        """Parse ACB output for key signals and alerts"""
        lines = output.split('\n')

        # State tracking to understand context
        in_five_star_section = False
        current_section = ""

        for line in lines:
            line_upper = line.upper().strip()

            # Track current section
            if '[9. FIVE-STAR SETUP PRIORITIZATION]' in line_upper:
                in_five_star_section = True
                current_section = "FIVE_STAR"
            elif line_upper.startswith('[') and 'FIVE-STAR' not in line_upper:
                in_five_star_section = False
                current_section = ""

            # Only parse for actual signals, not headers
            if not line_upper.startswith('[') and not line_upper.startswith('=') and not line_upper.startswith('-'):

                # FGD/FRD Patterns - Look for actual active triggers
                if 'FGD' in line_upper and ('TRIGGER' in line_upper or 'ACTIVE' in line_upper):
                    if 'GRADE:' in line_upper or 'CONFIDENCE:' in line_upper:
                        self._generate_alert(pair, 'FGD_PATTERN', line.strip())

                elif 'FRD' in line_upper and ('TRIGGER' in line_upper or 'ACTIVE' in line_upper):
                    if 'GRADE:' in line_upper or 'CONFIDENCE:' in line_upper:
                        self._generate_alert(pair, 'FRD_PATTERN', line.strip())

                # 5-Star Setups - Only if NOT a header and shows actual setup
                elif in_five_star_section and 'SETUP' in line_upper:
                    if 'NO SETUPS' not in line_upper and 'NOT FOUND' not in line_upper:
                        if any(word in line_upper for word in ['DETECTED', 'CONFIRMED', 'ACTIVE', 'IDENTIFIED']):
                            self._generate_alert(pair, 'HIGH_PRIORITY_SETUP', line.strip())

                # Manipulation - Look for actual detection
                elif 'MANIPULATION' in line_upper:
                    if 'PHASE:' in line_upper and 'CONFIDENCE' in line_upper:
                        phase = line_upper.split('PHASE:')[1].split()[0] if 'PHASE:' in line_upper else ''
                        if phase.upper() in ['MANIPULATION', 'ACCUMULATION', 'DISTRIBUTION']:
                            self._generate_alert(pair, 'MANIPULATION', line.strip())

                # Breakout Alerts
                elif 'BREAKOUT' in line_upper and 'ALERT' in line_upper:
                    if any(level in line_upper for level in ['HIGH', 'MEDIUM', 'CRITICAL']):
                        self._generate_alert(pair, 'BREAKOUT_ALERT', line.strip())

                # Inside Day Pattern
                elif 'INSIDE DAY' in line_upper and ('BREAKOUT' in line_upper or 'PATTERN' in line_upper):
                    if 'DETECTED' in line_upper or 'CONFIRMED' in line_upper:
                        self._generate_alert(pair, 'INSIDE_DAY', line.strip())

                # Pump & Dump
                elif 'PUMP' in line_upper and 'DUMP' in line_upper:
                    if 'DETECTED' in line_upper or 'PATTERN' in line_upper:
                        self._generate_alert(pair, 'PUMP_DUMP', line.strip())

                # Strong Bias Changes (only if significant)
                elif 'OVERALL BIAS:' in line_upper or 'PRIMARY INFLUENCES:' in line_upper:
                    bias_words = ['STRONGLY BULLISH', 'STRONGLY BEARISH', 'HIGH CONFIDENCE']
                    if any(word in line_upper for word in bias_words):
                        self._generate_alert(pair, 'BIAS_CHANGE', line.strip())

    def _generate_alert(self, pair: str, alert_type: str, message: str):
        """Generate and log an alert"""
        alert_key = f"{pair}_{alert_type}"
        current_time = datetime.now()

        # Check for duplicate alerts (5-minute cooldown)
        if alert_key in self.last_alerts:
            if (current_time - self.last_alerts[alert_key]).seconds < 300:
                return  # Skip duplicate

        self.last_alerts[alert_key] = current_time
        self.alerts_generated += 1

        # Add context based on alert type
        context = self._get_alert_context(alert_type)

        print(f"\n{'='*80}")
        print(f"[ALERT] {pair} - {alert_type}")
        print(f"[ALERT] {message}")
        if context:
            print(f"[ALERT] Context: {context}")
        print(f"[ALERT] Time: {current_time.strftime('%H:%M:%S UTC')}")
        print(f"{'='*80}")

        logger.warning(f"[ALERT] {pair} - {alert_type}: {message}")

    def _get_alert_context(self, alert_type: str) -> str:
        """Get contextual information for different alert types"""
        contexts = {
            'FGD_PATTERN': "FGD (First Green Day) - Bullish signal after red candles - Look for long entries",
            'FRD_PATTERN': "FRD (First Red Day) - Bearish signal after green candles - Look for short entries",
            'HIGH_PRIORITY_SETUP': "5-star setup - Highest probability trade opportunity",
            'MANIPULATION': "Smart money manipulation detected - Be cautious",
            'BREAKOUT_ALERT': "Breakout opportunity - Volume confirmation needed",
            'INSIDE_DAY': "Inside day pattern - Watch for breakout",
            'PUMP_DUMP': "Pump & dump pattern - Counter-trend opportunity",
            'BIAS_CHANGE': "Significant bias shift - Market structure change"
        }
        return contexts.get(alert_type, "")

    def quick_pair_analysis(self, pair: str, data: Dict) -> Dict:
        """Quick analysis for alert detection between full analyses"""
        df_h1 = data['df_h1']
        current_price = data['current_price']

        # Calculate key levels
        high_24h = df_h1['high'].tail(24).max()
        low_24h = df_h1['low'].tail(24).min()
        range_24h = (high_24h - low_24h) * 10000

        # Asian range (0-5 UTC)
        df_asian = df_h1[df_h1.index.hour < 5]
        if len(df_asian) > 0:
            asian_high = df_asian['high'].max()
            asian_low = df_asian['low'].min()
            asian_range = (asian_high - asian_low) * 10000
        else:
            asian_high = None
            asian_low = None
            asian_range = None

        # Volume analysis
        current_volume = df_h1.iloc[-1]['tick_volume']
        avg_volume = df_h1['tick_volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Determine position
        distance_to_high = (high_24h - current_price) * 10000
        distance_to_low = (current_price - low_24h) * 10000

        # Quick alert conditions
        alerts = []

        # 24h extreme alert
        if distance_to_high < 10:
            alerts.append(f"AT 24-HOUR HIGH ({high_24h:.5f})")
        elif distance_to_low < 10:
            alerts.append(f"AT 24-HOUR LOW ({low_24h:.5f})")

        # Asian range alert
        if asian_high and asian_low:
            if current_price > asian_high:
                alerts.append(f"ABOVE ASIAN HIGH (+{((current_price - asian_high) * 10000):.0f} pips)")
            elif current_price < asian_low:
                alerts.append(f"BELOW ASIAN LOW (-{((asian_low - current_price) * 10000):.0f} pips)")

        # Volume alert
        if volume_ratio > 2.0:
            alerts.append(f"EXTREME VOLUME ({volume_ratio:.1f}x)")
        elif volume_ratio < 0.5:
            alerts.append(f"LOW VOLUME ({volume_ratio:.1f}x)")

        return {
            'pair': pair,
            'price': current_price,
            'range_24h': range_24h,
            'asian_range': asian_range,
            'volume_ratio': volume_ratio,
            'alerts': alerts,
            'bias': self._determine_bias(df_h1, current_price)
        }

    def _determine_bias(self, df_h1: pd.DataFrame, current_price: float) -> str:
        """Simple bias determination"""
        # Simple moving averages
        ma20 = df_h1['close'].tail(20).mean()
        ma50 = df_h1['close'].tail(50).mean() if len(df_h1) >= 50 else ma20

        if current_price > ma20 > ma50:
            return "BULLISH"
        elif current_price < ma20 < ma50:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def check_for_alerts(self, analysis_results: List[Dict]) -> None:
        """Check and display alerts across all pairs"""
        total_alerts = 0
        pairs_with_alerts = []

        for result in analysis_results:
            if result['alerts']:
                total_alerts += len(result['alerts'])
                pairs_with_alerts.append(result)

        if pairs_with_alerts:
            print(f"\n[ALERTS DETECTED] {total_alerts} alerts across {len(pairs_with_alerts)} pairs")
            print("=" * 80)

            for result in pairs_with_alerts:
                print(f"\n{result['pair']} - {result['price']:.5f} ({result['bias']})")
                for alert in result['alerts']:
                    print(f"  - {alert}")

                self.alerts_generated += len(result['alerts'])

    def print_status_update(self, analysis_results: List[Dict]) -> None:
        """Print a summary of all pairs"""
        runtime = datetime.now() - self.start_time

        print(f"\n{'='*100}")
        print(f"MAJOR PAIRS MONITOR - Status Update")
        print(f"Time: {datetime.now().strftime('%H:%M:%S UTC')}")
        print(f"Runtime: {runtime} | Checks: {self.checks_completed} | Full Analyses: {self.full_analysis_count}")
        print(f"{'='*100}")

        # Group by bias
        bullish = [r for r in analysis_results if r['bias'] == 'BULLISH']
        bearish = [r for r in analysis_results if r['bias'] == 'BEARISH']
        neutral = [r for r in analysis_results if r['bias'] == 'NEUTRAL']

        print(f"\nBIAS SUMMARY:")
        if bullish:
            bull_pairs = ', '.join([f"{r['pair']} ({r['price']:.5f})" for r in bullish])
            print(f"  BULLISH ({len(bullish)}): {bull_pairs}")
        if bearish:
            bear_pairs = ', '.join([f"{r['pair']} ({r['price']:.5f})" for r in bearish])
            print(f"  BEARISH ({len(bearish)}): {bear_pairs}")
        if neutral:
            neut_pairs = ', '.join([f"{r['pair']} ({r['price']:.5f})" for r in neutral])
            print(f"  NEUTRAL ({len(neutral)}): {neut_pairs}")

        # Show pairs with alerts
        with_alerts = [r for r in analysis_results if r['alerts']]
        if with_alerts:
            print(f"\nPAIRS WITH ALERTS: {len(with_alerts)}")
            for result in with_alerts:
                print(f"  - {result['pair']}: {', '.join(result['alerts'])}")

    def run(self) -> None:
        """Main monitoring loop"""
        logger.info("Starting real-time Major Pairs monitoring service with full ACB integration...")
        self.running = True

        # Initialize MT5
        if not self.initialize_mt5():
            logger.error("Failed to initialize MT5. Exiting.")
            return

        try:
            # Run complete analysis immediately on startup
            print(f"\n{'='*100}")
            print(f"INITIAL FULL ACB ANALYSIS - STARTUP")
            print(f"{'='*100}")

            # Analyze all pairs immediately
            for pair, symbol in self.symbols.items():
                data = self.get_latest_data(symbol)
                if data:
                    print(f"\n--- Analyzing {pair} ---")
                    success = self.run_complete_acb_analysis(pair)
                    if success:
                        logger.info(f"Initial ACB analysis completed for {pair}")
                    else:
                        print(f"[ERROR] Failed to analyze {pair}")

            print(f"\n{'='*100}")
            print(f"INITIAL ANALYSIS COMPLETE - Starting 5-minute cycles")
            print(f"{'='*100}")

            # Reset for next cycle
            self.last_full_analysis = datetime.now()

            while self.running:
                cycle_start = time.time()
                analysis_results = []

                # Check if we should run complete analysis
                run_full_analysis = False
                if (datetime.now() - self.last_full_analysis).total_seconds() >= self.full_analysis_interval:
                    run_full_analysis = True
                    self.last_full_analysis = datetime.now()
                    self.full_analysis_count += 1

                if run_full_analysis:
                    print(f"\n{'='*100}")
                    print(f"FULL ACB ANALYSIS CYCLE - {datetime.now().strftime('%H:%M:%S UTC')}")
                    print(f"{'='*100}")

                # Fetch and analyze all pairs
                for pair, symbol in self.symbols.items():
                    data = self.get_latest_data(symbol)

                    if data:
                        # Run complete analysis for ALL pairs every 5 minutes
                        if run_full_analysis:
                            print(f"\n--- Analyzing {pair} ---")
                            success = self.run_complete_acb_analysis(pair)
                            if success:
                                logger.info(f"Full ACB analysis completed for {pair}")
                            else:
                                print(f"[ERROR] Failed to analyze {pair}")

                        # Always do quick analysis for all pairs
                        result = self.quick_pair_analysis(pair, data)
                        analysis_results.append(result)
                    else:
                        print(f"[WARNING] No data available for {pair}")

                # Check for alerts across all pairs
                self.check_for_alerts(analysis_results)

                # Print status update every cycle
                self.print_status_update(analysis_results)

                self.checks_completed += 1

                # Calculate sleep time
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.check_interval - cycle_time)

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\nReceived shutdown signal. Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown of the service"""
        logger.info("Shutting down Major Pairs monitoring service...")
        self.running = False

        # Print final statistics
        runtime = datetime.now() - self.start_time
        logger.info(f"\n{'='*80}")
        logger.info("SERVICE STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Runtime: {runtime}")
        logger.info(f"Checks Completed: {self.checks_completed}")
        logger.info(f"Full Analyses: {self.full_analysis_count}")
        logger.info(f"Alerts Generated: {self.alerts_generated}")
        logger.info(f"Pairs Monitored: {len(self.major_pairs)}")
        logger.info(f"Average Check Frequency: {self.checks_completed/max(1, runtime.total_seconds()/60):.1f} per minute")
        logger.info(f"ACB Features: All 14 features via unified_master_script.py")
        logger.info(f"{'='*80}")

        # Shutdown MT5
        mt5.shutdown()
        logger.info("Service stopped successfully")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Initiating shutdown...")
        self.running = False


def main():
    """Main entry point"""
    print("=" * 80)
    print("MAJOR PAIRS REAL-TIME MONITOR WITH FULL ACB INTEGRATION")
    print("=" * 80)
    print("Monitoring 6 major currency pairs with complete 14-feature ACB analysis:")
    print("  EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD")
    print("\nFeatures:")
    print("  - Real-time price monitoring")
    print("  - Complete ACB analysis using unified_master_script.py")
    print("    • All 14 ACB features")
    print("    • FGD/FRD pattern detection")
    print("    • Manipulation detection")
    print("    • 5-star setup identification")
    print("  - Asian range analysis")
    print("  - 24-hour extreme alerts")
    print("  - Volume spike detection")
    print("  - Full analysis for ALL pairs every 5 minutes")
    print("\nPress Ctrl+C to stop")
    print("=" * 80)

    # Create and start service
    service = RealtimeMajorPairsService()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, service.signal_handler)
    signal.signal(signal.SIGTERM, service.signal_handler)

    # Run the service
    service.run()


if __name__ == "__main__":
    main()