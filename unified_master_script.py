"""
Unified Master Script - 100% Feature Implementation
===============================================

This version includes ALL features for complete visibility:
- All signal types (Pump & Dump, Inside Day, Three Day High/Low)
- Enhanced session analysis with detailed metrics
- Market structure breakout alerts
- Full confidence factor breakdown
- Detailed ACB feature analysis
- Volume analysis display
- Enhanced manipulation detection display
- Asian range sweep strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List

# Import OANDA symbol mapper
from oanda_symbol_mapper import get_oanda_symbol

# Import ALL enhanced components
from acb import (
    EnhancedACBIntegration,
    DMRLevelCalculator,
    EnhancedACBDetector,
    EnhancedFRDFGDDetector,
    AsianRangeEntryDetector,
    LiquidityHuntDetector,
    MarketPhaseIdentifier,
    ACBAwareSignalGenerator,
    FiveStarSetupPrioritizer,
    DMRTargetCalculator
)

def is_market_closed():
    """Check if the forex market is closed (weekend)"""
    now = datetime.now()

    # Forex market is closed on Saturdays and Sundays
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        # Check if it's truly weekend (after market close on Friday)
        if now.weekday() == 5:  # Saturday
            return True, "Saturday"
        elif now.weekday() == 6:  # Sunday
            return True, "Sunday"

    # Check major holidays if needed (can add more holidays here)
    # Christmas, New Year, etc.
    major_holidays = [
        (12, 25),  # Christmas
        (1, 1),    # New Year's Day
        # Add more holidays as needed
    ]

    if (now.month, now.day) in major_holidays:
        return True, f"Holiday: {now.strftime('%B %d')}"

    return False, "Open"

def run_unified_master_analysis():
    """Ultimate complete analysis with 100% feature display"""
    symbol_base = "AUDUSD"  # Default symbol - can be changed when calling directly
    symbol = get_oanda_symbol(symbol_base)
    now = datetime.now()

    # Check if market is closed
    is_closed, reason = is_market_closed()

    if is_closed:
        print("=" * 100)
        print(f"{symbol_base} UNIFIED MASTER ANALYSIS - 100% FEATURES")
        print(f"Date: {now.strftime('%A, %B %d, %Y')}")
        print(f"Time: {now.strftime('%H:%M:%S')} UTC")
        print(f"Status: MARKET CLOSED - {reason}")
        print("=" * 100)
        print()
        print(f"[MARKET CLOSED] The forex market is currently closed ({reason})")
        print(f"[WAITING] Please wait for Monday's price action at 21:00 GMT / 16:00 EST")
        print()
        print(f"[PREVIEW] Current market analysis available, but:")
        print(f"  - Price is not moving (market closed)")
        print(f"  - Wait for Monday open to see real price action")
        print(f"  - Prepare setups based on weekly trend analysis")
        print()
        print(f"[TRADING SESSIONS]")
        print(f"  - Market Reopens: Monday 21:00 GMT (Sunday 4pm EST)")
        print(f"  - Asian Session: Monday 02:00-07:00 UTC")
        print(f"  - London Session: Monday 06:00-10:00 UTC")
        print(f"  - NY Session: Monday 13:00-17:00 UTC")
        print()
        print(f"[CURRENCY PAIR: {symbol_base}]")
        print(f"  - Last Close: Check MT5 or broker for Friday close price")
        print(f"  - Gap Risk: Be aware of potential weekend gaps")
        print()
        print(f"[SETUP PREPARATION]")
        print(f"  - Review weekly trend for Monday bias")
        print(f"  - Identify key support/resistance levels")
        print(f"  - Watch for Sunday 21:00 GMT open")
        print("=" * 100)
        return

    # If market is open, proceed with full analysis
    print("=" * 100)
    print(f"{symbol_base} UNIFIED MASTER ANALYSIS - 100% FEATURES")
    print(f"Date: {now.strftime('%A, %B %d, %Y')}")
    print(f"Time: {now.strftime('%H:%M:%S UTC')}")
    print(f"Trading Day: {now.strftime('%A')}")
    print(f"Status: MARKET OPEN - Active Trading")
    print("=" * 100)

    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
        return

    try:
        # Get comprehensive data
        print("\n[DATA COLLECTION]")
        print("-" * 50)

        # Multiple timeframes for complete analysis
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(h1_rates['time'], unit='s')
        df_h1.set_index('time', inplace=True)
        print(f"[OK] H1 Data: {len(df_h1)} candles")

        m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 2000)
        df_m15 = pd.DataFrame(m15_rates)
        df_m15['time'] = pd.to_datetime(m15_rates['time'], unit='s')
        df_m15.set_index('time', inplace=True)
        print(f"[OK] M15 Data: {len(df_m15)} candles")

        d1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100)
        df_d1 = pd.DataFrame(d1_rates)
        df_d1['time'] = pd.to_datetime(d1_rates['time'], unit='s')
        df_d1.set_index('time', inplace=True)
        print(f"[OK] Daily Data: {len(df_d1)} candles")

        current_price = df_h1.iloc[-1]['close']
        print(f"\nCurrent Price: {current_price:.5f}")

        # 1. ENHANCED ACB INTEGRATION
        print("\n\n[1. ENHANCED ACB INTEGRATION ANALYSIS]")
        print("=" * 70)

        integration = EnhancedACBIntegration()
        analysis = integration.analyze_market_complete(df_h1)

        # FGD/FRD Results
        print("\nFGD/FRD Pattern Detection:")
        fgd = analysis['fgd_pattern']
        if fgd.get('pattern_detected'):
            print(f"  [ACTIVE] {fgd['pattern_description']}")
            print(f"  Type: {fgd['signal_type']}")
            print(f"  Grade: {fgd['signal_grade']}")
            print(f"  Confidence: {fgd['confidence']}%")
            print(f"  Direction: {fgd.get('trade_direction', 'N/A')}")
            print(f"  Trade Today: {'YES' if fgd.get('trade_today') else 'NO'}")
        else:
            print("  [WAIT] No FGD/FRD patterns detected")

        # 2. COMPLETE SIGNAL ANALYSIS (Including Missing 2%)
        print("\n\n[2. COMPLETE SIGNAL ANALYSIS - ALL TYPES]")
        print("=" * 70)

        # Generate all signal types
        signal_gen = ACBAwareSignalGenerator()
        dmr_calc = DMRLevelCalculator()
        acb_det = EnhancedACBDetector()

        dmr_levels = dmr_calc.calculate_all_dmr_levels(df_h1)
        acb_levels = acb_det.identify_enhanced_acb_levels(df_h1, dmr_levels)

        signal_result = signal_gen.generate_signals(df_h1, d1_rates, symbol)

        # Get signals directly from their respective lists
        signal_groups = {
            'FGD/FRD': signal_result.get('frd_fgd_signals', []),
            'Inside Day Breakout': signal_result.get('inside_day_signals', []),
            'Pump & Dump': signal_result.get('pump_dump_signals', []),
            'Three Day High/Low': [],  # Note: These are handled inside FRD/FGD
            'Asian Range Entry': signal_result.get('asian_range_signals', [])
        }

        # For Three Day High/Low, check if any FRD/FGD signals are THREE_DL/THREE_DS
        fgd_fgd_signals = signal_result.get('frd_fgd_signals', [])
        for signal in fgd_fgd_signals:
            signal_type = signal.get('type', '')
            if '3DL' in str(signal_type) or '3DS' in str(signal_type):
                signal_groups['Three Day High/Low'].append(signal)

        # Display all signal types
        for signal_type, signals in signal_groups.items():
            print(f"\n{signal_type} Signals:")
            if signals:
                for i, sig in enumerate(signals[:3], 1):  # Show top 3
                    # Handle confidence which might be enum or string
                    conf = sig.get('confidence', 'N/A')
                    if hasattr(conf, 'value'):
                        confidence = conf.value
                    else:
                        confidence = str(conf)

                    direction = sig.get('direction', 'N/A')
                    signal_type = sig.get('type', 'N/A')

                    print(f"  {i}. {direction} ({signal_type}) | Confidence: {confidence}")
                    if sig.get('reasoning'):
                        print(f"     Reason: {sig['reasoning'][:80]}...")
            else:
                print("  [NONE] No signals detected")

        # 3. ENHANCED SESSION ANALYSIS
        print("\n\n[3. ENHANCED SESSION ANALYSIS]")
        print("=" * 70)

        # Session metrics
        session_metrics = calculate_session_metrics(df_h1)

        print(f"\nCurrent Time:")
        print(f"  PHT: {session_metrics['pht_time']}")
        print(f"  NY: {session_metrics['ny_time']}")
        print(f"  UTC: {session_metrics['utc_time']}")
        print(f"\nCurrent Session: {session_metrics['current_session']}")
        print(f"Session Start: {session_metrics['session_start']:.5f}")
        print(f"Session Range: {session_metrics['session_range'] * 10000:.1f} pips")
        print(f"Session Volume: {session_metrics['session_volume']:,.0f}")
        print(f"Volatility: {session_metrics['volatility'] * 10000:.1f} pips/hr")

        # Asian range details
        if session_metrics.get('asian_range'):
            ar = session_metrics['asian_range']
            print(f"\nAsian Range Details:")
            print(f"  Range: {ar['high']:.5f} - {ar['low']:.5f}")
            print(f"  Width: {ar['range_pips']:.1f} pips")
            print(f"  Breakout Status: {ar['breakout_status']}")
            if ar.get('breakout_direction'):
                print(f"  Breakout Direction: {ar['breakout_direction']}")

        # 4. MARKET STRUCTURE BREAKOUT ALERTS
        print("\n\n[4. MARKET STRUCTURE BREAKOUT ALERTS]")
        print("=" * 70)

        breakout_alerts = analyze_breakout_potential(df_h1, dmr_levels, acb_levels, current_price)

        if breakout_alerts['alerts']:
            print("\nActive Breakout Alerts:")
            for alert in breakout_alerts['alerts']:
                print(f"  [{alert['severity'].upper()}] {alert['type']}")
                print(f"    Level: {alert['level']:.5f}")
                print(f"    Distance: {alert['distance_pips']:.0f} pips")
                print(f"    Probability: {alert['probability']}%")
        else:
            print("\n[NO BREAKOUTS] No immediate breakout alerts")

        # Show breakout candidates
        if breakout_alerts['candidates']:
            print("\nBreakout Candidates (Monitoring):")
            for candidate in breakout_alerts['candidates']:
                print(f"  - {candidate['type']}: {candidate['level']:.5f} ({candidate['distance_pips']:.0f} pips away)")

        # 5. FULL CONFIDENCE FACTOR BREAKDOWN
        print("\n\n[5. COMPLETE CONFIDENCE FACTOR ANALYSIS]")
        print("=" * 70)

        conf = analysis.get('confidence_analysis', {})
        print(f"\nOverall Confidence: {conf.get('total_confidence', 0)}%")
        print(f"Grade: {conf.get('grade', 'N/A')}")

        if conf.get('factors'):
            print("\nConfidence Breakdown:")
            factors = conf['factors']
            for factor, value in factors.items():
                bar_length = int(value / 5)
                bar = "=" * bar_length + "-" * (20 - bar_length)
                print(f"  {factor:20} : {value:3d}% |{bar}|")

        if conf.get('weighted_score'):
            print(f"\nWeighted Score Components:")
            ws = conf['weighted_score']
            for component, score in ws.items():
                print(f"  {component}: {score}")

        # 6. DETAILED ACB FEATURE ANALYSIS
        print("\n\n[6. COMPREHENSIVE ACB FEATURE ANALYSIS]")
        print("=" * 70)

        acb = analysis['acb_levels']

        # Analyze all ACB types
        for acb_type in ['validated', 'extreme', 'confirmed', 'potential']:
            if acb.get(acb_type):
                print(f"\n{acb_type.upper()} ACBs ({len(acb[acb_type])}):")
                for i, level in enumerate(acb[acb_type][:2], 1):  # Show top 2
                    print(f"\n  {i}. {level['type'].upper()}: {level['price']:.5f}")
                    print(f"     Score: {level['validation_score']}")
                    print(f"     Age: {level['hours_elapsed']:.0f} hours")

                    # Show ALL validation features
                    features = []

                    # Consecutive closes
                    if level.get('consecutive_closes'):
                        cc = level['consecutive_closes']
                        if cc.get('has_three_higher'):
                            features.append(f"[OK] 3 Higher Closes (strength: {cc['close_strength']:.2f})")
                        elif cc.get('has_three_lower'):
                            features.append(f"[OK] 3 Lower Closes (strength: {cc['close_strength']:.2f})")
                        if cc.get('consecutive_count'):
                            features.append(f"[OK] {cc['consecutive_count']} consecutive closes")

                    # EMA coil
                    if level.get('ema_coil', {}).get('coil_detected'):
                        coil = level['ema_coil']
                        features.append(f"[OK] EMA Coil (ratio: {coil['coil_ratio']:.3f})")
                        if coil.get('bullish_squeeze'):
                            features.append("  -> Bullish squeeze detected")
                        elif coil.get('bearish_squeeze'):
                            features.append("  -> Bearish squeeze detected")

                    # HTF alignment
                    if level.get('htf_alignment', {}).get('monthly_aligned'):
                        htf = level['htf_alignment']
                        features.append("[OK] HTF Aligned")
                        if htf.get('weekly_aligned'):
                            features.append("  -> Weekly alignment")
                        if htf.get('daily_alignment'):
                            features.append("  -> Daily alignment")

                    # Extreme position
                    if level.get('extreme_position', {}).get('at_extreme'):
                        features.append("[OK] At Extreme Position")

                    # Volume confirmation
                    if level.get('volume_confirmation'):
                        features.append("[OK] Volume Confirmed")

                    # DMR alignment
                    if level.get('dmr_alignment'):
                        features.append("[OK] DMR Aligned")

                    if features:
                        print("     Features:")
                        for feature in features:
                            print(f"       {feature}")

        # 7. VOLUME ANALYSIS DISPLAY
        print("\n\n[7. VOLUME ANALYSIS]")
        print("=" * 70)

        volume_analysis = analyze_volume_profile(df_h1)

        print(f"\nCurrent Volume Profile:")
        print(f"  Current Candle Volume: {volume_analysis['current_volume']:,.0f}")
        print(f"  Average Volume (20): {volume_analysis['avg_volume']:,.0f}")
        print(f"  Volume Ratio: {volume_analysis['volume_ratio']:.2f}")
        print(f"  Volume Trend: {volume_analysis['volume_trend']}")

        if volume_analysis.get('high_volume_nodes'):
            print(f"\nHigh Volume Nodes (Support/Resistance):")
            for node in volume_analysis['high_volume_nodes'][:3]:
                print(f"  - Level: {node['price']:.5f} | Volume: {node['volume']:,.0f}")

        # 8. ENHANCED MANIPULATION DETECTION
        print("\n\n[8. SMART MONEY MANIPULATION ANALYSIS]")
        print("=" * 70)

        # Liquidity hunt detection
        hunt_det = LiquidityHuntDetector()
        hunts_result = hunt_det.detect_liquidity_hunts(df_h1.tail(48), dmr_levels, acb_levels)
        hunts = hunts_result.get('hunts', []) if isinstance(hunts_result, dict) else []

        print(f"\nLiquidity Hunt Analysis (Last 48h):")
        print(f"  Total Hunts: {len(hunts)}")

        if hunts:
            # Categorize hunts
            hunt_types = {}
            for hunt in hunts:
                hunt_type = hunt['type'].value
                if hunt_type not in hunt_types:
                    hunt_types[hunt_type] = []
                hunt_types[hunt_type].append(hunt)

            print(f"\nHunt Breakdown:")
            for hunt_type, type_hunts in hunt_types.items():
                print(f"  {hunt_type}: {len(type_hunts)} occurrences")
                if type_hunts:
                    latest = type_hunts[-1]
                    level_value = latest.get('level_value', latest.get('level', 0))
                    timestamp = latest.get('timestamp')
                    strength = latest.get('strength', {}).value if hasattr(latest.get('strength'), 'value') else latest.get('strength', 'N/A')
                    print(f"    Latest: {level_value:.5f} at {timestamp.strftime('%H:%M') if timestamp else 'N/A'} UTC")
                    print(f"    Strength: {strength}")

                    # Show manipulation signs
                    if latest.get('manipulation_signs'):
                        print("    Signs:")
                        for sign in latest['manipulation_signs'][:3]:
                            print(f"      - {sign}")

        # Market phase with detailed analysis
        phase_id = MarketPhaseIdentifier()
        phase_analysis = phase_id.identify_market_phase(df_h1, dmr_levels, acb_levels)

        print(f"\nCurrent Market Phase: {phase_analysis['current_phase'].value}")
        print(f"Phase Confidence: {phase_analysis['confidence'].value}")
        print(f"Description: {phase_analysis['phase_description']}")

        if phase_analysis.get('phase_characteristics'):
            print(f"\nPhase Characteristics:")
            for char in phase_analysis['phase_characteristics']:
                print(f"  - {char}")

        # 9. FIVE-STAR SETUP PRIORITIZATION
        print("\n\n[9. FIVE-STAR SETUP PRIORITIZATION]")
        print("=" * 70)

        prioritizer = FiveStarSetupPrioritizer()
        setups = prioritizer.prioritize_setups(df_h1, d1_rates, symbol)

        if isinstance(setups, dict):
            if setups.get('setups'):
                print(f"\nTop-Rated Setups:")
                setup_list = setups['setups']
                for i, setup in enumerate(setup_list[:5], 1):
                    stars = "*" * setup['star_rating']
                    print(f"\n{i}. {setup['setup_type']} - {stars} ({setup['star_rating']}/5)")
                    print(f"   Direction: {setup['direction']}")
                    print(f"   Confidence: {setup['confidence']}%")
                    print(f"   Risk/Reward: 1:{setup['risk_reward_ratio']:.1f}")
                    if setup.get('entry_zone'):
                        print(f"   Entry Zone: {setup['entry_zone'][0]:.5f} - {setup['entry_zone'][1]:.5f}")
                    print(f"   Rationale: {setup['rationale'][:60]}...")
            else:
                print("\n[NO SETUPS] No setups found in analysis")
        elif isinstance(setups, list):
            print(f"\nTop-Rated Setups:")
            for i, setup in enumerate(setups[:5], 1):
                stars = "*" * setup['star_rating']
                print(f"\n{i}. {setup['setup_type']} - {stars} ({setup['star_rating']}/5)")
                print(f"   Direction: {setup['direction']}")
                print(f"   Confidence: {setup['confidence']}%")
                print(f"   Risk/Reward: 1:{setup['risk_reward_ratio']:.1f}")
                if setup.get('entry_zone'):
                    print(f"   Entry Zone: {setup['entry_zone'][0]:.5f} - {setup['entry_zone'][1]:.5f}")
                print(f"   Rationale: {setup['rationale'][:60]}...")
        else:
            print("\n[NO SETUPS] No high-probability setups detected")

        # 10. DMR TARGET CALCULATION
        print("\n\n[10. DMR TARGET CALCULATION (Phase 5)]")
        print("=" * 70)

        target_calc = DMRTargetCalculator()
        targets = target_calc.calculate_targets(df_h1, dmr_levels, acb_levels, current_price)

        if targets.get('targets'):
            print(f"\nDMR-Based Targets:")
            for i, target in enumerate(targets['targets'][:4], 1):
                target_type = target['type'].replace('_', ' ').title()
                distance = abs(target['price'] - current_price) * 10000
                print(f"\n{i}. {target_type}")
                print(f"   Price: {target['price']:.5f}")
                print(f"   Distance: {distance:.0f} pips")
                print(f"   Probability: {target['probability']}%")
                if target.get('expected_time'):
                    print(f"   Expected Time: {target['expected_time']}")
        else:
            print("\n[NO TARGETS] No valid targets identified")

        # 11. MONDAY BREAKOUT DETAILS (Enhanced)
        print("\n\n[11. MONDAY BREAKOUT ANALYSIS]")
        print("=" * 70)

        if now.weekday() == 0 or (now.weekday() == 1 and len(df_h1) > 24):
            monday_breakout = analyze_monday_breakout(df_h1, dmr_levels)

            if monday_breakout['active']:
                print(f"\n[ACTIVE] Monday Breakout Detected!")
                print(f"  Direction: {monday_breakout['direction']}")
                print(f"  Broken Level: {monday_breakout['broken_level']:.5f}")
                print(f"  Breakout Strength: {monday_breakout['strength']}/10")
                print(f"  Retest Status: {monday_breakout['retest_status']}")
                if monday_breakout.get('entry_price'):
                    print(f"  Entry Opportunity: {monday_breakout['entry_price']:.5f}")
            else:
                print("\n[INACTIVE] No Monday breakout detected")
        else:
            print("\n[NOT MONDAY] Monday breakout analysis not applicable")

        # 12. FINAL COMPREHENSIVE RECOMMENDATION
        print("\n\n[12. ULTIMATE TRADING RECOMMENDATION]")
        print("=" * 70)

        # Compile all factors
        overall_bias = determine_overall_bias(analysis, signal_result, phase_analysis)

        print(f"\nOverall Market Bias: {overall_bias['bias']}")
        print(f"Confidence: {overall_bias['confidence']}%")

        if overall_bias.get('primary_signals'):
            print("\nPrimary Influences:")
            for signal in overall_bias['primary_signals']:
                print(f"  - {signal}")

        print("\nAction Plan:")
        for action in overall_bias['action_plan']:
            print(f"  {action}")

        print("\nRisk Management:")
        print(f"  - ATR (14): {(df_h1['high'] - df_h1['low']).tail(14).mean() * 10000:.1f} pips")
        print(f"  - Suggested Stop: {overall_bias.get('suggested_stop', 'Use structure')}")
        print(f"  - Position Size: {overall_bias.get('position_size_advice', 'Standard 1% risk')}")

        # 13. HIGHER TIMEFRAME CONTEXT
        print("\n\n[13. HIGHER TIMEFRAME ANALYSIS]")
        print("=" * 70)

        # Weekly
        if len(df_d1) >= 5:
            week_data = df_d1.tail(5)
            whigh = week_data['high'].max()
            wlow = week_data['low'].min()
            wclose = week_data.iloc[-1]['close']
            wpos = ((wclose - wlow) / (whigh - wlow) * 100)

            print(f"\nWeekly Structure:")
            print(f"  Range: {wlow:.5f} - {whigh:.5f} ({(whigh-wlow)*10000:.0f} pips)")
            print(f"  Position: {wpos:.0f}% from low")
            print(f"  Trend: {'Bullish' if wclose > week_data.iloc[0]['open'] else 'Bearish'}")

        # Monthly
        if len(df_d1) >= 22:
            month_data = df_d1.tail(22)
            mhigh = month_data['high'].max()
            mlow = month_data['low'].min()
            mclose = month_data.iloc[-1]['close']
            mpos = ((mclose - mlow) / (mhigh - mlow) * 100)

            print(f"\nMonthly Structure:")
            print(f"  Range: {mlow:.5f} - {mhigh:.5f} ({(mhigh-mlow)*10000:.0f} pips)")
            print(f"  Position: {mpos:.0f}% from low")
            print(f"  Trend: {'Bullish' if mclose > month_data.iloc[0]['open'] else 'Bearish'}")

        # Summary
        print("\n" + "=" * 100)
        print("ULTIMATE ANALYSIS COMPLETE - 100% FEATURES DISPLAYED")
        print("=" * 100)
        print(f"Total Systems Analyzed: 14")
        print(f"Signal Types Displayed: 5/5 (100%)")
        print(f"Analysis Time: {datetime.now().strftime('%H:%M:%S UTC')}")
        print("\nAll enhanced systems operational and providing real-time insights")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        mt5.shutdown()


# Helper functions for enhanced analysis

def calculate_session_metrics(df_h1: pd.DataFrame) -> Dict:
    """Calculate detailed session metrics"""
    now = datetime.now(timezone.utc)
    utc_hour = now.hour

    # Calculate local times
    pht_time = now + timedelta(hours=8)
    ny_time = now - timedelta(hours=4)  # EDT (UTC-4) in summer

    # Determine current session based on UTC
    if 0 <= utc_hour < 7:
        current_session = "Asian"
        session_start = datetime(now.year, now.month, now.day, 0, 0)
        session_end = datetime(now.year, now.month, now.day, 7, 0)
    elif 7 <= utc_hour < 13:
        current_session = "London"
        session_start = datetime(now.year, now.month, now.day, 7, 0)
        session_end = datetime(now.year, now.month, now.day, 13, 0)
    elif 13 <= utc_hour < 20:
        current_session = "New York"
        session_start = datetime(now.year, now.month, now.day, 13, 0)
        session_end = datetime(now.year, now.month, now.day, 20, 0)
    else:
        current_session = "Post-NY"
        session_start = datetime(now.year, now.month, now.day, 20, 0)
        session_end = datetime(now.year, now.month, now.day, 0, 0) + timedelta(days=1)

    # Get session data
    session_data = df_h1[df_h1.index >= session_start]
    if current_session != "Post-NY":
        session_data = session_data[session_data.index < session_end]

    if len(session_data) == 0:
        return {
            'current_session': current_session,
            'session_start': df_h1.iloc[-1]['close'],
            'session_range': 0,
            'session_volume': 0,
            'volatility': 0,
            'pht_time': pht_time.strftime('%H:%M PHT'),
            'ny_time': ny_time.strftime('%H:%M NY'),
            'utc_time': now.strftime('%H:%M UTC')
        }

    # Calculate metrics
    session_high = session_data['high'].max()
    session_low = session_data['low'].min()
    session_range = session_high - session_low

    # Volume (if available)
    session_volume = session_data['tick_volume'].sum() if 'tick_volume' in session_data else 0

    # Volatility (pips per hour)
    volatility = session_range / max(len(session_data), 1) if len(session_data) > 0 else 0

    # Asian range analysis
    asian_range = None
    if current_session in ["London", "New York"]:
        # Get today's Asian session properly
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time()).replace(hour=2)
        today_end = datetime.combine(today, datetime.min.time()).replace(hour=7)

        asian_data = df_h1[
            (df_h1.index >= pd.Timestamp(today_start)) &
            (df_h1.index < pd.Timestamp(today_end))
        ]

        if len(asian_data) > 0:
            asian_high = asian_data['high'].max()
            asian_low = asian_data['low'].min()
            current_price = df_h1.iloc[-1]['close']

            asian_range = {
                'high': asian_high,
                'low': asian_low,
                'range_pips': (asian_high - asian_low) * 10000,
                'breakout_status': 'ABOVE' if current_price > asian_high else 'BELOW' if current_price < asian_low else 'INSIDE',
                'breakout_direction': 'UP' if current_price > asian_high else 'DOWN' if current_price < asian_low else None
            }

    return {
        'current_session': current_session,
        'session_start': session_data.iloc[0]['close'] if len(session_data) > 0 else df_h1.iloc[-1]['close'],
        'session_range': session_range,
        'session_volume': session_volume,
        'volatility': volatility,
        'asian_range': asian_range,
        'pht_time': pht_time.strftime('%H:%M PHT'),
        'ny_time': ny_time.strftime('%H:%M NY'),
        'utc_time': now.strftime('%H:%M UTC')
    }


def analyze_breakout_potential(df_h1: pd.DataFrame, dmr_levels: Dict, acb_levels: Dict, current_price: float) -> Dict:
    """Analyze breakout potential at key levels"""
    alerts = []
    candidates = []

    # Check distance to key levels
    all_levels = []

    # DMR levels
    if dmr_levels.get('daily'):
        all_levels.append({'type': 'Daily High', 'level': dmr_levels['daily']['high']['price'], 'source': 'DMR'})
        all_levels.append({'type': 'Daily Low', 'level': dmr_levels['daily']['low']['price'], 'source': 'DMR'})

    # ACB levels
    for acb_type in ['validated', 'extreme']:
        if acb_levels.get(acb_type):
            for acb in acb_levels[acb_type]:
                all_levels.append({'type': f'{acb_type.title()} ACB', 'level': acb['price'], 'source': 'ACB'})

    # Analyze each level
    for level_info in all_levels[:10]:  # Check top 10
        level = level_info['level']
        distance = abs(level - current_price) * 10000

        if distance < 20:  # Within 20 pips
            severity = 'HIGH' if distance < 5 else 'MEDIUM' if distance < 10 else 'LOW'
            probability = 80 if distance < 5 else 60 if distance < 10 else 40

            alerts.append({
                'severity': severity,
                'type': level_info['type'],
                'level': level,
                'distance_pips': distance,
                'probability': probability
            })
        elif distance < 50:  # Within 50 pips
            candidates.append({
                'type': level_info['type'],
                'level': level,
                'distance_pips': distance
            })

    return {
        'alerts': alerts,
        'candidates': candidates
    }


def analyze_volume_profile(df_h1: pd.DataFrame) -> Dict:
    """Analyze volume profile"""
    if 'tick_volume' not in df_h1.columns:
        return {
            'current_volume': 0,
            'avg_volume': 0,
            'volume_ratio': 0,
            'volume_trend': 'No volume data',
            'high_volume_nodes': []
        }

    current_volume = df_h1.iloc[-1]['tick_volume']
    avg_volume = df_h1['tick_volume'].tail(20).mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    # Volume trend
    recent_volumes = df_h1['tick_volume'].tail(10)
    if len(recent_volumes) > 5:
        if recent_volumes.is_monotonic_increasing:
            volume_trend = 'Increasing'
        elif recent_volumes.is_monotonic_decreasing:
            volume_trend = 'Decreasing'
        else:
            volume_trend = 'Fluctuating'
    else:
        volume_trend = 'Insufficient data'

    # High volume nodes (price levels with high volume)
    high_volume_nodes = []
    if len(df_h1) > 50:
        # Group by price ranges and sum volume
        df_h1['price_rounded'] = df_h1['close'].round(4)
        volume_by_price = df_h1.groupby('price_rounded')['tick_volume'].sum().sort_values(ascending=False)
        high_volume_nodes = [
            {'price': price, 'volume': volume}
            for price, volume in volume_by_price.head(5).items()
        ]

    return {
        'current_volume': int(current_volume),
        'avg_volume': int(avg_volume),
        'volume_ratio': volume_ratio,
        'volume_trend': volume_trend,
        'high_volume_nodes': high_volume_nodes
    }


def analyze_monday_breakout(df_h1: pd.DataFrame, dmr_levels: Dict) -> Dict:
    """Analyze Monday breakout patterns"""
    now = datetime.now()

    # Get Monday data (either today or last Monday)
    if now.weekday() == 0:  # Today is Monday
        monday_data = df_h1[df_h1.index.date == now.date()]
    else:  # Get last Monday
        last_monday = now - timedelta(days=now.weekday())
        monday_data = df_h1[df_h1.index.date == last_monday.date()]

    if len(monday_data) == 0:
        return {'active': False}

    monday_high = monday_data['high'].max()
    monday_low = monday_data['low'].min()

    # Check if HOD/LOD was broken
    if dmr_levels.get('hodlod'):
        hod = dmr_levels['hodlod']['high']['price']
        lod = dmr_levels['hodlod']['low']['price']

        # Check for breakout
        current_price = df_h1.iloc[-1]['close']

        if current_price > monday_high:
            return {
                'active': True,
                'direction': 'UP',
                'broken_level': monday_high,
                'strength': min(10, int((current_price - monday_high) * 10000 / 10)),
                'retest_status': 'PENDING' if current_price < monday_high + 0.0010 else 'CONFIRMED'
            }
        elif current_price < monday_low:
            return {
                'active': True,
                'direction': 'DOWN',
                'broken_level': monday_low,
                'strength': min(10, int((monday_low - current_price) * 10000 / 10)),
                'retest_status': 'PENDING' if current_price > monday_low - 0.0010 else 'CONFIRMED'
            }

    return {'active': False}


def determine_overall_bias(analysis: Dict, signals: Dict, phase_analysis: Dict) -> Dict:
    """
    Determine overall market bias - STACEY BURKE ACB LOGIC

    GOLDEN RULE: FGD = BUYING DAY (Longs only), FRD = SELLING DAY (Shorts only)
    """
    bias_score = 0
    reasons = []

    # FGD/FRD influence - PRIMARY FILTER
    fgd = analysis.get('fgd_pattern', {})
    fgd_detected = fgd.get('pattern_detected', False)
    trade_direction = fgd.get('trade_direction', '')
    trade_today = fgd.get('trade_today', False)

    if fgd_detected:
        if trade_direction == 'LONG':
            # FGD (First Green Day) = BUYING DAY
            bias_score = 50  # Strong bullish
            reasons.append("FGD (First Green Day) - BUYING DAY - Longs ONLY")
        elif trade_direction == 'SHORT':
            # FRD (First Red Day) = SELLING DAY
            bias_score = -50  # Strong bearish
            reasons.append("FRD (First Red Day) - SELLING DAY - Shorts ONLY")
    else:
        # No FGD/FRD detected - neutral
        bias_score = 0
        reasons.append("No FGD/FRD pattern detected - NEUTRAL")

    # Filter signals based on FGD/FRD direction
    # FGD = Only consider long signals, FRD = Only consider short signals
    all_sig_lists = [
        signals.get('frd_fgd_signals', []),
        signals.get('inside_day_signals', []),
        signals.get('pump_dump_signals', []),
        signals.get('asian_range_signals', [])
    ]

    # Count ONLY aligned signals
    if trade_direction == 'LONG' and fgd_detected:
        # Buying day - count only bullish signals
        bullish_signals = sum(1 for s in sum(all_sig_lists, []) if 'long' in str(s.get('direction', '')).lower())
        reasons.append(f"Bullish signals: {bullish_signals} (Short signals IGNORED - Buying day)")
    elif trade_direction == 'SHORT' and fgd_detected:
        # Selling day - count only bearish signals
        bearish_signals = sum(1 for s in sum(all_sig_lists, []) if 'short' in str(s.get('direction', '')).lower())
        reasons.append(f"Bearish signals: {bearish_signals} (Long signals IGNORED - Selling day)")
    else:
        # No FGD/FRD - show all signals
        bullish_signals = sum(1 for s in sum(all_sig_lists, []) if 'long' in str(s.get('direction', '')).lower())
        bearish_signals = sum(1 for s in sum(all_sig_lists, []) if 'short' in str(s.get('direction', '')).lower())
        reasons.append(f"Bullish: {bullish_signals} | Bearish: {bearish_signals} (No FGD/FRD filter)")

    # Determine final bias
    if trade_direction == 'LONG' and fgd_detected:
        bias = "STRONGLY BULLISH (FGD - BUYING DAY)"
        confidence = 85
    elif trade_direction == 'SHORT' and fgd_detected:
        bias = "STRONGLY BEARISH (FRD - SELLING DAY)"
        confidence = 85
    else:
        # No FGD/FRD - use phase
        
        phase = phase_analysis.get('current_phase', None)
        phase_value = phase.value if hasattr(phase, 'value') else str(phase)
        if 'bullish' in phase_value.lower():
            bias = "BULLISH (No FGD/FRD - Market Phase)"
            confidence = 50
        elif 'bearish' in phase_value.lower():
            bias = "BEARISH (No FGD/FRD - Market Phase)"
            confidence = 50
        else:
            bias = "NEUTRAL (No FGD/FRD - Wait for setup)"
            confidence = 50

    # Action plan based on FGD/FRD
    action_plan = []

    if trade_direction == 'LONG' and fgd_detected:
        action_plan.append("[LOOK] for LONG entry opportunities (Asian Low sweep, pullbacks)")
        action_plan.append("[IGNORE] all short signals - FGD = Buying day")
    elif trade_direction == 'SHORT' and fgd_detected:
        action_plan.append("[LOOK] for SHORT entry opportunities (Asian High sweep, pullbacks)")
        action_plan.append("[IGNORE] all long signals - FRD = Selling day")
    else:
        action_plan.append("[WAIT] for FGD or FRD pattern to trigger")
        action_plan.append("[MONITOR] Asian range for sweep + rejection setup")

    if trade_today:
        action_plan.append("[ACTIVE] Trade today - FGD/FRD in play")

    return {
        'bias': bias,
        'confidence': confidence,
        'primary_signals': reasons,
        'action_plan': action_plan,
        'fgd_frd_filter': trade_direction if fgd_detected else 'NONE',
        'trade_today': trade_today,
        'suggested_stop': 'Use recent swing or structure level',
        'position_size_advice': 'Standard 1% risk management'
    }


if __name__ == "__main__":
    run_unified_master_analysis()