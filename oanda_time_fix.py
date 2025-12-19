"""
OANDA Timezone Fix
==================

Handles OANDA's timezone offset (UTC+2 instead of UTC).

OANDA data shows times in UTC+2, so we need to:
1. Convert OANDA timestamps to actual UTC
2. Use correct UTC for our session calculations
3. Ensure proper time alignment for pump & dump detection
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import pytz


def convert_oanda_to_utc(df):
    """
    Convert OANDA timestamps from UTC+2 to actual UTC

    Args:
        df: DataFrame with datetime index from OANDA

    Returns:
        DataFrame with corrected UTC timestamps
    """
    df_corrected = df.copy()

    # OANDA uses UTC+2, so we subtract 2 hours to get actual UTC
    df_corrected.index = df_corrected.index - timedelta(hours=2)

    return df_corrected


def get_session_corrected_time(timestamp_utc):
    """
    Convert UTC timestamp to EST for display

    Args:
        timestamp_utc: datetime in UTC

    Returns:
        Dict with time in different formats
    """
    # EST timezone (UTC-5 in December)
    est_tz = pytz.timezone('US/Eastern')

    # Ensure timestamp is timezone-aware
    if timestamp_utc.tzinfo is None:
        timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)

    return {
        'utc': timestamp_utc.strftime('%H:%M UTC'),
        'est': timestamp_utc.astimezone(est_tz).strftime('%I:%M %p EST'),
        'local_oanda': (timestamp_utc + timedelta(hours=2)).strftime('%H:%M OANDA')
    }


def display_time_info(df):
    """
    Display time conversion information for OANDA data
    """
    print("\n" + "=" * 60)
    print("OANDA TIMEZONE INFORMATION")
    print("=" * 60)
    print("OANDA data timestamp format: UTC+2 (server time)")
    print("Actual UTC time: OANDA time - 2 hours")
    print("EST time: UTC - 5 hours")
    print()

    # Show sample conversions
    if len(df) > 0:
        sample_time = df.index[0]
        print(f"Example:")
        print(f"  OANDA shows: {sample_time.strftime('%Y-%m-%d %H:%M')}")

        utc_time = sample_time - timedelta(hours=2)
        est_time = utc_time - timedelta(hours=5)

        print(f"  Actual UTC:  {utc_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  EST time:    {est_time.strftime('%Y-%m-%d %I:%M %p')}")
        print()
        print(f"Difference from UTC: +2 hours")
        print(f"Difference from EST: +7 hours")


def correct_session_times(df):
    """
    Return session-aware time columns for analysis

    Args:
        df: DataFrame with corrected UTC timestamps

    Returns:
        DataFrame with session information
    """
    df_session = df.copy()

    # Add time columns
    df_session['utc_hour'] = df_session.index.hour
    df_session['est_hour'] = (df_session.index - timedelta(hours=5)).hour

    # Session labels based on actual UTC
    def get_utc_session(utc_hour):
        if 0 <= utc_hour < 7:
            return 'Asian'
        elif 7 <= utc_hour < 13:
            return 'London'
        elif 13 <= utc_hour < 22:
            return 'New York'
        else:
            return 'Overlap'

    df_session['session'] = df_session['utc_hour'].apply(get_utc_session)

    return df_session


# Export the main functions
__all__ = [
    'convert_oanda_to_utc',
    'get_session_corrected_time',
    'display_time_info',
    'correct_session_times'
]