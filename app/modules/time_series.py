"""
Time Series Analysis Module.
Provides time series detection, trend analysis, seasonality, and rolling statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain or can be parsed as datetime."""
    datetime_cols = []
    
    for col in df.columns:
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
        
        # Try parsing object columns as datetime
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(50)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
            except (ValueError, TypeError):
                continue
    
    return datetime_cols


def is_time_series(df: pd.DataFrame, datetime_col: str) -> Dict[str, Any]:
    """
    Determine if the data represents a time series.
    
    Returns dict with:
        - is_time_series: bool
        - frequency: detected frequency (D, H, M, etc.)
        - is_regular: whether intervals are regular
        - gaps: number of gaps in the series
    """
    try:
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            dt_series = pd.to_datetime(df[datetime_col])
        else:
            dt_series = df[datetime_col]
        
        # Sort by datetime
        dt_sorted = dt_series.dropna().sort_values()
        
        if len(dt_sorted) < 3:
            return {'is_time_series': False, 'reason': 'Insufficient data points'}
        
        # Calculate time differences
        diffs = dt_sorted.diff().dropna()
        
        # Check if mostly monotonic (allowing some irregularity)
        increasing = (diffs > pd.Timedelta(0)).sum()
        is_mostly_monotonic = increasing / len(diffs) > 0.8
        
        if not is_mostly_monotonic:
            return {'is_time_series': False, 'reason': 'Not monotonic'}
        
        # Infer frequency
        try:
            freq = pd.infer_freq(dt_sorted)
        except (ValueError, TypeError):
            freq = None
        
        # Calculate regularity
        median_diff = diffs.median()
        is_regular = (diffs == median_diff).sum() / len(diffs) > 0.9
        
        # Detect gaps
        if median_diff > pd.Timedelta(0):
            gaps = (diffs > median_diff * 2).sum()
        else:
            gaps = 0
        
        return {
            'is_time_series': True,
            'frequency': freq,
            'is_regular': is_regular,
            'gaps': gaps,
            'median_interval': str(median_diff),
            'total_duration': str(dt_sorted.max() - dt_sorted.min()),
        }
    
    except Exception as e:
        return {'is_time_series': False, 'reason': str(e)}


def prepare_time_series(df: pd.DataFrame, datetime_col: str, value_col: str) -> pd.DataFrame:
    """Prepare DataFrame for time series analysis."""
    ts_df = df[[datetime_col, value_col]].copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(ts_df[datetime_col]):
        ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col])
    
    # Sort by datetime
    ts_df = ts_df.sort_values(datetime_col)
    
    # Set datetime as index
    ts_df = ts_df.set_index(datetime_col)
    
    return ts_df


def calculate_rolling_statistics(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    windows: List[int] = [7, 14, 30]
) -> pd.DataFrame:
    """Calculate rolling mean and standard deviation."""
    ts_df = prepare_time_series(df, datetime_col, value_col)
    
    result = ts_df.copy()
    result.columns = ['Value']
    
    for window in windows:
        result[f'Rolling_Mean_{window}'] = result['Value'].rolling(window=window).mean()
        result[f'Rolling_Std_{window}'] = result['Value'].rolling(window=window).std()
    
    return result.reset_index()


def calculate_trend(df: pd.DataFrame, datetime_col: str, value_col: str) -> Dict[str, Any]:
    """Calculate linear trend of the time series."""
    ts_df = prepare_time_series(df, datetime_col, value_col)
    
    # Convert datetime to numeric for regression
    x = np.arange(len(ts_df))
    y = ts_df[value_col].values
    
    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return {'trend': 'insufficient_data'}
    
    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Determine trend direction
    if slope > 0.01 * np.std(y):
        trend = 'upward'
    elif slope < -0.01 * np.std(y):
        trend = 'downward'
    else:
        trend = 'flat'
    
    return {
        'trend': trend,
        'slope': slope,
        'intercept': intercept,
        'trend_line': {'x': x.tolist(), 'y': (slope * x + intercept).tolist()},
    }


def resample_time_series(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    freq: str = 'D',
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """Resample time series to different frequency."""
    ts_df = prepare_time_series(df, datetime_col, value_col)
    
    agg_funcs = {
        'mean': 'mean',
        'sum': 'sum',
        'min': 'min',
        'max': 'max',
        'first': 'first',
        'last': 'last',
        'count': 'count',
    }
    
    func = agg_funcs.get(agg_func, 'mean')
    resampled = ts_df.resample(freq).agg(func)
    
    return resampled.reset_index()


def calculate_period_stats(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    period: str = 'month'
) -> pd.DataFrame:
    """Calculate statistics by time period (day, week, month, year)."""
    ts_df = prepare_time_series(df, datetime_col, value_col)
    ts_df = ts_df.reset_index()
    
    if period == 'hour':
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.hour
    elif period == 'dayofweek':
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.day_name()
    elif period == 'day':
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.day
    elif period == 'month':
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.month_name()
    elif period == 'quarter':
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.quarter
    elif period == 'year':
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.year
    else:
        ts_df['period'] = pd.to_datetime(ts_df[datetime_col]).dt.month
    
    stats = ts_df.groupby('period')[value_col].agg(['mean', 'std', 'min', 'max', 'count'])
    stats = stats.round(2)
    
    return stats.reset_index()


def detect_seasonality(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str
) -> Dict[str, Any]:
    """Detect basic seasonality patterns."""
    ts_df = prepare_time_series(df, datetime_col, value_col)
    ts_df = ts_df.reset_index()
    
    results = {}
    
    # Daily pattern (by hour)
    if len(ts_df) > 24:
        try:
            hourly = ts_df.copy()
            hourly['hour'] = pd.to_datetime(hourly[datetime_col]).dt.hour
            hourly_mean = hourly.groupby('hour')[value_col].mean()
            if hourly_mean.std() > hourly_mean.mean() * 0.1:
                results['daily_pattern'] = True
            else:
                results['daily_pattern'] = False
        except Exception:
            results['daily_pattern'] = None
    
    # Weekly pattern
    if len(ts_df) > 7:
        try:
            weekly = ts_df.copy()
            weekly['dayofweek'] = pd.to_datetime(weekly[datetime_col]).dt.dayofweek
            weekly_mean = weekly.groupby('dayofweek')[value_col].mean()
            if weekly_mean.std() > weekly_mean.mean() * 0.1:
                results['weekly_pattern'] = True
            else:
                results['weekly_pattern'] = False
        except Exception:
            results['weekly_pattern'] = None
    
    # Monthly pattern
    if len(ts_df) > 30:
        try:
            monthly = ts_df.copy()
            monthly['month'] = pd.to_datetime(monthly[datetime_col]).dt.month
            monthly_mean = monthly.groupby('month')[value_col].mean()
            if monthly_mean.std() > monthly_mean.mean() * 0.1:
                results['monthly_pattern'] = True
            else:
                results['monthly_pattern'] = False
        except Exception:
            results['monthly_pattern'] = None
    
    return results


def get_time_series_summary(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str
) -> Dict[str, Any]:
    """Get comprehensive time series summary."""
    ts_info = is_time_series(df, datetime_col)
    
    if not ts_info.get('is_time_series', False):
        return {'error': 'Not a valid time series', 'details': ts_info}
    
    ts_df = prepare_time_series(df, datetime_col, value_col)
    
    summary = {
        'time_series_info': ts_info,
        'value_stats': {
            'mean': ts_df[value_col].mean(),
            'std': ts_df[value_col].std(),
            'min': ts_df[value_col].min(),
            'max': ts_df[value_col].max(),
        },
        'trend': calculate_trend(df, datetime_col, value_col),
        'seasonality': detect_seasonality(df, datetime_col, value_col),
    }
    
    return summary
 
