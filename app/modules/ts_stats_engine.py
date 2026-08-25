"""
DataPilot AI — Time-Series Statistical Tests
ACF/PACF, stationarity tests (ADF/KPSS), seasonal decomposition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def run_stationarity_test(series: pd.Series) -> Dict:
    """Run ADF and KPSS stationarity tests."""
    from statsmodels.tsa.stattools import adfuller, kpss

    series = series.dropna()
    if len(series) < 20:
        return {"error": "Need at least 20 data points"}

    result = {}

    # ADF Test
    try:
        adf_stat, adf_p, adf_lags, nobs, crit, _ = adfuller(series, autolag="AIC")
        result["adf"] = {
            "statistic": round(float(adf_stat), 4),
            "p_value": round(float(adf_p), 4),
            "lags_used": int(adf_lags),
            "critical_values": {k: round(float(v), 4) for k, v in crit.items()},
            "stationary": float(adf_p) < 0.05,
            "interpretation": (
                "✅ Series is **stationary** (p < 0.05)" if adf_p < 0.05
                else "❌ Series is **non-stationary** (p ≥ 0.05) — consider differencing"
            ),
        }
    except Exception as e:
        result["adf"] = {"error": str(e)}

    # KPSS Test
    try:
        kpss_stat, kpss_p, kpss_lags, crit = kpss(series, regression="c", nlags="auto")
        result["kpss"] = {
            "statistic": round(float(kpss_stat), 4),
            "p_value": round(float(kpss_p), 4),
            "lags_used": int(kpss_lags),
            "critical_values": {k: round(float(v), 4) for k, v in crit.items()},
            "stationary": float(kpss_p) > 0.05,
            "interpretation": (
                "✅ Series is **stationary** (p > 0.05)" if kpss_p > 0.05
                else "❌ Series is **non-stationary** (p ≤ 0.05)"
            ),
        }
    except Exception as e:
        result["kpss"] = {"error": str(e)}

    # Combined interpretation
    adf_stat_result = result.get("adf", {}).get("stationary", None)
    kpss_stat_result = result.get("kpss", {}).get("stationary", None)

    if adf_stat_result and kpss_stat_result:
        result["conclusion"] = "Both tests agree: series is **stationary**."
    elif not adf_stat_result and not kpss_stat_result:
        result["conclusion"] = "Both tests agree: series is **non-stationary**. Apply differencing."
    elif adf_stat_result and not kpss_stat_result:
        result["conclusion"] = "Tests disagree: series may be **trend-stationary**. Consider detrending."
    else:
        result["conclusion"] = "Tests disagree: series may have a **unit root with drift**. Investigate further."

    return result


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
) -> Dict:
    """Compute ACF and PACF with confidence intervals."""
    from statsmodels.tsa.stattools import acf, pacf

    series = series.dropna()
    nlags = min(nlags, len(series) // 2 - 1)
    if nlags < 2:
        return {"error": "Insufficient data points"}

    try:
        acf_vals, acf_ci = acf(series, nlags=nlags, alpha=0.05)
        pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=0.05)

        ci_bound = 1.96 / np.sqrt(len(series))

        return {
            "acf": acf_vals.tolist(),
            "pacf": pacf_vals.tolist(),
            "acf_ci_lower": acf_ci[:, 0].tolist(),
            "acf_ci_upper": acf_ci[:, 1].tolist(),
            "pacf_ci_lower": pacf_ci[:, 0].tolist(),
            "pacf_ci_upper": pacf_ci[:, 1].tolist(),
            "confidence_bound": round(ci_bound, 4),
            "nlags": nlags,
            "significant_acf_lags": [
                i for i, v in enumerate(acf_vals) if i > 0 and abs(v) > ci_bound
            ],
            "significant_pacf_lags": [
                i for i, v in enumerate(pacf_vals) if i > 0 and abs(v) > ci_bound
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def seasonal_decomposition(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    period: Optional[int] = None,
    model: str = "additive",
) -> Dict:
    """Perform seasonal decomposition."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    ts_df = df[[datetime_col, value_col]].dropna().copy()
    if not pd.api.types.is_datetime64_any_dtype(ts_df[datetime_col]):
        ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col], errors="coerce")
    ts_df = ts_df.dropna().sort_values(datetime_col).set_index(datetime_col)

    if len(ts_df) < 10:
        return {"error": "Need at least 10 data points"}

    # Auto-detect period if not provided
    if period is None:
        period = _auto_detect_period(ts_df[value_col])

    if period < 2 or period >= len(ts_df) // 2:
        period = min(12, len(ts_df) // 3)

    try:
        decomposition = seasonal_decompose(ts_df[value_col], model=model, period=period)

        return {
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.dropna().tolist(),
            "residual": decomposition.resid.dropna().tolist(),
            "observed": decomposition.observed.tolist(),
            "dates": [str(d) for d in decomposition.observed.index],
            "period": period,
            "model": model,
            "trend_strength": round(float(
                1 - decomposition.resid.dropna().var() /
                (decomposition.trend.dropna() + decomposition.resid.dropna()).dropna().var()
            ), 3) if decomposition.resid.dropna().var() > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def _auto_detect_period(series: pd.Series) -> int:
    """Auto-detect the dominant period using FFT."""
    try:
        values = series.dropna().values
        if len(values) < 20:
            return 7

        fft = np.fft.rfft(values - values.mean())
        magnitudes = np.abs(fft)
        # Skip DC component (index 0) and very low frequencies
        magnitudes[:2] = 0
        dominant_freq = np.argmax(magnitudes)
        period = max(2, len(values) // dominant_freq if dominant_freq > 0 else 7)
        return min(period, len(values) // 3)
    except Exception:
        return 7
