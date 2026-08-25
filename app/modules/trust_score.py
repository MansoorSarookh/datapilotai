"""
DataPilot AI — Dataset Trust Score Engine
Computes a multi-dimensional trust score with actionable flags.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def compute_trust_score(df: pd.DataFrame) -> Dict:
    """
    Compute a multi-dimensional Dataset Trust Score.
    
    Returns:
        dict with keys: overall, dimensions, flags, label, color, recommendations
    """
    scores = {}
    flags = []
    recommendations = []

    # ── 1. Completeness ───────────────────────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isna().sum().sum()
    completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 1.0
    scores["completeness"] = completeness
    
    if completeness < 0.95:
        cols_with_nulls = df.columns[df.isna().any()].tolist()
        flags.append(f"⚠️ {null_cells} missing values across {len(cols_with_nulls)} column(s)")
        recommendations.append(f"Impute or drop null values in: {', '.join(cols_with_nulls[:5])}")

    # ── 2. Consistency (type violations) ─────────────────────────────────────
    type_violations = 0
    for col in df.select_dtypes(include="object").columns:
        # Try coercing to numeric — violations = could-be-numeric cells mixed with text
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null_orig = df[col].notna().sum()
        if non_null_orig > 0:
            converted = coerced.notna().sum()
            ratio = converted / non_null_orig
            if 0.1 < ratio < 0.9:  # Mixed types
                type_violations += int(min(coerced.isna().sum(), converted))
    
    consistency = 1 - (type_violations / total_cells) if total_cells > 0 else 1.0
    consistency = max(0.0, min(1.0, consistency))
    scores["consistency"] = consistency
    
    if consistency < 0.95:
        flags.append(f"⚠️ Possible mixed data types detected in object columns")
        recommendations.append("Review object columns for mixed numeric/text values")

    # ── 3. Variance (information content) ────────────────────────────────────
    variance_scores = []
    numeric_df = df.select_dtypes(include=np.number)
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) > 1 and series.std() > 0:
            # Coefficient of variation captures relative variance
            cv = series.std() / (abs(series.mean()) + 1e-9)
            variance_scores.append(min(1.0, cv / 10))  # normalize
        else:
            variance_scores.append(0.0)
            flags.append(f"⚠️ Column '{col}' has zero variance (constant)")
            recommendations.append(f"Drop constant column '{col}' — it provides no information")
    
    variance = np.mean(variance_scores) if variance_scores else 0.5
    variance = max(0.1, min(1.0, variance))  # floor at 0.1 for non-numeric datasets
    scores["variance"] = variance

    # ── 4. Balance (class imbalance for detecting target bias) ───────────────
    balance_scores = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 1:
            # 1.0 = perfectly balanced, 0.0 = all in one class
            balance_score = 1 - (vc.max() - vc.min())
            balance_scores.append(balance_score)
            if vc.max() > 0.85:
                flags.append(f"⚠️ Column '{col}' is highly imbalanced ({vc.max():.0%} in dominant class)")
                recommendations.append(f"Consider SMOTE or class weighting for '{col}' if used as target")

    balance = np.mean(balance_scores) if balance_scores else 0.85
    balance = max(0.0, min(1.0, balance))
    scores["balance"] = balance

    # ── 5. Uniqueness (duplicate rows) ───────────────────────────────────────
    dup_count = df.duplicated().sum()
    uniqueness = 1 - (dup_count / len(df)) if len(df) > 0 else 1.0
    scores["uniqueness"] = uniqueness
    
    if dup_count > 0:
        flags.append(f"⚠️ {dup_count} duplicate rows detected")
        recommendations.append(f"Remove {dup_count} duplicate rows before analysis")

    # ── Weighted overall score ────────────────────────────────────────────────
    weights = {
        "completeness": 0.30,
        "consistency": 0.25,
        "variance": 0.20,
        "balance": 0.15,
        "uniqueness": 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights)

    # ── Label & color ─────────────────────────────────────────────────────────
    if overall >= 0.82:
        label = "🟢 Reliable for ML"
        color = "green"
        status = "reliable"
    elif overall >= 0.62:
        label = "🟡 Needs Cleaning"
        color = "orange"
        status = "warning"
    else:
        label = "🔴 Unsafe for Modeling"
        color = "red"
        status = "danger"

    # ── PII / leakage hints ───────────────────────────────────────────────────
    pii_hints = _detect_pii_hints(df)
    if pii_hints:
        flags.extend(pii_hints)
        recommendations.append("Review flagged columns for PII before sharing data")

    # ── Sample size check ─────────────────────────────────────────────────────
    if len(df) < 100:
        flags.append(f"⚠️ Very small dataset ({len(df)} rows) — ML results may be unreliable")
        recommendations.append("Collect more data or use cross-validation carefully")

    return {
        "overall": overall,
        "dimensions": scores,
        "flags": flags,
        "recommendations": recommendations,
        "label": label,
        "color": color,
        "status": status,
        "pii_columns": pii_hints,
    }


def _detect_pii_hints(df: pd.DataFrame) -> List[str]:
    """Detect potentially PII-containing columns by name patterns."""
    pii_patterns = [
        "name", "email", "phone", "mobile", "ssn", "passport", "address",
        "zip", "postal", "dob", "birth", "age", "gender", "sex", "race",
        "ip", "credit", "card", "bank", "iban", "pan", "nid", "id"
    ]
    flags = []
    for col in df.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        for pattern in pii_patterns:
            if pattern in col_lower:
                flags.append(f"🔐 Possible PII detected: '{col}' (matches '{pattern}')")
                break
    return flags


def get_trust_score_summary(trust: dict) -> str:
    """Return a human-readable summary string."""
    score = trust['overall']
    dims = trust['dimensions']
    return (
        f"Overall: {score:.0%} | "
        f"Completeness: {dims['completeness']:.0%} | "
        f"Consistency: {dims['consistency']:.0%} | "
        f"Variance: {dims['variance']:.0%} | "
        f"Balance: {dims['balance']:.0%} | "
        f"Uniqueness: {dims['uniqueness']:.0%}"
    )
 
