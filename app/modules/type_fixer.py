"""
DataPilot AI — Smart Column Type Fixer
Auto-detect and fix wrong column types: string numbers, date strings,
boolean-as-int, currency-as-string, etc.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


def detect_type_issues(df: pd.DataFrame) -> List[Dict]:
    """
    Scan all columns and detect type mismatches.

    Returns list of dicts:
    [{"column": str, "current_type": str, "suggested_type": str,
      "confidence": float, "reason": str, "sample_values": list}]
    """
    issues = []

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        current_dtype = str(df[col].dtype)

        # ── Object columns that might be numeric ──────────────────────────────
        if df[col].dtype == "object":
            # Check for currency strings ($1,234.56)
            currency_check = _detect_currency(series)
            if currency_check["is_currency"]:
                issues.append({
                    "column": col,
                    "current_type": current_dtype,
                    "suggested_type": "float64 (currency)",
                    "confidence": currency_check["confidence"],
                    "reason": f"Detected currency format ({currency_check['symbol']})",
                    "sample_values": series.head(3).tolist(),
                    "fix_type": "currency",
                })
                continue

            # Check for numeric strings with thousand separators
            numeric_check = _detect_numeric_string(series)
            if numeric_check["is_numeric"]:
                issues.append({
                    "column": col,
                    "current_type": current_dtype,
                    "suggested_type": numeric_check["suggested_type"],
                    "confidence": numeric_check["confidence"],
                    "reason": "Values are numeric but stored as text",
                    "sample_values": series.head(3).tolist(),
                    "fix_type": "numeric",
                })
                continue

            # Check for boolean strings
            bool_check = _detect_boolean_string(series)
            if bool_check["is_boolean"]:
                issues.append({
                    "column": col,
                    "current_type": current_dtype,
                    "suggested_type": "bool",
                    "confidence": bool_check["confidence"],
                    "reason": f"Boolean-like values detected ({bool_check['variants']})",
                    "sample_values": series.head(3).tolist(),
                    "fix_type": "boolean",
                })
                continue

            # Check for datetime strings
            datetime_check = _detect_datetime_string(series)
            if datetime_check["is_datetime"]:
                issues.append({
                    "column": col,
                    "current_type": current_dtype,
                    "suggested_type": "datetime64",
                    "confidence": datetime_check["confidence"],
                    "reason": "Date/time values stored as text",
                    "sample_values": series.head(3).tolist(),
                    "fix_type": "datetime",
                })
                continue

            # Check for percentage strings
            pct_check = _detect_percentage(series)
            if pct_check["is_percentage"]:
                issues.append({
                    "column": col,
                    "current_type": current_dtype,
                    "suggested_type": "float64 (percentage)",
                    "confidence": pct_check["confidence"],
                    "reason": "Percentage values stored as text (e.g., '45%')",
                    "sample_values": series.head(3).tolist(),
                    "fix_type": "percentage",
                })
                continue

        # ── Numeric columns that might be IDs or categories ───────────────────
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if integer column is actually an ID
            if pd.api.types.is_integer_dtype(df[col]):
                if _looks_like_id(series, col):
                    issues.append({
                        "column": col,
                        "current_type": current_dtype,
                        "suggested_type": "object (ID/category)",
                        "confidence": 0.75,
                        "reason": "Appears to be an ID column (sequential, high cardinality)",
                        "sample_values": series.head(3).tolist(),
                        "fix_type": "to_string",
                    })
                    continue

                # Check if it's actually boolean (0/1 only)
                unique_vals = set(series.unique())
                if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
                    issues.append({
                        "column": col,
                        "current_type": current_dtype,
                        "suggested_type": "bool",
                        "confidence": 0.85,
                        "reason": "Only contains 0 and 1 — likely boolean",
                        "sample_values": series.head(3).tolist(),
                        "fix_type": "int_to_bool",
                    })

    return issues


def apply_type_fixes(df: pd.DataFrame, fixes: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply selected type fixes.

    fixes: {column_name: fix_type}
    Returns (fixed_df, log_messages)
    """
    df = df.copy()
    logs = []

    for col, fix_type in fixes.items():
        if col not in df.columns:
            continue

        try:
            if fix_type == "currency":
                df[col] = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logs.append(f"✅ '{col}': currency string → float64")

            elif fix_type == "numeric":
                df[col] = df[col].astype(str).str.replace(",", "")
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logs.append(f"✅ '{col}': text → numeric")

            elif fix_type == "boolean":
                bool_map = {
                    "true": True, "false": False,
                    "yes": True, "no": False,
                    "1": True, "0": False,
                    "t": True, "f": False,
                    "y": True, "n": False,
                }
                df[col] = df[col].astype(str).str.lower().str.strip().map(bool_map)
                logs.append(f"✅ '{col}': text → boolean")

            elif fix_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                logs.append(f"✅ '{col}': text → datetime")

            elif fix_type == "percentage":
                df[col] = df[col].astype(str).str.replace("%", "").str.strip()
                df[col] = pd.to_numeric(df[col], errors="coerce") / 100
                logs.append(f"✅ '{col}': percentage string → float (decimal)")

            elif fix_type == "to_string":
                df[col] = df[col].astype(str)
                logs.append(f"✅ '{col}': numeric → string (ID)")

            elif fix_type == "int_to_bool":
                df[col] = df[col].astype(bool)
                logs.append(f"✅ '{col}': int (0/1) → boolean")

        except Exception as e:
            logs.append(f"❌ '{col}': fix failed — {e}")

    return df, logs


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_currency(series: pd.Series) -> Dict:
    sample = series.head(100).astype(str)
    currency_pattern = r"^[\$€£¥₹₽]?\s?-?[\d,]+\.?\d*$|^-?[\d,]+\.?\d*\s?[\$€£¥₹₽]$"
    matches = sample.str.match(currency_pattern, na=False).sum()
    confidence = matches / len(sample) if len(sample) > 0 else 0

    symbol = ""
    for s in ["$", "€", "£", "¥", "₹", "₽"]:
        if sample.str.contains(re.escape(s), na=False).any():
            symbol = s
            break

    return {"is_currency": confidence > 0.7, "confidence": round(confidence, 2), "symbol": symbol}


def _detect_numeric_string(series: pd.Series) -> Dict:
    sample = series.head(200).astype(str)
    coerced = pd.to_numeric(sample.str.replace(",", ""), errors="coerce")
    valid = coerced.notna().sum()
    confidence = valid / len(sample) if len(sample) > 0 else 0

    # Check if integers or floats
    if confidence > 0.8:
        has_decimal = sample.str.contains(r"\.", na=False).any()
        suggested = "float64" if has_decimal else "int64"
    else:
        suggested = "float64"

    return {"is_numeric": confidence > 0.8, "confidence": round(confidence, 2), "suggested_type": suggested}


def _detect_boolean_string(series: pd.Series) -> Dict:
    sample = series.head(200).astype(str).str.lower().str.strip()
    bool_vals = {"true", "false", "yes", "no", "t", "f", "y", "n", "1", "0"}
    matches = sample.isin(bool_vals).sum()
    confidence = matches / len(sample) if len(sample) > 0 else 0

    unique_lower = set(sample.unique())
    variants = ", ".join(sorted(unique_lower & bool_vals)[:4])

    return {"is_boolean": confidence > 0.9 and series.nunique() <= 3, "confidence": round(confidence, 2), "variants": variants}


def _detect_datetime_string(series: pd.Series) -> Dict:
    sample = series.head(100)
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        valid = parsed.notna().sum()
        confidence = valid / len(sample) if len(sample) > 0 else 0
        return {"is_datetime": confidence > 0.7, "confidence": round(confidence, 2)}
    except Exception:
        return {"is_datetime": False, "confidence": 0}


def _detect_percentage(series: pd.Series) -> Dict:
    sample = series.head(100).astype(str)
    pct_pattern = r"^\s*-?\d+\.?\d*\s*%\s*$"
    matches = sample.str.match(pct_pattern, na=False).sum()
    confidence = matches / len(sample) if len(sample) > 0 else 0
    return {"is_percentage": confidence > 0.7, "confidence": round(confidence, 2)}


def _looks_like_id(series: pd.Series, col_name: str) -> bool:
    """Heuristic: column looks like an ID if it's sequential, high cardinality, and name suggests it."""
    name_lower = col_name.lower().replace("_", "").replace(" ", "")
    id_patterns = ["id", "index", "key", "code", "num", "no", "serial"]

    name_matches = any(p in name_lower for p in id_patterns)
    high_unique = series.nunique() / len(series) > 0.9 if len(series) > 0 else False

    return name_matches and high_unique
