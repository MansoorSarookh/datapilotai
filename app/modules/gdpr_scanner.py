"""
DataPilot AI — GDPR / PII Scanner
Deep PII detection using regex patterns and column analysis.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List


# PII detection patterns
PII_PATTERNS = {
    "email": {
        "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "risk": "High",
        "category": "Contact",
        "remediation": "Hash or anonymize email addresses",
    },
    "phone": {
        "regex": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "risk": "High",
        "category": "Contact",
        "remediation": "Mask or remove phone numbers",
    },
    "ssn": {
        "regex": r"\b\d{3}-\d{2}-\d{4}\b",
        "risk": "Critical",
        "category": "Identity",
        "remediation": "Remove immediately — highly sensitive",
    },
    "credit_card": {
        "regex": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "risk": "Critical",
        "category": "Financial",
        "remediation": "Remove or tokenize credit card numbers",
    },
    "ip_address": {
        "regex": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "risk": "Medium",
        "category": "Technical",
        "remediation": "Anonymize IP addresses",
    },
    "date_of_birth": {
        "regex": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
        "risk": "Medium",
        "category": "Identity",
        "remediation": "Convert to age or age range",
    },
}

# Column name patterns suggesting PII
PII_COLUMN_PATTERNS = {
    "name": {"patterns": ["name", "first_name", "last_name", "full_name", "fname", "lname", "surname"],
             "risk": "High", "category": "Identity"},
    "email": {"patterns": ["email", "e_mail", "mail", "email_address"],
              "risk": "High", "category": "Contact"},
    "phone": {"patterns": ["phone", "telephone", "mobile", "cell", "tel", "fax"],
              "risk": "High", "category": "Contact"},
    "address": {"patterns": ["address", "street", "zip", "postal", "city", "state"],
                "risk": "High", "category": "Location"},
    "id_number": {"patterns": ["ssn", "social_security", "passport", "license", "national_id", "tax_id", "ein"],
                  "risk": "Critical", "category": "Identity"},
    "financial": {"patterns": ["credit_card", "card_number", "account", "bank", "iban", "routing"],
                  "risk": "Critical", "category": "Financial"},
    "health": {"patterns": ["diagnosis", "medication", "medical", "health", "patient", "blood_type"],
               "risk": "Critical", "category": "Health"},
    "demographic": {"patterns": ["age", "gender", "sex", "race", "ethnicity", "religion", "dob", "birth"],
                    "risk": "Medium", "category": "Demographic"},
}


def scan_for_pii(df: pd.DataFrame) -> Dict:
    """
    Deep scan dataset for PII/GDPR-sensitive data.

    Returns column-level risk assessment with detection details.
    """
    results = {}
    overall_risk = "Low"
    total_pii_columns = 0

    for col in df.columns:
        col_result = {
            "column": col,
            "risk_level": "Low",
            "detections": [],
            "remediation": [],
        }

        col_lower = col.lower().replace(" ", "_").replace("-", "_")

        # ── Check column name ─────────────────────────────────────────────────
        for pii_type, config in PII_COLUMN_PATTERNS.items():
            if any(p in col_lower for p in config["patterns"]):
                col_result["detections"].append({
                    "type": f"Column name suggests: {pii_type}",
                    "method": "name_pattern",
                    "risk": config["risk"],
                    "category": config["category"],
                })
                col_result["risk_level"] = _max_risk(col_result["risk_level"], config["risk"])

        # ── Check column values (sample) ──────────────────────────────────────
        if df[col].dtype == "object":
            sample = df[col].dropna().head(200).astype(str)

            for pii_type, config in PII_PATTERNS.items():
                try:
                    matches = sample.str.contains(config["regex"], regex=True, na=False)
                    match_count = int(matches.sum())
                    match_pct = match_count / len(sample) * 100 if len(sample) > 0 else 0

                    if match_pct > 10:
                        col_result["detections"].append({
                            "type": pii_type.replace("_", " ").title(),
                            "method": "value_pattern",
                            "match_count": match_count,
                            "match_percentage": round(match_pct, 1),
                            "risk": config["risk"],
                            "category": config["category"],
                            "sample_masked": _mask_value(sample[matches].iloc[0]) if match_count > 0 else "",
                        })
                        col_result["risk_level"] = _max_risk(col_result["risk_level"], config["risk"])
                        col_result["remediation"].append(config["remediation"])
                except Exception:
                    continue

        if col_result["risk_level"] != "Low":
            total_pii_columns += 1
            overall_risk = _max_risk(overall_risk, col_result["risk_level"])

        results[col] = col_result

    # Build risk heatmap data
    heatmap_data = _build_heatmap_data(results)

    return {
        "column_risks": results,
        "overall_risk": overall_risk,
        "total_pii_columns": total_pii_columns,
        "total_columns": len(df.columns),
        "risk_summary": {
            "Critical": sum(1 for r in results.values() if r["risk_level"] == "Critical"),
            "High": sum(1 for r in results.values() if r["risk_level"] == "High"),
            "Medium": sum(1 for r in results.values() if r["risk_level"] == "Medium"),
            "Low": sum(1 for r in results.values() if r["risk_level"] == "Low"),
        },
        "heatmap_data": heatmap_data,
    }


def _max_risk(current: str, new: str) -> str:
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    return new if order.get(new, 0) > order.get(current, 0) else current


def _mask_value(value: str) -> str:
    """Mask a PII value for display."""
    if len(value) <= 4:
        return "***"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _build_heatmap_data(results: Dict) -> Dict:
    """Build heatmap data for visualization."""
    categories = ["Identity", "Contact", "Financial", "Location", "Health", "Demographic", "Technical"]
    columns = []
    matrix = []

    for col, result in results.items():
        if result["risk_level"] == "Low":
            continue
        columns.append(col)
        row = [0] * len(categories)
        for detection in result["detections"]:
            cat = detection.get("category", "")
            if cat in categories:
                risk_score = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}.get(detection.get("risk", "Low"), 0)
                idx = categories.index(cat)
                row[idx] = max(row[idx], risk_score)
        matrix.append(row)

    return {
        "categories": categories,
        "columns": columns,
        "matrix": matrix,
    }
