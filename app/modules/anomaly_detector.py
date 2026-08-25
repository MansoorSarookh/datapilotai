"""
DataPilot AI — Anomaly Detection Engine
Advanced anomaly detection using Isolation Forest, LOF, and Z-Score ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    columns: Optional[List[str]] = None,
) -> Dict:
    """
    Detect anomalies in the dataset.

    method: 'isolation_forest', 'lof', 'zscore', 'ensemble'
    contamination: expected proportion of outliers (0.01-0.20)
    columns: specific numeric columns to analyze (None = all numeric)

    Returns dict with anomaly labels, scores, and summary.
    """
    numeric_df = df.select_dtypes(include=np.number)
    if columns:
        valid_cols = [c for c in columns if c in numeric_df.columns]
        if valid_cols:
            numeric_df = numeric_df[valid_cols]

    if numeric_df.empty or len(numeric_df) < 10:
        return {"error": "Insufficient numeric data for anomaly detection"}

    # Prepare data
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(numeric_df))

    if method == "isolation_forest":
        labels, scores = _isolation_forest(X, contamination)
    elif method == "lof":
        labels, scores = _local_outlier_factor(X, contamination)
    elif method == "zscore":
        labels, scores = _zscore_detection(X, contamination)
    elif method == "ensemble":
        labels, scores = _ensemble_detection(X, contamination)
    else:
        return {"error": f"Unknown method: {method}"}

    # Build results
    anomaly_mask = labels == -1
    n_anomalies = int(anomaly_mask.sum())
    anomaly_indices = np.where(anomaly_mask)[0].tolist()

    # Per-column outlier contribution
    column_contributions = {}
    for i, col in enumerate(numeric_df.columns):
        col_data = X[:, i]
        anomaly_vals = col_data[anomaly_mask]
        normal_vals = col_data[~anomaly_mask]
        if len(anomaly_vals) > 0 and len(normal_vals) > 0:
            deviation = abs(anomaly_vals.mean() - normal_vals.mean())
            column_contributions[col] = round(float(deviation), 3)

    # Sort columns by contribution
    column_contributions = dict(sorted(column_contributions.items(), key=lambda x: x[1], reverse=True))

    return {
        "method": method,
        "contamination": contamination,
        "n_total": len(df),
        "n_anomalies": n_anomalies,
        "anomaly_percentage": round(n_anomalies / len(df) * 100, 2),
        "anomaly_indices": anomaly_indices[:500],  # Cap at 500
        "anomaly_scores": scores.tolist() if len(scores) <= 5000 else scores[:5000].tolist(),
        "labels": labels.tolist(),
        "column_contributions": column_contributions,
        "columns_analyzed": numeric_df.columns.tolist(),
    }


def flag_anomalies(df: pd.DataFrame, anomaly_result: Dict) -> pd.DataFrame:
    """Add an anomaly flag column to the DataFrame."""
    df = df.copy()
    labels = anomaly_result.get("labels", [])
    if len(labels) == len(df):
        df["_anomaly_flag"] = ["Anomaly" if l == -1 else "Normal" for l in labels]
        scores = anomaly_result.get("anomaly_scores", [])
        if len(scores) == len(df):
            df["_anomaly_score"] = scores
    return df


def remove_anomalies(df: pd.DataFrame, anomaly_result: Dict) -> Tuple[pd.DataFrame, int]:
    """Remove detected anomalies from the DataFrame."""
    indices = anomaly_result.get("anomaly_indices", [])
    if not indices:
        return df, 0
    df_clean = df.drop(index=df.index[indices]).reset_index(drop=True)
    return df_clean, len(indices)


# ── Detection algorithms ──────────────────────────────────────────────────────

def _isolation_forest(X: np.ndarray, contamination: float):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = clf.fit_predict(X)
    scores = clf.decision_function(X)
    return labels, -scores  # Negate so higher = more anomalous

def _local_outlier_factor(X: np.ndarray, contamination: float):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(contamination=contamination, n_neighbors=min(20, len(X) - 1))
    labels = clf.fit_predict(X)
    scores = -clf.negative_outlier_factor_
    return labels, scores

def _zscore_detection(X: np.ndarray, contamination: float):
    # Multi-dimensional z-score: flag if any feature exceeds threshold
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, axis=0, nan_policy="omit"))
    max_z = np.nanmax(z_scores, axis=1)
    # Determine threshold from contamination
    threshold = np.percentile(max_z, (1 - contamination) * 100)
    labels = np.where(max_z > threshold, -1, 1)
    return labels, max_z

def _ensemble_detection(X: np.ndarray, contamination: float):
    """Combine IF and LOF votes — anomaly if both agree."""
    if_labels, if_scores = _isolation_forest(X, contamination)
    lof_labels, lof_scores = _local_outlier_factor(X, contamination)

    # Normalize scores to [0, 1]
    if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-9)
    lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-9)

    combined_scores = (if_norm + lof_norm) / 2
    threshold = np.percentile(combined_scores, (1 - contamination) * 100)
    labels = np.where(combined_scores > threshold, -1, 1)
    return labels, combined_scores
