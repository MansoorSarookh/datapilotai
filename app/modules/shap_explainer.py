"""
DataPilot AI — SHAP Explainability Engine
Generates SHAP explanations for trained ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any


def compute_shap_values(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_samples: int = 500,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP values for a trained model.

    Returns dict with:
        - shap_values: array of SHAP values
        - expected_value: base value
        - feature_names: list of feature names
        - feature_importance: dict of mean absolute SHAP values
    """
    try:
        import shap

        # Sample data if too large
        if len(X_train) > max_samples:
            X_bg = X_train.sample(n=min(100, len(X_train)), random_state=42)
        else:
            X_bg = X_train

        if len(X_test) > max_samples:
            X_explain = X_test.sample(n=max_samples, random_state=42)
        else:
            X_explain = X_test

        # Choose appropriate explainer
        model_type = type(model).__name__.lower()

        if any(t in model_type for t in ["forest", "xgb", "lgbm", "gradient", "catboost"]):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(
                model.predict,
                X_bg,
                link="identity",
            )

        shap_values = explainer.shap_values(X_explain)

        # Handle multi-class (take first class or average)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_vals = shap_values[1]  # Binary: use positive class
            else:
                shap_vals = np.mean(np.abs(np.array(shap_values)), axis=0)
        else:
            shap_vals = shap_values

        # Compute feature importance from SHAP
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        feature_importance = dict(zip(X_explain.columns, mean_abs_shap))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        # Get expected value
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            if len(expected) == 2:
                expected = expected[1]
            else:
                expected = expected[0]

        return {
            "shap_values": shap_vals,
            "expected_value": float(expected),
            "feature_names": X_explain.columns.tolist(),
            "feature_importance": feature_importance,
            "X_explain": X_explain,
            "explainer": explainer,
        }

    except ImportError:
        return None
    except Exception as e:
        return {"error": str(e)}


def get_single_prediction_explanation(
    shap_result: Dict,
    row_index: int = 0,
) -> Optional[Dict]:
    """Get SHAP explanation for a single prediction."""
    if not shap_result or "error" in shap_result:
        return None

    try:
        shap_vals = shap_result["shap_values"]
        features = shap_result["feature_names"]
        expected = shap_result["expected_value"]
        X = shap_result["X_explain"]

        if row_index >= len(shap_vals):
            row_index = 0

        row_shap = shap_vals[row_index]
        row_values = X.iloc[row_index]

        explanation = []
        for feat, sv, fv in zip(features, row_shap, row_values):
            explanation.append({
                "feature": feat,
                "shap_value": float(sv),
                "feature_value": float(fv) if isinstance(fv, (int, float, np.number)) else str(fv),
                "direction": "↑ increases" if sv > 0 else "↓ decreases",
            })

        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": expected,
            "prediction": expected + sum(row_shap),
            "features": explanation[:15],
        }
    except Exception:
        return None


def create_shap_summary_data(shap_result: Dict) -> Optional[pd.DataFrame]:
    """Create a DataFrame suitable for plotting SHAP summary."""
    if not shap_result or "error" in shap_result:
        return None

    try:
        shap_vals = shap_result["shap_values"]
        features = shap_result["feature_names"]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)

        df = pd.DataFrame({
            "Feature": features,
            "Mean |SHAP|": mean_abs,
            "Mean SHAP": mean_signed,
            "Direction": ["Positive ↑" if s > 0 else "Negative ↓" for s in mean_signed],
        })
        df = df.sort_values("Mean |SHAP|", ascending=False)
        return df
    except Exception:
        return None
