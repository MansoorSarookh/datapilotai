"""
DataPilot AI — Full Regression Suite
OLS, Logistic, Ridge, Lasso, Poisson regression with diagnostics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


def run_regression(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    regression_type: str = "OLS",
) -> Dict:
    """
    Run regression analysis with full diagnostics.

    regression_type: 'OLS', 'Logistic', 'Ridge', 'Lasso', 'Poisson'
    """
    import statsmodels.api as sm
    from sklearn.preprocessing import LabelEncoder

    df_clean = df[predictor_cols + [target_col]].dropna().copy()
    if len(df_clean) < 10:
        return {"error": "Insufficient data points (need at least 10)"}

    # Encode categorical predictors
    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    X = df_clean[predictor_cols].astype(float)
    y = df_clean[target_col]

    if regression_type == "Logistic":
        if pd.api.types.is_object_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        y = y.astype(float)

    X_const = sm.add_constant(X)

    try:
        if regression_type == "OLS":
            model = sm.OLS(y.astype(float), X_const).fit()
        elif regression_type == "Logistic":
            model = sm.Logit(y, X_const).fit(disp=0)
        elif regression_type == "Ridge":
            model = sm.OLS(y.astype(float), X_const).fit_regularized(alpha=1.0, L1_wt=0)
        elif regression_type == "Lasso":
            model = sm.OLS(y.astype(float), X_const).fit_regularized(alpha=1.0, L1_wt=1)
        elif regression_type == "Poisson":
            model = sm.GLM(y.astype(float), X_const, family=sm.families.Poisson()).fit()
        else:
            return {"error": f"Unknown regression type: {regression_type}"}
    except Exception as e:
        return {"error": str(e)}

    result = {
        "regression_type": regression_type,
        "n_observations": len(df_clean),
        "n_predictors": len(predictor_cols),
        "predictors": predictor_cols,
        "target": target_col,
    }

    # Extract results based on model type
    if regression_type in ("Ridge", "Lasso"):
        # Regularized models don't have the full summary
        result["coefficients"] = {
            name: round(float(coef), 6)
            for name, coef in zip(["const"] + predictor_cols, model.params)
        }
        # Compute R² manually
        y_pred = X_const @ model.params
        ss_res = ((y.astype(float) - y_pred) ** 2).sum()
        ss_tot = ((y.astype(float) - y.astype(float).mean()) ** 2).sum()
        result["r_squared"] = round(float(1 - ss_res / ss_tot), 4) if ss_tot > 0 else 0
        result["residuals"] = (y.astype(float) - y_pred).values
    else:
        # Full statsmodels results
        result["coefficients"] = {}
        coef_table = []
        param_names = ["const"] + predictor_cols

        for i, name in enumerate(param_names):
            try:
                coef = float(model.params.iloc[i]) if hasattr(model.params, 'iloc') else float(model.params[i])
                try:
                    pval = float(model.pvalues.iloc[i]) if hasattr(model.pvalues, 'iloc') else float(model.pvalues[i])
                except Exception:
                    pval = None
                try:
                    se = float(model.bse.iloc[i]) if hasattr(model.bse, 'iloc') else float(model.bse[i])
                except Exception:
                    se = None

                result["coefficients"][name] = round(coef, 6)
                coef_table.append({
                    "Variable": name,
                    "Coefficient": round(coef, 4),
                    "Std Error": round(se, 4) if se else "N/A",
                    "P-Value": round(pval, 4) if pval else "N/A",
                    "Significant": "✅" if (pval and pval < 0.05) else "❌",
                })
            except (IndexError, KeyError):
                continue

        result["coefficient_table"] = coef_table

        if regression_type == "OLS":
            result["r_squared"] = round(float(model.rsquared), 4)
            result["r_squared_adj"] = round(float(model.rsquared_adj), 4)
            result["f_statistic"] = round(float(model.fvalue), 4)
            result["f_pvalue"] = round(float(model.f_pvalue), 6)
            result["aic"] = round(float(model.aic), 2)
            result["bic"] = round(float(model.bic), 2)
            result["durbin_watson"] = round(float(sm.stats.stattools.durbin_watson(model.resid)), 4)
            result["residuals"] = model.resid.values
            result["fitted_values"] = model.fittedvalues.values

        elif regression_type == "Logistic":
            result["pseudo_r_squared"] = round(float(model.prsquared), 4)
            result["log_likelihood"] = round(float(model.llf), 2)
            result["aic"] = round(float(model.aic), 2)
            result["bic"] = round(float(model.bic), 2)
            # Accuracy
            y_pred_class = (model.predict(X_const) > 0.5).astype(int)
            from sklearn.metrics import accuracy_score
            result["accuracy"] = round(float(accuracy_score(y, y_pred_class)), 4)

        elif regression_type == "Poisson":
            result["pseudo_r_squared"] = round(float(1 - model.deviance / model.null_deviance), 4)
            result["aic"] = round(float(model.aic), 2)

    # VIF computation
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = []
        for i, col in enumerate(predictor_cols):
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({"Feature": col, "VIF": round(float(vif), 2)})
        result["vif"] = vif_data
    except Exception:
        result["vif"] = []

    return result


def compute_residual_diagnostics(result: Dict) -> Dict:
    """Compute residual diagnostics for regression results."""
    residuals = result.get("residuals")
    fitted = result.get("fitted_values")

    if residuals is None:
        return {}

    from scipy import stats

    residuals = np.array(residuals)

    # Normality of residuals
    if len(residuals) >= 8:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])
            normality = {
                "test": "Shapiro-Wilk",
                "statistic": round(float(shapiro_stat), 4),
                "p_value": round(float(shapiro_p), 4),
                "normal": float(shapiro_p) > 0.05,
            }
        except Exception:
            normality = {"test": "N/A", "normal": True}
    else:
        normality = {"test": "N/A", "normal": True}

    # Homoscedasticity (Breusch-Pagan)
    homoscedasticity = {"test": "N/A"}
    if fitted is not None:
        try:
            import statsmodels.stats.diagnostic as diag
            bp_stat, bp_p, _, _ = diag.het_breuschpagan(residuals, np.column_stack([np.ones(len(fitted)), fitted]))
            homoscedasticity = {
                "test": "Breusch-Pagan",
                "statistic": round(float(bp_stat), 4),
                "p_value": round(float(bp_p), 4),
                "homoscedastic": float(bp_p) > 0.05,
            }
        except Exception:
            pass

    return {
        "normality": normality,
        "homoscedasticity": homoscedasticity,
        "mean_residual": round(float(residuals.mean()), 6),
        "std_residual": round(float(residuals.std()), 4),
    }
