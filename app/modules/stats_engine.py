"""
DataPilot AI — Statistical Intelligence Engine
Auto statistical test recommender, distribution analyzer, hypothesis builder.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TEST RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

def recommend_statistical_test(
    df: pd.DataFrame, col1: str, col2: str
) -> Dict:
    """
    Auto-recommend the appropriate statistical test for two columns.
    Uses a decision tree: variable types → normality → group count.
    """
    is_num1 = pd.api.types.is_numeric_dtype(df[col1])
    is_num2 = pd.api.types.is_numeric_dtype(df[col2])

    result = {
        "test": None,
        "variables": [col1, col2],
        "assumptions": {},
        "result": {},
        "interpretation": "",
        "recommendation": "",
    }

    # ── Both numeric → correlation ─────────────────────────────────────────
    if is_num1 and is_num2:
        s1 = df[col1].dropna()
        s2 = df[col2].dropna()
        n = min(len(s1), len(s2))
        s1, s2 = s1.iloc[:n], s2.iloc[:n]

        normal1 = _normality_check(s1)
        normal2 = _normality_check(s2)

        if normal1["passed"] and normal2["passed"]:
            r, p = stats.pearsonr(s1, s2)
            test_name = "Pearson Correlation"
        else:
            r, p = stats.spearmanr(s1, s2)
            test_name = "Spearman Correlation"

        strength = _correlation_strength(r)
        sig = p < 0.05

        result.update({
            "test": test_name,
            "assumptions": {"normality_col1": normal1, "normality_col2": normal2},
            "result": {
                "statistic": round(r, 4),
                "p_value": round(p, 6),
                "significant": sig,
                "strength": strength,
            },
            "interpretation": (
                f"{'Significant' if sig else 'No significant'} {strength} {'positive' if r > 0 else 'negative'} "
                f"relationship between '{col1}' and '{col2}' (r={r:.3f}, p={'<0.001' if p < 0.001 else f'{p:.3f}'})."
            ),
            "recommendation": (
                "Consider linear regression if '{col1}' is a predictor variable."
                if sig and abs(r) > 0.3
                else "Relationship may not be practically significant despite statistical significance."
            ),
        })

    # ── Numeric vs Categorical ─────────────────────────────────────────────
    elif (is_num1 and not is_num2) or (not is_num1 and is_num2):
        num_col = col1 if is_num1 else col2
        cat_col = col2 if is_num1 else col1
        groups = [group.dropna() for _, group in df.groupby(cat_col)[num_col]]
        n_groups = len(groups)

        if n_groups < 2:
            result["interpretation"] = f"Need at least 2 groups in '{cat_col}' to run this test."
            return result

        normality_ok = all(_normality_check(g)["passed"] for g in groups if len(g) >= 3)

        if n_groups == 2:
            g1, g2 = groups[0], groups[1]
            if normality_ok:
                stat, p = stats.ttest_ind(g1, g2)
                test_name = "Independent t-test"
            else:
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_name = "Mann-Whitney U Test"
            
            effect = _cohens_d(g1, g2)
            effect_label = "large" if abs(effect) > 0.8 else "medium" if abs(effect) > 0.5 else "small"
        else:
            if normality_ok:
                stat, p = stats.f_oneway(*groups)
                test_name = "One-Way ANOVA"
            else:
                stat, p = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis H Test"
            effect = None
            effect_label = "N/A"

        sig = p < 0.05
        result.update({
            "test": test_name,
            "assumptions": {"normality_ok": normality_ok, "n_groups": n_groups},
            "result": {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "significant": sig,
                "effect_size": f"{effect_label} (d={abs(effect):.2f})" if effect is not None else effect_label,
            },
            "interpretation": (
                f"{'Significant' if sig else 'No significant'} difference in '{num_col}' across '{cat_col}' groups "
                f"({test_name}: stat={stat:.3f}, p={'<0.001' if p < 0.001 else f'{p:.3f}'})."
            ),
            "recommendation": (
                "Investigate which groups differ using post-hoc tests (e.g., Tukey HSD)."
                if sig and n_groups > 2
                else "Consider using this grouping as a feature in your ML model."
                if sig
                else "Grouping may not add predictive value."
            ),
        })

    # ── Both categorical → Chi-square ─────────────────────────────────────
    else:
        ct = pd.crosstab(df[col1].dropna(), df[col2].dropna())
        if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
            result["interpretation"] = "Insufficient data for chi-square test."
            return result

        chi2, p, dof, expected = stats.chi2_contingency(ct)
        sig = p < 0.05
        cramer_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))

        result.update({
            "test": "Chi-Square Test of Independence",
            "assumptions": {"min_expected_freq": float(expected.min())},
            "result": {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p), 6),
                "dof": int(dof),
                "significant": sig,
                "cramers_v": round(float(cramer_v), 3),
            },
            "interpretation": (
                f"{'Significant' if sig else 'No significant'} association between '{col1}' and '{col2}' "
                f"(χ²={chi2:.2f}, df={dof}, p={'<0.001' if p < 0.001 else f'{p:.3f}'}, V={cramer_v:.3f})."
            ),
            "recommendation": (
                "Strong categorical association — consider interaction features for ML."
                if sig and cramer_v > 0.3
                else "Weak association." if not sig else "Moderate association detected."
            ),
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

def analyze_distribution(df: pd.DataFrame, column: str) -> Dict:
    """Full distribution analysis for a numeric column."""
    series = df[column].dropna()
    if len(series) < 3:
        return {"error": "Too few data points"}

    n = len(series)
    normality = _normality_check(series)
    skewness = float(series.skew())
    kurtosis = float(series.kurtosis())

    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_label = "approximately symmetric"
        transform_suggestion = "No transformation needed"
    elif skewness > 0.5:
        skew_label = "right-skewed (positive)"
        transform_suggestion = "Apply log or square-root transformation"
    else:
        skew_label = "left-skewed (negative)"
        transform_suggestion = "Apply square or exponential transformation"

    # Outlier counts
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    z_outliers = int((np.abs(stats.zscore(series)) > 3).sum())

    return {
        "column": column,
        "n": n,
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4),
        "skewness": round(skewness, 4),
        "skewness_label": skew_label,
        "kurtosis": round(kurtosis, 4),
        "normality": normality,
        "iqr_outliers": iqr_outliers,
        "zscore_outliers": z_outliers,
        "transform_suggestion": transform_suggestion,
        "percentiles": {
            "p5": round(float(series.quantile(0.05)), 4),
            "p25": round(float(q1), 4),
            "p75": round(float(q3), 4),
            "p95": round(float(series.quantile(0.95)), 4),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_hypothesis(df: pd.DataFrame, col1: str, relationship: str, col2: str) -> Dict:
    """
    No-code hypothesis builder. Maps natural language relationship to statistical test.
    relationship options: 'affects', 'differs by', 'associated with', 'predicts'
    """
    rel_lower = relationship.lower()

    # Map relationship to test type
    if any(w in rel_lower for w in ["affect", "impact", "cause", "predict"]):
        test_result = recommend_statistical_test(df, col1, col2)
        h0 = f"'{col1}' has no effect on '{col2}' (β = 0)"
        h1 = f"'{col1}' significantly influences '{col2}' (β ≠ 0)"
    elif any(w in rel_lower for w in ["differ", "vary", "compare"]):
        test_result = recommend_statistical_test(df, col2, col1)
        h0 = f"'{col1}' does not differ across groups of '{col2}'"
        h1 = f"'{col1}' significantly differs across groups of '{col2}'"
    else:
        test_result = recommend_statistical_test(df, col1, col2)
        h0 = f"There is no association between '{col1}' and '{col2}'"
        h1 = f"'{col1}' and '{col2}' are significantly associated"

    sig = test_result.get("result", {}).get("significant", False)
    conclusion = (
        f"✅ **Reject H₀:** {h1} — statistically supported."
        if sig
        else f"❌ **Fail to Reject H₀:** Insufficient evidence that {h1.lower()}."
    )

    return {
        "null_hypothesis": h0,
        "alternative_hypothesis": h1,
        "test_used": test_result.get("test", "N/A"),
        "result": test_result.get("result", {}),
        "assumptions": test_result.get("assumptions", {}),
        "interpretation": test_result.get("interpretation", ""),
        "conclusion": conclusion,
        "recommendation": test_result.get("recommendation", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normality_check(series: pd.Series) -> Dict:
    """Run Shapiro-Wilk (n<5000) or K-S test. Returns dict with passed bool."""
    n = len(series)
    if n < 3:
        return {"test": "N/A", "p_value": None, "passed": True}
    try:
        if n <= 5000:
            stat, p = stats.shapiro(series.sample(min(n, 5000), random_state=42))
            test_name = "Shapiro-Wilk"
        else:
            stat, p = stats.kstest(series, "norm", args=(series.mean(), series.std()))
            test_name = "Kolmogorov-Smirnov"
        return {"test": test_name, "statistic": round(float(stat), 4), "p_value": round(float(p), 4), "passed": float(p) > 0.05}
    except Exception:
        return {"test": "N/A", "p_value": None, "passed": True}


def _cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std() ** 2 + (n2 - 1) * g2.std() ** 2) / (n1 + n2 - 2))
    return float((g1.mean() - g2.mean()) / (pooled_std + 1e-9))


def _correlation_strength(r: float) -> str:
    r = abs(r)
    if r >= 0.7:
        return "strong"
    elif r >= 0.4:
        return "moderate"
    elif r >= 0.2:
        return "weak"
    return "negligible"
 
