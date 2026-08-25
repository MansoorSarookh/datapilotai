"""
DataPilot AI — AI Chart Recommender
Auto-suggests the best chart types for selected columns based on data characteristics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def recommend_charts(
    df: pd.DataFrame,
    selected_columns: List[str],
    max_recommendations: int = 3,
) -> List[Dict]:
    """
    Recommend the best chart types for selected columns.

    Returns a list of dicts: [{"chart": str, "reason": str, "score": float}]
    """
    if not selected_columns:
        return []

    n_cols = len(selected_columns)
    recommendations = []

    # Classify selected columns
    col_types = []
    for col in selected_columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            if nunique <= 10:
                col_types.append(("numeric_low_card", col))
            elif nunique <= 50:
                col_types.append(("numeric_med_card", col))
            else:
                col_types.append(("numeric_high_card", col))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types.append(("datetime", col))
        else:
            nunique = df[col].nunique()
            if nunique <= 10:
                col_types.append(("categorical_low", col))
            elif nunique <= 50:
                col_types.append(("categorical_med", col))
            else:
                col_types.append(("categorical_high", col))

    if not col_types:
        return [{"chart": "Bar Chart", "reason": "Default for unknown data types", "score": 0.5}]

    type_names = [t[0] for t in col_types]

    # ── Single column ─────────────────────────────────────────────────────────
    if n_cols == 1:
        ctype, cname = col_types[0]
        series = df[cname].dropna()

        if "numeric" in ctype:
            skewness = abs(series.skew()) if len(series) > 2 else 0

            if skewness > 1.5:
                recommendations.append({
                    "chart": "Box Plot",
                    "reason": f"'{cname}' is heavily skewed (skew={series.skew():.2f}) — box plot reveals outliers and spread",
                    "score": 0.95,
                })
                recommendations.append({
                    "chart": "Histogram",
                    "reason": f"Shows the full distribution shape, revealing the skewness pattern",
                    "score": 0.90,
                })
                recommendations.append({
                    "chart": "Violin Plot",
                    "reason": f"Combines box plot and density — ideal for skewed data visualization",
                    "score": 0.85,
                })
            else:
                recommendations.append({
                    "chart": "Histogram",
                    "reason": f"Best for seeing the distribution shape of '{cname}' ({series.nunique()} unique values)",
                    "score": 0.95,
                })
                recommendations.append({
                    "chart": "KDE Plot",
                    "reason": f"Smooth density estimate — shows distribution without bin sensitivity",
                    "score": 0.88,
                })
                recommendations.append({
                    "chart": "Box Plot",
                    "reason": f"Quick summary of median, quartiles, and outliers",
                    "score": 0.80,
                })

        elif "categorical" in ctype:
            nunique = series.nunique()
            if nunique <= 6:
                recommendations.append({
                    "chart": "Pie Chart",
                    "reason": f"'{cname}' has only {nunique} categories — pie chart shows proportions clearly",
                    "score": 0.92,
                })
                recommendations.append({
                    "chart": "Bar Chart",
                    "reason": f"Precise comparison of category counts",
                    "score": 0.88,
                })
            else:
                recommendations.append({
                    "chart": "Bar Chart",
                    "reason": f"'{cname}' has {nunique} categories — bar chart handles many categories well",
                    "score": 0.95,
                })
                recommendations.append({
                    "chart": "Pie Chart",
                    "reason": f"Shows proportional breakdown (best with top N filter)",
                    "score": 0.70,
                })

        elif ctype == "datetime":
            recommendations.append({
                "chart": "Line with Range Slider",
                "reason": f"Time-series data — line chart with interactive range exploration",
                "score": 0.95,
            })

    # ── Two columns ───────────────────────────────────────────────────────────
    elif n_cols == 2:
        t1, c1 = col_types[0]
        t2, c2 = col_types[1]

        if "numeric" in t1 and "numeric" in t2:
            # Check correlation
            try:
                corr = abs(df[c1].corr(df[c2]))
            except Exception:
                corr = 0

            recommendations.append({
                "chart": "Scatter Plot",
                "reason": f"Two numeric columns — scatter reveals relationship pattern (correlation: {corr:.2f})",
                "score": 0.95,
            })
            if corr > 0.5:
                recommendations.append({
                    "chart": "Scatter Plot (+ Trendline)",
                    "reason": f"Strong correlation ({corr:.2f}) — trendline shows linear relationship",
                    "score": 0.93,
                })
            recommendations.append({
                "chart": "Heatmap",
                "reason": f"Correlation matrix shows overall relationship strength",
                "score": 0.75,
            })

        elif ("numeric" in t1 and "categorical" in t2) or ("categorical" in t1 and "numeric" in t2):
            num_col = c1 if "numeric" in t1 else c2
            cat_col = c2 if "numeric" in t1 else c1
            n_groups = df[cat_col].nunique()

            recommendations.append({
                "chart": "Violin Plot",
                "reason": f"Shows distribution of '{num_col}' across {n_groups} groups of '{cat_col}' — reveals shape differences",
                "score": 0.95,
            })
            recommendations.append({
                "chart": "Box Plot",
                "reason": f"Compare medians and spread across groups",
                "score": 0.90,
            })
            recommendations.append({
                "chart": "Grouped Bar",
                "reason": f"Direct value comparison across categories",
                "score": 0.80,
            })

        elif "categorical" in t1 and "categorical" in t2:
            recommendations.append({
                "chart": "Grouped Bar",
                "reason": f"Compare distribution of '{c1}' across groups of '{c2}'",
                "score": 0.92,
            })
            recommendations.append({
                "chart": "Heatmap",
                "reason": f"Cross-tabulation heatmap shows association patterns",
                "score": 0.85,
            })

        elif "datetime" in t1 or "datetime" in t2:
            recommendations.append({
                "chart": "Line with Range Slider",
                "reason": f"Time-series relationship — interactive line chart",
                "score": 0.95,
            })
            recommendations.append({
                "chart": "Area Chart",
                "reason": f"Area chart emphasizes magnitude over time",
                "score": 0.85,
            })

    # ── Three+ columns ────────────────────────────────────────────────────────
    elif n_cols >= 3:
        numeric_count = sum(1 for t, _ in col_types if "numeric" in t)
        cat_count = sum(1 for t, _ in col_types if "categorical" in t)

        if numeric_count >= 3:
            recommendations.append({
                "chart": "3D Scatter Plot",
                "reason": f"Three numeric dimensions — 3D scatter reveals multivariate patterns",
                "score": 0.90,
            })
            recommendations.append({
                "chart": "Parallel Coordinates",
                "reason": f"Compare {numeric_count} numeric features simultaneously across rows",
                "score": 0.92,
            })
            recommendations.append({
                "chart": "Bubble Chart",
                "reason": f"Third dimension encoded as bubble size",
                "score": 0.85,
            })
        elif cat_count >= 2:
            recommendations.append({
                "chart": "Sunburst",
                "reason": f"Hierarchical breakdown of {cat_count} categorical variables",
                "score": 0.90,
            })
            recommendations.append({
                "chart": "Treemap",
                "reason": f"Space-efficient hierarchical visualization",
                "score": 0.88,
            })
        else:
            recommendations.append({
                "chart": "Pair Plot",
                "reason": f"Overview of all pairwise relationships between selected columns",
                "score": 0.90,
            })

    # Sort by score and return top N
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:max_recommendations]


def get_quick_insight(df: pd.DataFrame, column: str) -> str:
    """Generate a one-line quick insight about a column."""
    series = df[column].dropna()
    if len(series) == 0:
        return "No data available."

    if pd.api.types.is_numeric_dtype(series):
        skew = series.skew()
        if abs(skew) > 1.5:
            direction = "right" if skew > 0 else "left"
            return f"⚠️ Heavily {direction}-skewed (skew={skew:.2f}) — consider log transform"
        outlier_pct = ((series > series.quantile(0.99)) | (series < series.quantile(0.01))).mean() * 100
        if outlier_pct > 5:
            return f"⚠️ {outlier_pct:.1f}% potential outliers detected"
        return f"✅ Well-distributed (mean={series.mean():.2f}, std={series.std():.2f})"
    else:
        nunique = series.nunique()
        top_val = series.value_counts().index[0]
        top_pct = series.value_counts(normalize=True).iloc[0] * 100
        if top_pct > 70:
            return f"⚠️ Dominated by '{top_val}' ({top_pct:.0f}%) — highly imbalanced"
        return f"✅ {nunique} categories, top='{top_val}' ({top_pct:.0f}%)"
