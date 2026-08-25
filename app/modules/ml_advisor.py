"""
DataPilot AI — ML Readiness Advisor
ML preparation advisor, feature engineering suggestions, feasibility predictor, and built-in algorithm execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# ML PREPARATION ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def assess_ml_readiness(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
    """
    Holistic ML readiness assessment. Returns score, checks, and recommendations.
    """
    checks = []
    warnings_list = []
    errors = []
    score = 100

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Sample size ───────────────────────────────────────────────────────────
    if n_rows >= 1000:
        checks.append("✅ Sufficient sample size (n={:,})".format(n_rows))
    elif n_rows >= 200:
        warnings_list.append("⚠️ Moderate sample size ({} rows) — use cross-validation".format(n_rows))
        score -= 10
    else:
        errors.append("❌ Very small dataset ({} rows) — results may be unreliable".format(n_rows))
        score -= 25

    # ── Missing values ────────────────────────────────────────────────────────
    null_pct = df.isna().sum().sum() / (n_rows * n_cols) * 100
    if null_pct == 0:
        checks.append("✅ No missing values")
    elif null_pct < 5:
        warnings_list.append(f"⚠️ {null_pct:.1f}% missing values — imputation recommended")
        score -= 5
    elif null_pct < 20:
        warnings_list.append(f"⚠️ {null_pct:.1f}% missing values — significant imputation needed")
        score -= 15
    else:
        errors.append(f"❌ {null_pct:.1f}% missing values — high risk for ML")
        score -= 25

    # ── Duplicates ────────────────────────────────────────────────────────────
    dups = df.duplicated().sum()
    if dups == 0:
        checks.append("✅ No duplicate rows")
    else:
        warnings_list.append(f"⚠️ {dups} duplicate rows — consider removing")
        score -= 5

    # ── Target analysis ───────────────────────────────────────────────────────
    problem_type = "unknown"
    recommended_algorithms = []
    expected_performance = {}

    if target_col and target_col in df.columns:
        target = df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(target):
            n_unique_ratio = target.nunique() / len(target)
            if n_unique_ratio < 0.05 and target.nunique() <= 20:
                problem_type = "classification"
            else:
                problem_type = "regression"
        else:
            problem_type = "classification"

        if problem_type == "classification":
            vc = target.value_counts(normalize=True)
            if vc.max() > 0.85:
                errors.append(f"❌ Severe class imbalance ({vc.max():.0%} dominant class) — use SMOTE or class_weight")
                score -= 20
            elif vc.max() > 0.70:
                warnings_list.append(f"⚠️ Moderate class imbalance ({vc.max():.0%}) — consider oversampling")
                score -= 10
            else:
                checks.append("✅ Balanced class distribution")
            
            recommended_algorithms = ["Logistic Regression", "Random Forest", "XGBoost"]
            baseline = max(vc.values)
            expected_performance = {
                "baseline_accuracy": f"{baseline:.0%}",
                "expected_range": f"{max(baseline + 0.05, 0.6):.0%} – {min(baseline + 0.15, 0.95):.0%}",
            }
        else:
            recommended_algorithms = ["Linear Regression", "Random Forest", "XGBoost"]
            expected_performance = {"metric": "R² / RMSE", "expected_r2_range": "0.60 – 0.85"}

        # Leakage detection
        leakage_cols = _detect_leakage(df, target_col)
        if leakage_cols:
            errors.append(f"❌ Potential leakage columns: {leakage_cols}")
            score -= 15
    else:
        warnings_list.append("⚠️ No target column selected — cannot assess classification/regression suitability")

    # ── High cardinality features ─────────────────────────────────────────────
    high_card = [c for c in cat_cols if df[c].nunique() > 50]
    if high_card:
        warnings_list.append(f"⚠️ High-cardinality categoricals: {high_card} — use target/hash encoding")
        score -= 5

    # ── Encoding needed ───────────────────────────────────────────────────────
    encode_cols = [c for c in cat_cols if c != target_col]
    if encode_cols:
        warnings_list.append(f"⚠️ {len(encode_cols)} categorical column(s) need encoding: {encode_cols[:5]}")

    # ── Scaling ───────────────────────────────────────────────────────────────
    scale_cols = _suggest_scaling(df, numeric_cols)
    if scale_cols:
        warnings_list.append(f"⚠️ Columns with large scale variance (suggest normalization): {scale_cols[:5]}")

    score = max(0, min(100, score))

    return {
        "score": score,
        "problem_type": problem_type,
        "checks": checks,
        "warnings": warnings_list,
        "errors": errors,
        "recommended_algorithms": recommended_algorithms,
        "expected_performance": expected_performance,
        "columns_to_encode": encode_cols,
        "columns_to_scale": scale_cols,
        "high_cardinality_cols": high_card,
        "leakage_risk_cols": _detect_leakage(df, target_col) if target_col else [],
        "feature_count": len([c for c in df.columns if c != target_col]),
    }


def _detect_leakage(df: pd.DataFrame, target_col: str) -> List[str]:
    """Detect columns with suspiciously high correlation to target (potential leakage)."""
    leaky = []
    if not target_col or target_col not in df.columns:
        return leaky
    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        return leaky
    for col in df.select_dtypes(include=np.number).columns:
        if col == target_col:
            continue
        try:
            corr = abs(df[col].fillna(0).corr(target.fillna(0)))
            if corr > 0.95:
                leaky.append(col)
        except Exception:
            pass
    return leaky


def _suggest_scaling(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    """Find columns where std/mean ratio suggests scaling would be beneficial."""
    scale_needed = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        if series.std() > 100 or (series.max() - series.min()) > 1000:
            scale_needed.append(col)
    return scale_needed


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════

def suggest_feature_engineering(df: pd.DataFrame) -> List[Dict]:
    """Return a list of feature engineering suggestions."""
    suggestions = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Datetime features
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            suggestions.append({
                "type": "DateTime Decomposition",
                "column": col,
                "suggestion": f"Extract: year, month, day_of_week, is_weekend, quarter from '{col}'",
                "impact": "High",
                "code_hint": f"df['{col}_year'] = df['{col}'].dt.year",
            })
        # Try parsing string as datetime
        elif col in cat_cols:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / len(df) > 0.7:
                    suggestions.append({
                        "type": "DateTime Parsing",
                        "column": col,
                        "suggestion": f"'{col}' may be a datetime column — parse and extract date features",
                        "impact": "High",
                        "code_hint": f"df['{col}'] = pd.to_datetime(df['{col}'])",
                    })
            except Exception:
                pass

    # High-cardinality encoding suggestions
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > 50:
            suggestions.append({
                "type": "Frequency Encoding",
                "column": col,
                "suggestion": f"'{col}' has {n_unique} unique values — use frequency encoding or hashing",
                "impact": "Medium",
                "code_hint": f"df['{col}_freq'] = df['{col}'].map(df['{col}'].value_counts())",
            })
        elif 2 < n_unique <= 50:
            suggestions.append({
                "type": "One-Hot Encoding",
                "column": col,
                "suggestion": f"'{col}' has {n_unique} unique values — one-hot encoding recommended",
                "impact": "Medium",
                "code_hint": f"pd.get_dummies(df['{col}'], prefix='{col}')",
            })

    # Numeric pair interactions
    if len(numeric_cols) >= 2:
        col_a, col_b = numeric_cols[0], numeric_cols[1]
        suggestions.append({
            "type": "Interaction Feature",
            "column": f"{col_a} × {col_b}",
            "suggestion": f"Create interaction feature: '{col_a}' × '{col_b}'",
            "impact": "Medium",
            "code_hint": f"df['{col_a}_x_{col_b}'] = df['{col_a}'] * df['{col_b}']",
        })

    # Log transform for skewed numerics
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0 and series.min() > 0 and abs(series.skew()) > 1.5:
            suggestions.append({
                "type": "Log Transform",
                "column": col,
                "suggestion": f"'{col}' is heavily skewed (skew={series.skew():.2f}) — apply log transform",
                "impact": "High",
                "code_hint": f"df['log_{col}'] = np.log1p(df['{col}'])",
            })

    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FEASIBILITY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

def predict_model_feasibility(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Heuristic model feasibility predictor.
    """
    n_rows = len(df)
    readiness = assess_ml_readiness(df, target_col)
    problem_type = readiness["problem_type"]
    score = readiness["score"]

    feasibility = "High" if score >= 75 else "Medium" if score >= 50 else "Low"

    models = readiness["recommended_algorithms"]
    expected = readiness.get("expected_performance", {})

    return {
        "feasibility": feasibility,
        "score": score,
        "problem_type": problem_type,
        "recommended_models": models,
        "expected_performance": expected,
        "data_sufficiency": (
            "Excellent" if n_rows >= 10000 else
            "Good" if n_rows >= 1000 else
            "Adequate" if n_rows >= 200 else "Insufficient"
        ),
        "risk_factors": readiness["errors"] + readiness["warnings"],
        "readiness_checks": readiness["checks"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN ML EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_pipeline(
    df: pd.DataFrame,
    target_col: str,
    algorithm: str = "Random Forest",
    test_size: float = 0.2,
) -> Dict:
    """
    Run a full ML pipeline: preprocessing → train/test split → model → metrics.
    Supports: Logistic/Linear Regression, Random Forest, XGBoost, K-Means.
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        r2_score, mean_squared_error, mean_absolute_error,
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression

    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False

    # ── Prepare data ──────────────────────────────────────────────────────────
    df_clean = df.copy()
    feature_cols = [c for c in df_clean.columns if c != target_col]

    # Encode categoricals
    le_dict = {}
    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le

    # Impute missing
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df_clean[feature_cols]), columns=feature_cols)
    y = df_clean[target_col].fillna(df_clean[target_col].mode().iloc[0] if not pd.api.types.is_numeric_dtype(df_clean[target_col]) else df_clean[target_col].median())

    # Determine problem type
    is_classification = (
        not pd.api.types.is_numeric_dtype(df[target_col]) or
        df[target_col].nunique() <= 20
    )

    if is_classification and pd.api.types.is_object_dtype(df[target_col]):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # ── Select model ──────────────────────────────────────────────────────────
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) if is_classification else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif algorithm == "Logistic Regression" and is_classification:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "Linear Regression" and not is_classification:
        model = LinearRegression()
    elif algorithm == "XGBoost" and XGB_AVAILABLE:
        model = (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            if is_classification
            else xgb.XGBRegressor(random_state=42)
        )
    else:
        # Fallback to Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42) if is_classification else RandomForestRegressor(n_estimators=100, random_state=42)

    # ── Train ─────────────────────────────────────────────────────────────────
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {}
    if is_classification:
        metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        try:
            if len(set(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
        except Exception:
            pass
    else:
        metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)

    # ── Feature importance ────────────────────────────────────────────────────
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        feature_importance = fi.head(15).to_dict()
    elif hasattr(model, "coef_"):
        fi = pd.Series(np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_), index=feature_cols).sort_values(ascending=False)
        feature_importance = fi.head(15).to_dict()

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_scores = []
    try:
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy" if is_classification else "r2")
        cv_scores = cv.tolist()
    except Exception:
        pass

    return {
        "algorithm": algorithm,
        "problem_type": "classification" if is_classification else "regression",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "cv_scores": cv_scores,
        "cv_mean": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
        "predictions": y_pred.tolist()[:50],  # sample predictions
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING (Unsupervised)
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> Dict:
    """Run K-Means clustering on numeric columns."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty or len(numeric_df) < n_clusters:
        return {"error": "Insufficient numeric data for clustering"}

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(numeric_df))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0

    return {
        "labels": labels.tolist(),
        "cluster_centers": km.cluster_centers_.tolist(),
        "silhouette_score": round(float(sil), 4),
        "inertia": round(float(km.inertia_), 2),
        "n_clusters": n_clusters,
        "cluster_counts": pd.Series(labels).value_counts().to_dict(),
    }
 
