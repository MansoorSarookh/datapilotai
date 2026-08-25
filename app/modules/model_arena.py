"""
DataPilot AI — Model Comparison Arena
Train multiple models simultaneously and compare metrics in a leaderboard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
)
import warnings
warnings.filterwarnings("ignore")


# All available algorithms
CLASSIFICATION_ALGORITHMS = {
    "Random Forest": "sklearn.ensemble.RandomForestClassifier",
    "Logistic Regression": "sklearn.linear_model.LogisticRegression",
    "XGBoost": "xgboost.XGBClassifier",
    "LightGBM": "lightgbm.LGBMClassifier",
    "SVM": "sklearn.svm.SVC",
    "KNN": "sklearn.neighbors.KNeighborsClassifier",
    "Decision Tree": "sklearn.tree.DecisionTreeClassifier",
    "Naive Bayes": "sklearn.naive_bayes.GaussianNB",
    "Gradient Boosting": "sklearn.ensemble.GradientBoostingClassifier",
}

REGRESSION_ALGORITHMS = {
    "Random Forest": "sklearn.ensemble.RandomForestRegressor",
    "Linear Regression": "sklearn.linear_model.LinearRegression",
    "XGBoost": "xgboost.XGBRegressor",
    "LightGBM": "lightgbm.LGBMRegressor",
    "SVR": "sklearn.svm.SVR",
    "KNN": "sklearn.neighbors.KNeighborsRegressor",
    "Decision Tree": "sklearn.tree.DecisionTreeRegressor",
    "Ridge": "sklearn.linear_model.Ridge",
    "Lasso": "sklearn.linear_model.Lasso",
    "ElasticNet": "sklearn.linear_model.ElasticNet",
    "Gradient Boosting": "sklearn.ensemble.GradientBoostingRegressor",
}


def _get_model_instance(algorithm: str, is_classification: bool):
    """Dynamically create model instance."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB

    models = {
        # Classification
        ("Random Forest", True): RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        ("Logistic Regression", True): LogisticRegression(max_iter=1000, random_state=42),
        ("Decision Tree", True): DecisionTreeClassifier(random_state=42),
        ("KNN", True): KNeighborsClassifier(),
        ("Naive Bayes", True): GaussianNB(),
        ("Gradient Boosting", True): GradientBoostingClassifier(n_estimators=100, random_state=42),
        # Regression
        ("Random Forest", False): RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        ("Linear Regression", False): LinearRegression(),
        ("Decision Tree", False): DecisionTreeRegressor(random_state=42),
        ("KNN", False): KNeighborsRegressor(),
        ("Ridge", False): Ridge(random_state=42),
        ("Lasso", False): Lasso(random_state=42),
        ("ElasticNet", False): ElasticNet(random_state=42),
        ("Gradient Boosting", False): GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    key = (algorithm, is_classification)
    if key in models:
        return models[key]

    # Try SVM
    if algorithm in ("SVM", "SVR"):
        from sklearn.svm import SVC, SVR
        if is_classification:
            return SVC(probability=True, random_state=42)
        return SVR()

    # Try XGBoost
    if algorithm == "XGBoost":
        try:
            import xgboost as xgb
            if is_classification:
                return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0)
            return xgb.XGBRegressor(random_state=42, verbosity=0)
        except ImportError:
            return None

    # Try LightGBM
    if algorithm == "LightGBM":
        try:
            import lightgbm as lgb
            if is_classification:
                return lgb.LGBMClassifier(random_state=42, verbose=-1)
            return lgb.LGBMRegressor(random_state=42, verbose=-1)
        except ImportError:
            return None

    return None


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
):
    """Prepare data for ML training."""
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
    y = df_clean[target_col].fillna(
        df_clean[target_col].mode().iloc[0]
        if not pd.api.types.is_numeric_dtype(df_clean[target_col])
        else df_clean[target_col].median()
    )

    # Determine problem type
    is_classification = (
        not pd.api.types.is_numeric_dtype(df[target_col]) or
        df[target_col].nunique() <= 20
    )

    if is_classification and pd.api.types.is_object_dtype(df[target_col]):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_cols, is_classification


def run_model_arena(
    df: pd.DataFrame,
    target_col: str,
    algorithms: List[str],
    test_size: float = 0.2,
    progress_callback=None,
) -> Dict:
    """
    Train multiple models and return a comparison leaderboard.

    Returns dict with 'leaderboard' (sorted results) and 'models' (trained model objects).
    """
    X_train, X_test, y_train, y_test, feature_cols, is_classification = prepare_data(
        df, target_col, test_size
    )

    results = []
    trained_models = {}
    total = len(algorithms)

    for i, algo in enumerate(algorithms):
        if progress_callback:
            progress_callback(i / total, f"Training {algo}...")

        model = _get_model_instance(algo, is_classification)
        if model is None:
            results.append({"algorithm": algo, "error": "Not available (missing dependency)"})
            continue

        try:
            # Scale for algorithms that need it
            needs_scaling = algo in ["SVM", "SVR", "KNN", "Logistic Regression", "Ridge", "Lasso", "ElasticNet"]
            if needs_scaling:
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
                X_te = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
            else:
                X_tr, X_te = X_train, X_test

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            trained_models[algo] = model

            metrics = {}
            if is_classification:
                metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                metrics["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                metrics["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                try:
                    if len(set(y_test)) == 2 and hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_te)[:, 1]
                        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
                except Exception:
                    pass
                primary_metric = metrics["accuracy"]
            else:
                metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
                metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
                primary_metric = metrics["r2"]

            # Feature importance
            fi = {}
            if hasattr(model, "feature_importances_"):
                fi = dict(zip(feature_cols, model.feature_importances_))
            elif hasattr(model, "coef_"):
                coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                fi = dict(zip(feature_cols, np.abs(coefs)))

            # Cross-validation
            try:
                scoring = "accuracy" if is_classification else "r2"
                cv = cross_val_score(model, X_tr, y_train, cv=min(5, len(X_tr)), scoring=scoring)
                cv_mean = round(float(cv.mean()), 4)
                cv_std = round(float(cv.std()), 4)
            except Exception:
                cv_mean = None
                cv_std = None

            results.append({
                "algorithm": algo,
                "metrics": metrics,
                "primary_metric": primary_metric,
                "feature_importance": dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]),
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "train_size": len(X_tr),
                "test_size": len(X_te),
            })

        except Exception as e:
            results.append({"algorithm": algo, "error": str(e)})

    if progress_callback:
        progress_callback(1.0, "Complete!")

    # Sort by primary metric
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["primary_metric"], reverse=True)
    error_results = [r for r in results if "error" in r]

    return {
        "leaderboard": valid_results + error_results,
        "models": trained_models,
        "problem_type": "classification" if is_classification else "regression",
        "best_model": valid_results[0]["algorithm"] if valid_results else None,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }
