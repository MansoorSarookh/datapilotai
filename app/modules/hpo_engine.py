"""
DataPilot AI — Hyperparameter Optimization Engine
Uses Optuna for efficient hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


def get_search_space(algorithm: str, is_classification: bool) -> Dict:
    """Define hyperparameter search spaces for each algorithm."""
    spaces = {
        "Random Forest": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int_or_none", 3, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", None]),
        },
        "XGBoost": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float_log", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("float_log", 1e-3, 10),
            "reg_lambda": ("float_log", 1e-3, 10),
        },
        "LightGBM": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float_log", 0.01, 0.3),
            "num_leaves": ("int", 10, 150),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("float_log", 1e-3, 10),
            "reg_lambda": ("float_log", 1e-3, 10),
        },
        "SVM": {
            "C": ("float_log", 0.01, 100),
            "kernel": ("categorical", ["rbf", "linear", "poly"]),
            "gamma": ("categorical", ["scale", "auto"]),
        },
        "KNN": {
            "n_neighbors": ("int", 3, 30),
            "weights": ("categorical", ["uniform", "distance"]),
            "metric": ("categorical", ["euclidean", "manhattan", "minkowski"]),
        },
        "Logistic Regression": {
            "C": ("float_log", 0.01, 100),
            "solver": ("categorical", ["lbfgs", "liblinear", "saga"]),
            "max_iter": ("int", 100, 2000),
        },
        "Decision Tree": {
            "max_depth": ("int_or_none", 3, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "criterion": ("categorical", ["gini", "entropy"] if is_classification else ["squared_error", "friedman_mse"]),
        },
        "Gradient Boosting": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float_log", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "min_samples_split": ("int", 2, 20),
        },
        "Ridge": {
            "alpha": ("float_log", 0.01, 100),
        },
        "Lasso": {
            "alpha": ("float_log", 0.001, 10),
        },
        "ElasticNet": {
            "alpha": ("float_log", 0.001, 10),
            "l1_ratio": ("float", 0.1, 0.9),
        },
    }
    return spaces.get(algorithm, {})


def run_hpo(
    X_train: pd.DataFrame,
    y_train,
    algorithm: str,
    is_classification: bool,
    n_trials: int = 50,
    cv_folds: int = 5,
    progress_callback=None,
) -> Dict:
    """
    Run hyperparameter optimization using Optuna.

    Returns dict with best_params, best_score, trial_history, and best_model.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {"error": "Optuna not installed. Run: pip install optuna"}

    from app.modules.model_arena import _get_model_instance

    search_space = get_search_space(algorithm, is_classification)
    if not search_space:
        return {"error": f"No search space defined for {algorithm}"}

    scoring = "accuracy" if is_classification else "r2"
    trial_history = []

    def objective(trial):
        params = {}
        for param_name, param_config in search_space.items():
            ptype = param_config[0]
            if ptype == "int":
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif ptype == "int_or_none":
                use_none = trial.suggest_categorical(f"{param_name}_none", [True, False])
                if use_none:
                    params[param_name] = None
                else:
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif ptype == "float":
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif ptype == "float_log":
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
            elif ptype == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])

        model = _get_model_instance(algorithm, is_classification)
        if model is None:
            return float("-inf")

        # Apply parameters
        valid_params = {k: v for k, v in params.items() if hasattr(model, k) or k in model.get_params()}
        model.set_params(**valid_params)

        try:
            cv = cross_val_score(model, X_train, y_train, cv=min(cv_folds, len(X_train)), scoring=scoring)
            score = cv.mean()
        except Exception:
            score = float("-inf")

        trial_history.append({
            "trial": trial.number,
            "score": round(score, 4) if score != float("-inf") else None,
            "params": params,
        })

        if progress_callback and n_trials > 0:
            progress_callback(trial.number / n_trials, f"Trial {trial.number}/{n_trials}: score={score:.4f}")

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Train best model
    best_model = _get_model_instance(algorithm, is_classification)
    if best_model is not None:
        valid_params = {}
        for k, v in study.best_params.items():
            if not k.endswith("_none"):
                valid_params[k] = v
        try:
            best_model.set_params(**valid_params)
            best_model.fit(X_train, y_train)
        except Exception:
            best_model = None

    return {
        "best_params": study.best_params,
        "best_score": round(study.best_value, 4),
        "n_trials": n_trials,
        "trial_history": trial_history,
        "best_model": best_model,
        "optimization_history": [
            {"trial": t.number, "value": round(t.value, 4) if t.value else None}
            for t in study.trials
            if t.value is not None
        ],
    }
