"""
DataPilot AI — Intelligent AI-Assisted Data Cleaning Engine
"""

import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.modules.ai_engine import chat_with_data, get_groq_client

warnings.filterwarnings("ignore")


# ============================================================
# VALIDATION LAYER
# ============================================================

def validate_dataframe(df: pd.DataFrame):
    if df is None:
        raise ValueError("Dataset is None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Dataset is empty")


# ============================================================
# DATA QUALITY & ML READINESS
# ============================================================

def compute_data_quality_score(df: pd.DataFrame, issues: Dict) -> int:
    score = 100

    score -= len(issues.get("missing_values", {})) * 2
    score -= issues.get("duplicates", 0) / max(len(df), 1) * 20
    score -= len(issues.get("outliers", {})) * 3
    score -= len(issues.get("constant_columns", [])) * 2

    return int(max(0, min(100, score)))


def compute_ml_readiness(df: pd.DataFrame) -> int:
    score = 100

    if df.isna().sum().sum() > 0:
        score -= 20

    if len(df.select_dtypes(include="object").columns) > 0:
        score -= 15

    if len(df) < 50:
        score -= 20

    return int(max(0, score))


# ============================================================
# DETECTION ENGINE
# ============================================================

def detect_cleaning_opportunities(df: pd.DataFrame) -> Dict:
    issues = {}

    # Missing
    null_counts = df.isna().sum()
    missing = null_counts[null_counts > 0]
    if not missing.empty:
        issues["missing_values"] = {
            col: {"count": int(ct), "pct": round(ct / len(df) * 100, 2)}
            for col, ct in missing.items()
        }

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues["duplicates"] = dup_count

    # Outliers
    outliers = {}
    for col in df.select_dtypes(include=np.number):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) |
                     (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            outliers[col] = n_out
    if outliers:
        issues["outliers"] = outliers

    # Constant
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        issues["constant_columns"] = const_cols

    # High cardinality
    high_card = {
        c: df[c].nunique()
        for c in df.select_dtypes(include="object")
        if df[c].nunique() > 50
    }
    if high_card:
        issues["high_cardinality"] = high_card

    return issues


# ============================================================
# AI RECOMMENDATION ENGINE
# ============================================================

def generate_ai_cleaning_plan(df: pd.DataFrame, issues: Dict) -> Dict:
    client = get_groq_client()
    if not client:
        return {}

    try:
        prompt = f"""
        You are an expert data scientist.

        Dataset shape: {df.shape}
        Issues: {issues}

        Return ONLY JSON:
        {{
            "missing_strategy": {{col: method}},
            "remove_duplicates": true/false,
            "drop_constants": true/false,
            "outlier_strategy": "iqr_clip"|"iqr_drop"|"zscore_clip"|"winsorize",
            "encode_method": "onehot"|"label"|"frequency",
            "scale_method": "minmax"|"zscore"|"robust"
        }}
        """

        response = chat_with_data(prompt, df, [])

        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])

    except Exception:
        return {}


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def handle_missing_values(df, strategy):
    df = df.copy()
    for col, method in strategy.items():
        if col not in df.columns:
            continue

        try:
            if method == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "ffill":
                df[col].fillna(method="ffill", inplace=True)
            elif method == "bfill":
                df[col].fillna(method="bfill", inplace=True)
            elif method == "zero":
                df[col].fillna(0, inplace=True)
            elif method == "drop":
                df = df[df[col].notna()]
        except Exception:
            continue

    return df


def handle_outliers(df, strategy):
    df = df.copy()
    for col in df.select_dtypes(include=np.number):
        try:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            if strategy == "iqr_clip":
                df[col] = df[col].clip(lower, upper)
            elif strategy == "iqr_drop":
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            elif strategy == "zscore_clip":
                mean, std = df[col].mean(), df[col].std()
                df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)
            elif strategy == "winsorize":
                p5, p95 = df[col].quantile(0.05), df[col].quantile(0.95)
                df[col] = df[col].clip(p5, p95)
        except Exception:
            continue
    return df


def encode_categoricals(df, method):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if method == "onehot":
        cols = [c for c in cat_cols if df[c].nunique() <= 20]
        df = pd.get_dummies(df, columns=cols)
    elif method == "label":
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    elif method == "frequency":
        for col in cat_cols:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)

    return df


def scale_features(df, method):
    df = df.copy()
    for col in df.select_dtypes(include=np.number):
        try:
            if method == "minmax":
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
            elif method == "zscore":
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            elif method == "robust":
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                df[col] = (df[col] - median) / (iqr + 1e-9)
        except Exception:
            continue
    return df


# ============================================================
# PIPELINE ENGINE (WITH IMPACT + LOGGING)
# ============================================================

def apply_cleaning_pipeline(df: pd.DataFrame, config: Dict):
    validate_dataframe(df)

    logs = []
    before_shape = df.shape

    try:
        if config.get("remove_duplicates"):
            before = len(df)
            df = df.drop_duplicates()
            logs.append(f"Removed {before - len(df)} duplicates")

        if config.get("drop_constants"):
            cols = [c for c in df.columns if df[c].nunique() <= 1]
            df.drop(columns=cols, inplace=True)
            logs.append(f"Dropped constants: {cols}")

        if config.get("missing_strategy"):
            df = handle_missing_values(df, config["missing_strategy"])
            logs.append("Handled missing values")

        if config.get("outlier_strategy"):
            df = handle_outliers(df, config["outlier_strategy"])
            logs.append(f"Outliers handled: {config['outlier_strategy']}")

        if config.get("encode_method"):
            df = encode_categoricals(df, config["encode_method"])
            logs.append("Encoded categoricals")

        if config.get("scale_method"):
            df = scale_features(df, config["scale_method"])
            logs.append("Scaled features")

        if df.empty:
            raise ValueError("All rows removed during cleaning")

        after_shape = df.shape
        logs.append(f"Shape: {before_shape} → {after_shape}")

        return df, logs

    except Exception as e:
        logs.append(f"ERROR: {str(e)}")
        return df, logs


# ============================================================
# AUTO AI CLEANING
# ============================================================

def auto_clean_dataset(df: pd.DataFrame):
    validate_dataframe(df)

    issues = detect_cleaning_opportunities(df)
    quality_before = compute_data_quality_score(df, issues)

    ai_plan = generate_ai_cleaning_plan(df, issues)

    if not ai_plan:
        return df, ["AI failed"]

    cleaned_df, logs = apply_cleaning_pipeline(df, ai_plan)

    quality_after = compute_data_quality_score(
        cleaned_df,
        detect_cleaning_opportunities(cleaned_df)
    )

    logs.append(f"Quality Score: {quality_before} → {quality_after}")
    logs.append(f"ML Readiness: {compute_ml_readiness(cleaned_df)}")

    return cleaned_df, logs 


# """
# DataPilot AI — Automated Data Cleaning Module
# Handles missing values, outliers, duplicates, scaling, encoding, and date features.
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Optional
# import warnings
# warnings.filterwarnings("ignore")


# def detect_cleaning_opportunities(df: pd.DataFrame) -> Dict:
#     """Scan dataset and return a summary of cleaning opportunities."""
#     issues = {}

#     # Missing values
#     null_counts = df.isna().sum()
#     missing_cols = null_counts[null_counts > 0].to_dict()
#     if missing_cols:
#         issues["missing_values"] = {
#             col: {"count": int(ct), "pct": round(ct / len(df) * 100, 1)}
#             for col, ct in missing_cols.items()
#         }

#     # Duplicates
#     dup_count = int(df.duplicated().sum())
#     if dup_count:
#         issues["duplicates"] = dup_count

#     # Outliers per numeric column (IQR method)
#     outlier_info = {}
#     for col in df.select_dtypes(include=np.number).columns:
#         q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
#         iqr = q3 - q1
#         n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
#         if n_out > 0:
#             outlier_info[col] = n_out
#     if outlier_info:
#         issues["outliers"] = outlier_info

#     # Constant columns
#     const_cols = [col for col in df.columns if df[col].nunique() <= 1]
#     if const_cols:
#         issues["constant_columns"] = const_cols

#     # High-cardinality object columns
#     high_card = {col: int(df[col].nunique()) for col in df.select_dtypes(include="object").columns if df[col].nunique() > 50}
#     if high_card:
#         issues["high_cardinality"] = high_card

#     return issues


# def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
#     """
#     Handle missing values per column.
#     strategy: {col_name: 'mean'|'median'|'mode'|'ffill'|'bfill'|'drop'|'zero'}
#     """
#     df = df.copy()
#     drop_rows = []

#     for col, method in strategy.items():
#         if col not in df.columns:
#             continue
#         if method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(df[col].mean())
#         elif method == "median" and pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(df[col].median())
#         elif method == "mode":
#             mode_val = df[col].mode()
#             if not mode_val.empty:
#                 df[col] = df[col].fillna(mode_val[0])
#         elif method == "ffill":
#             df[col] = df[col].fillna(method="ffill")
#         elif method == "bfill":
#             df[col] = df[col].fillna(method="bfill")
#         elif method == "zero":
#             df[col] = df[col].fillna(0)
#         elif method == "drop":
#             drop_rows.append(df[df[col].isna()].index)

#     if drop_rows:
#         all_drop = drop_rows[0]
#         for idx in drop_rows[1:]:
#             all_drop = all_drop.union(idx)
#         df = df.drop(index=all_drop)

#     return df


# def handle_outliers(df: pd.DataFrame, strategy: str = "iqr_clip", columns: Optional[List[str]] = None) -> pd.DataFrame:
#     """
#     Handle outliers.
#     strategy: 'iqr_clip' | 'zscore_clip' | 'iqr_drop' | 'winsorize'
#     """
#     df = df.copy()
#     num_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

#     for col in num_cols:
#         if col not in df.columns:
#             continue
#         series = df[col].dropna()
#         if len(series) < 4:
#             continue

#         if strategy in ("iqr_clip", "iqr_drop"):
#             q1, q3 = series.quantile(0.25), series.quantile(0.75)
#             iqr = q3 - q1
#             lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
#             if strategy == "iqr_clip":
#                 df[col] = df[col].clip(lower=lower, upper=upper)
#             else:
#                 df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]
#         elif strategy == "zscore_clip":
#             mean, std = series.mean(), series.std()
#             df[col] = df[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)
#         elif strategy == "winsorize":
#             p5, p95 = series.quantile(0.05), series.quantile(0.95)
#             df[col] = df[col].clip(lower=p5, upper=p95)

#     return df


# def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
#     return df.drop_duplicates().reset_index(drop=True)


# def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
#     const_cols = [col for col in df.columns if df[col].nunique() <= 1]
#     return df.drop(columns=const_cols)


# def scale_features(df: pd.DataFrame, method: str = "minmax", columns: Optional[List[str]] = None) -> pd.DataFrame:
#     """Scale numeric columns. method: 'minmax' | 'zscore' | 'robust'"""
#     df = df.copy()
#     cols = columns or df.select_dtypes(include=np.number).columns.tolist()

#     for col in cols:
#         if col not in df.columns:
#             continue
#         series = df[col].dropna()
#         if len(series) == 0:
#             continue

#         if method == "minmax":
#             min_v, max_v = series.min(), series.max()
#             if max_v > min_v:
#                 df[col] = (df[col] - min_v) / (max_v - min_v)
#         elif method == "zscore":
#             df[col] = (df[col] - series.mean()) / (series.std() + 1e-9)
#         elif method == "robust":
#             median = series.median()
#             q1, q3 = series.quantile(0.25), series.quantile(0.75)
#             iqr = q3 - q1
#             if iqr > 0:
#                 df[col] = (df[col] - median) / iqr

#     return df


# def encode_categoricals(df: pd.DataFrame, method: str = "onehot", columns: Optional[List[str]] = None, target_col: str = None) -> pd.DataFrame:
#     """Encode categorical columns. method: 'onehot' | 'label' | 'frequency'"""
#     df = df.copy()
#     cat_cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
#     if target_col and target_col in cat_cols:
#         cat_cols.remove(target_col)

#     if method == "onehot":
#         # Only one-hot encode low cardinality columns
#         cols_to_encode = [c for c in cat_cols if df[c].nunique() <= 20]
#         df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)
#     elif method == "label":
#         from sklearn.preprocessing import LabelEncoder
#         for col in cat_cols:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#     elif method == "frequency":
#         for col in cat_cols:
#             freq_map = df[col].value_counts(normalize=True).to_dict()
#             df[col] = df[col].map(freq_map)

#     return df


# def extract_date_features(df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
#     """Extract year, month, day, dayofweek, quarter, is_weekend from datetime columns."""
#     df = df.copy()
#     if date_columns is None:
#         date_columns = df.select_dtypes(include="datetime").columns.tolist()

#     for col in date_columns:
#         if col not in df.columns:
#             continue
#         try:
#             dt = pd.to_datetime(df[col], errors="coerce")
#             df[f"{col}_year"] = dt.dt.year
#             df[f"{col}_month"] = dt.dt.month
#             df[f"{col}_day"] = dt.dt.day
#             df[f"{col}_dayofweek"] = dt.dt.dayofweek
#             df[f"{col}_quarter"] = dt.dt.quarter
#             df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
#             df = df.drop(columns=[col])  # Replace original with features
#         except Exception:
#             pass

#     return df


# def apply_cleaning_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
#     """
#     Apply a full cleaning pipeline based on a configuration dict.
#     config keys:
#         - missing_strategy: {col: method}
#         - outlier_strategy: 'iqr_clip' | 'zscore_clip' | 'winsorize' | None
#         - remove_duplicates: bool
#         - drop_constants: bool
#         - scale_method: 'minmax' | 'zscore' | 'robust' | None
#         - encode_method: 'onehot' | 'label' | 'frequency' | None
#         - extract_dates: bool
#     Returns cleaned DataFrame
#     """
#     operations_log = []

#     if config.get("remove_duplicates", False):
#         before = len(df)
#         df = remove_duplicates(df)
#         operations_log.append(f"Removed {before - len(df)} duplicate rows")

#     if config.get("drop_constants", False):
#         before_cols = set(df.columns)
#         df = drop_constant_columns(df)
#         dropped = before_cols - set(df.columns)
#         if dropped:
#             operations_log.append(f"Dropped constant columns: {list(dropped)}")

#     if config.get("missing_strategy"):
#         df = handle_missing_values(df, config["missing_strategy"])
#         operations_log.append(f"Applied missing value strategies to {len(config['missing_strategy'])} columns")

#     if config.get("outlier_strategy"):
#         df = handle_outliers(df, strategy=config["outlier_strategy"])
#         operations_log.append(f"Handled outliers using {config['outlier_strategy']}")

#     if config.get("extract_dates", False):
#         df = extract_date_features(df)
#         operations_log.append("Extracted date features")

#     if config.get("encode_method"):
#         df = encode_categoricals(df, method=config["encode_method"], target_col=config.get("target_col"))
#         operations_log.append(f"Encoded categoricals using {config['encode_method']} encoding")

#     if config.get("scale_method"):
#         df = scale_features(df, method=config["scale_method"])
#         operations_log.append(f"Scaled numeric features using {config['scale_method']}")

#     return df, operations_log
 
