"""
EDA Engine - Core analysis functions for exploratory data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic information about the DataFrame."""
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicates': df.duplicated().sum(),
        'total_missing': df.isnull().sum().sum(),
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Get detailed information about each column."""
    info_data = []
    
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        missing = col_data.isnull().sum()
        missing_pct = (missing / len(df)) * 100 if len(df) > 0 else 0
        unique = col_data.nunique()
        
        # Infer data type category
        if pd.api.types.is_numeric_dtype(col_data):
            type_category = 'Numeric'
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            type_category = 'DateTime'
        elif pd.api.types.is_bool_dtype(col_data):
            type_category = 'Boolean'
        elif unique <= 20:
            type_category = 'Categorical'
        else:
            type_category = 'Text'
        
        info_data.append({
            'Column': col,
            'Data Type': dtype,
            'Category': type_category,
            'Missing': missing,
            'Missing %': f"{missing_pct:.1f}%",
            'Unique': unique,
        })
    
    return pd.DataFrame(info_data)


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats_df = numeric_df.describe().T
    
    # Add additional statistics
    stats_df['median'] = numeric_df.median()
    stats_df['mode'] = numeric_df.mode().iloc[0] if not numeric_df.mode().empty else np.nan
    stats_df['skew'] = numeric_df.skew()
    stats_df['kurtosis'] = numeric_df.kurtosis()
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
    
    # Rename columns for clarity
    stats_df = stats_df.rename(columns={
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Std Dev',
        'min': 'Min',
        '25%': 'Q1',
        '50%': 'Median',
        '75%': 'Q3',
        'max': 'Max',
        'median': 'Median_calc',
        'mode': 'Mode',
        'skew': 'Skewness',
        'kurtosis': 'Kurtosis',
        'range': 'Range',
        'iqr': 'IQR',
    })
    
    # Select and order columns
    cols = ['Count', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max', 
            'Range', 'IQR', 'Skewness', 'Kurtosis', 'Mode']
    
    return stats_df[[c for c in cols if c in stats_df.columns]]


def get_categorical_stats(df: pd.DataFrame, max_categories: int = 20) -> Dict[str, pd.DataFrame]:
    """Get statistics for categorical columns."""
    categorical_stats = {}
    
    for col in df.columns:
        if df[col].nunique() <= max_categories:
            value_counts = df[col].value_counts()
            pct = (value_counts / len(df) * 100).round(2)
            
            stats_df = pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': pct.values,
            })
            categorical_stats[col] = stats_df
    
    return categorical_stats


def get_missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values in the DataFrame."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values,
        'Present Count': (len(df) - missing).values,
        'Present %': (100 - missing_pct).values,
    })
    
    # Sort by missing percentage descending
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    
    return missing_df


def get_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return pd.DataFrame()
    
    return numeric_df.corr(method=method)


def compute_vif(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute Variance Inflation Factor for numeric columns.
    
    Returns a DataFrame with columns ['Feature', 'VIF'] sorted by VIF descending.
    Returns None if insufficient data.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None
    
    # Remove constant columns
    non_const = [col for col in numeric_df.columns if numeric_df[col].std() > 0]
    if len(non_const) < 2:
        return None
    
    numeric_df = numeric_df[non_const]
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X = numeric_df.values
        vif_data = []
        for i in range(X.shape[1]):
            try:
                vif_val = variance_inflation_factor(X, i)
                vif_data.append({
                    "Feature": numeric_df.columns[i],
                    "VIF": round(float(vif_val), 2),
                })
            except Exception:
                vif_data.append({
                    "Feature": numeric_df.columns[i],
                    "VIF": float('inf'),
                })
        
        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
        
        # Filter out infinite VIF values
        vif_df = vif_df[vif_df["VIF"] != float('inf')]
        
        return vif_df if not vif_df.empty else None
    except ImportError:
        return None
    except Exception:
        return None


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Dict]:
    """Detect outliers in numeric columns."""
    outlier_info = {}
    numeric_df = df.select_dtypes(include=[np.number])
    
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        else:  # z-score
            z_scores = np.abs(stats.zscore(col_data))
            outliers = col_data[z_scores > 3]
        
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(col_data) * 100) if len(col_data) > 0 else 0,
            'lower_bound': lower_bound if method == 'iqr' else None,
            'upper_bound': upper_bound if method == 'iqr' else None,
        }
    
    return outlier_info


def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer semantic data types for columns."""
    type_map = {}
    
    for col in df.columns:
        col_data = df[col]
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            type_map[col] = 'datetime'
            continue
        
        # Try parsing as datetime
        if col_data.dtype == 'object':
            try:
                pd.to_datetime(col_data.dropna().head(100))
                type_map[col] = 'datetime_parseable'
                continue
            except (ValueError, TypeError):
                pass
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                type_map[col] = 'integer'
            else:
                type_map[col] = 'float'
            continue
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(col_data):
            type_map[col] = 'boolean'
            continue
        
        # Check cardinality for categorical
        unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
        if unique_ratio < 0.05 or col_data.nunique() <= 20:
            type_map[col] = 'categorical'
        else:
            type_map[col] = 'text'
    
    return type_map


def get_column_types_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns by their inferred type."""
    type_map = detect_data_types(df)
    
    summary = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': [],
    }
    
    for col, dtype in type_map.items():
        if dtype in ['integer', 'float']:
            summary['numeric'].append(col)
        elif dtype == 'categorical':
            summary['categorical'].append(col)
        elif dtype in ['datetime', 'datetime_parseable']:
            summary['datetime'].append(col)
        elif dtype == 'boolean':
            summary['boolean'].append(col)
        else:
            summary['text'].append(col)
    
    return summary


def get_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall data quality metrics."""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
    uniqueness = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 0
    
    # Overall quality score (simple average)
    quality_score = (completeness + uniqueness) / 2
    
    return {
        'completeness': round(completeness, 2),
        'uniqueness': round(uniqueness, 2),
        'overall_score': round(quality_score, 2),
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows,
    }


def perform_full_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform complete EDA and return all analyses."""
    return {
        'basic_info': get_basic_info(df),
        'column_info': get_column_info(df),
        'descriptive_stats': get_descriptive_stats(df),
        'missing_analysis': get_missing_analysis(df),
        'correlation': get_correlation_matrix(df),
        'data_types': detect_data_types(df),
        'column_types': get_column_types_summary(df),
        'quality_score': get_data_quality_score(df),
        'outliers': detect_outliers(df),
    }

