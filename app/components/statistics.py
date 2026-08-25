"""
Statistics display component.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_descriptive_stats(stats_df: pd.DataFrame) -> None:
    """Render descriptive statistics table."""
    
    if stats_df.empty:
        st.info("No numeric columns found for descriptive statistics.")
        return
    
    st.subheader("📈 Descriptive Statistics")
    
    # Round for display
    display_df = stats_df.round(3)
    
    st.dataframe(
        display_df,
        use_container_width=True,
    )
    
    # Quick insights
    with st.expander("📊 Quick Insights"):
        for col in stats_df.index:
            skewness = stats_df.loc[col, 'Skewness'] if 'Skewness' in stats_df.columns else None
            
            if skewness is not None:
                if abs(skewness) < 0.5:
                    dist = "approximately symmetric"
                elif skewness > 0:
                    dist = "right-skewed (positive skew)"
                else:
                    dist = "left-skewed (negative skew)"
                
                st.write(f"**{col}**: Distribution is {dist}")


def render_missing_values(missing_df: pd.DataFrame) -> None:
    """Render missing values analysis."""
    
    st.subheader("❓ Missing Values Analysis")
    
    # Filter to only show columns with missing values
    missing_only = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_only.empty:
        st.success("✅ No missing values found in the dataset!")
        return
    
    # Summary metrics
    total_missing = missing_df['Missing Count'].sum()
    cols_with_missing = len(missing_only)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Missing Values", f"{total_missing:,}")
    
    with col2:
        st.metric("Columns with Missing", f"{cols_with_missing}")
    
    # Table of missing values
    st.dataframe(
        missing_only[['Column', 'Missing Count', 'Missing %']],
        use_container_width=True,
        hide_index=True,
    )
    
    # Warnings for high missing percentages
    high_missing = missing_only[missing_only['Missing Count'] / missing_df['Missing Count'].max() > 0.5]
    
    if not high_missing.empty:
        st.warning(f"⚠️ {len(high_missing)} column(s) have more than 50% missing values!")


def render_correlation_insights(correlation_df: pd.DataFrame, threshold: float = 0.7) -> None:
    """Render correlation insights."""
    
    if correlation_df.empty:
        return
    
    st.subheader("🔗 Correlation Insights")
    
    # Find highly correlated pairs
    high_corr_pairs = []
    
    for i in range(len(correlation_df.columns)):
        for j in range(i + 1, len(correlation_df.columns)):
            col1 = correlation_df.columns[i]
            col2 = correlation_df.columns[j]
            corr = correlation_df.iloc[i, j]
            
            if abs(corr) >= threshold:
                high_corr_pairs.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': corr,
                    'Strength': 'Strong Positive' if corr > 0 else 'Strong Negative',
                })
    
    if high_corr_pairs:
        st.write(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| ≥ {threshold}):")
        st.dataframe(
            pd.DataFrame(high_corr_pairs),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(f"No highly correlated pairs found (threshold: |r| ≥ {threshold})")


def render_outlier_analysis(outlier_info: Dict[str, Dict]) -> None:
    """Render outlier analysis."""
    
    st.subheader("🎯 Outlier Detection (IQR Method)")
    
    outlier_data = []
    
    for col, info in outlier_info.items():
        if info['count'] > 0:
            outlier_data.append({
                'Column': col,
                'Outliers': info['count'],
                'Percentage': f"{info['percentage']:.2f}%",
            })
    
    if outlier_data:
        st.dataframe(
            pd.DataFrame(outlier_data),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ No significant outliers detected!")


def render_categorical_stats(categorical_stats: Dict[str, pd.DataFrame]) -> None:
    """Render categorical column statistics."""
    
    if not categorical_stats:
        return
    
    st.subheader("🏷️ Categorical Columns")
    
    for col, stats_df in categorical_stats.items():
        with st.expander(f"📊 {col} ({len(stats_df)} categories)"):
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
            )


def render_statistics_panel(
    descriptive_stats: pd.DataFrame,
    missing_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    outlier_info: Dict[str, Dict],
    categorical_stats: Dict[str, pd.DataFrame] = None
) -> None:
    """Render complete statistics panel."""
    
    # Create tabs for different statistics
    tabs = st.tabs([
        "📈 Descriptive",
        "❓ Missing Values",
        "🔗 Correlations",
        "🎯 Outliers",
        "🏷️ Categorical",
    ])
    
    with tabs[0]:
        render_descriptive_stats(descriptive_stats)
    
    with tabs[1]:
        render_missing_values(missing_df)
    
    with tabs[2]:
        render_correlation_insights(correlation_df)
    
    with tabs[3]:
        render_outlier_analysis(outlier_info)
    
    with tabs[4]:
        if categorical_stats:
            render_categorical_stats(categorical_stats)
        else:
            st.info("No categorical columns found.")
 
