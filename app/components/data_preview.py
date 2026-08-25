"""
Data preview component for displaying dataset information.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_data_preview(df: pd.DataFrame, n_rows: int = 10) -> None:
    """Render a preview of the dataset."""
    
    st.subheader("📋 Data Preview")
    
    # Tabs for head and tail
    tab1, tab2 = st.tabs(["First rows", "Last rows"])
    
    with tab1:
        st.dataframe(df.head(n_rows), use_container_width=True)
    
    with tab2:
        st.dataframe(df.tail(n_rows), use_container_width=True)


def render_data_shape(df: pd.DataFrame) -> None:
    """Render dataset shape information."""
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Rows", f"{len(df):,}")
    
    with cols[1]:
        st.metric("Columns", f"{len(df.columns):,}")
    
    with cols[2]:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory", f"{memory_mb:.2f} MB")
    
    with cols[3]:
        duplicates = df.duplicated().sum()
        st.metric("Duplicates", f"{duplicates:,}")


def render_column_info(column_info: pd.DataFrame) -> None:
    """Render column information table."""
    
    st.subheader("📊 Column Information")
    
    # Add interactive filtering
    col1, col2 = st.columns(2)
    
    with col1:
        type_filter = st.multiselect(
            "Filter by type:",
            options=column_info['Category'].unique(),
            default=column_info['Category'].unique(),
        )
    
    with col2:
        show_missing_only = st.checkbox("Show columns with missing values only")
    
    filtered_info = column_info[column_info['Category'].isin(type_filter)]
    
    if show_missing_only:
        filtered_info = filtered_info[filtered_info['Missing'] > 0]
    
    st.dataframe(
        filtered_info,
        use_container_width=True,
        hide_index=True,
    )


def render_data_quality_card(quality_score: Dict[str, Any]) -> None:
    """Render data quality metrics."""
    
    st.subheader("🎯 Data Quality")
    
    # Defensive check for non-dict quality_score
    if not isinstance(quality_score, dict):
        st.error(f"⚠️ Data Quality Error: Expected dictionary, got {type(quality_score).__name__}.")
        st.info("💡 **Fix:** Please click the **♻️ Force App Refresh** button in the sidebar or restart the server.")
        return

    cols = st.columns(3)
    
    with cols[0]:
        score = quality_score.get('overall_score', 0)
        color = 'normal' if score >= 80 else 'inverse' if score < 50 else 'off'
        st.metric(
            "Quality Score",
            f"{score:.1f}%",
            delta=None,
        )
    
    with cols[1]:
        completeness = quality_score.get('completeness', 0)
        st.metric(
            "Completeness",
            f"{completeness:.1f}%",
        )
    
    with cols[2]:
        uniqueness = quality_score.get('uniqueness', 0)
        st.metric(
            "Uniqueness",
            f"{uniqueness:.1f}%",
        )
    
    # Additional details in expander
    with st.expander("View details"):
        st.write(f"**Total cells:** {quality_score.get('total_cells', 0):,}")
        st.write(f"**Missing cells:** {quality_score.get('missing_cells', 0):,}")
        st.write(f"**Duplicate rows:** {quality_score.get('duplicate_rows', 0):,}")


def render_data_types_summary(column_types: Dict[str, list]) -> None:
    """Render summary of column types."""
    
    st.subheader("🏷️ Column Types")
    
    cols = st.columns(5)
    
    types_info = [
        ("Numeric", "🔢", column_types.get('numeric', [])),
        ("Categorical", "🏷️", column_types.get('categorical', [])),
        ("DateTime", "📅", column_types.get('datetime', [])),
        ("Text", "📝", column_types.get('text', [])),
        ("Boolean", "✓/✗", column_types.get('boolean', [])),
    ]
    
    for i, (type_name, icon, columns) in enumerate(types_info):
        with cols[i]:
            st.metric(f"{icon} {type_name}", len(columns))
            if columns:
                with st.expander("View"):
                    for col in columns:
                        st.write(f"• {col}")


def render_full_data_preview(
    df: pd.DataFrame,
    column_info: pd.DataFrame,
    quality_score: Dict[str, Any],
    column_types: Dict[str, list]
) -> None:
    """Render complete data preview section."""
    
    # Shape metrics
    render_data_shape(df)
    
    st.divider()
    
    # Data quality
    render_data_quality_card(quality_score)
    
    st.divider()
    
    # Column types summary
    render_data_types_summary(column_types)
    
    st.divider()
    
    # Data preview
    render_data_preview(df)
    
    st.divider()
    
    # Column information
    render_column_info(column_info)
 
