"""
Sidebar component for the EDA application.
"""

import streamlit as st
from typing import Optional, List, Tuple, Any
from ..modules.file_parser import get_supported_extensions
from ..config import (
    UNIVARIATE_CHARTS,
    BIVARIATE_CHARTS,
    MULTIVARIATE_CHARTS,
    TIME_SERIES_CHARTS,
    PLOTLY_THEMES,
)


def render_file_upload() -> Any:
    """Render file upload widget and return uploaded file."""
    st.sidebar.header("📁 Data Upload")
    
    supported_exts = get_supported_extensions()
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=supported_exts,
        help=f"Supported formats: {', '.join(supported_exts)}",
    )
    
    if uploaded_file:
        st.sidebar.success(f"📄 {uploaded_file.name}")
    
    return uploaded_file


def render_analysis_type_selector() -> str:
    """Render analysis type selector."""
    st.sidebar.header("📊 Analysis Type")
    
    analysis_type = st.sidebar.radio(
        "Select analysis type:",
        options=['Univariate', 'Bivariate', 'Multivariate', 'Time Series'],
        help="Choose the type of analysis to perform",
    )
    
    return analysis_type


def render_chart_selector(analysis_type: str) -> str:
    """Render chart type selector based on analysis type."""
    st.sidebar.header("🎨 Chart Selection")
    
    chart_options = {
        'Univariate': UNIVARIATE_CHARTS,
        'Bivariate': BIVARIATE_CHARTS,
        'Multivariate': MULTIVARIATE_CHARTS,
        'Time Series': TIME_SERIES_CHARTS,
    }
    
    charts = chart_options.get(analysis_type, UNIVARIATE_CHARTS)
    
    chart_type = st.sidebar.selectbox(
        "Select chart type:",
        options=charts,
    )
    
    return chart_type


def render_column_selectors(
    columns: List[str],
    numeric_columns: List[str],
    categorical_columns: List[str],
    datetime_columns: List[str],
    analysis_type: str
) -> dict:
    """Render column selection widgets based on analysis type."""
    
    st.sidebar.header("📋 Column Selection")
    selections = {}
    
    if analysis_type == 'Univariate':
        all_cols = numeric_columns + categorical_columns
        if all_cols:
            selections['column'] = st.sidebar.selectbox(
                "Select column:",
                options=all_cols,
            )
    
    elif analysis_type == 'Bivariate':
        if numeric_columns:
            selections['x_column'] = st.sidebar.selectbox(
                "X-axis column:",
                options=columns,
            )
            selections['y_column'] = st.sidebar.selectbox(
                "Y-axis column:",
                options=columns,
            )
        if categorical_columns:
            selections['color_column'] = st.sidebar.selectbox(
                "Color by (optional):",
                options=['None'] + categorical_columns,
            )
            if selections['color_column'] == 'None':
                selections['color_column'] = None
    
    elif analysis_type == 'Multivariate':
        if len(numeric_columns) >= 2:
            selections['columns'] = st.sidebar.multiselect(
                "Select columns (2-5):",
                options=numeric_columns,
                max_selections=5,
            )
            if categorical_columns:
                selections['color_column'] = st.sidebar.selectbox(
                    "Color by (optional):",
                    options=['None'] + categorical_columns,
                )
                if selections['color_column'] == 'None':
                    selections['color_column'] = None
    
    elif analysis_type == 'Time Series':
        if datetime_columns:
            selections['datetime_column'] = st.sidebar.selectbox(
                "DateTime column:",
                options=datetime_columns,
            )
        elif columns:
            selections['datetime_column'] = st.sidebar.selectbox(
                "DateTime column (will attempt parsing):",
                options=columns,
            )
        
        if numeric_columns:
            selections['value_column'] = st.sidebar.selectbox(
                "Value column:",
                options=numeric_columns,
            )
    
    return selections


def render_chart_options() -> dict:
    """Render additional chart configuration options."""
    st.sidebar.header("⚙️ Chart Options")
    
    options = {}
    
    # Theme selection
    options['theme'] = st.sidebar.selectbox(
        "Theme:",
        options=PLOTLY_THEMES,
        index=2,  # Default to plotly_dark
    )
    
    # Advanced options in expander
    with st.sidebar.expander("Advanced Options"):
        options['show_grid'] = st.checkbox("Show grid", value=True)
        options['histogram_bins'] = st.slider("Histogram bins", 10, 100, 30)
        options['top_n_categories'] = st.slider("Top N categories", 5, 50, 20)
        options['add_trendline'] = st.checkbox("Add trendline", value=False)
        options['sample_data'] = st.checkbox("Sample large datasets", value=True)
        options['sample_size'] = st.number_input("Sample size", 1000, 100000, 10000, step=1000)
    
    return options


def render_time_series_options() -> dict:
    """Render time series specific options."""
    st.sidebar.header("⏱️ Time Series Options")
    
    options = {}
    
    options['resample_freq'] = st.sidebar.selectbox(
        "Resample frequency:",
        options=['None', 'H', 'D', 'W', 'M', 'Q', 'Y'],
        format_func=lambda x: {
            'None': 'No resampling',
            'H': 'Hourly',
            'D': 'Daily',
            'W': 'Weekly',
            'M': 'Monthly',
            'Q': 'Quarterly',
            'Y': 'Yearly',
        }.get(x, x),
    )
    
    options['add_range_slider'] = st.sidebar.checkbox("Add range slider", value=True)
    
    options['rolling_window'] = st.sidebar.number_input(
        "Rolling window size:",
        min_value=1,
        max_value=365,
        value=7,
    )
    
    options['show_rolling_stats'] = st.sidebar.checkbox(
        "Show rolling statistics",
        value=False,
    )
    
    return options


def render_sidebar() -> Tuple[Any, str, str, dict, dict]:
    """
    Render complete sidebar and return all selections.
    
    Returns:
        Tuple of (uploaded_file, analysis_type, chart_type, column_selections, options)
    """
    # File upload
    uploaded_file = render_file_upload()
    
    st.sidebar.divider()
    
    # Analysis type
    analysis_type = render_analysis_type_selector()
    
    st.sidebar.divider()
    
    # Chart type
    chart_type = render_chart_selector(analysis_type)
    
    st.sidebar.divider()
    
    # Chart options
    options = render_chart_options()
    
    # Time series options (if applicable)
    if analysis_type == 'Time Series':
        st.sidebar.divider()
        ts_options = render_time_series_options()
        options.update(ts_options)
    
    return uploaded_file, analysis_type, chart_type, options
 
