"""
Visualization Engine - Plotly chart generators for EDA.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from ..config import DEFAULT_COLOR_PALETTE


def apply_theme(fig: go.Figure, theme: str = 'plotly_dark') -> go.Figure:
    """Apply consistent theme to figure."""
    fig.update_layout(
        template=theme,
        font=dict(family="Inter, sans-serif"),
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ============== Univariate Charts ==============

def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    color: str = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create histogram for numeric column."""
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        color_discrete_sequence=[color or DEFAULT_COLOR_PALETTE[0]],
        title=f'Distribution of {column}',
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Frequency',
        bargap=0.05,
    )
    return apply_theme(fig, theme)


def create_box_plot(
    df: pd.DataFrame,
    column: str,
    color: str = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create box plot for outlier detection."""
    fig = px.box(
        df,
        y=column,
        color_discrete_sequence=[color or DEFAULT_COLOR_PALETTE[1]],
        title=f'Box Plot of {column}',
    )
    fig.update_layout(yaxis_title=column)
    return apply_theme(fig, theme)


def create_violin_plot(
    df: pd.DataFrame,
    column: str,
    color: str = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create violin plot for distribution shape."""
    fig = px.violin(
        df,
        y=column,
        box=True,
        color_discrete_sequence=[color or DEFAULT_COLOR_PALETTE[2]],
        title=f'Violin Plot of {column}',
    )
    fig.update_layout(yaxis_title=column)
    return apply_theme(fig, theme)


def create_bar_chart(
    df: pd.DataFrame,
    column: str,
    top_n: int = 20,
    color: str = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create bar chart for categorical frequency."""
    value_counts = df[column].value_counts().head(top_n)
    
    fig = px.bar(
        x=value_counts.index.astype(str),
        y=value_counts.values,
        color_discrete_sequence=[color or DEFAULT_COLOR_PALETTE[4]],
        title=f'Top {top_n} Categories in {column}',
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Count',
        xaxis_tickangle=-45,
    )
    return apply_theme(fig, theme)


def create_pie_chart(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create pie chart for category proportions."""
    value_counts = df[column].value_counts().head(top_n)
    
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index.astype(str),
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'Distribution of {column}',
        hole=0.4,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return apply_theme(fig, theme)


def create_kde_plot(
    df: pd.DataFrame,
    column: str,
    color: str = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create KDE (density) plot."""
    fig = px.histogram(
        df,
        x=column,
        marginal='violin',
        color_discrete_sequence=[color or DEFAULT_COLOR_PALETTE[0]],
        title=f'Density Plot of {column}',
    )
    return apply_theme(fig, theme)


# ============== Bivariate Charts ==============

def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    trendline: str = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create scatter plot for numeric vs numeric."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        trendline=trendline,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'{y_col} vs {x_col}',
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return apply_theme(fig, theme)


def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create line chart."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'{y_col} over {x_col}',
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return apply_theme(fig, theme)


def create_grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    barmode: str = 'group',
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create grouped bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode=barmode,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'{y_col} by {x_col} (grouped by {color_col})',
    )
    fig.update_layout(xaxis_tickangle=-45)
    return apply_theme(fig, theme)


def create_heatmap(
    correlation_matrix: pd.DataFrame,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create correlation heatmap."""
    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Correlation Matrix',
    )
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
    )
    return apply_theme(fig, theme)


# ============== Multivariate Charts ==============

def create_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create 3D scatter plot."""
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'3D Scatter: {x_col} × {y_col} × {z_col}',
    )
    return apply_theme(fig, theme)


def create_parallel_coordinates(
    df: pd.DataFrame,
    columns: List[str],
    color_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create parallel coordinates plot."""
    fig = px.parallel_coordinates(
        df,
        dimensions=columns,
        color=color_col,
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Parallel Coordinates Plot',
    )
    return apply_theme(fig, theme)


def create_bubble_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    color_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create bubble chart."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'Bubble Chart: {x_col} × {y_col} (size: {size_col})',
    )
    return apply_theme(fig, theme)


def create_sunburst(
    df: pd.DataFrame,
    path: List[str],
    values: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create sunburst chart for hierarchical data."""
    fig = px.sunburst(
        df,
        path=path,
        values=values,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title='Sunburst Chart',
    )
    return apply_theme(fig, theme)


def create_treemap(
    df: pd.DataFrame,
    path: List[str],
    values: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create treemap for hierarchical data."""
    fig = px.treemap(
        df,
        path=path,
        values=values,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title='Treemap',
    )
    return apply_theme(fig, theme)


def create_pair_plot(
    df: pd.DataFrame,
    columns: List[str],
    color_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create pair plot (scatter matrix)."""
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title='Pair Plot',
    )
    fig.update_traces(diagonal_visible=False)
    return apply_theme(fig, theme)


# ============== Time Series Charts ==============

def create_time_series_line(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    add_range_slider: bool = True,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create time series line chart with optional range slider."""
    fig = px.line(
        df,
        x=datetime_col,
        y=value_col,
        color_discrete_sequence=[DEFAULT_COLOR_PALETTE[0]],
        title=f'{value_col} Over Time',
    )
    
    if add_range_slider:
        fig.update_xaxes(rangeslider_visible=True)
    
    return apply_theme(fig, theme)


def create_area_chart(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create area chart for time series."""
    fig = px.area(
        df,
        x=datetime_col,
        y=value_col,
        color_discrete_sequence=[DEFAULT_COLOR_PALETTE[0]],
        title=f'{value_col} Over Time (Area)',
    )
    return apply_theme(fig, theme)


def create_candlestick(
    df: pd.DataFrame,
    datetime_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create candlestick chart for OHLC data."""
    fig = go.Figure(data=[go.Candlestick(
        x=df[datetime_col],
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col],
    )])
    
    fig.update_layout(
        title='Candlestick Chart (OHLC)',
        xaxis_rangeslider_visible=False,
    )
    return apply_theme(fig, theme)


def create_rolling_stats_chart(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    rolling_mean_col: str,
    rolling_std_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create chart with original values and rolling statistics."""
    fig = go.Figure()
    
    # Original values
    fig.add_trace(go.Scatter(
        x=df[datetime_col],
        y=df[value_col],
        mode='lines',
        name='Original',
        line=dict(color=DEFAULT_COLOR_PALETTE[0], width=1),
        opacity=0.5,
    ))
    
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=df[datetime_col],
        y=df[rolling_mean_col],
        mode='lines',
        name='Rolling Mean',
        line=dict(color=DEFAULT_COLOR_PALETTE[3], width=2),
    ))
    
    # Rolling std (if provided)
    if rolling_std_col and rolling_std_col in df.columns:
        upper = df[rolling_mean_col] + df[rolling_std_col]
        lower = df[rolling_mean_col] - df[rolling_std_col]
        
        fig.add_trace(go.Scatter(
            x=df[datetime_col],
            y=upper,
            mode='lines',
            name='Upper Band',
            line=dict(width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df[datetime_col],
            y=lower,
            mode='lines',
            name='Lower Band',
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(width=0),
            showlegend=False,
        ))
    
    fig.update_layout(title=f'{value_col} with Rolling Statistics')
    return apply_theme(fig, theme)


# ============== Animated Charts ==============

def create_animated_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    animation_frame: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create animated scatter plot."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        animation_frame=animation_frame,
        color=color_col,
        size=size_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'Animated Scatter: {y_col} vs {x_col}',
    )
    return apply_theme(fig, theme)


def create_animated_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    animation_frame: str,
    color_col: Optional[str] = None,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create animated bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        animation_frame=animation_frame,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        title=f'Animated Bar Chart: {y_col} by {x_col}',
    )
    return apply_theme(fig, theme)


# ============== Missing Values Visualization ==============

def create_missing_heatmap(
    df: pd.DataFrame,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create heatmap showing missing value patterns."""
    missing_matrix = df.isnull().astype(int)
    
    # Sample if too large
    if len(missing_matrix) > 100:
        missing_matrix = missing_matrix.sample(n=100, random_state=42)
    
    fig = px.imshow(
        missing_matrix.T,
        color_continuous_scale=['#667eea', '#f5576c'],
        aspect='auto',
        title='Missing Values Pattern (Dark = Missing)',
    )
    fig.update_layout(
        xaxis_title='Row Index',
        yaxis_title='Column',
    )
    return apply_theme(fig, theme)


def create_missing_bar(
    missing_df: pd.DataFrame,
    theme: str = 'plotly_dark'
) -> go.Figure:
    """Create bar chart of missing values by column."""
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing %',
        color='Missing %',
        color_continuous_scale='Reds',
        title='Missing Values by Column',
    )
    fig.update_layout(xaxis_tickangle=-45)
    return apply_theme(fig, theme)
 
