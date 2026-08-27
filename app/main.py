"""
DataPilot AI — Main Application
The AI Data Intelligence Copilot
Version: 3.0 | Author: Mansoor Sarookh
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DataPilot AI — AI Data Intelligence Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto", # Options: "expanded", "collapsed", or "auto"
    
)


# ── CSS loader ────────────────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Imports after page config ─────────────────────────────────────────────────
from app.modules.file_parser import parse_file, get_supported_extensions
from app.modules.eda_engine import (
    get_column_info,
    get_descriptive_stats,
    get_missing_analysis,
    get_correlation_matrix,
    get_column_types_summary,
    detect_outliers,
    get_categorical_stats,
)
from app.modules.time_series import detect_datetime_columns, is_time_series, calculate_rolling_statistics, get_time_series_summary
from app.modules.viz_engine import (
    create_histogram,
    create_box_plot,
    create_violin_plot,
    create_bar_chart,
    create_pie_chart,
    create_kde_plot,
    create_scatter_plot,
    create_line_chart,
    create_grouped_bar,
    create_heatmap,
    create_3d_scatter,
    create_parallel_coordinates,
    create_bubble_chart,
    create_sunburst,
    create_treemap,
    create_pair_plot,
    create_time_series_line,
    create_area_chart,
    create_rolling_stats_chart,
    create_missing_heatmap,
    create_missing_bar,
)
from app.modules.trust_score import compute_trust_score
from app.modules.session_manager import get_session_manager

# Components
from app.components.ai_chat import render_ai_chat
from app.components.trust_score_display import render_trust_score
from app.components.ml_studio import render_ml_studio
from app.components.stats_panel import render_stats_panel
from app.components.clean_panel import render_clean_panel
from app.components.report_panel import render_report_panel
from app.components.data_preview import render_full_data_preview
from app.components.statistics import render_statistics_panel
from app.components.export import create_export_panel, create_data_export_panel

from app.config import PLOTLY_THEMES

# ── Session state initialization ──────────────────────────────────────────────
session_mgr = get_session_manager()

for key, default in {
    "df": None,
    "file_name": None,
    "trust_score": None,
    "chat_history": [],
    "cleaned_df": None,
    "cleaning_ops": [],
    "audit_log": [],
    "df_snapshots": [],
    "type_issues": None,
    "anomaly_result": None,
    "fuzzy_result": None,
    "cluster_result": None,
    "cluster_labels": [],
    "cluster_profiles": None,
    "cluster_personas": {},
    "arena_result": None,
    "hpo_result": None,
    "ml_result": None,
    "shap_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="datapilot-header">
    <div class="header-logo">🧠</div>
    <div class="header-text">
        <h1 class="header-title">DataPilot AI</h1>
        <p class="header-subtitle">The AI Data Intelligence Copilot — Understand · Clean · Model · Decide</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILE UPLOAD & SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=get_supported_extensions(),
        help="CSV, Excel, JSON, Parquet, PDF tables, Word tables, and more.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.session_state.file_name != uploaded_file.name:
            with st.spinner("🔄 Parsing file..."):
                try:
                    df, status = parse_file(uploaded_file)
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.cleaned_df = None
                    st.session_state.cleaning_ops = []
                    st.session_state.chat_history = []
                    st.session_state.audit_log = []

                    with st.spinner("🛡️ Computing Trust Score..."):
                        st.session_state.trust_score = compute_trust_score(df)
                    st.success(status)
                except Exception as e:
                    st.error(f"❌ Parse error: {e}")
                    st.session_state.df = None

    # Dataset info
    if st.session_state.df is not None:
        df = st.session_state.df
        trust = st.session_state.trust_score or {}
        score = trust.get("overall", 0)
        label = trust.get("label", "Computing...")
        color = trust.get("color", "gray")
        color_emoji = {"green": "🟢", "orange": "🟡", "red": "🔴"}.get(color, "⚪")

        st.markdown(f"""
        <div class="sidebar-stat-card">
            <div class="stat-row"><span>📄 File</span><strong>{df.shape[0]:,} × {df.shape[1]}</strong></div>
            <div class="stat-row"><span>🛡️ Trust</span><strong>{color_emoji} {score:.0%}</strong></div>
            <div class="stat-row"><span>❓ Missing</span><strong>{df.isna().sum().sum():,}</strong></div>
            <div class="stat-row"><span>🔢 Numeric</span><strong>{len(df.select_dtypes(include='number').columns)}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Theme
        st.markdown("**🎨 Chart Theme**")
        theme = st.selectbox("Theme:", PLOTLY_THEMES, index=2, label_visibility="collapsed")

        st.divider()

        # LLM Provider selector
        st.markdown("**🤖 AI Provider**")
        try:
            from app.modules.llm_router import get_llm_router, PROVIDERS
            router = get_llm_router()
            provider_options = ["Auto (Fallback)"] + [PROVIDERS[p]["display"] for p in PROVIDERS]
            provider_keys = ["auto"] + list(PROVIDERS.keys())
            selected_idx = st.selectbox(
                "LLM Provider:",
                range(len(provider_options)),
                format_func=lambda i: provider_options[i],
                key="llm_provider_select",
                label_visibility="collapsed",
            )
            if selected_idx > 0:
                router.set_preferred_provider(provider_keys[selected_idx])
            else:
                router.set_preferred_provider(None)
        except Exception:
            st.caption("LLM Router not available")

        st.divider()

        if st.button("♻️ Force App Refresh", use_container_width=True, help="Clears session state and reloads the app"):
            st.session_state.clear()
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:8px;color:#94a3b8;font-size:11px;">
        Made with 🧠 by <strong>Mansoor Sarookh</strong><br>
        DataPilot AI v3.0
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN (no file uploaded)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    st.markdown("""
    <div class="welcome-hero">
        <h2>🚀 Upload a Dataset to Get Started</h2>
        <p>DataPilot AI transforms raw data into insights, statistical conclusions, ML-ready features, and shareable reports — in minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "🤖", "AI Copilot", "Ask questions about your data in plain English"),
        (c2, "🛡️", "Trust Score", "Instant data quality assessment with 5-dimension scoring"),
        (c3, "🎯", "ML Studio", "Train 10+ ML algorithms with SHAP explainability"),
        (c4, "📄", "Reports", "PDF, interactive HTML dashboards & Jupyter notebook export"),
    ]:
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Supported Formats")
    fmt_c1, fmt_c2, fmt_c3 = st.columns(3)
    with fmt_c1:
        st.markdown("**Tabular Data:** CSV · Excel · TSV · ODS · Parquet · Feather · JSON")
    with fmt_c2:
        st.markdown("**Documents:** PDF (tables) · Word (tables) · HTML tables")
    with fmt_c3:
        st.markdown("**Formats:** Up to 500MB · Auto-encoding detection · Smart type inference")

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS — DATA LOADED
# ══════════════════════════════════════════════════════════════════════════════
df = st.session_state.df
trust = st.session_state.trust_score or compute_trust_score(df)
file_name = st.session_state.file_name or "dataset"

# Quick Trust Score badge in header area
score = trust.get("overall", 0)
score_color = "#10b981" if score > 0.80 else "#f59e0b" if score > 0.60 else "#ef4444"
st.markdown(f"""
<div class="trust-banner">
    🛡️ <strong>Dataset Trust Score:</strong> <span style="color:{score_color};font-weight:800;">{score:.0%}</span> — {trust.get('label', '')}
    &nbsp;|&nbsp; 📄 <strong>{file_name}</strong> &nbsp;|&nbsp; 📐 {df.shape[0]:,} × {df.shape[1]}
</div>
""", unsafe_allow_html=True)

# Column type info
column_types = get_column_types_summary(df)
numeric_cols = column_types.get("numeric", [])
categorical_cols = column_types.get("categorical", [])
datetime_cols = detect_datetime_columns(df)

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab_data, tab_analyze, tab_ai, tab_stats, tab_clean, tab_ml, tab_export = st.tabs([
    "📁 Overview",
    "📊 Visualize",
    "🤖 AI Copilot",
    "📐 Statistics",
    "🧹 Clean",
    "🎯 ML Studio",
    "📥 Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    # Trust Score widget
    render_trust_score(trust)
    st.divider()

    # Data preview
    st.markdown("#### 📋 Dataset Preview")
    column_info = get_column_info(df)
    quality_score = {
        "overall_score": round(score * 100, 1),
        "completeness": round(trust.get("dimensions", {}).get("completeness", score) * 100, 1),
        "uniqueness": round(trust.get("dimensions", {}).get("uniqueness", score) * 100, 1),
        "total_cells": df.shape[0] * df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    render_full_data_preview(df, column_info, quality_score, column_types)

    st.divider()

    # Statistics
    st.markdown("#### 📈 Descriptive Statistics")
    descriptive_stats = get_descriptive_stats(df)
    missing_df = get_missing_analysis(df)
    correlation = get_correlation_matrix(df)
    outliers = detect_outliers(df)
    categorical_stats = get_categorical_stats(df)
    render_statistics_panel(descriptive_stats, missing_df, correlation, outliers, categorical_stats)

    # ── VIF Analysis (NEW v3.0) ───────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.divider()
        st.markdown("#### 📐 Variance Inflation Factor (VIF)")
        try:
            from app.modules.eda_engine import compute_vif
            vif_df = compute_vif(df)
            if vif_df is not None and not vif_df.empty:
                def _vif_color(val):
                    if val < 5:
                        return "background-color: #065f46; color: white"
                    elif val < 10:
                        return "background-color: #92400e; color: white"
                    else:
                        return "background-color: #991b1b; color: white"

                try:
                    # pandas >= 1.3 uses map, older uses applymap
                    styled = vif_df.style.map(_vif_color, subset=["VIF"])
                except AttributeError:
                    styled = vif_df.style.applymap(_vif_color, subset=["VIF"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                high_vif = vif_df[vif_df["VIF"] > 10]
                if not high_vif.empty:
                    st.warning(f"⚠️ **High multicollinearity detected** in: {', '.join(high_vif['Feature'].tolist())}. Consider removing or combining these features.")
                elif (vif_df["VIF"] > 5).any():
                    st.info("ℹ️ Moderate multicollinearity detected in some features. Monitor these during modeling.")
                else:
                    st.success("✅ No significant multicollinearity detected.")
            else:
                st.info("VIF requires at least 2 numeric columns with no missing values.")
        except Exception as e:
            st.caption(f"VIF computation skipped: {e}")

    # ── Correlation Heatmap with method toggle (ENHANCED v3.0) ────────────
    if not correlation.empty:
        st.divider()
        st.markdown("#### 🔗 Correlation Matrix")

        corr_method = st.radio(
            "Method:",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True,
            key="corr_method_toggle",
        )
        corr_matrix = get_correlation_matrix(df, method=corr_method.lower())
        fig = create_heatmap(corr_matrix, theme=theme)
        st.plotly_chart(fig, use_container_width=True)
        create_export_panel(fig, f"correlation_{corr_method.lower()}", "corr_heat")

    # Missing values visualization
    if missing_df["Missing Count"].sum() > 0:
        st.divider()
        st.markdown("#### ❓ Missing Values")
        c1, c2 = st.columns(2)
        with c1:
            fig_bar = create_missing_bar(missing_df, theme=theme)
            st.plotly_chart(fig_bar, use_container_width=True)
            create_export_panel(fig_bar, "missing_values_bar", "miss_bar")
        with c2:
            fig_heat = create_missing_heatmap(df, theme=theme)
            st.plotly_chart(fig_heat, use_container_width=True)
            create_export_panel(fig_heat, "missing_values_heatmap", "miss_heat")

    # ── GDPR / PII Scanner (NEW v3.0) ─────────────────────────────────────
    st.divider()
    st.markdown("#### 🔒 GDPR / PII Scanner")
    with st.expander("🔒 Scan dataset for Personally Identifiable Information (PII)", expanded=False):
        if st.button("🔍 Run PII Scan", key="btn_gdpr_scan", use_container_width=True):
            with st.spinner("Scanning for PII/GDPR-sensitive data..."):
                try:
                    from app.modules.gdpr_scanner import scan_for_pii
                    pii_result = scan_for_pii(df)
                    st.session_state["pii_result"] = pii_result
                except ImportError:
                    st.error("GDPR scanner module not available.")
                except Exception as e:
                    st.error(f"PII scan failed: {e}")

        pii_result = st.session_state.get("pii_result")
        if pii_result:
            overall_risk = pii_result.get("overall_risk", "Low")
            risk_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Critical": "#dc2626"}
            risk_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🚨"}

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Overall Risk", f"{risk_emojis.get(overall_risk, '⚪')} {overall_risk}")
            rc2.metric("PII Columns", f"{pii_result.get('total_pii_columns', 0)}/{pii_result.get('total_columns', 0)}")

            risk_summary = pii_result.get("risk_summary", {})
            rc3.metric("Critical/High", f"{risk_summary.get('Critical', 0) + risk_summary.get('High', 0)}")

            # Per-column details
            column_risks = pii_result.get("column_risks", {})
            flagged = {k: v for k, v in column_risks.items() if v.get("risk_level") != "Low"}
            if flagged:
                st.markdown("**Flagged Columns:**")
                for col_name, col_info in flagged.items():
                    risk = col_info.get("risk_level", "Unknown")
                    emoji = risk_emojis.get(risk, "⚪")
                    detections = col_info.get("detections", [])
                    det_types = ", ".join([d.get("type", "") for d in detections[:3]])
                    remediations = col_info.get("remediation", [])

                    with st.expander(f"{emoji} `{col_name}` — **{risk}** risk"):
                        st.write(f"**Detections:** {det_types}")
                        for det in detections:
                            if "match_percentage" in det:
                                st.write(f"  - {det['type']}: {det['match_percentage']}% match rate")
                            else:
                                st.write(f"  - {det['type']} ({det.get('method', 'name_pattern')})")
                        if remediations:
                            st.info(f"💡 **Remediation:** {'; '.join(remediations)}")
            else:
                st.success("✅ No PII/GDPR-sensitive data detected!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    # ── AI Chart Recommender (NEW v3.0) ───────────────────────────────────
    try:
        from app.modules.chart_recommender import recommend_charts
        with st.expander("🤖 AI Chart Recommender — Get smart chart suggestions", expanded=False):
            rec_cols = st.multiselect(
                "Select columns for recommendation:",
                df.columns.tolist(),
                default=df.columns.tolist()[:min(3, len(df.columns))],
                key="chart_rec_cols",
            )
            if rec_cols and st.button("🔍 Recommend Charts", key="btn_recommend"):
                recommendations = recommend_charts(df, rec_cols)
                if recommendations:
                    for i, rec in enumerate(recommendations[:3]):
                        st.markdown(f"**{i+1}. {rec.get('chart', rec.get('chart_type', 'Unknown'))}** — {rec.get('reason', '')}")
                else:
                    st.info("No specific recommendations for these columns.")
    except Exception:
        pass  # Chart recommender not critical

    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Univariate", "Bivariate", "Multivariate", "Time Series"],
        key="viz_analysis_type",
    )

    if analysis_type == "Univariate":
        st.markdown("#### 📊 Univariate Analysis")
        all_cols = numeric_cols + categorical_cols
        if not all_cols:
            st.warning("No suitable columns found.")
        else:
            v1, v2 = st.columns([2, 1])
            with v1:
                selected_col = st.selectbox("Column:", all_cols)
            with v2:
                chart_type = st.selectbox("Chart:", ["Histogram", "Box Plot", "Violin Plot", "Bar Chart", "Pie Chart", "KDE Plot"])

            with st.expander("⚙️ Options"):
                bins = st.slider("Bins:", 10, 100, 30)
                top_n = st.slider("Top N:", 5, 50, 15)

            is_num = selected_col in numeric_cols
            if chart_type == "Histogram" and is_num:
                fig = create_histogram(df, selected_col, bins=bins, theme=theme)
            elif chart_type == "Box Plot" and is_num:
                fig = create_box_plot(df, selected_col, theme=theme)
            elif chart_type == "Violin Plot" and is_num:
                fig = create_violin_plot(df, selected_col, theme=theme)
            elif chart_type == "KDE Plot" and is_num:
                fig = create_kde_plot(df, selected_col, theme=theme)
            elif chart_type == "Bar Chart":
                fig = create_bar_chart(df, selected_col, top_n=top_n, theme=theme)
            elif chart_type == "Pie Chart":
                fig = create_pie_chart(df, selected_col, top_n=top_n, theme=theme)
            else:
                st.warning(f"{chart_type} requires {'numeric' if not is_num else 'categorical'} data.")
                fig = create_bar_chart(df, selected_col, top_n=top_n, theme=theme)

            st.plotly_chart(fig, use_container_width=True)
            create_export_panel(fig, f"univariate_{selected_col}", "uni")

    elif analysis_type == "Bivariate":
        st.markdown("#### 📈 Bivariate Analysis")
        if len(df.columns) < 2:
            st.warning("Need at least 2 columns.")
        else:
            b1, b2, b3 = st.columns(3)
            with b1:
                x_col = st.selectbox("X-axis:", df.columns.tolist(), key="bv_x")
            with b2:
                y_col = st.selectbox("Y-axis:", df.columns.tolist(), index=min(1, len(df.columns)-1), key="bv_y")
            with b3:
                chart_type = st.selectbox("Chart:", ["Scatter Plot", "Line Chart", "Grouped Bar", "Heatmap"], key="bv_chart")

            with st.expander("⚙️ Options"):
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols, key="bv_color")
                color_col = None if color_col == "None" else color_col
                add_trendline = st.checkbox("Add trendline") if chart_type == "Scatter Plot" else False

            if chart_type == "Scatter Plot":
                fig = create_scatter_plot(df, x_col, y_col, color_col=color_col, trendline="ols" if add_trendline else None, theme=theme)
            elif chart_type == "Line Chart":
                fig = create_line_chart(df, x_col, y_col, color_col=color_col, theme=theme)
            elif chart_type == "Grouped Bar":
                fig = create_grouped_bar(df, x_col, y_col, color_col or (categorical_cols[0] if categorical_cols else x_col), theme=theme)
            else:
                fig = create_heatmap(get_correlation_matrix(df), theme=theme)

            st.plotly_chart(fig, use_container_width=True)
            create_export_panel(fig, f"bivariate_{x_col}_{y_col}", "bv")

    elif analysis_type == "Multivariate":
        st.markdown("#### 🔮 Multivariate Analysis")
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            chart_type = st.selectbox("Chart type:", ["Pair Plot", "3D Scatter", "Parallel Coordinates", "Bubble Chart", "Treemap", "Sunburst"], key="mv_chart")

            if chart_type in ["Pair Plot", "Parallel Coordinates"]:
                selected_cols = st.multiselect("Columns (2-5):", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], max_selections=5)
                if len(selected_cols) < 2:
                    st.warning("Select at least 2 columns.")
                    st.stop()
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols, key="mv_color")
                color_col = None if color_col == "None" else color_col
                fig = create_pair_plot(df, selected_cols, color_col=color_col, theme=theme) if chart_type == "Pair Plot" else create_parallel_coordinates(df, selected_cols, color_col=color_col, theme=theme)

            elif chart_type == "3D Scatter":
                if len(numeric_cols) < 3:
                    st.warning("Need 3+ numeric columns.")
                else:
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        x_col = st.selectbox("X:", numeric_cols, key="3d_x")
                    with mc2:
                        y_col = st.selectbox("Y:", numeric_cols, index=min(1, len(numeric_cols)-1), key="3d_y")
                    with mc3:
                        z_col = st.selectbox("Z:", numeric_cols, index=min(2, len(numeric_cols)-1), key="3d_z")
                    color_col = st.selectbox("Color by:", ["None"] + categorical_cols, key="3d_color")
                    color_col = None if color_col == "None" else color_col
                    fig = create_3d_scatter(df, x_col, y_col, z_col, color_col=color_col, theme=theme)

            elif chart_type == "Bubble Chart":
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    x_col = st.selectbox("X:", numeric_cols, key="bb_x")
                with bc2:
                    y_col = st.selectbox("Y:", numeric_cols, index=min(1, len(numeric_cols)-1), key="bb_y")
                with bc3:
                    size_col = st.selectbox("Size:", numeric_cols, index=min(2, len(numeric_cols)-1), key="bb_size")
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols, key="bb_color")
                color_col = None if color_col == "None" else color_col
                fig = create_bubble_chart(df, x_col, y_col, size_col, color_col=color_col, theme=theme)

            else:  # Treemap / Sunburst
                if not categorical_cols:
                    st.warning("Need categorical columns.")
                else:
                    path_cols = st.multiselect("Hierarchy path:", categorical_cols, default=[categorical_cols[0]])
                    value_col = st.selectbox("Values:", ["Count"] + numeric_cols, key="tree_val")
                    if path_cols:
                        if value_col == "Count":
                            plot_df = df.groupby(path_cols).size().reset_index(name="Count")
                            value_col = "Count"
                        else:
                            plot_df = df.groupby(path_cols)[value_col].sum().reset_index()
                        fig = create_treemap(plot_df, path_cols, values=value_col, theme=theme) if chart_type == "Treemap" else create_sunburst(plot_df, path_cols, values=value_col, theme=theme)

            if 'fig' in dir():
                st.plotly_chart(fig, use_container_width=True)
                create_export_panel(fig, f"multivariate_{chart_type.lower().replace(' ', '_')}", "mv")

    elif analysis_type == "Time Series":
        st.markdown("#### ⏱️ Time Series Analysis")
        if datetime_cols:
            dt_col = st.selectbox("DateTime column:", datetime_cols)
        else:
            dt_col = st.selectbox("DateTime column (will try to parse):", df.columns.tolist())
            st.info("No datetime columns auto-detected. Will attempt parsing.")

        if not numeric_cols:
            st.warning("Need numeric columns.")
        else:
            val_col = st.selectbox("Value column:", numeric_cols, key="ts_val")
            ts_info = is_time_series(df, dt_col)

            if ts_info.get("is_time_series"):
                st.success(f"✅ Valid time series | Frequency: {ts_info.get('frequency', 'irregular')}")
            else:
                st.warning(f"⚠️ {ts_info.get('reason', 'May not be a proper time series')}")

            chart_type = st.selectbox("Chart:", ["Line with Range Slider", "Area Chart", "Rolling Statistics"], key="ts_chart")

            with st.expander("⚙️ Options"):
                rolling_window = st.slider("Rolling window:", 2, 30, 7)

            try:
                plot_df = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(plot_df[dt_col]):
                    # Use robust datetime parser (v3.0)
                    try:
                        from app.modules.time_series import robust_datetime_parse
                        plot_df[dt_col] = robust_datetime_parse(plot_df[dt_col])
                    except (ImportError, Exception):
                        plot_df[dt_col] = pd.to_datetime(plot_df[dt_col], format="mixed", dayfirst=False, errors="coerce")
                plot_df = plot_df.dropna(subset=[dt_col]).sort_values(dt_col)
            except Exception as e:
                st.error(f"Datetime parse error: {e}")
                st.stop()

            if chart_type == "Line with Range Slider":
                fig = create_time_series_line(plot_df, dt_col, val_col, add_range_slider=True, theme=theme)
            elif chart_type == "Area Chart":
                fig = create_area_chart(plot_df, dt_col, val_col, theme=theme)
            else:
                rolling_df = calculate_rolling_statistics(plot_df, dt_col, val_col, windows=[rolling_window])
                fig = create_rolling_stats_chart(rolling_df, dt_col, "Value", f"Rolling_Mean_{rolling_window}", f"Rolling_Std_{rolling_window}", theme=theme)

            st.plotly_chart(fig, use_container_width=True)
            create_export_panel(fig, f"timeseries_{val_col}", "ts")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AI COPILOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    render_ai_chat(df)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    render_stats_panel(df)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: CLEAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_clean:
    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = None
    if "cleaning_ops" not in st.session_state:
        st.session_state["cleaning_ops"] = []

    render_clean_panel(df)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: ML STUDIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    render_ml_studio(df)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    render_report_panel(
        df=df,
        trust_score=trust,
        file_name=file_name.replace(".", "_"),
    )
