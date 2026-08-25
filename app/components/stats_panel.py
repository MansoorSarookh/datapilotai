"""
DataPilot AI — Stats Panel Component
Statistical test recommender, distribution analyzer, hypothesis builder,
regression analysis, time-series stats, and power analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.modules.stats_engine import (
    recommend_statistical_test,
    analyze_distribution,
    build_hypothesis,
)


def render_stats_panel(df: pd.DataFrame):
    """Render the Statistical Intelligence Engine UI."""
    st.markdown("### 📐 Statistical Intelligence Engine")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔬 Test Recommender",
        "📊 Distribution Analyzer",
        "🧪 Hypothesis Builder",
        "📈 Regression Analysis",
        "⏱️ Time-Series Stats",
        "⚡ Power Analysis",
    ])

    # ── Tab 1: Statistical Test Recommender ───────────────────────────────────
    with tab1:
        st.markdown("**Auto-recommend the right statistical test for any two columns.**")
        col1, col2 = st.columns(2)
        with col1:
            col_a = st.selectbox("Variable 1:", df.columns.tolist(), key="stat_col_a")
        with col2:
            remaining = [c for c in df.columns if c != col_a]
            col_b = st.selectbox("Variable 2:", remaining, key="stat_col_b")

        if st.button("🔬 Run Statistical Test", use_container_width=True):
            with st.spinner("Running analysis..."):
                result = recommend_statistical_test(df, col_a, col_b)

            st.markdown(f"**Test Selected:** `{result.get('test', 'N/A')}`")

            # Result metrics
            r = result.get("result", {})
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Test Statistic", f"{r.get('statistic', r.get('chi2', 'N/A'))}")
            with m_col2:
                p = r.get("p_value", None)
                st.metric("P-Value", f"{p:.4f}" if p is not None else "N/A")
            with m_col3:
                sig = r.get("significant", False)
                st.metric("Significant?", "✅ Yes" if sig else "❌ No")

            # Effect size display
            effect = r.get("effect_size", r.get("cramers_v", r.get("strength", None)))
            if effect:
                st.markdown(f"📏 **Effect Size:** {effect}")

            # Assumptions
            assumptions = result.get("assumptions", {})
            if assumptions:
                with st.expander("📋 Assumptions Check"):
                    for k, v in assumptions.items():
                        if isinstance(v, dict):
                            st.write(f"**{k}:** {v.get('test', '')} — {'✅ Passed' if v.get('passed') else '❌ Failed'} (p={v.get('p_value', 'N/A')})")
                        else:
                            st.write(f"**{k}:** {v}")

            # Interpretation
            st.info(f"📊 **Interpretation:** {result.get('interpretation', '')}")
            if result.get("recommendation"):
                st.success(f"✅ **Recommendation:** {result['recommendation']}")

            # AI Statistical Narrator
            try:
                from app.modules.stats_narrator import narrate_test_result
                with st.expander("🧠 AI Plain-English Explanation", expanded=True):
                    narration = narrate_test_result(result, col_a, col_b)
                    st.markdown(narration)
            except Exception:
                pass

    # ── Tab 2: Distribution Analyzer ──────────────────────────────────────────
    with tab2:
        st.markdown("**Analyze the distribution of any numeric column.**")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            dist_col = st.selectbox("Select column:", numeric_cols, key="dist_col")

            if st.button("📊 Analyze Distribution", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analysis = analyze_distribution(df, dist_col)

                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Mean", f"{analysis['mean']:.3f}")
                    d2.metric("Std Dev", f"{analysis['std']:.3f}")
                    d3.metric("Skewness", f"{analysis['skewness']:.3f}")
                    d4.metric("Kurtosis", f"{analysis['kurtosis']:.3f}")

                    st.markdown(f"- **Shape:** {analysis['skewness_label']}")
                    st.markdown(f"- **Normality ({analysis['normality']['test']}):** {'✅ Normal' if analysis['normality']['passed'] else '❌ Non-normal'} (p={analysis['normality'].get('p_value', 'N/A')})")
                    st.markdown(f"- **IQR Outliers:** {analysis['iqr_outliers']} | **Z-Score Outliers:** {analysis['zscore_outliers']}")
                    st.info(f"💡 **Transform Suggestion:** {analysis['transform_suggestion']}")

                    # AI Distribution Narrator
                    try:
                        from app.modules.stats_narrator import narrate_distribution
                        with st.expander("🧠 AI Plain-English Explanation", expanded=True):
                            narration = narrate_distribution(analysis, dist_col)
                            st.markdown(narration)
                    except Exception:
                        pass

                    # Histogram + KDE
                    series = df[dist_col].dropna()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=series, name="Distribution", nbinsx=40, opacity=0.7, 
                                               marker_color="#6366f1"))
                    fig.update_layout(title=f"Distribution: {dist_col}", xaxis_title=dist_col, yaxis_title="Count",
                                      template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    # Q-Q Plot
                    from scipy import stats as scipy_stats
                    q_theory, q_data = scipy_stats.probplot(series, dist="norm")[:2]
                    qq_fig = go.Figure()
                    qq_fig.add_trace(go.Scatter(x=q_theory[0], y=q_theory[1], mode="markers", name="Data", marker_color="#10b981"))
                    qq_fig.add_trace(go.Scatter(x=[q_theory[0].min(), q_theory[0].max()],
                                                y=[q_theory[0].min() * q_data[0] + q_data[1],
                                                   q_theory[0].max() * q_data[0] + q_data[1]],
                                                mode="lines", name="Normal Line", line=dict(color="#ef4444")))
                    qq_fig.update_layout(title="Q-Q Plot (Normality Check)", template="plotly_dark", height=350)
                    st.plotly_chart(qq_fig, use_container_width=True)

    # ── Tab 3: Hypothesis Builder ─────────────────────────────────────────────
    with tab3:
        st.markdown("**No-code hypothesis testing. Select variables and relationship — we do the rest.**")

        h_col1, h_rel, h_col2 = st.columns([2, 1, 2])
        with h_col1:
            h_var1 = st.selectbox("Variable 1:", df.columns.tolist(), key="h_var1")
        with h_rel:
            relationship = st.selectbox("Relationship:", ["affects", "differs by", "associated with"], key="h_rel")
        with h_col2:
            remaining_h = [c for c in df.columns if c != h_var1]
            h_var2 = st.selectbox("Variable 2:", remaining_h, key="h_var2")

        if st.button("🧪 Build & Test Hypothesis", use_container_width=True):
            with st.spinner("Building hypothesis..."):
                result = build_hypothesis(df, h_var1, relationship, h_var2)

            st.markdown(f"**H₀:** {result['null_hypothesis']}")
            st.markdown(f"**H₁:** {result['alternative_hypothesis']}")
            st.markdown(f"**Test:** `{result['test_used']}`")

            r = result.get("result", {})
            hc1, hc2 = st.columns(2)
            with hc1:
                p = r.get("p_value")
                st.metric("P-Value", f"{p:.4f}" if p else "N/A")
            with hc2:
                significant = r.get("significant", False)
                st.metric("Decision", "Reject H₀" if significant else "Fail to Reject H₀")

            if significant:
                st.success(result["conclusion"])
            else:
                st.warning(result["conclusion"])

            st.info(f"📊 {result.get('interpretation', '')}")
            if result.get("recommendation"):
                st.markdown(f"✅ **Next Step:** {result['recommendation']}")

            # AI Hypothesis Narrator
            try:
                from app.modules.stats_narrator import narrate_hypothesis_result
                with st.expander("🧠 AI Plain-English Explanation", expanded=True):
                    narration = narrate_hypothesis_result(result)
                    st.markdown(narration)
            except Exception:
                pass

    # ── Tab 4: Regression Analysis ────────────────────────────────────────────
    with tab4:
        _render_regression_tab(df)

    # ── Tab 5: Time-Series Statistics ─────────────────────────────────────────
    with tab5:
        _render_timeseries_tab(df)

    # ── Tab 6: Power Analysis ─────────────────────────────────────────────────
    with tab6:
        _render_power_tab(df)


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION TAB
# ══════════════════════════════════════════════════════════════════════════════

def _render_regression_tab(df: pd.DataFrame):
    """Regression analysis UI with full diagnostics."""
    st.markdown("**Run OLS, Logistic, Ridge, Lasso, or Poisson regression with diagnostics.**")

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    target_col = st.selectbox("Target (Dependent) Variable:", all_cols, key="reg_target")
    available_predictors = [c for c in all_cols if c != target_col]
    predictor_cols = st.multiselect(
        "Predictor (Independent) Variables:",
        available_predictors,
        default=available_predictors[:min(5, len(available_predictors))],
        key="reg_predictors",
    )

    reg_type = st.selectbox(
        "Regression Type:",
        ["OLS", "Logistic", "Ridge", "Lasso", "Poisson"],
        key="reg_type",
    )

    if not predictor_cols:
        st.warning("Select at least one predictor variable.")
        return

    if st.button("📈 Run Regression", use_container_width=True, type="primary"):
        with st.spinner(f"Running {reg_type} regression..."):
            try:
                from app.modules.regression_engine import run_regression, compute_residual_diagnostics
                result = run_regression(df, target_col, predictor_cols, reg_type)
            except Exception as e:
                st.error(f"Regression failed: {e}")
                return

        if "error" in result:
            st.error(f"⚠️ {result['error']}")
            return

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("N Obs", f"{result.get('n_observations', 0):,}")
        mc2.metric("N Predictors", f"{result.get('n_predictors', 0)}")

        if reg_type == "OLS":
            mc3.metric("R²", f"{result.get('r_squared', 0):.4f}")
            mc4.metric("Adj R²", f"{result.get('r_squared_adj', 0):.4f}")
        elif reg_type in ("Logistic", "Poisson"):
            mc3.metric("Pseudo R²", f"{result.get('pseudo_r_squared', 0):.4f}")
            mc4.metric("AIC", f"{result.get('aic', 0):.1f}")
        elif reg_type in ("Ridge", "Lasso"):
            mc3.metric("R²", f"{result.get('r_squared', 0):.4f}")

        # Coefficient table
        coef_table = result.get("coefficient_table", [])
        if coef_table:
            st.markdown("**📊 Coefficient Table:**")
            st.dataframe(pd.DataFrame(coef_table), use_container_width=True, hide_index=True)

        # VIF
        vif_data = result.get("vif", [])
        if vif_data:
            with st.expander("📏 Variance Inflation Factor (VIF)"):
                vif_df = pd.DataFrame(vif_data)
                vif_df["Status"] = vif_df["VIF"].apply(
                    lambda v: "🔴 High" if v > 10 else "🟡 Moderate" if v > 5 else "🟢 OK"
                )
                st.dataframe(vif_df, use_container_width=True, hide_index=True)

        # Residual diagnostics
        if reg_type == "OLS":
            try:
                diagnostics = compute_residual_diagnostics(result)
                with st.expander("🔍 Residual Diagnostics", expanded=True):
                    dc1, dc2, dc3 = st.columns(3)
                    norm = diagnostics.get("normality", {})
                    dc1.metric(
                        "Residual Normality",
                        "✅ Normal" if norm.get("normal") else "❌ Non-normal",
                        help=f"{norm.get('test', '')}: p={norm.get('p_value', 'N/A')}",
                    )
                    homo = diagnostics.get("homoscedasticity", {})
                    dc2.metric(
                        "Homoscedasticity",
                        "✅ OK" if homo.get("homoscedastic", True) else "⚠️ Violated",
                        help=f"{homo.get('test', '')}: p={homo.get('p_value', 'N/A')}",
                    )
                    dw = result.get("durbin_watson", None)
                    if dw is not None:
                        dw_ok = 1.5 < dw < 2.5
                        dc3.metric(
                            "Durbin-Watson",
                            f"{dw:.3f}",
                            help="Values near 2 = no autocorrelation",
                        )

                # Residuals vs Fitted plot
                residuals = result.get("residuals")
                fitted = result.get("fitted_values")
                if residuals is not None and fitted is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fitted, y=residuals, mode="markers",
                        marker=dict(color="#667eea", opacity=0.5, size=5),
                        name="Residuals",
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title="Residuals vs Fitted",
                        xaxis_title="Fitted Values",
                        yaxis_title="Residuals",
                        template="plotly_dark",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES TAB
# ══════════════════════════════════════════════════════════════════════════════

def _render_timeseries_tab(df: pd.DataFrame):
    """Time-series statistics: stationarity tests, ACF/PACF, seasonal decomposition."""
    st.markdown("**Stationarity tests, ACF/PACF, and seasonal decomposition.**")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Also check for parseable datetime columns
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(20))
            datetime_cols.append(col)
        except Exception:
            pass

    if not numeric_cols:
        st.warning("No numeric columns found for time-series analysis.")
        return

    ts_col = st.selectbox("Numeric column to analyze:", numeric_cols, key="ts_value_col")

    ts_tab1, ts_tab2, ts_tab3 = st.tabs(["📉 Stationarity", "📊 ACF/PACF", "🔄 Decomposition"])

    # Stationarity
    with ts_tab1:
        if st.button("📉 Run Stationarity Tests", use_container_width=True):
            with st.spinner("Running ADF & KPSS tests..."):
                try:
                    from app.modules.ts_stats_engine import run_stationarity_test
                    result = run_stationarity_test(df[ts_col])
                except Exception as e:
                    st.error(f"Test failed: {e}")
                    return

            if "error" in result:
                st.error(result["error"])
                return

            # ADF results
            adf = result.get("adf", {})
            if "error" not in adf:
                st.markdown("**ADF Test (Augmented Dickey-Fuller):**")
                ac1, ac2 = st.columns(2)
                ac1.metric("ADF Statistic", f"{adf.get('statistic', 'N/A')}")
                ac2.metric("P-Value", f"{adf.get('p_value', 'N/A')}")
                st.markdown(adf.get("interpretation", ""))

            # KPSS results
            kpss = result.get("kpss", {})
            if "error" not in kpss:
                st.markdown("**KPSS Test:**")
                kc1, kc2 = st.columns(2)
                kc1.metric("KPSS Statistic", f"{kpss.get('statistic', 'N/A')}")
                kc2.metric("P-Value", f"{kpss.get('p_value', 'N/A')}")
                st.markdown(kpss.get("interpretation", ""))

            # Combined conclusion
            conclusion = result.get("conclusion", "")
            if conclusion:
                st.info(f"🧭 **Conclusion:** {conclusion}")

    # ACF/PACF
    with ts_tab2:
        nlags = st.slider("Number of lags:", 5, 100, 40, key="acf_lags")
        if st.button("📊 Compute ACF/PACF", use_container_width=True):
            with st.spinner("Computing..."):
                try:
                    from app.modules.ts_stats_engine import compute_acf_pacf
                    result = compute_acf_pacf(df[ts_col], nlags=nlags)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

            if "error" in result:
                st.error(result["error"])
                return

            ci = result.get("confidence_bound", 0)

            # ACF plot
            acf_vals = result.get("acf", [])
            if acf_vals:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF", marker_color="#667eea"))
                fig.add_hline(y=ci, line_dash="dash", line_color="#ef4444")
                fig.add_hline(y=-ci, line_dash="dash", line_color="#ef4444")
                fig.update_layout(title="Autocorrelation Function (ACF)", template="plotly_dark", height=300,
                                  xaxis_title="Lag", yaxis_title="ACF")
                st.plotly_chart(fig, use_container_width=True)

            # PACF plot
            pacf_vals = result.get("pacf", [])
            if pacf_vals:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF", marker_color="#10b981"))
                fig.add_hline(y=ci, line_dash="dash", line_color="#ef4444")
                fig.add_hline(y=-ci, line_dash="dash", line_color="#ef4444")
                fig.update_layout(title="Partial Autocorrelation Function (PACF)", template="plotly_dark", height=300,
                                  xaxis_title="Lag", yaxis_title="PACF")
                st.plotly_chart(fig, use_container_width=True)

            sig_acf = result.get("significant_acf_lags", [])
            sig_pacf = result.get("significant_pacf_lags", [])
            if sig_acf:
                st.info(f"📊 Significant ACF lags: {sig_acf[:10]}")
            if sig_pacf:
                st.info(f"📊 Significant PACF lags: {sig_pacf[:10]}")

    # Decomposition
    with ts_tab3:
        if not datetime_cols:
            st.warning("No datetime columns found. Parse dates first in the Clean tab.")
        else:
            dt_col = st.selectbox("DateTime column:", datetime_cols, key="decomp_dt")
            model = st.selectbox("Decomposition model:", ["additive", "multiplicative"], key="decomp_model")
            period = st.number_input("Period (0=auto-detect):", min_value=0, value=0, key="decomp_period")
            actual_period = period if period > 0 else None

            if st.button("🔄 Run Decomposition", use_container_width=True):
                with st.spinner("Decomposing..."):
                    try:
                        from app.modules.ts_stats_engine import seasonal_decomposition
                        result = seasonal_decomposition(df, dt_col, ts_col, period=actual_period, model=model)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return

                if "error" in result:
                    st.error(result["error"])
                    return

                st.metric("Detected Period", f"{result.get('period', 'N/A')}")
                st.metric("Trend Strength", f"{result.get('trend_strength', 0):.3f}")

                # Plot components
                dates = result.get("dates", [])
                for comp_name, comp_key in [("Trend", "trend"), ("Seasonal", "seasonal"), ("Residual", "residual")]:
                    comp = result.get(comp_key, [])
                    if comp:
                        fig = go.Figure()
                        x = dates[:len(comp)] if dates else list(range(len(comp)))
                        fig.add_trace(go.Scatter(x=x, y=comp, mode="lines", name=comp_name,
                                                line=dict(color="#667eea" if comp_name == "Trend" else "#10b981" if comp_name == "Seasonal" else "#f59e0b")))
                        fig.update_layout(title=f"{comp_name} Component", template="plotly_dark", height=250)
                        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# POWER ANALYSIS TAB
# ══════════════════════════════════════════════════════════════════════════════

def _render_power_tab(df: pd.DataFrame):
    """Statistical power analysis UI."""
    st.markdown("**Compute required sample sizes and statistical power for study planning.**")

    try:
        from app.modules.power_analysis import compute_power_analysis, EFFECT_SIZE_PRESETS
    except ImportError:
        st.error("Power analysis module not available.")
        return

    pc1, pc2 = st.columns(2)
    with pc1:
        test_type = st.selectbox("Test Type:", ["t-test", "anova", "chi-square", "correlation"], key="power_test")
    with pc2:
        mode = st.radio("Mode:", ["Compute Sample Size", "Compute Power"], key="power_mode")

    presets = EFFECT_SIZE_PRESETS.get(test_type, {"Small": 0.2, "Medium": 0.5, "Large": 0.8})
    preset_choice = st.selectbox("Effect Size Preset:", list(presets.keys()), index=1, key="power_preset")
    effect_size = st.number_input("Effect Size:", min_value=0.01, max_value=3.0, value=presets[preset_choice], step=0.05, key="power_effect")
    alpha = st.number_input("Significance Level (α):", min_value=0.001, max_value=0.20, value=0.05, step=0.01, key="power_alpha")

    if mode == "Compute Sample Size":
        power = st.number_input("Desired Power:", min_value=0.50, max_value=0.99, value=0.80, step=0.05, key="power_val")
        n_input = None
    else:
        n_input = st.number_input("Sample Size (per group):", min_value=5, max_value=10000, value=50, step=10, key="power_n")
        power = 0.80

    if st.button("⚡ Run Power Analysis", use_container_width=True, type="primary"):
        with st.spinner("Computing..."):
            result = compute_power_analysis(
                test_type=test_type,
                effect_size=effect_size,
                alpha=alpha,
                power=power if n_input is None else None,
                n=n_input,
            )

        if "error" in result:
            st.error(f"⚠️ {result['error']}")
            return

        # Display results
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Effect Size", f"{result.get('effect_size', 0):.3f}")
        rc2.metric("Effect Label", result.get("effect_label", ""))

        if n_input is None:
            req_n = result.get("required_n_per_group", result.get("required_n", 0))
            total_n = result.get("total_n", req_n)
            rc3.metric("Required N (per group)", f"{req_n:,}")
            st.info(f"📊 You need at least **{req_n:,} samples per group** (total: ~{total_n:,}) to detect a {result.get('effect_label', '').lower()} effect with {power:.0%} power.")
        else:
            achieved_power = result.get("achieved_power", 0)
            sufficient = result.get("sufficient", False)
            rc3.metric("Achieved Power", f"{achieved_power:.1%}")
            if sufficient:
                st.success(f"✅ Sample size of {n_input:,} provides **sufficient power** ({achieved_power:.1%} ≥ 80%).")
            else:
                st.warning(f"⚠️ Sample size of {n_input:,} provides **insufficient power** ({achieved_power:.1%} < 80%). Consider increasing sample size.")

        # Power curve
        power_curve = result.get("power_curve", {})
        n_vals = power_curve.get("n_values", [])
        p_vals = power_curve.get("power_values", [])
        if n_vals and p_vals:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=n_vals, y=p_vals, mode="lines", name="Power",
                                     line=dict(color="#667eea", width=3)))
            fig.add_hline(y=0.80, line_dash="dash", line_color="#ef4444",
                          annotation_text="80% Power Threshold")
            fig.update_layout(
                title="Power Curve (Sample Size vs Statistical Power)",
                xaxis_title="Sample Size (per group)",
                yaxis_title="Statistical Power",
                template="plotly_dark", height=400,
                yaxis=dict(range=[0, 1.05]),
            )
            st.plotly_chart(fig, use_container_width=True)
