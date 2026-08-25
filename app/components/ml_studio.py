"""
DataPilot AI — ML Studio Component (v3.0)
Integrates: Model Arena, SHAP Explainability, HPO (Optuna), Cluster Profiler.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.modules.ml_advisor import (
    assess_ml_readiness,
    suggest_feature_engineering,
    run_ml_pipeline,
    run_kmeans,
    predict_model_feasibility,
)


def render_ml_studio(df: pd.DataFrame):
    """Render the ML Studio UI."""
    st.markdown("### 🎯 ML Studio")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Readiness Advisor",
        "⚗️ Train Model",
        "🏟️ Model Arena",
        "🔵 Clustering",
        "⚙️ Hyperparameter Tuning",
    ])

    # ── Tab 1: ML Readiness ───────────────────────────────────────────────────
    with tab1:
        _render_readiness(df)

    # ── Tab 2: Train Model + SHAP ─────────────────────────────────────────────
    with tab2:
        _render_train_model(df)

    # ── Tab 3: Model Arena ────────────────────────────────────────────────────
    with tab3:
        _render_model_arena(df)

    # ── Tab 4: Clustering + Profiler ──────────────────────────────────────────
    with tab4:
        _render_clustering(df)

    # ── Tab 5: Hyperparameter Tuning ──────────────────────────────────────────
    with tab5:
        _render_hpo(df)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: READINESS ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def _render_readiness(df: pd.DataFrame):
    st.markdown("**Assess your dataset's readiness for machine learning.**")
    all_cols = df.columns.tolist()
    target_col = st.selectbox("Select Target Column (optional):", ["None"] + all_cols, key="ml_target_readiness")
    target = None if target_col == "None" else target_col

    if st.button("🔍 Assess ML Readiness", use_container_width=True):
        with st.spinner("Analyzing ML readiness..."):
            readiness = assess_ml_readiness(df, target)
            feasibility = predict_model_feasibility(df, target) if target else None

        score = readiness["score"]
        score_color = "#10b981" if score >= 75 else "#f59e0b" if score >= 50 else "#ef4444"
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:12px;margin-bottom:16px;">
          <div style="font-size:52px;font-weight:900;color:{score_color};">{score}/100</div>
          <div style="color:#94a3b8;font-size:14px;">ML Readiness Score</div>
          <div style="color:white;font-size:16px;font-weight:600;margin-top:4px;">{readiness['problem_type'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        if readiness["checks"]:
            for c in readiness["checks"]:
                st.success(c)
        if readiness["warnings"]:
            for w in readiness["warnings"]:
                st.warning(w)
        if readiness["errors"]:
            for e in readiness["errors"]:
                st.error(e)

        st.markdown("---")
        st.markdown("**💡 Feature Engineering Suggestions:**")
        suggestions = suggest_feature_engineering(df)
        if suggestions:
            for sug in suggestions[:6]:
                with st.expander(f"[{sug['impact']}] {sug['type']} — `{sug['column']}`"):
                    st.write(sug["suggestion"])
                    st.code(sug.get("code_hint", ""), language="python")
        else:
            st.info("No feature engineering suggestions at this time.")

        if feasibility:
            st.markdown("---")
            st.markdown(f"**🤖 Recommended Algorithms:** {', '.join(feasibility['recommended_models'])}")
            perf = feasibility.get("expected_performance", {})
            if perf:
                for k, v in perf.items():
                    st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRAIN MODEL + SHAP
# ══════════════════════════════════════════════════════════════════════════════

def _render_train_model(df: pd.DataFrame):
    st.markdown("**Train an ML model directly in your browser with SHAP explainability.**")

    all_cols = df.columns.tolist()
    target_col = st.selectbox("Target Column:", all_cols, key="ml_target_train")

    # Auto-detect problem type
    if target_col:
        target_data = df[target_col].dropna()
        is_class = not pd.api.types.is_numeric_dtype(target_data) or target_data.nunique() <= 20
        problem_type = "Classification" if is_class else "Regression"
        st.info(f"🔍 Detected Problem Type: **{problem_type}** ({target_data.nunique()} unique values)")

    algo_options = {
        "Classification": ["Random Forest", "Logistic Regression", "XGBoost"],
        "Regression": ["Random Forest", "Linear Regression", "XGBoost"],
    }
    algo = st.selectbox("Algorithm:", algo_options.get(problem_type, ["Random Forest"]), key="ml_algo")
    test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="ml_test_size")

    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        with st.spinner(f"Training {algo}... Please wait ⏳"):
            try:
                result = run_ml_pipeline(df, target_col, algorithm=algo, test_size=test_size)
                st.session_state["ml_result"] = result
            except Exception as e:
                st.error(f"Training failed: {e}")
                result = None

        if result:
            st.success("✅ Model trained successfully!")
            _display_ml_results(result, df, target_col, algo)

    # Display previous results if available
    elif st.session_state.get("ml_result"):
        result = st.session_state["ml_result"]
        _display_ml_results(result, df, target_col, algo)


def _display_ml_results(result, df, target_col, algo):
    """Display ML training results with SHAP explanation."""
    # Metrics
    metrics = result.get("metrics", {})
    metric_cols = st.columns(min(len(metrics), 4))
    for i, (k, v) in enumerate(list(metrics.items())[:4]):
        metric_cols[i].metric(k.replace("_", " ").upper(), f"{v:.4f}")

    # Cross-validation
    cv = result.get("cv_scores", [])
    if cv:
        st.markdown(f"**5-Fold CV Score:** {result['cv_mean']:.4f} ± {np.std(cv):.4f}")

    # Feature importance
    fi = result.get("feature_importance", {})
    if fi:
        st.markdown("**Top Feature Importances:**")
        fi_df = pd.DataFrame.from_dict(fi, orient="index", columns=["Importance"]).sort_values("Importance", ascending=True).tail(15)
        fig = px.bar(fi_df, x="Importance", y=fi_df.index, orientation="h",
                     title="Feature Importance", color="Importance",
                     color_continuous_scale="viridis")
        fig.update_layout(template="plotly_dark", height=400, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # SHAP Explainability
    model = result.get("model")
    X_train = result.get("X_train")
    X_test = result.get("X_test")

    if model is not None and X_train is not None and X_test is not None:
        with st.expander("🧠 SHAP Explainability (Model Interpretation)", expanded=False):
            if st.button("🔍 Compute SHAP Values", key="btn_compute_shap"):
                with st.spinner("Computing SHAP values (this may take a moment)..."):
                    try:
                        from app.modules.shap_explainer import compute_shap_values, create_shap_summary_data
                        shap_result = compute_shap_values(model, X_train, X_test)
                        st.session_state["shap_result"] = shap_result
                    except ImportError:
                        st.error("SHAP not installed. Install with: `pip install shap`")
                        return
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")
                        return

            shap_result = st.session_state.get("shap_result")
            if shap_result and "error" not in shap_result:
                _render_shap_display(shap_result)
            elif shap_result and "error" in shap_result:
                st.error(f"SHAP error: {shap_result['error']}")

    # Download predictions
    predictions = result.get("predictions", [])
    if len(predictions) > 0:
        pred_df = pd.DataFrame({"prediction": predictions})
        csv_bytes = pred_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Predictions (CSV)",
            csv_bytes, "datapilot_predictions.csv", "text/csv",
            key="dl_predictions",
        )


def _render_shap_display(shap_result):
    """Render SHAP analysis visualizations."""
    from app.modules.shap_explainer import create_shap_summary_data, get_single_prediction_explanation

    # Feature importance from SHAP
    fi = shap_result.get("feature_importance", {})
    if fi:
        st.markdown("**🎯 SHAP Feature Importance (more reliable than model-intrinsic):**")
        shap_fi_df = pd.DataFrame([
            {"Feature": k, "Mean |SHAP|": round(v, 4)} for k, v in fi.items()
        ]).sort_values("Mean |SHAP|", ascending=True).tail(15)

        fig = px.bar(shap_fi_df, x="Mean |SHAP|", y="Feature", orientation="h",
                     title="SHAP Feature Importance", color="Mean |SHAP|",
                     color_continuous_scale="Plasma")
        fig.update_layout(template="plotly_dark", height=400, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Summary data
    summary_df = create_shap_summary_data(shap_result)
    if summary_df is not None:
        with st.expander("📊 SHAP Summary Table"):
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Single prediction explanation
    st.markdown("**🔍 Explain Single Prediction:**")
    max_idx = len(shap_result.get("shap_values", [])) - 1
    if max_idx >= 0:
        row_idx = st.number_input("Row index:", 0, max_idx, 0, key="shap_row_idx")
        explanation = get_single_prediction_explanation(shap_result, int(row_idx))
        if explanation:
            st.markdown(f"**Base value:** {explanation['base_value']:.4f} → **Prediction:** {explanation['prediction']:.4f}")
            for feat in explanation["features"][:10]:
                emoji = "🔼" if feat["shap_value"] > 0 else "🔽"
                st.write(f"{emoji} **{feat['feature']}** = {feat['feature_value']} → SHAP: {feat['shap_value']:+.4f} ({feat['direction']})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL ARENA
# ══════════════════════════════════════════════════════════════════════════════

def _render_model_arena(df: pd.DataFrame):
    st.markdown("**🏟️ Train multiple models and compare them side-by-side.**")

    all_cols = df.columns.tolist()
    target_col = st.selectbox("Target Column:", all_cols, key="arena_target")

    # Detect problem type
    target_data = df[target_col].dropna()
    is_class = not pd.api.types.is_numeric_dtype(target_data) or target_data.nunique() <= 20
    problem_type = "classification" if is_class else "regression"
    st.info(f"🔍 Problem Type: **{problem_type.title()}** ({target_data.nunique()} unique target values)")

    try:
        from app.modules.model_arena import CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS
        algo_dict = CLASSIFICATION_ALGORITHMS if is_class else REGRESSION_ALGORITHMS
    except ImportError:
        st.error("Model Arena module not available.")
        return

    available_algos = list(algo_dict.keys())
    selected_algos = st.multiselect(
        "Select algorithms to compare:",
        available_algos,
        default=available_algos[:min(5, len(available_algos))],
        key="arena_algos",
    )

    test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="arena_test_size")

    if not selected_algos:
        st.warning("Select at least one algorithm.")
        return

    if st.button("🏟️ Run Model Arena", use_container_width=True, type="primary"):
        progress_bar = st.progress(0, text="Initializing...")

        def progress_callback(pct, text):
            progress_bar.progress(min(pct, 1.0), text=text)

        with st.spinner("Training models..."):
            try:
                from app.modules.model_arena import run_model_arena
                arena_result = run_model_arena(
                    df, target_col, selected_algos,
                    test_size=test_size,
                    progress_callback=progress_callback,
                )
                st.session_state["arena_result"] = arena_result
            except Exception as e:
                st.error(f"Arena failed: {e}")
                return

        progress_bar.empty()

    # Display results
    arena_result = st.session_state.get("arena_result")
    if arena_result:
        _display_arena_results(arena_result)


def _display_arena_results(arena_result):
    """Display model arena comparison."""
    leaderboard = arena_result.get("leaderboard", [])
    best = arena_result.get("best_model")
    problem_type = arena_result.get("problem_type", "classification")

    if best:
        st.success(f"🏆 **Best Model: {best}**")

    # Leaderboard table
    st.markdown("**📊 Model Leaderboard:**")
    rows = []
    for entry in leaderboard:
        if "error" in entry:
            rows.append({
                "🏅 Rank": "—",
                "Algorithm": entry["algorithm"],
                "Status": f"❌ {entry['error']}",
            })
        else:
            metrics = entry.get("metrics", {})
            row = {
                "🏅 Rank": "🥇" if entry["algorithm"] == best else "",
                "Algorithm": entry["algorithm"],
            }
            for k, v in metrics.items():
                row[k.upper()] = f"{v:.4f}"
            if entry.get("cv_mean") is not None:
                row["CV Mean"] = f"{entry['cv_mean']:.4f}"
            rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Metric comparison chart
    valid = [e for e in leaderboard if "error" not in e]
    if len(valid) >= 2:
        primary_key = list(valid[0].get("metrics", {}).keys())[0] if valid[0].get("metrics") else None
        if primary_key:
            chart_data = pd.DataFrame([
                {"Algorithm": e["algorithm"], primary_key.upper(): e["metrics"][primary_key]}
                for e in valid
            ]).sort_values(primary_key.upper(), ascending=False)

            fig = px.bar(chart_data, x="Algorithm", y=primary_key.upper(),
                         title=f"Model Comparison — {primary_key.upper()}",
                         color=primary_key.upper(), color_continuous_scale="viridis")
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance comparison (top model)
    if valid:
        top = valid[0]
        fi = top.get("feature_importance", {})
        if fi:
            with st.expander(f"📊 Feature Importance — {top['algorithm']}"):
                fi_df = pd.DataFrame.from_dict(fi, orient="index", columns=["Importance"]).sort_values("Importance", ascending=True).tail(10)
                fig = px.bar(fi_df, x="Importance", y=fi_df.index, orientation="h",
                             color="Importance", color_continuous_scale="viridis")
                fig.update_layout(template="plotly_dark", height=350, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CLUSTERING + PROFILER
# ══════════════════════════════════════════════════════════════════════════════

def _render_clustering(df: pd.DataFrame):
    st.markdown("**Discover natural groupings with K-Means clustering and AI cluster profiles.**")
    n_clusters = st.slider("Number of clusters:", 2, 10, 3, key="km_n")

    if st.button("🔵 Run K-Means Clustering", use_container_width=True):
        with st.spinner("Clustering..."):
            result = run_kmeans(df, n_clusters=n_clusters)

        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state["cluster_result"] = result
            st.session_state["cluster_labels"] = result.get("labels", [])

    # Display results
    result = st.session_state.get("cluster_result")
    if result and "error" not in result:
        sil = result.get("silhouette_score", 0)
        km_c1, km_c2 = st.columns(2)
        km_c1.metric("Silhouette Score", f"{sil:.4f}", help="Closer to 1.0 = better clusters")
        km_c2.metric("Inertia", f"{result['inertia']:.2f}")

        st.markdown("**Cluster Sizes:**")
        cluster_df = pd.DataFrame.from_dict(result["cluster_counts"], orient="index", columns=["Count"]).reset_index()
        cluster_df.columns = ["Cluster", "Count"]
        fig = px.bar(cluster_df, x="Cluster", y="Count", title="Cluster Distribution",
                     color="Count", color_continuous_scale="viridis")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # ── Cluster Profiler ──────────────────────────────────────────────
        labels = st.session_state.get("cluster_labels", [])
        if len(labels) == len(df):
            with st.expander("🧬 AI Cluster Profiler — Persona Generation", expanded=False):
                if st.button("🧬 Generate Cluster Profiles", key="btn_cluster_profile"):
                    with st.spinner("Profiling clusters..."):
                        try:
                            from app.modules.cluster_profiler import profile_clusters, generate_ai_personas
                            profiles = profile_clusters(df, labels)
                            st.session_state["cluster_profiles"] = profiles

                            # Try AI personas
                            try:
                                personas = generate_ai_personas(profiles, df)
                                st.session_state["cluster_personas"] = personas
                            except Exception:
                                st.session_state["cluster_personas"] = {}

                        except ImportError:
                            st.error("Cluster profiler module not available.")
                        except Exception as e:
                            st.error(f"Profiling failed: {e}")

                # Display profiles
                profiles = st.session_state.get("cluster_profiles")
                if profiles:
                    for name, profile in profiles.get("profiles", {}).items():
                        st.markdown(f"#### {name} ({profile['size']} members, {profile['percentage']}%)")

                        # Persona
                        persona = st.session_state.get("cluster_personas", {}).get(name, profile.get("persona", ""))
                        if persona:
                            st.info(f"🧬 {persona}")

                        # Distinguishing features
                        dist = profile.get("distinguishing_features", [])
                        if dist:
                            for d in dist[:5]:
                                emoji = "📈" if d["direction"] == "high" else "📉"
                                st.write(f"{emoji} **{d['feature']}**: {d['value']} (z={d['z_score']:+.2f})")

                        st.markdown("---")

            # Download clustered dataset
            df_clustered = df.copy()
            df_clustered["cluster"] = labels
            csv_bytes = df_clustered.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Clustered Dataset",
                csv_bytes, "datapilot_clustered.csv", "text/csv",
                key="dl_clustered",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

def _render_hpo(df: pd.DataFrame):
    st.markdown("**⚙️ Optimize model hyperparameters with Optuna (Bayesian optimization).**")

    all_cols = df.columns.tolist()
    target_col = st.selectbox("Target Column:", all_cols, key="hpo_target")

    # Detect problem type
    target_data = df[target_col].dropna()
    is_class = not pd.api.types.is_numeric_dtype(target_data) or target_data.nunique() <= 20

    try:
        from app.modules.model_arena import CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS
        algo_dict = CLASSIFICATION_ALGORITHMS if is_class else REGRESSION_ALGORITHMS
    except ImportError:
        st.error("Model Arena module not available.")
        return

    algo = st.selectbox("Algorithm to optimize:", list(algo_dict.keys()), key="hpo_algo")

    hc1, hc2 = st.columns(2)
    with hc1:
        n_trials = st.slider("Number of trials:", 10, 200, 50, 10, key="hpo_trials")
    with hc2:
        cv_folds = st.slider("CV Folds:", 2, 10, 5, key="hpo_cv")

    # Show search space
    try:
        from app.modules.hpo_engine import get_search_space
        space = get_search_space(algo, is_class)
        if space:
            with st.expander("📋 Search Space"):
                for param, config in space.items():
                    st.write(f"- **{param}**: {config}")
    except Exception:
        pass

    if st.button("⚙️ Start Hyperparameter Tuning", use_container_width=True, type="primary"):
        progress_bar = st.progress(0, text="Initializing Optuna...")

        def progress_cb(pct, text):
            progress_bar.progress(min(pct, 1.0), text=text)

        with st.spinner(f"Optimizing {algo} ({n_trials} trials)..."):
            try:
                from app.modules.model_arena import prepare_data
                from app.modules.hpo_engine import run_hpo

                X_train, X_test, y_train, y_test, feature_cols, _ = prepare_data(df, target_col)
                hpo_result = run_hpo(
                    X_train, y_train, algo,
                    is_classification=is_class,
                    n_trials=n_trials,
                    cv_folds=cv_folds,
                    progress_callback=progress_cb,
                )
                st.session_state["hpo_result"] = hpo_result
            except ImportError:
                st.error("HPO requires `optuna`. Install with: `pip install optuna`")
                return
            except Exception as e:
                st.error(f"HPO failed: {e}")
                return

        progress_bar.empty()

    # Display HPO results
    hpo_result = st.session_state.get("hpo_result")
    if hpo_result:
        if "error" in hpo_result:
            st.error(hpo_result["error"])
            return

        st.success(f"🏆 **Best Score: {hpo_result['best_score']:.4f}** after {hpo_result['n_trials']} trials")

        # Best parameters
        st.markdown("**🔧 Best Hyperparameters:**")
        best_params = hpo_result.get("best_params", {})
        # Filter out internal params
        display_params = {k: v for k, v in best_params.items() if not k.endswith("_none")}
        for k, v in display_params.items():
            st.write(f"- **{k}**: `{v}`")

        # Optimization history plot
        opt_history = hpo_result.get("optimization_history", [])
        if opt_history:
            hist_df = pd.DataFrame(opt_history)
            hist_df = hist_df.dropna(subset=["value"])
            if not hist_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scattiser(
                    x=hist_df["trial"], y=hist_df["value"],
                    mode="lines+markers", name="Score",
                    line=dict(color="#667eea", width=2),
                    marker=dict(size=4),
                ))
                # Running best
                hist_df["best_so_far"] = hist_df["value"].cummax()
                fig.add_trace(go.Scatter(
                    x=hist_df["trial"], y=hist_df["best_so_far"],
                    mode="lines", name="Best So Far",
                    line=dict(color="#10b981", width=2, dash="dash"),
                ))
                fig.update_layout(
                    title="Optimization History",
                    xaxis_title="Trial", yaxis_title="Score",
                    template="plotly_dark", height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Code snippet for best params
        with st.expander("📋 Copy Best Parameters (Python)"):
            code = f"# Best hyperparameters for {algo}\nbest_params = {display_params}\n"
            st.code(code, language="python")
