"""
DataPilot AI — Data Cleaning Panel Component (v3.0)
Integrates: audit trail, anomaly detection, type fixer, fuzzy dedup.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from app.modules.cleaner import (
    detect_cleaning_opportunities,
    apply_cleaning_pipeline,
)


def render_clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Render the data cleaning UI. Returns the cleaned dataframe."""

    # ── Ensure session state keys exist ─────────────────────────────
    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = None
    if "cleaning_ops" not in st.session_state:
        st.session_state["cleaning_ops"] = []

    st.markdown("### 🧹 Data Cleaning Studio")
    st.caption("Clean your dataset interactively. Preview changes before applying.")

    # ── Dataset Status ──────────────────────────────────────────────
    current_df = st.session_state.get("cleaned_df")
    if current_df is None or (isinstance(current_df, pd.DataFrame) and current_df.empty):
        current_df = df

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Rows", f"{current_df.shape[0]:,}")
    col2.metric("📊 Columns", f"{current_df.shape[1]:,}")
    col3.metric("❓ Missing", f"{int(current_df.isna().sum().sum()):,}")

    # ── Audit Trail + Undo Button ───────────────────────────────────
    _render_audit_trail(df)

    st.divider()

    # ── Main Cleaning Tabs ──────────────────────────────────────────
    ctab1, ctab2, ctab3, ctab4, ctab5 = st.tabs([
        "🔧 Standard Cleaning",
        "🔍 Type Fixer",
        "🎯 Anomaly Detection",
        "🔗 Fuzzy Dedup",
        "📋 Audit Log",
    ])

    # ── Tab 1: Standard Cleaning ────────────────────────────────────
    with ctab1:
        cleaned_df = _render_standard_cleaning(df, current_df)

    # ── Tab 2: Type Fixer ───────────────────────────────────────────
    with ctab2:
        _render_type_fixer(current_df)

    # ── Tab 3: Anomaly Detection ────────────────────────────────────
    with ctab3:
        _render_anomaly_detection(current_df)

    # ── Tab 4: Fuzzy Dedup ──────────────────────────────────────────
    with ctab4:
        _render_fuzzy_dedup(current_df)

    # ── Tab 5: Audit Log ────────────────────────────────────────────
    with ctab5:
        _render_audit_log()

    # ── Download Section ────────────────────────────────────────────
    dataset_for_download = st.session_state.get("cleaned_df")
    if dataset_for_download is None or (isinstance(dataset_for_download, pd.DataFrame) and dataset_for_download.empty):
        dataset_for_download = df

    if dataset_for_download is not None and not dataset_for_download.empty:
        st.markdown("### 📥 Export Dataset")
        col1, col2 = st.columns(2)
        with col1:
            csv_bytes = dataset_for_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", csv_bytes, "datapilot_dataset.csv",
                mime="text/csv", key="dl_csv",
            )
        with col2:
            buffer = io.BytesIO()
            try:
                dataset_for_download.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)
                st.download_button(
                    "Download Excel", buffer.getvalue(), "datapilot_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xlsx",
                )
            except Exception as e:
                st.error(f"Excel export failed: {e}")

    return dataset_for_download


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _render_standard_cleaning(df: pd.DataFrame, current_df: pd.DataFrame):
    """Original cleaning pipeline with issue detection."""

    with st.spinner("Scanning for cleaning opportunities..."):
        issues = detect_cleaning_opportunities(df)

    if not issues:
        st.success("✅ No major data quality issues detected! Dataset looks clean.")
    else:
        st.markdown(f"**Found {len(issues)} area(s) needing attention:**")
        if "missing_values" in issues:
            st.warning(f"⚠️ Missing values in {len(issues['missing_values'])} column(s)")
        if "duplicates" in issues:
            st.warning(f"⚠️ {issues['duplicates']} duplicate rows")
        if "outliers" in issues:
            st.warning(f"⚠️ Outliers detected in {len(issues['outliers'])} column(s)")
        if "constant_columns" in issues:
            st.warning(f"⚠️ Constant columns: {issues['constant_columns']}")

    st.divider()
    config = {}

    # Missing values
    st.markdown("**1️⃣ Missing Value Handling**")
    missing_info = issues.get("missing_values", {})
    if missing_info:
        missing_strategy = {}
        for col, info in missing_info.items():
            mc1, mc2 = st.columns([2, 1])
            with mc1:
                st.write(f"`{col}` — {info['pct']}% missing ({info['count']} rows)")
            with mc2:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                options = (
                    ["mean", "median", "mode", "ffill", "bfill", "zero", "drop"]
                    if is_numeric
                    else ["mode", "ffill", "bfill", "drop"]
                )
                method = st.selectbox("Method:", options, key=f"miss_{col}")
                missing_strategy[col] = method
        config["missing_strategy"] = missing_strategy
    else:
        st.success("✅ No missing values")

    # Duplicates
    st.markdown("**2️⃣ Duplicate Rows**")
    dup_count = issues.get("duplicates", 0)
    if dup_count:
        config["remove_duplicates"] = st.checkbox(f"Remove {dup_count} duplicate rows", value=True)
    else:
        st.success("✅ No duplicates")

    # Constant columns
    if issues.get("constant_columns"):
        st.markdown("**3️⃣ Constant Columns**")
        config["drop_constants"] = st.checkbox(
            f"Drop constant columns: {issues['constant_columns']}", value=True
        )

    # Outliers
    st.markdown("**4️⃣ Outlier Handling**")
    outlier_cols = issues.get("outliers", {})
    if outlier_cols:
        st.write(f"Outliers detected in: {list(outlier_cols.keys())[:5]}")
        outlier_method = st.selectbox(
            "Outlier strategy:",
            ["None", "iqr_clip", "zscore_clip", "winsorize", "iqr_drop"],
            index=1,
        )
        if outlier_method != "None":
            config["outlier_strategy"] = outlier_method
    else:
        st.success("✅ No significant outliers")

    # Encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.markdown("**5️⃣ Categorical Encoding**")
        encode_method = st.selectbox(
            "Encoding method:", ["None", "onehot", "label", "frequency"], index=0
        )
        if encode_method != "None":
            config["encode_method"] = encode_method

    # Scaling
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.markdown("**6️⃣ Feature Scaling**")
        scale_method = st.selectbox(
            "Scaling method:", ["None", "minmax", "zscore", "robust"], index=0
        )
        if scale_method != "None":
            config["scale_method"] = scale_method

    st.divider()

    # Preview + Apply Cleaning
    if st.button("👁️ Preview Cleaned Data", use_container_width=True, key="btn_preview_clean"):
        with st.spinner("Applying cleaning steps..."):
            try:
                # Save snapshot for undo
                _save_audit_snapshot(current_df)

                cleaned_df, ops_log = apply_cleaning_pipeline(df, config)
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_ops"] = ops_log

                # Record in audit trail
                _record_audit("Standard Cleaning Pipeline", ops_log,
                              len(df), len(cleaned_df), df.shape[1], cleaned_df.shape[1])

            except Exception as e:
                st.error(f"Cleaning error: {e}")
                cleaned_df = df.copy()
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_ops"] = []

    # Show metrics if cleaned
    if st.session_state["cleaned_df"] is not None:
        cleaned_df = st.session_state["cleaned_df"]
        st.markdown("**Before vs After:**")
        c1, c2 = st.columns(2)
        c1.metric("Original Rows", f"{len(df):,}")
        c2.metric("Cleaned Rows", f"{len(cleaned_df):,}", delta=f"{len(cleaned_df) - len(df):,}")
        c1.metric("Original Cols", f"{df.shape[1]}")
        c2.metric("Cleaned Cols", f"{cleaned_df.shape[1]}", delta=f"{cleaned_df.shape[1] - df.shape[1]:,}")

        if st.session_state["cleaning_ops"]:
            with st.expander("📋 Operations Applied"):
                for op in st.session_state["cleaning_ops"]:
                    st.write(f"→ {op}")

        st.dataframe(cleaned_df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TYPE FIXER
# ══════════════════════════════════════════════════════════════════════════════

def _render_type_fixer(current_df: pd.DataFrame):
    """Smart column type detection and fixing."""
    st.markdown("**🔍 Auto-detect and fix wrong column types** (currency strings, date text, boolean-as-int, etc.)")

    if st.button("🔎 Scan for Type Issues", use_container_width=True, key="btn_type_scan"):
        try:
            from app.modules.type_fixer import detect_type_issues
            issues = detect_type_issues(current_df)
        except ImportError:
            st.error("Type fixer module not available.")
            return
        except Exception as e:
            st.error(f"Scan error: {e}")
            return

        st.session_state["type_issues"] = issues

    issues = st.session_state.get("type_issues", [])

    if not issues:
        if st.session_state.get("type_issues") is not None:
            st.success("✅ No type issues detected — all columns look correct!")
        return

    st.markdown(f"**Found {len(issues)} type issue(s):**")

    fixes_to_apply = {}
    for i, issue in enumerate(issues):
        col = issue["column"]
        with st.expander(
            f"⚠️ `{col}` — {issue['current_type']} → **{issue['suggested_type']}** "
            f"(confidence: {issue['confidence']:.0%})",
            expanded=i < 3,
        ):
            st.write(f"**Reason:** {issue['reason']}")
            st.write(f"**Sample values:** {issue['sample_values'][:3]}")
            if st.checkbox(f"Fix `{col}`", value=True, key=f"fix_type_{col}"):
                fixes_to_apply[col] = issue["fix_type"]

    if fixes_to_apply:
        if st.button("🔧 Apply Selected Fixes", use_container_width=True, key="btn_apply_types"):
            try:
                from app.modules.type_fixer import apply_type_fixes
                _save_audit_snapshot(current_df)
                fixed_df, logs = apply_type_fixes(current_df, fixes_to_apply)
                st.session_state["cleaned_df"] = fixed_df
                _record_audit("Type Fix", logs, len(current_df), len(fixed_df),
                              current_df.shape[1], fixed_df.shape[1])
                for log in logs:
                    st.write(log)
                st.success("✅ Type fixes applied!")
                st.rerun()
            except Exception as e:
                st.error(f"Fix error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_anomaly_detection(current_df: pd.DataFrame):
    """Advanced anomaly detection with multiple methods."""
    st.markdown("**🎯 Detect anomalies using Isolation Forest, LOF, Z-Score, or Ensemble.**")

    numeric_cols = current_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for anomaly detection.")
        return

    ac1, ac2 = st.columns(2)
    with ac1:
        method = st.selectbox(
            "Detection Method:",
            ["isolation_forest", "lof", "zscore", "ensemble"],
            format_func=lambda x: {
                "isolation_forest": "🌲 Isolation Forest",
                "lof": "📍 Local Outlier Factor",
                "zscore": "📊 Z-Score",
                "ensemble": "🎯 Ensemble (IF + LOF)",
            }.get(x, x),
            key="anomaly_method",
        )
    with ac2:
        contamination = st.slider(
            "Expected Outlier Fraction:",
            0.01, 0.20, 0.05, 0.01,
            key="anomaly_contamination",
        )

    selected_cols = st.multiselect(
        "Columns to analyze (leave empty for all numeric):",
        numeric_cols, key="anomaly_cols",
    )

    if st.button("🔍 Detect Anomalies", use_container_width=True, key="btn_detect_anomalies"):
        with st.spinner("Running anomaly detection..."):
            try:
                from app.modules.anomaly_detector import detect_anomalies
                result = detect_anomalies(
                    current_df,
                    method=method,
                    contamination=contamination,
                    columns=selected_cols if selected_cols else None,
                )
            except ImportError:
                st.error("Anomaly detector module not available.")
                return
            except Exception as e:
                st.error(f"Detection error: {e}")
                return

        if "error" in result:
            st.error(result["error"])
            return

        st.session_state["anomaly_result"] = result

        # Display results
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Total Rows", f"{result['n_total']:,}")
        rc2.metric("Anomalies Found", f"{result['n_anomalies']:,}")
        rc3.metric("Anomaly %", f"{result['anomaly_percentage']:.1f}%")

        # Column contributions
        contributions = result.get("column_contributions", {})
        if contributions:
            with st.expander("📊 Column Contributions to Anomalies"):
                import plotly.express as px
                contrib_df = pd.DataFrame([
                    {"Column": k, "Deviation": v} for k, v in contributions.items()
                ]).sort_values("Deviation", ascending=True).tail(10)
                fig = px.bar(contrib_df, x="Deviation", y="Column", orientation="h",
                             title="Top Anomaly-Contributing Columns",
                             color="Deviation", color_continuous_scale="Reds")
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ Found {result['n_anomalies']} anomalies ({result['anomaly_percentage']:.1f}%)")

    # Actions on detected anomalies
    anomaly_result = st.session_state.get("anomaly_result")
    if anomaly_result and anomaly_result.get("n_anomalies", 0) > 0:
        st.divider()
        action = st.radio(
            "Action:", ["Flag anomalies (add column)", "Remove anomalies"],
            key="anomaly_action", horizontal=True,
        )
        if st.button("✅ Apply Action", key="btn_apply_anomaly"):
            try:
                _save_audit_snapshot(current_df)
                if action == "Flag anomalies (add column)":
                    from app.modules.anomaly_detector import flag_anomalies
                    flagged_df = flag_anomalies(current_df, anomaly_result)
                    st.session_state["cleaned_df"] = flagged_df
                    _record_audit("Anomaly Flagging", [f"Added _anomaly_flag column ({anomaly_result['n_anomalies']} anomalies)"],
                                  len(current_df), len(flagged_df), current_df.shape[1], flagged_df.shape[1])
                    st.success("✅ Anomaly flags added!")
                else:
                    from app.modules.anomaly_detector import remove_anomalies
                    cleaned_df, n_removed = remove_anomalies(current_df, anomaly_result)
                    st.session_state["cleaned_df"] = cleaned_df
                    _record_audit("Anomaly Removal", [f"Removed {n_removed} anomalous rows"],
                                  len(current_df), len(cleaned_df), current_df.shape[1], cleaned_df.shape[1])
                    st.success(f"✅ Removed {n_removed} anomalous rows!")
                st.rerun()
            except Exception as e:
                st.error(f"Action failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY DEDUP
# ══════════════════════════════════════════════════════════════════════════════

def _render_fuzzy_dedup(current_df: pd.DataFrame):
    """Near-duplicate detection using text similarity."""
    st.markdown("**🔗 Find near-duplicate rows using fuzzy text matching.**")

    text_cols = current_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not text_cols:
        st.warning("No text/categorical columns available for fuzzy matching.")
        return

    columns = st.multiselect(
        "Columns to compare:",
        text_cols, default=text_cols[:min(2, len(text_cols))],
        key="fuzzy_cols",
    )
    threshold = st.slider(
        "Similarity Threshold:",
        0.50, 1.00, 0.85, 0.05,
        help="Higher = stricter matching (1.0 = exact match)",
        key="fuzzy_threshold",
    )

    if not columns:
        st.info("Select at least one column.")
        return

    if st.button("🔍 Find Fuzzy Duplicates", use_container_width=True, key="btn_fuzzy_scan"):
        with st.spinner("Scanning for near-duplicates..."):
            try:
                from app.modules.fuzzy_dedup import find_fuzzy_duplicates
                result = find_fuzzy_duplicates(current_df, columns, threshold=threshold)
            except ImportError:
                st.error("Fuzzy dedup module not available. Install: `pip install rapidfuzz`")
                return
            except Exception as e:
                st.error(f"Scan error: {e}")
                return

        if "error" in result:
            st.error(result["error"])
            return

        st.session_state["fuzzy_result"] = result

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Groups Found", f"{result['n_groups']}")
        rc2.metric("Near-Duplicates", f"{result['total_fuzzy_duplicates']}")
        rc3.metric("Rows Scanned", f"{result['rows_scanned']:,}")

        # Show groups
        groups = result.get("groups", [])
        if groups:
            with st.expander(f"📋 View {min(len(groups), 10)} Duplicate Groups"):
                for i, group in enumerate(groups[:10]):
                    st.markdown(f"**Group {i+1}** — {group['size']} rows")
                    sample = group.get("sample_values", {})
                    if sample:
                        st.json(sample)
                    st.markdown("---")

    # Merge action
    fuzzy_result = st.session_state.get("fuzzy_result")
    if fuzzy_result and fuzzy_result.get("total_fuzzy_duplicates", 0) > 0:
        st.divider()
        strategy = st.selectbox(
            "Merge Strategy:",
            ["keep_first", "keep_last", "remove_all"],
            format_func=lambda x: {
                "keep_first": "Keep first occurrence",
                "keep_last": "Keep last occurrence",
                "remove_all": "Remove all duplicates",
            }.get(x, x),
            key="fuzzy_strategy",
        )
        if st.button("🔗 Merge Duplicates", key="btn_fuzzy_merge"):
            try:
                from app.modules.fuzzy_dedup import merge_fuzzy_duplicates
                _save_audit_snapshot(current_df)
                merged_df, n_removed = merge_fuzzy_duplicates(
                    current_df, fuzzy_result["groups"], strategy=strategy
                )
                st.session_state["cleaned_df"] = merged_df
                _record_audit("Fuzzy Deduplication",
                              [f"Removed {n_removed} near-duplicate rows (strategy: {strategy})"],
                              len(current_df), len(merged_df), current_df.shape[1], merged_df.shape[1])
                st.success(f"✅ Removed {n_removed} near-duplicate rows!")
                st.rerun()
            except Exception as e:
                st.error(f"Merge failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_audit_trail(df: pd.DataFrame):
    """Render undo button in header area."""
    try:
        from app.modules.audit_trail import get_audit_trail
        audit = get_audit_trail()
        if audit.can_undo():
            if st.button("↩️ Undo Last Change", key="btn_undo", type="secondary"):
                restored = audit.undo()
                if restored is not None:
                    st.session_state["cleaned_df"] = restored
                    st.success("✅ Reverted to previous state!")
                    st.rerun()
    except Exception:
        pass


def _render_audit_log():
    """Display the full audit log."""
    st.markdown("**📋 Complete Cleaning Audit Log**")
    try:
        from app.modules.audit_trail import get_audit_trail
        audit = get_audit_trail()
        log_df = audit.export_log()
        if log_df.empty:
            st.info("No cleaning operations recorded yet.")
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            summary = audit.get_summary()
            sc1, sc2 = st.columns(2)
            sc1.metric("Total Operations", summary.get("total_operations", 0))
            sc2.metric("Total Rows Affected", f"{summary.get('total_rows_affected', 0):,}")

            # Export audit log
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Audit Log", csv, "datapilot_audit_log.csv",
                                "text/csv", key="dl_audit_log")

            if st.button("🗑️ Clear Audit History", key="btn_clear_audit"):
                audit.clear()
                st.rerun()
    except Exception:
        st.info("Audit trail not available.")


def _save_audit_snapshot(df: pd.DataFrame):
    """Save a DataFrame snapshot for undo support."""
    try:
        from app.modules.audit_trail import get_audit_trail
        audit = get_audit_trail()
        audit.save_snapshot(df)
    except Exception:
        pass


def _record_audit(operation: str, details_list, rows_before, rows_after, cols_before, cols_after):
    """Record a cleaning operation in the audit trail."""
    try:
        from app.modules.audit_trail import get_audit_trail
        audit = get_audit_trail()
        details = " | ".join(details_list) if isinstance(details_list, list) else str(details_list)
        audit.record(
            operation=operation,
            rows_before=rows_before,
            rows_after=rows_after,
            cols_before=cols_before,
            cols_after=cols_after,
            details=details,
        )
    except Exception:
        pass
