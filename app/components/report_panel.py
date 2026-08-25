"""
DataPilot AI — Export & Report Panel Component (v3.0)
Integrates: PDF/HTML reports, Jupyter notebooks, interactive dashboard,
data dictionary (Word), and all data export formats.
"""

import streamlit as st
import pandas as pd
import io

from app.modules.report_generator import generate_pdf_report, generate_html_report
from app.modules.notebook_exporter import generate_notebook
from app.modules.ai_engine import generate_executive_summary


def render_report_panel(df: pd.DataFrame, trust_score: dict, file_name: str = "dataset"):
    """Render the report generation and export UI."""

    st.markdown("### 📥 Export & Report Center")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Executive Report",
        "📓 Jupyter Notebook",
        "📊 Interactive Dashboard",
        "📖 Data Dictionary",
        "💾 Data Export",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: Executive Report
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Generate a one-click professional analysis report.**")

        include_ai = st.checkbox(
            "✨ Include AI-generated executive summary",
            value=True,
        )

        custom_insights = []
        with st.expander("📝 Add Custom Insights (Optional)"):
            for i in range(3):
                insight = st.text_input(
                    f"Insight {i+1}:", key=f"insight_{i}",
                    placeholder="e.g. Q3 revenue dropped due to seasonal factors",
                )
                if insight:
                    custom_insights.append(insight)

        col1, col2 = st.columns(2)

        # PDF Report
        with col1:
            if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
                with st.spinner("Generating executive report..."):
                    ai_summary = ""
                    if include_ai:
                        ai_summary = generate_executive_summary(df, trust_score, custom_insights)
                    try:
                        pdf_bytes = generate_pdf_report(
                            df=df, trust_score=trust_score, insights=custom_insights,
                            file_name=file_name, ai_summary=ai_summary,
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        pdf_bytes = None

                if isinstance(pdf_bytes, bytes):
                    st.download_button(
                        "⬇️ Download PDF Report", pdf_bytes,
                        f"datapilot_report_{file_name}.pdf", "application/pdf",
                        key="dl_pdf_report",
                    )
                    st.success("✅ PDF Report ready!")
                elif pdf_bytes is not None:
                    st.error("PDF generation failed. Try HTML format instead.")

        # HTML Report
        with col2:
            if st.button("🌐 Generate HTML Report", use_container_width=True):
                with st.spinner("Generating HTML report..."):
                    ai_summary = ""
                    if include_ai:
                        ai_summary = generate_executive_summary(df, trust_score, custom_insights)
                    try:
                        html_content = generate_html_report(
                            df=df, trust_score=trust_score, insights=custom_insights,
                            file_name=file_name, ai_summary=ai_summary,
                        )
                        st.download_button(
                            "⬇️ Download HTML Report",
                            html_content.encode("utf-8"),
                            f"datapilot_report_{file_name}.html", "text/html",
                            key="dl_html_report",
                        )
                        st.success("✅ HTML Report ready! Open in browser for interactive charts.")
                    except Exception as e:
                        st.error(f"HTML generation failed: {e}")

        # AI Summary Preview
        if include_ai:
            if st.button("👁️ Preview AI Summary", key="preview_ai"):
                with st.spinner("Generating AI narrative..."):
                    summary = generate_executive_summary(df, trust_score, custom_insights)
                st.markdown("---")
                st.markdown("**AI Executive Summary:**")
                st.markdown(summary)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: Jupyter Notebook Export
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("**Export a fully reproducible Jupyter notebook with all your analysis steps.**")
        st.info("💡 The notebook includes: setup, data loading, trust score, EDA, cleaning, and ML training code.")

        include_ml = st.checkbox("Include ML training code", value=False)
        ml_target = None
        ml_algo = "Random Forest"

        if include_ml:
            cols = df.columns.tolist()
            ml_target = st.selectbox("Target column for ML:", cols, key="nb_target")
            ml_algo = st.selectbox(
                "Algorithm:",
                ["Random Forest", "Logistic Regression", "XGBoost", "Linear Regression"],
                key="nb_algo",
            )

        if st.button("📓 Generate Jupyter Notebook", use_container_width=True, type="primary"):
            with st.spinner("Building notebook..."):
                cleaning_ops = st.session_state.get("cleaning_ops", [])
                ml_config = (
                    {"target_col": ml_target, "algorithm": ml_algo}
                    if include_ml and ml_target
                    else None
                )
                try:
                    nb_json = generate_notebook(
                        file_name=file_name, df=df, cleaning_steps=cleaning_ops,
                        ml_config=ml_config, trust_score=trust_score,
                    )
                    st.download_button(
                        "⬇️ Download .ipynb Notebook",
                        nb_json.encode("utf-8"),
                        f"datapilot_analysis_{file_name.replace('.', '_')}.ipynb",
                        "application/json", key="dl_notebook",
                    )
                    st.success("✅ Notebook generated! Open with Jupyter Lab or VS Code.")
                except Exception as e:
                    st.error(f"Notebook generation failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: Interactive Dashboard
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("**Generate a self-contained interactive HTML dashboard with embedded Plotly charts.**")
        st.info("📊 The dashboard includes: data overview, trust score, auto-generated charts, and data table.")

        include_table = st.checkbox("Include data table (first 100 rows)", value=True, key="dash_table")

        if st.button("📊 Generate Interactive Dashboard", use_container_width=True, type="primary"):
            with st.spinner("Building interactive dashboard..."):
                try:
                    from app.modules.dashboard_exporter import generate_dashboard_html
                    dashboard_html = generate_dashboard_html(
                        df=df,
                        trust_score=trust_score,
                        file_name=file_name,
                        include_data_table=include_table,
                    )
                    st.download_button(
                        "⬇️ Download Interactive Dashboard (.html)",
                        dashboard_html.encode("utf-8"),
                        f"datapilot_dashboard_{file_name}.html", "text/html",
                        key="dl_dashboard",
                    )
                    st.success("✅ Interactive dashboard ready! Open in any modern browser.")
                except ImportError:
                    st.error("Dashboard exporter module not available.")
                except Exception as e:
                    st.error(f"Dashboard generation failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: Data Dictionary (Word)
    # ──────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("**Auto-generate a professional Word (.docx) data dictionary.**")
        st.info("📖 Includes: dataset overview, trust score, column descriptions, numeric statistics, and data types.")

        if st.button("📖 Generate Data Dictionary", use_container_width=True, type="primary"):
            with st.spinner("Building data dictionary..."):
                try:
                    from app.modules.data_dictionary import generate_data_dictionary
                    docx_bytes = generate_data_dictionary(
                        df=df,
                        file_name=file_name,
                        trust_score=trust_score,
                    )
                    st.download_button(
                        "⬇️ Download Data Dictionary (.docx)",
                        docx_bytes,
                        f"datapilot_data_dictionary_{file_name}.docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="dl_data_dict",
                    )
                    st.success("✅ Data dictionary generated!")
                except ImportError:
                    st.error("Data dictionary requires `python-docx`. Install with: `pip install python-docx`")
                except Exception as e:
                    st.error(f"Data dictionary generation failed: {e}")

        # Quick preview
        with st.expander("👁️ Preview Column Summary"):
            preview_data = []
            for col in df.columns:
                series = df[col]
                preview_data.append({
                    "Column": col,
                    "Type": str(series.dtype),
                    "Non-Null": int(series.notna().sum()),
                    "Unique": int(series.nunique()),
                    "Missing %": f"{series.isna().sum() / len(df) * 100:.1f}%",
                    "Sample": str(series.dropna().head(1).values[0]) if series.notna().any() else "N/A",
                })
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5: Data Export
    # ──────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("**Download your data in multiple formats.**")

        cleaned = st.session_state.get("cleaned_df")
        if cleaned is None or (isinstance(cleaned, pd.DataFrame) and cleaned.empty):
            export_df = df
            label = "Original Dataset"
        else:
            export_df = cleaned
            label = "Cleaned Dataset"

        st.info(f"Exporting: **{label}** ({export_df.shape[0]:,} rows × {export_df.shape[1]} columns)")

        c1, c2, c3, c4 = st.columns(4)

        # CSV
        with c1:
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV", csv_bytes, f"{file_name}_export.csv", "text/csv",
                use_container_width=True, key="dl_exp_csv",
            )

        # Excel
        with c2:
            try:
                buf = io.BytesIO()
                export_df.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                st.download_button(
                    "📥 Excel", buf.getvalue(), f"{file_name}_export.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, key="dl_exp_xlsx",
                )
            except Exception as e:
                st.error(f"Excel: {e}")

        # JSON
        with c3:
            json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "📥 JSON", json_bytes, f"{file_name}_export.json", "application/json",
                use_container_width=True, key="dl_exp_json",
            )

        # Parquet
        with c4:
            try:
                buf = io.BytesIO()
                export_df.to_parquet(buf, index=False)
                buf.seek(0)
                st.download_button(
                    "📥 Parquet", buf.getvalue(), f"{file_name}_export.parquet",
                    "application/octet-stream",
                    use_container_width=True, key="dl_exp_parquet",
                )
            except Exception as e:
                st.error(f"Parquet: {e}")

        # Statistics Export
        st.markdown("**📊 Statistics Export:**")
        stats_df = export_df.describe(include="all").round(4)
        stats_csv = stats_df.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Statistics CSV", stats_csv,
            f"{file_name}_stats.csv", "text/csv", key="dl_stats_csv",
        )
