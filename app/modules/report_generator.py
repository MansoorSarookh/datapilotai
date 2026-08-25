"""
DataPilot AI - Executive Report Generator
Produces PDF and HTML analysis reports.
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, List


# -------------------------------------------------------------------
# UNIVERSAL TEXT SANITIZER (prevents FPDF unicode crashes forever)
# -------------------------------------------------------------------
def safe_text(text) -> str:
    if text is None:
        return ""
    text = str(text)

    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "’": "'",
        "→": "->",
        "•": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # strip unsupported characters
    return text.encode("latin-1", "ignore").decode("latin-1")


# -------------------------------------------------------------------
# PDF REPORT
# -------------------------------------------------------------------
def generate_pdf_report(
    df: pd.DataFrame,
    trust_score: Dict,
    insights: List[str],
    file_name: str = "dataset",
    ai_summary: str = "",
) -> bytes:

    from fpdf import FPDF

    file_name = safe_text(file_name)
    ai_summary = safe_text(ai_summary)
    insights = [safe_text(i) for i in insights]

    class DataPilotPDF(FPDF):

        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 20, 'F')
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 14)
            self.set_xy(10, 5)
            self.cell(0, 10, safe_text("DataPilot AI - Analysis Report"), align="L")
            self.set_font("Helvetica", "", 9)
            self.set_xy(150, 5)
            self.cell(0, 10, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), align="R")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_text_color(150, 150, 150)
            self.set_font("Helvetica", "I", 8)
            self.cell(
                0,
                10,
                safe_text(f"DataPilot AI | Page {self.page_no()} | Generated for: {file_name}"),
                align="C"
            )

        def section_title(self, title: str, color=(15, 23, 42)):
            self.set_fill_color(*color)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 8, "  " + safe_text(title), ln=True, fill=True)
            self.set_text_color(0, 0, 0)
            self.ln(2)

        def body_text(self, text: str, size: int = 10):
            self.set_font("Helvetica", "", size)
            self.set_text_color(40, 40, 40)
            self.multi_cell(0, 6, safe_text(text))
            self.ln(2)

        def metric_row(self, label: str, value: str):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(15, 23, 42)
            self.cell(70, 7, safe_text(label) + ":", border='B')
            self.set_font("Helvetica", "", 10)
            self.set_text_color(60, 60, 60)
            self.cell(0, 7, safe_text(value), border='B', ln=True)

    pdf = DataPilotPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 25, 15)

    # Title block
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(15, 22, 180, 22, 'F')
    pdf.set_xy(15, 24)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "AI Data Intelligence Report", ln=True)
    pdf.set_xy(15, 34)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0,
        8,
        safe_text(
            f"Dataset: {file_name} | Generated: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}"
        ),
        ln=True,
    )
    pdf.ln(10)

    # Executive summary
    pdf.section_title("1. Executive Summary")
    if ai_summary:
        pdf.body_text(ai_summary)
    else:
        score = trust_score.get("overall", 0)
        label = "Reliable" if score > 0.80 else "Needs Cleaning" if score > 0.60 else "High Risk"
        pdf.body_text(
            f"Dataset '{file_name}' contains {df.shape[0]:,} rows and {df.shape[1]} columns. "
            f"Overall quality rated {label} with trust score {score:.0%}."
        )

    # Dataset overview
    pdf.section_title("2. Dataset Overview", color=(30, 64, 175))
    pdf.metric_row("Total Rows", f"{df.shape[0]:,}")
    pdf.metric_row("Total Columns", str(df.shape[1]))
    pdf.metric_row("Missing Values", f"{df.isna().sum().sum():,}")
    pdf.metric_row("Duplicate Rows", str(df.duplicated().sum()))

    # Insights
    if insights:
        pdf.add_page()
        pdf.section_title("3. Key Insights", color=(124, 58, 237))
        pdf.set_font("Helvetica", "", 10)
        for i, ins in enumerate(insights[:10], 1):
            pdf.multi_cell(0, 6, safe_text(f"{i}. {ins}"))
            pdf.ln(1)

    # Recommendations
    recs = trust_score.get("recommendations", [])
    if recs:
        pdf.section_title("4. Recommendations", color=(234, 88, 12))
        for i, r in enumerate(recs[:8], 1):
            pdf.multi_cell(0, 6, safe_text(f"{i}. {r}"))
            pdf.ln(1)

    # Footer note
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "Generated by DataPilot AI - The AI Data Intelligence Copilot", align="C", ln=True)

    # IMPORTANT FIX FOR STREAMLIT CLOUD
    # return pdf.output(dest="S").encode("latin-1", "ignore")
    pdf_output = pdf.output(dest="S")

    # Handle both old (str) and new (bytes/bytearray) FPDF behavior
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1", "ignore")
    else:
        return bytes(pdf_output) 


# -------------------------------------------------------------------
# HTML REPORT (unchanged, unicode-safe in browsers)
# -------------------------------------------------------------------
def generate_html_report(
    df: pd.DataFrame,
    trust_score: Dict,
    insights: List[str],
    file_name: str = "dataset",
    ai_summary: str = "",
) -> str:

    score = trust_score.get("overall", 0)
    score_pct = f"{score:.0%}"
    now = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    return f"""
    <html>
    <head>
    <title>DataPilot AI Report</title>
    </head>
    <body style="font-family:Arial;padding:40px;">
        <h1>DataPilot AI - Analysis Report</h1>
        <p><b>Dataset:</b> {file_name}</p>
        <p><b>Generated:</b> {now}</p>
        <p><b>Rows:</b> {df.shape[0]:,} | <b>Columns:</b> {df.shape[1]}</p>
        <p><b>Trust Score:</b> {score_pct}</p>
    </body>
    </html>
    """ 


# """
# DataPilot AI — Executive Report Generator
# Produces PDF and HTML analysis reports.
# """

# import pandas as pd
# import numpy as np
# import io
# import base64
# import datetime
# from typing import Dict, List, Optional


# def generate_pdf_report(
#     df: pd.DataFrame,
#     trust_score: Dict,
#     insights: List[str],
#     file_name: str = "dataset",
#     ai_summary: str = "",
# ) -> bytes:
#     """
#     Generate a professional PDF executive report using fpdf2.
#     Returns bytes of the PDF file.
#     """
#     try:
#         from fpdf import FPDF

#         class DataPilotPDF(FPDF):
#             def header(self):
#                 self.set_fill_color(15, 23, 42)
#                 self.rect(0, 0, 210, 20, 'F')
#                 self.set_text_color(255, 255, 255)
#                 self.set_font("Helvetica", "B", 14)
#                 self.set_xy(10, 5)
#                 self.cell(0, 10, "DataPilot AI — Analysis Report", align="L")
#                 self.set_font("Helvetica", "", 9)
#                 self.set_xy(150, 5)
#                 self.cell(0, 10, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), align="R")
#                 self.ln(10)

#             def footer(self):
#                 self.set_y(-15)
#                 self.set_text_color(150, 150, 150)
#                 self.set_font("Helvetica", "I", 8)
#                 self.cell(0, 10, f"DataPilot AI | Page {self.page_no()} | Generated for: {file_name}", align="C")

#             def section_title(self, title: str, color=(15, 23, 42)):
#                 self.set_fill_color(*color)
#                 self.set_text_color(255, 255, 255)
#                 self.set_font("Helvetica", "B", 11)
#                 self.cell(0, 8, f"  {title}", ln=True, fill=True)
#                 self.set_text_color(0, 0, 0)
#                 self.ln(2)

#             def body_text(self, text: str, size: int = 10):
#                 self.set_font("Helvetica", "", size)
#                 self.set_text_color(40, 40, 40)
#                 self.multi_cell(0, 6, text)
#                 self.ln(2)

#             def metric_row(self, label: str, value: str):
#                 self.set_font("Helvetica", "B", 10)
#                 self.set_text_color(15, 23, 42)
#                 self.cell(70, 7, label + ":", border='B')
#                 self.set_font("Helvetica", "", 10)
#                 self.set_text_color(60, 60, 60)
#                 self.cell(0, 7, str(value), border='B', ln=True)

#         pdf = DataPilotPDF()
#         pdf.set_auto_page_break(auto=True, margin=15)
#         pdf.add_page()
#         pdf.set_margins(15, 25, 15)

#         # Title block
#         pdf.set_fill_color(248, 250, 252)
#         pdf.rect(15, 22, 180, 22, 'F')
#         pdf.set_xy(15, 24)
#         pdf.set_font("Helvetica", "B", 18)
#         pdf.set_text_color(15, 23, 42)
#         pdf.cell(0, 10, "AI Data Intelligence Report", ln=True)
#         pdf.set_xy(15, 34)
#         pdf.set_font("Helvetica", "", 10)
#         pdf.set_text_color(100, 100, 100)
#         pdf.cell(0, 8, f"Dataset: {file_name}  |  Generated: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", ln=True)
#         pdf.ln(10)

#         # ── Section 1: Executive Summary ──────────────────────────────────────
#         pdf.section_title("1. Executive Summary")
#         if ai_summary:
#             pdf.body_text(ai_summary)
#         else:
#             score = trust_score.get("overall", 0)
#             label = "Reliable" if score > 0.80 else "Needs Cleaning" if score > 0.60 else "High Risk"
#             pdf.body_text(
#                 f"Dataset '{file_name}' contains {df.shape[0]:,} records across {df.shape[1]} variables. "
#                 f"Overall data quality is rated {label} with a trust score of {score:.0%}. "
#                 "Key areas requiring attention are highlighted in the quality assessment section below."
#             )

#         # ── Section 2: Dataset Overview ───────────────────────────────────────
#         pdf.section_title("2. Dataset Overview", color=(30, 64, 175))
#         pdf.metric_row("Total Rows", f"{df.shape[0]:,}")
#         pdf.metric_row("Total Columns", str(df.shape[1]))
#         pdf.metric_row("Numeric Columns", str(len(df.select_dtypes(include=np.number).columns)))
#         pdf.metric_row("Categorical Columns", str(len(df.select_dtypes(include="object").columns)))
#         pdf.metric_row("Missing Values", f"{df.isna().sum().sum():,} ({df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%)")
#         pdf.metric_row("Duplicate Rows", str(df.duplicated().sum()))
#         pdf.ln(5)

#         # ── Section 3: Trust Score ────────────────────────────────────────────
#         pdf.section_title("3. Dataset Trust Score", color=(5, 150, 105))
#         score = trust_score.get("overall", 0)
#         dims = trust_score.get("dimensions", {})
#         pdf.set_font("Helvetica", "B", 24)
#         pdf.set_text_color(5, 150, 105) if score > 0.80 else (pdf.set_text_color(217, 119, 6) if score > 0.60 else pdf.set_text_color(220, 38, 38))
#         pdf.cell(0, 12, f"{score:.0%}", ln=True)
#         pdf.set_text_color(0, 0, 0)
#         for dim, val in dims.items():
#             bar_w = int(val * 80)
#             pdf.set_font("Helvetica", "", 9)
#             pdf.cell(45, 6, dim.capitalize() + ":")
#             pdf.set_fill_color(99, 102, 241) if val > 0.80 else (pdf.set_fill_color(234, 179, 8) if val > 0.60 else pdf.set_fill_color(239, 68, 68))
#             pdf.cell(bar_w, 5, "", fill=True)
#             pdf.set_font("Helvetica", "B", 9)
#             pdf.cell(0, 5, f"  {val:.0%}", ln=True)
#         pdf.ln(3)

#         # Issues
#         flags = trust_score.get("flags", [])
#         if flags:
#             pdf.set_font("Helvetica", "B", 10)
#             pdf.cell(0, 7, "Issues Detected:", ln=True)
#             pdf.set_font("Helvetica", "", 9)
#             pdf.set_text_color(180, 50, 50)
#             for flag in flags[:6]:
#                 # Strip emoji for PDF compatibility
#                 clean_flag = flag.encode('ascii', 'ignore').decode()
#                 pdf.cell(0, 6, "  " + clean_flag.strip(), ln=True)
#             pdf.set_text_color(0, 0, 0)
#         pdf.ln(3)

#         # ── Section 4: Key Insights ───────────────────────────────────────────
#         if insights:
#             pdf.add_page()
#             pdf.section_title("4. Key Insights & Findings", color=(124, 58, 237))
#             pdf.set_font("Helvetica", "", 10)
#             for i, insight in enumerate(insights[:10], 1):
#                 clean_insight = insight.encode('ascii', 'ignore').decode()
#                 pdf.multi_cell(0, 6, f"{i}. {clean_insight.strip()}")
#                 pdf.ln(1)
#             pdf.ln(4)

#         # ── Section 5: Recommendations ────────────────────────────────────────
#         pdf.section_title("5. Recommendations", color=(234, 88, 12))
#         recs = trust_score.get("recommendations", [])
#         if recs:
#             pdf.set_font("Helvetica", "", 10)
#             for i, rec in enumerate(recs[:8], 1):
#                 clean_rec = rec.encode('ascii', 'ignore').decode()
#                 pdf.multi_cell(0, 6, f"{i}. {clean_rec.strip()}")
#                 pdf.ln(1)
#         else:
#             pdf.body_text("Dataset appears ready for analysis. Proceed with ML pipeline.")

#         # ── Section 6: Descriptive Stats ──────────────────────────────────────
#         numeric_df = df.select_dtypes(include=np.number)
#         if not numeric_df.empty:
#             pdf.section_title("6. Descriptive Statistics (Numeric)", color=(15, 23, 42))
#             stats = numeric_df.describe().T[["mean", "std", "min", "50%", "max"]]
#             stats.columns = ["Mean", "Std", "Min", "Median", "Max"]

#             pdf.set_font("Helvetica", "B", 8)
#             col_widths = [45, 25, 25, 25, 25, 25]
#             headers = ["Column", "Mean", "Std", "Min", "Median", "Max"]
#             for i, h in enumerate(headers):
#                 pdf.set_fill_color(15, 23, 42)
#                 pdf.set_text_color(255, 255, 255)
#                 pdf.cell(col_widths[i], 6, h, border=1, fill=True)
#             pdf.ln()

#             pdf.set_font("Helvetica", "", 8)
#             for row_idx, (col, row) in enumerate(stats.iterrows()):
#                 fill = row_idx % 2 == 0
#                 pdf.set_fill_color(240, 240, 248)
#                 pdf.set_text_color(30, 30, 30)
#                 pdf.cell(col_widths[0], 6, str(col)[:20], border=1, fill=fill)
#                 for i, val in enumerate(row, 1):
#                     pdf.cell(col_widths[i], 6, f"{val:.3f}", border=1, fill=fill)
#                 pdf.ln()

#         # Footer note
#         pdf.ln(10)
#         pdf.set_font("Helvetica", "I", 8)
#         pdf.set_text_color(150, 150, 150)
#         pdf.cell(0, 5, "Generated by DataPilot AI — The AI Data Intelligence Copilot", align="C", ln=True)
#         pdf.cell(0, 5, "Powered by Groq LLM | Built on Python + Streamlit", align="C")

#         return bytes(pdf.output())

#     except ImportError:
#         return _fallback_pdf_text(df, trust_score, file_name)


# def _fallback_pdf_text(df, trust_score, file_name):
#     """Plain text fallback if fpdf2 not available."""
#     content = f"""DataPilot AI Analysis Report
# ===========================
# Dataset: {file_name}
# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

# Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
# Trust Score: {trust_score.get('overall', 0):.0%}

# Statistics:
# {df.describe().to_string()}
# """
#     return content.encode("utf-8")


# def generate_html_report(
#     df: pd.DataFrame,
#     trust_score: Dict,
#     insights: List[str],
#     file_name: str = "dataset",
#     ai_summary: str = "",
# ) -> str:
#     """Generate an interactive HTML report."""
#     score = trust_score.get("overall", 0)
#     score_pct = f"{score:.0%}"
#     label = trust_score.get("label", "")
#     color = trust_score.get("color", "gray")
#     dims = trust_score.get("dimensions", {})
#     flags = trust_score.get("flags", [])
#     recs = trust_score.get("recommendations", [])
#     now = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

#     # Dimension bars
#     dim_bars_html = "".join([
#         f"""<div style="display:flex;align-items:center;margin:6px 0;">
#             <span style="width:120px;font-size:13px;color:#374151;">{k.capitalize()}</span>
#             <div style="background:#e5e7eb;border-radius:999px;height:10px;flex:1;overflow:hidden;">
#               <div style="background:{'#10b981' if v > 0.8 else '#f59e0b' if v > 0.6 else '#ef4444'};width:{v*100:.0f}%;height:100%;border-radius:999px;"></div>
#             </div>
#             <span style="margin-left:8px;font-weight:600;font-size:13px;">{v:.0%}</span>
#           </div>"""
#         for k, v in dims.items()
#     ])

#     flag_items = "".join([f"<li>{f}</li>" for f in flags[:8]])
#     rec_items = "".join([f"<li>{r}</li>" for r in recs[:8]])
#     insight_items = "".join([f"<li>{ins}</li>" for ins in insights[:10]])

#     # Stats table
#     numeric_df = df.select_dtypes(include=np.number)
#     stats_html = ""
#     if not numeric_df.empty:
#         stats = numeric_df.describe().T[["mean", "std", "min", "50%", "max"]].round(3)
#         stats.columns = ["Mean", "Std", "Min", "Median", "Max"]
#         stats_html = stats.to_html(classes="stats-table", border=0)

#     score_color = "#10b981" if score > 0.80 else "#f59e0b" if score > 0.60 else "#ef4444"

#     return f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>DataPilot AI — Analysis Report</title>
# <style>
#   * {{ box-sizing: border-box; margin: 0; padding: 0; }}
#   body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; }}
#   .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%); color: white; padding: 32px; }}
#   .header h1 {{ font-size: 28px; font-weight: 800; letter-spacing: -0.5px; }}
#   .header p {{ color: #94a3b8; margin-top: 8px; }}
#   .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
#   .card {{ background: white; border-radius: 16px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
#   .card h2 {{ font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }}
#   .score-badge {{ display: inline-block; font-size: 48px; font-weight: 900; color: {score_color}; }}
#   .label-pill {{ display: inline-block; padding: 4px 16px; border-radius: 999px; background: {score_color}22; color: {score_color}; font-weight: 600; font-size: 14px; margin-left: 12px; }}
#   .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
#   .metric {{ background: #f8fafc; border-radius: 10px; padding: 16px; text-align: center; }}
#   .metric-value {{ font-size: 24px; font-weight: 800; color: #0f172a; }}
#   .metric-label {{ font-size: 12px; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
#   .flag-list li {{ padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px; color: #374151; }}
#   .rec-list li {{ padding: 6px 0; font-size: 13px; color: #374151; }}
#   .rec-list li::before {{ content: "→ "; color: #6366f1; font-weight: 700; }}
#   .stats-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
#   .stats-table th {{ background: #0f172a; color: white; padding: 8px 12px; text-align: left; }}
#   .stats-table td {{ padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }}
#   .stats-table tr:nth-child(even) td {{ background: #f8fafc; }}
#   .footer {{ text-align: center; padding: 24px; color: #94a3b8; font-size: 12px; }}
#   @media print {{ body {{ background: white; }} .card {{ box-shadow: none; }} }}
# </style>
# </head>
# <body>
# <div class="header">
#   <h1>🧠 DataPilot AI — Analysis Report</h1>
#   <p>Dataset: <strong>{file_name}</strong> | Generated: {now}</p>
# </div>
# <div class="container">

#   <!-- Executive Summary -->
#   <div class="card">
#     <h2>Executive Summary</h2>
#     <p style="line-height:1.7;color:#374151;">{ai_summary or f"Dataset '{file_name}' contains {df.shape[0]:,} records across {df.shape[1]} variables. Trust score: {score_pct}."}</p>
#   </div>

#   <!-- Dataset Stats -->
#   <div class="card">
#     <h2>Dataset Overview</h2>
#     <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
#       <div class="metric"><div class="metric-value">{df.shape[0]:,}</div><div class="metric-label">Total Rows</div></div>
#       <div class="metric"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Columns</div></div>
#       <div class="metric"><div class="metric-value">{df.isna().sum().sum():,}</div><div class="metric-label">Missing Values</div></div>
#       <div class="metric"><div class="metric-value">{len(df.select_dtypes(include='number').columns)}</div><div class="metric-label">Numeric Cols</div></div>
#       <div class="metric"><div class="metric-value">{len(df.select_dtypes(include='object').columns)}</div><div class="metric-label">Categorical Cols</div></div>
#       <div class="metric"><div class="metric-value">{df.duplicated().sum()}</div><div class="metric-label">Duplicates</div></div>
#     </div>
#   </div>

#   <!-- Trust Score -->
#   <div class="card">
#     <h2>Dataset Trust Score</h2>
#     <div style="margin-bottom:16px;">
#       <span class="score-badge">{score_pct}</span>
#       <span class="label-pill">{label}</span>
#     </div>
#     {dim_bars_html}
#     {f'<div style="margin-top:16px;"><h3 style="font-size:14px;margin-bottom:8px;color:#dc2626;">Issues Detected</h3><ul class="flag-list">{flag_items}</ul></div>' if flags else ""}
#   </div>

#   <!-- Key Insights -->
#   {f'<div class="card"><h2>Key Insights</h2><ul class="rec-list">{insight_items}</ul></div>' if insights else ""}

#   <!-- Recommendations -->
#   {f'<div class="card"><h2>Recommendations</h2><ul class="rec-list">{rec_items}</ul></div>' if recs else ""}

#   <!-- Descriptive Stats -->
#   {f'<div class="card"><h2>Descriptive Statistics</h2>{stats_html}</div>' if stats_html else ""}

# </div>
# <div class="footer">
#   Generated by <strong>DataPilot AI</strong> — The AI Data Intelligence Copilot | 
#   Powered by Groq LLM &amp; Python
# </div>
# </body></html>"""
 
