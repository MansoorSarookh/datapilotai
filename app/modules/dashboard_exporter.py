"""
DataPilot AI — Interactive HTML Dashboard Exporter
Generates self-contained HTML dashboards with embedded Plotly charts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def generate_dashboard_html(
    df: pd.DataFrame,
    trust_score: Dict,
    charts: List = None,
    file_name: str = "dataset",
    include_data_table: bool = True,
) -> str:
    """
    Generate a self-contained interactive HTML dashboard.

    Returns HTML string with embedded Plotly charts, data tables, and navigation.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    score = trust_score.get("overall", 0)
    dims = trust_score.get("dimensions", {})
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Generate auto charts if none provided
    chart_htmls = []
    if charts:
        for fig in charts[:8]:
            try:
                chart_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))
            except Exception:
                pass

    # Auto-generate key charts
    if not chart_htmls:
        # Missing values bar
        if df.isna().sum().sum() > 0:
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            fig = px.bar(x=missing.index, y=missing.values, title="Missing Values by Column",
                         color=missing.values, color_continuous_scale="Reds")
            fig.update_layout(template="plotly_dark", height=400)
            chart_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           title="Correlation Matrix", aspect="auto")
            fig.update_layout(template="plotly_dark", height=500)
            chart_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # Distribution of first 3 numeric columns
        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col, title=f"Distribution: {col}",
                              color_discrete_sequence=["#667eea"])
            fig.update_layout(template="plotly_dark", height=350)
            chart_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # Data table HTML (first 100 rows)
    table_html = ""
    if include_data_table:
        table_df = df.head(100)
        table_html = table_df.to_html(
            classes="data-table", index=False, border=0,
            max_cols=20, max_rows=100
        )

    # Build trust score dimensions HTML
    dim_cards = ""
    for dim_name, dim_val in dims.items():
        val = dim_val if isinstance(dim_val, (int, float)) else 0
        color = "#10b981" if val > 0.8 else "#f59e0b" if val > 0.6 else "#ef4444"
        dim_cards += f"""
        <div class="dim-card">
            <div class="dim-name">{dim_name.replace('_', ' ').title()}</div>
            <div class="dim-bar"><div class="dim-fill" style="width:{val*100:.0f}%;background:{color}"></div></div>
            <div class="dim-val" style="color:{color}">{val:.0%}</div>
        </div>"""

    charts_section = "\n".join(f'<div class="chart-container">{c}</div>' for c in chart_htmls)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DataPilot AI — Dashboard: {file_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #667eea; --border: #334155; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter','Segoe UI',sans-serif; background:var(--bg); color:var(--text); }}
  .navbar {{ background:linear-gradient(135deg,#667eea,#764ba2); padding:16px 32px; display:flex; justify-content:space-between; align-items:center; }}
  .navbar h1 {{ font-size:20px; color:#fff; }} .navbar span {{ color:rgba(255,255,255,0.8); font-size:13px; }}
  .tabs {{ display:flex; gap:4px; background:var(--card); padding:8px; border-radius:12px; margin:24px 32px 0; }}
  .tab {{ padding:10px 24px; border-radius:8px; cursor:pointer; font-size:14px; font-weight:500; color:#94a3b8; transition:all 0.2s; }}
  .tab.active {{ background:var(--accent); color:#fff; }}
  .tab:hover:not(.active) {{ background:#334155; color:#fff; }}
  .section {{ display:none; padding:24px 32px; }} .section.active {{ display:block; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin:16px 0; }}
  .card {{ background:var(--card); border-radius:12px; padding:20px; border:1px solid var(--border); }}
  .card h3 {{ font-size:14px; color:#94a3b8; margin-bottom:8px; }}
  .card .value {{ font-size:28px; font-weight:700; }}
  .trust-badge {{ font-size:48px; font-weight:800; text-align:center; margin:24px 0; }}
  .dim-card {{ display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); }}
  .dim-name {{ width:120px; font-size:13px; color:#94a3b8; text-transform:capitalize; }}
  .dim-bar {{ flex:1; height:8px; background:#1e293b; border-radius:4px; overflow:hidden; }}
  .dim-fill {{ height:100%; border-radius:4px; transition:width 0.5s; }}
  .dim-val {{ width:50px; text-align:right; font-weight:600; font-size:14px; }}
  .chart-container {{ background:var(--card); border-radius:12px; padding:16px; margin:16px 0; border:1px solid var(--border); }}
  .data-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .data-table th {{ background:var(--accent); color:#fff; padding:10px; text-align:left; position:sticky; top:0; }}
  .data-table td {{ padding:8px 10px; border-bottom:1px solid var(--border); }}
  .data-table tr:hover {{ background:rgba(102,126,234,0.1); }}
  .table-wrap {{ max-height:500px; overflow:auto; border-radius:12px; border:1px solid var(--border); }}
  .footer {{ text-align:center; padding:32px; color:#64748b; font-size:12px; }}
</style>
</head>
<body>
<div class="navbar">
    <h1>🧠 DataPilot AI — Interactive Dashboard</h1>
    <span>Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} · {file_name}</span>
</div>

<div class="tabs" id="tabBar">
    <div class="tab active" onclick="showSection('overview')">📊 Overview</div>
    <div class="tab" onclick="showSection('charts')">📈 Charts</div>
    <div class="tab" onclick="showSection('data')">📋 Data</div>
</div>

<div class="section active" id="overview">
    <div class="grid">
        <div class="card"><h3>Total Rows</h3><div class="value">{df.shape[0]:,}</div></div>
        <div class="card"><h3>Total Columns</h3><div class="value">{df.shape[1]}</div></div>
        <div class="card"><h3>Missing Values</h3><div class="value">{int(df.isna().sum().sum()):,}</div></div>
        <div class="card"><h3>Duplicates</h3><div class="value">{int(df.duplicated().sum()):,}</div></div>
    </div>
    <div class="card" style="max-width:500px;margin:24px auto;">
        <h3 style="text-align:center">🛡️ Trust Score</h3>
        <div class="trust-badge" style="color:{'#10b981' if score>0.8 else '#f59e0b' if score>0.6 else '#ef4444'}">{score:.0%}</div>
        {dim_cards}
    </div>
</div>

<div class="section" id="charts">
    <h2 style="margin-bottom:16px">📈 Analysis Charts</h2>
    {charts_section if charts_section else '<p style="color:#94a3b8">No charts generated. Upload data and create visualizations to include them here.</p>'}
</div>

<div class="section" id="data">
    <h2 style="margin-bottom:16px">📋 Data Preview (first 100 rows)</h2>
    <div class="table-wrap">{table_html}</div>
</div>

<div class="footer">Generated by DataPilot AI v3.0 — Made with 🧠 by Mansoor Sarookh</div>

<script>
function showSection(id) {{
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    return html
