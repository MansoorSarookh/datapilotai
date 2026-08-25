"""
DataPilot AI — Trust Score Display Component
"""

import streamlit as st
import plotly.graph_objects as go


def render_trust_score(trust: dict):
    """Render the full Trust Score widget."""
    score = trust.get("overall", 0)
    dims = trust.get("dimensions", {})
    flags = trust.get("flags", [])
    recs = trust.get("recommendations", [])
    label = trust.get("label", "")
    color = trust.get("color", "gray")
    color_map = {"green": "#10b981", "orange": "#f59e0b", "red": "#ef4444"}
    hex_color = color_map.get(color, "#6366f1")

    st.markdown("### 🛡️ Dataset Trust Score")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": label, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": hex_color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 60], "color": "#fee2e2"},
                    {"range": [60, 82], "color": "#fef9c3"},
                    {"range": [82, 100], "color": "#d1fae5"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": round(score * 100, 1),
                },
            },
        ))
        fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dimension bars
        for dim_name, dim_val in dims.items():
            bar_color = "#10b981" if dim_val > 0.80 else "#f59e0b" if dim_val > 0.60 else "#ef4444"
            pct = round(dim_val * 100)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                <span style="font-weight:600;color:#374151;">{dim_name.capitalize()}</span>
                <span style="font-weight:700;color:{bar_color};">{pct}%</span>
              </div>
              <div style="background:#e5e7eb;border-radius:999px;height:8px;">
                <div style="background:{bar_color};width:{pct}%;height:8px;border-radius:999px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Flags & recommendations
    if flags:
        with st.expander(f"⚠️ {len(flags)} Issue(s) Detected", expanded=len(flags) > 0):
            for flag in flags[:8]:
                st.warning(flag)

    if recs:
        with st.expander("✅ Recommendations"):
            for rec in recs[:6]:
                st.info(f"→ {rec}")
 
