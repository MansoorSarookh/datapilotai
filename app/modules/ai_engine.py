"""
DataPilot AI — AI Engine
Integrates LLM Router for multi-provider AI with fallback to rule-based heuristics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re


# ── LLM Router integration ───────────────────────────────────────────────────
def get_groq_client():
    """Initialize Groq client (backward compatibility). Prefer LLM Router."""
    try:
        from groq import Groq
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            return Groq(api_key=api_key)
    except Exception:
        pass
    return None


def _get_router():
    """Get the LLM Router instance (returns None if not available)."""
    try:
        from app.modules.llm_router import get_llm_router
        return get_llm_router()
    except Exception:
        return None


def _router_chat(messages: list, max_tokens: int = 600, temperature: float = 0.4) -> str:
    """Try LLM Router first, fall back to direct Groq, then return None."""
    # Try LLM Router (handles multi-provider fallback)
    router = _get_router()
    if router:
        try:
            result = router.chat(messages, max_tokens=max_tokens, temperature=temperature)
            if result:
                return result
        except Exception:
            pass

    # Direct Groq fallback
    client = get_groq_client()
    if client:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    return None


def get_active_provider_display() -> str:
    """Get the display name of the active LLM provider."""
    router = _get_router()
    if router:
        return router.get_active_provider_display()
    if get_groq_client():
        return "Groq (llama-3.3-70b)"
    return "Rule-based Fallback"


# ── Context builder ───────────────────────────────────────────────────────────
def build_dataset_context(df: pd.DataFrame, max_rows: int = 5) -> dict:
    """Build a structured dataset context dict for LLM prompts."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    stats = {}
    for col in numeric_cols[:10]:
        stats[col] = {
            "mean": round(float(df[col].mean()), 3),
            "std": round(float(df[col].std()), 3),
            "min": round(float(df[col].min()), 3),
            "max": round(float(df[col].max()), 3),
            "nulls": int(df[col].isna().sum()),
        }

    return {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": datetime_cols,
        "null_counts": df.isna().sum().to_dict(),
        "numeric_stats": stats,
        "sample": df.head(max_rows).to_dict(orient="records"),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }


def build_system_prompt(code_mode: bool = False) -> str:
    base = """You are DataPilot, an expert AI data scientist assistant. Your role is to:
1. Analyze datasets and provide concise, numbered insights backed by actual numbers from the data.
2. Suggest actionable next steps (cleaning, feature engineering, model selection).
3. Identify data quality issues, bias, and ML suitability.
4. Recommend appropriate statistical tests and visualizations.
5. Be direct and professional. NEVER make up numbers — only cite figures from the dataset context provided.

Format responses with:
- 📊 **Insight:** <main finding>
- ⚠️ **Risk:** <any data quality concern>
- ✅ **Action:** <recommended next step>

Keep responses under 300 words unless specifically asked for detail."""

    if code_mode:
        base += """

Additionally, after your analysis, provide a Python code block showing how to reproduce the analysis using pandas/plotly.
Wrap code in ```python ... ``` blocks. Use the variable `df` for the DataFrame."""

    return base


def build_user_prompt(question: str, context: dict, chat_history: list) -> str:
    """Construct the full prompt with dataset context + history."""
    history_text = ""
    for msg in chat_history[-4:]:  # last 4 messages for context window management
        role = "User" if msg["role"] == "user" else "DataPilot"
        history_text += f"{role}: {msg['content']}\n"

    ctx_json = json.dumps(context, default=str, indent=2)
    return f"""Dataset Context:
{ctx_json}

Previous conversation:
{history_text}

User Question: {question}

Provide a data-driven, concise response."""


# ── Main chat function ────────────────────────────────────────────────────────
def chat_with_data(
    question: str,
    df: pd.DataFrame,
    chat_history: list,
    code_mode: bool = False,
) -> str:
    """
    Send a question about the dataset to LLM (via Router or Groq).
    Falls back to rule-based heuristics if all providers are unavailable.
    """
    context = build_dataset_context(df)
    user_prompt = build_user_prompt(question, context, chat_history)
    messages = [
        {"role": "system", "content": build_system_prompt(code_mode=code_mode)},
        {"role": "user", "content": user_prompt},
    ]

    result = _router_chat(messages, max_tokens=600, temperature=0.4)
    if result:
        return result

    return _fallback_response(question, df, context)


# ── Narrative insight generator ───────────────────────────────────────────────
def generate_chart_narrative(
    chart_type: str,
    column: str,
    df: pd.DataFrame,
    secondary_column: str = None,
) -> str:
    """
    Generate an AI narrative for a given chart. Uses LLM Router if available,
    otherwise uses rule-based templates.
    """
    context = build_dataset_context(df)
    prompt = f"""I just created a {chart_type} chart for the column '{column}'{f" vs '{secondary_column}'" if secondary_column else ""}.

Dataset context: {json.dumps(context, default=str)}

Generate a short chart narrative (max 100 words) with:
- 📊 Explanation of what the chart shows
- 📈 Key insight or finding  
- ⚠️ Any risk or warning
- ✅ Suggested action"""

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    result = _router_chat(messages, max_tokens=200, temperature=0.3)
    if result:
        return result

    return _template_narrative(chart_type, column, df, secondary_column)


def generate_executive_summary(df: pd.DataFrame, trust_score: dict, insights: list) -> str:
    """Generate an AI executive summary for the report."""
    context = build_dataset_context(df)
    prompt = f"""Generate a professional executive summary (200 words max) for a data analysis report.

Dataset: {context['shape']['rows']} rows × {context['shape']['cols']} columns
Trust Score: {trust_score.get('overall', 0):.0%}
Key Flags: {trust_score.get('flags', [])}
Top Insights: {insights[:5]}

Include: dataset overview, quality assessment, key findings, and recommendations."""

    messages = [
        {"role": "system", "content": "You are a senior data analyst writing executive reports."},
        {"role": "user", "content": prompt},
    ]

    result = _router_chat(messages, max_tokens=350, temperature=0.3)
    if result:
        return result

    return _fallback_executive_summary(df, trust_score)


# ── Fallback heuristics ───────────────────────────────────────────────────────
def _fallback_response(question: str, df: pd.DataFrame, context: dict) -> str:
    """Rule-based fallback when all LLM providers are unavailable."""
    q = question.lower()
    rows, cols = context["shape"]["rows"], context["shape"]["cols"]
    numeric_cols = context["numeric_columns"]
    null_counts = context["null_counts"]
    
    total_nulls = sum(null_counts.values())
    null_pct = total_nulls / (rows * cols) * 100 if rows * cols > 0 else 0

    if any(w in q for w in ["reliable", "trust", "quality", "good"]):
        return (
            f"📊 **Dataset Quality Assessment**\n\n"
            f"Your dataset has **{rows:,} rows** and **{cols} columns**.\n\n"
            f"- Missing values: **{null_pct:.1f}%** of all cells\n"
            f"- Duplicate rows: **{context.get('duplicates', 0)}**\n"
            f"- Numeric columns: **{len(numeric_cols)}**\n\n"
            f"{'⚠️ **Risk:** High missing data may affect analysis.' if null_pct > 10 else '✅ Missing data is within acceptable range.'}\n\n"
            f"✅ **Action:** Review columns with high missing rates before modeling."
        )
    elif any(w in q for w in ["trend", "pattern", "insight", "tell"]):
        if numeric_cols:
            col = numeric_cols[0]
            try:
                series = df[col].dropna()
                trend = "increasing" if series.iloc[-1] > series.iloc[0] else "decreasing"
                return (
                    f"📊 **Key Patterns Detected**\n\n"
                    f"- Column **'{col}'**: {trend} trend (first={series.iloc[0]:.2f} → last={series.iloc[-1]:.2f})\n"
                    f"- Mean: **{series.mean():.3f}**, Std: **{series.std():.3f}**\n\n"
                    f"✅ **Action:** Explore correlation heatmap to discover inter-variable relationships."
                )
            except Exception:
                pass
        return f"📊 Dataset has {cols} columns. Go to the **Analyze** tab for visualizations and trends."
    elif any(w in q for w in ["column", "feature", "important", "matter"]):
        return (
            f"📊 **Feature Overview**\n\n"
            f"- **{len(numeric_cols)} numeric** columns available for modeling.\n"
            f"- **{len(context['categorical_columns'])} categorical** columns (may need encoding).\n\n"
            f"⚠️ **Risk:** High-cardinality categorical features can cause overfitting.\n\n"
            f"✅ **Action:** Use the **ML Studio** tab to get auto feature importance after training."
        )
    elif any(w in q for w in ["ml", "model", "predict", "machine"]):
        return (
            f"🎯 **ML Readiness**\n\n"
            f"- Dataset: **{rows:,} rows** ({'✅ sufficient' if rows > 200 else '⚠️ may be too small'} for ML)\n"
            f"- Features: **{len(numeric_cols)}** numeric columns available\n\n"
            f"✅ **Action:** Go to **ML Studio** → select your target column → run Auto-ML."
        )
    else:
        return (
            f"📊 **Dataset Summary**\n\n"
            f"- Shape: **{rows:,} rows × {cols} columns**\n"
            f"- Numeric features: **{len(numeric_cols)}**\n"
            f"- Missing data: **{null_pct:.1f}%**\n\n"
            f"Try asking: *'What trends exist?'*, *'Is this dataset reliable?'*, or *'Prepare ML-ready data'*"
        )


def _template_narrative(chart_type: str, column: str, df: pd.DataFrame, secondary: str = None) -> str:
    """Template-based chart narrative."""
    try:
        if column in df.select_dtypes(include=np.number).columns:
            series = df[column].dropna()
            mean_val = series.mean()
            std_val = series.std()
            skew = series.skew()
            skew_dir = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "approximately normal"
            outliers = len(series[(series - mean_val).abs() > 3 * std_val])
            return (
                f"📊 **{chart_type} — {column}**\n\n"
                f"Distribution is **{skew_dir}** (skewness={skew:.2f}). "
                f"Mean = {mean_val:.3f}, Std = {std_val:.3f}.\n\n"
                f"{'⚠️ **Risk:** ' + str(outliers) + ' potential outliers detected (>3σ).' if outliers > 0 else '✅ No significant outliers detected.'}\n\n"
                f"✅ **Action:** {'Consider log transformation to normalize distribution.' if abs(skew) > 1 else 'Distribution is suitable for parametric analysis.'}"
            )
    except Exception:
        pass
    return f"📊 **{chart_type}** visualization for **{column}**."


def _fallback_executive_summary(df: pd.DataFrame, trust_score: dict) -> str:
    rows, cols = df.shape
    score = trust_score.get("overall", 0)
    label = "🟢 Reliable" if score > 0.80 else "🟡 Needs Review" if score > 0.60 else "🔴 High Risk"
    return (
        f"**Executive Summary — DataPilot AI Analysis**\n\n"
        f"Dataset contains {rows:,} records across {cols} variables. "
        f"Overall data quality is rated **{label}** with a trust score of **{score:.0%}**. "
        f"Key focus areas include addressing missing values, validating data types, "
        f"and reviewing class balance before proceeding with machine learning workflows.\n\n"
        f"Recommendation: Complete the data cleaning pipeline in the Clean tab and "
        f"validate the dataset using the Statistical Intelligence Engine before modeling."
    )
