"""
DataPilot AI — AI Statistical Narrator
Generates plain-English explanations of statistical test results.
Falls back to template-based narration when LLM is unavailable.
"""

import pandas as pd
from typing import Dict, Optional


def narrate_test_result(test_result: Dict, col1: str, col2: str) -> str:
    """
    Generate a plain-English narrative for a statistical test result.
    Uses LLM if available, otherwise uses templates.
    """
    try:
        from app.modules.llm_router import get_llm_router
        router = get_llm_router()
        prompt = _build_narration_prompt(test_result, col1, col2)
        messages = [
            {"role": "system", "content": (
                "You are a statistics tutor explaining results to a non-technical audience. "
                "Use simple, everyday language. Avoid jargon. Keep it under 120 words. "
                "Structure: 1) What was tested, 2) What was found, 3) What it means practically."
            )},
            {"role": "user", "content": prompt},
        ]
        response = router.chat(messages, max_tokens=250, temperature=0.3)
        if response:
            return response
    except Exception:
        pass

    return _template_narration(test_result, col1, col2)


def narrate_distribution(analysis: Dict, column: str) -> str:
    """Generate plain-English narrative for a distribution analysis."""
    try:
        from app.modules.llm_router import get_llm_router
        router = get_llm_router()
        prompt = (
            f"Explain this distribution analysis to a beginner:\n"
            f"Column: '{column}'\n"
            f"Mean: {analysis.get('mean')}, Std: {analysis.get('std')}\n"
            f"Skewness: {analysis.get('skewness')} ({analysis.get('skewness_label')})\n"
            f"Normality test: {'Passed' if analysis.get('normality', {}).get('passed') else 'Failed'}\n"
            f"IQR Outliers: {analysis.get('iqr_outliers')}, Z-Score Outliers: {analysis.get('zscore_outliers')}\n"
            f"Transform suggestion: {analysis.get('transform_suggestion')}\n"
            f"Explain in simple terms what this means for analysis."
        )
        messages = [
            {"role": "system", "content": "You are a statistics tutor. Use simple language. Under 100 words."},
            {"role": "user", "content": prompt},
        ]
        response = router.chat(messages, max_tokens=200, temperature=0.3)
        if response:
            return response
    except Exception:
        pass

    return _template_distribution_narration(analysis, column)


def narrate_hypothesis_result(result: Dict) -> str:
    """Generate plain-English narrative for a hypothesis test."""
    try:
        from app.modules.llm_router import get_llm_router
        router = get_llm_router()
        prompt = (
            f"Explain this hypothesis test to a non-statistician:\n"
            f"H₀: {result.get('null_hypothesis')}\n"
            f"H₁: {result.get('alternative_hypothesis')}\n"
            f"Test: {result.get('test_used')}\n"
            f"P-value: {result.get('result', {}).get('p_value')}\n"
            f"Decision: {result.get('conclusion')}\n"
            f"Use everyday language. Under 100 words."
        )
        messages = [
            {"role": "system", "content": "You are a statistics tutor. Explain simply."},
            {"role": "user", "content": prompt},
        ]
        response = router.chat(messages, max_tokens=200, temperature=0.3)
        if response:
            return response
    except Exception:
        pass

    return result.get("conclusion", "")


# ── Template fallbacks ────────────────────────────────────────────────────────

def _build_narration_prompt(result: Dict, col1: str, col2: str) -> str:
    test = result.get("test", "Unknown")
    r = result.get("result", {})
    p = r.get("p_value", "N/A")
    sig = r.get("significant", False)
    stat = r.get("statistic", r.get("chi2", "N/A"))
    effect = r.get("effect_size", r.get("cramers_v", r.get("strength", "")))

    return (
        f"Explain this statistical test result in plain English:\n"
        f"Test: {test}\n"
        f"Variables: '{col1}' and '{col2}'\n"
        f"Test Statistic: {stat}\n"
        f"P-value: {p}\n"
        f"Significant: {'Yes' if sig else 'No'}\n"
        f"Effect Size: {effect}\n"
        f"Interpretation: {result.get('interpretation', '')}"
    )


def _template_narration(result: Dict, col1: str, col2: str) -> str:
    test = result.get("test", "statistical test")
    r = result.get("result", {})
    p = r.get("p_value")
    sig = r.get("significant", False)
    effect = r.get("effect_size", r.get("cramers_v", r.get("strength", "")))

    if sig:
        verdict = (
            f"📝 **In plain English:** We used the {test} to check if there's a real "
            f"connection between **{col1}** and **{col2}**. The answer is **yes** — "
            f"the data shows a statistically significant relationship "
            f"(p-value = {p:.4f if isinstance(p, (int, float)) else p}). "
        )
        if effect:
            verdict += f"The effect size is **{effect}**, which tells us how strong this relationship is. "
        verdict += (
            "This means the pattern we see is unlikely to be just due to chance. "
            "You can have confidence that this finding is real."
        )
    else:
        verdict = (
            f"📝 **In plain English:** We used the {test} to check if there's a real "
            f"connection between **{col1}** and **{col2}**. The answer is **no** — "
            f"we didn't find enough evidence of a significant relationship "
            f"(p-value = {p:.4f if isinstance(p, (int, float)) else p}). "
            "This doesn't necessarily mean there's no relationship at all — "
            "it just means our data isn't strong enough to prove one exists."
        )
    return verdict


def _template_distribution_narration(analysis: Dict, column: str) -> str:
    skew_label = analysis.get("skewness_label", "unknown")
    normal = analysis.get("normality", {}).get("passed", False)
    outliers = analysis.get("iqr_outliers", 0) + analysis.get("zscore_outliers", 0)
    transform = analysis.get("transform_suggestion", "None needed")

    text = (
        f"📝 **In plain English:** The values in **{column}** are "
        f"**{skew_label}**, meaning "
    )
    if "right" in skew_label:
        text += "most values are on the lower end with a few very high values pulling the average up. "
    elif "left" in skew_label:
        text += "most values are on the higher end with a few very low values. "
    else:
        text += "they're fairly evenly spread around the average — like a bell curve. "

    if normal:
        text += "The distribution **passes the normality test**, so you can safely use standard statistical methods. "
    else:
        text += "The distribution **does not follow a normal (bell curve) pattern**, so some statistical methods may not work well. "

    if outliers > 0:
        text += f"There are **{outliers} potential outliers** — unusual values that are far from the rest. "

    text += f"**Suggestion:** {transform}."
    return text
