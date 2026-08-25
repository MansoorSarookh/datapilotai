"""
DataPilot AI — Cluster Profiler
Auto-generates cluster personas with dominant features and descriptions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def profile_clusters(
    df: pd.DataFrame,
    labels: List[int],
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Generate cluster profiles from K-Means or other clustering results.

    Returns per-cluster statistics, dominant features, and AI-friendly descriptions.
    """
    df_profiled = df.copy()
    df_profiled["_cluster"] = labels

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()

    profiles = {}
    n_clusters = len(set(labels))

    # Global stats for comparison
    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std()

    for cluster_id in sorted(set(labels)):
        cluster_data = df_profiled[df_profiled["_cluster"] == cluster_id]
        n = len(cluster_data)

        # Numeric statistics
        numeric_stats = {}
        distinguishing = []
        for col in feature_cols:
            if col in cluster_data.columns:
                cluster_mean = cluster_data[col].mean()
                cluster_std = cluster_data[col].std()
                global_mean = global_means.get(col, 0)
                global_std = global_stds.get(col, 1)

                # Z-score of cluster mean vs global
                z = (cluster_mean - global_mean) / (global_std + 1e-9)

                numeric_stats[col] = {
                    "mean": round(float(cluster_mean), 3),
                    "std": round(float(cluster_std), 3),
                    "global_mean": round(float(global_mean), 3),
                    "z_score": round(float(z), 2),
                    "direction": "▲ Above" if z > 0.5 else "▼ Below" if z < -0.5 else "● Average",
                }

                if abs(z) > 0.5:
                    distinguishing.append({
                        "feature": col,
                        "z_score": round(float(z), 2),
                        "direction": "high" if z > 0 else "low",
                        "value": round(float(cluster_mean), 2),
                    })

        # Sort distinguishing features by absolute z-score
        distinguishing.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        # Categorical breakdown (top categories per cat column)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_profiles = {}
        for col in cat_cols[:5]:  # Limit to 5 cat columns
            if col in cluster_data.columns:
                top = cluster_data[col].value_counts(normalize=True).head(3)
                cat_profiles[col] = {
                    str(k): round(float(v) * 100, 1) for k, v in top.items()
                }

        # Generate persona description
        persona = _generate_persona(cluster_id, n, distinguishing, cat_profiles)

        profiles[f"Cluster {cluster_id}"] = {
            "size": n,
            "percentage": round(n / len(df) * 100, 1),
            "numeric_stats": numeric_stats,
            "distinguishing_features": distinguishing[:10],
            "categorical_profiles": cat_profiles,
            "persona": persona,
        }

    # Radar chart data (for visualization)
    radar_data = _build_radar_data(profiles, feature_cols[:8])

    return {
        "profiles": profiles,
        "n_clusters": n_clusters,
        "feature_cols": feature_cols,
        "radar_data": radar_data,
    }


def generate_ai_personas(cluster_result: Dict, df: pd.DataFrame) -> Dict[str, str]:
    """Use LLM to generate natural-language cluster personas."""
    personas = {}
    try:
        from app.modules.llm_router import get_llm_router
        router = get_llm_router()

        for name, profile in cluster_result.get("profiles", {}).items():
            distinguishing = profile.get("distinguishing_features", [])
            if not distinguishing:
                personas[name] = profile.get("persona", "Average cluster")
                continue

            features_desc = ", ".join([
                f"{f['feature']} is {f['direction']} ({f['value']})"
                for f in distinguishing[:5]
            ])

            prompt = (
                f"Generate a short, engaging 2-sentence persona for this data cluster:\n"
                f"Cluster size: {profile['size']} ({profile['percentage']}% of data)\n"
                f"Key features: {features_desc}\n"
                f"Give it a creative name and describe what makes this group unique."
            )

            messages = [
                {"role": "system", "content": "You're a data analyst creating customer personas. Be concise and insightful."},
                {"role": "user", "content": prompt},
            ]

            response = router.chat(messages, max_tokens=100, temperature=0.5)
            if response:
                personas[name] = response
            else:
                personas[name] = profile.get("persona", "")
    except Exception:
        for name, profile in cluster_result.get("profiles", {}).items():
            personas[name] = profile.get("persona", "")

    return personas


def _generate_persona(cluster_id: int, size: int, distinguishing: List, cat_profiles: Dict) -> str:
    """Generate a template-based persona description."""
    if not distinguishing:
        return f"Cluster {cluster_id}: Average group ({size} members) with no strongly distinguishing features."

    high_features = [f for f in distinguishing if f["direction"] == "high"][:3]
    low_features = [f for f in distinguishing if f["direction"] == "low"][:3]

    parts = [f"**Cluster {cluster_id}** ({size} members): "]

    if high_features:
        feat_str = ", ".join([f"'{f['feature']}'" for f in high_features])
        parts.append(f"Above-average in {feat_str}. ")

    if low_features:
        feat_str = ", ".join([f"'{f['feature']}'" for f in low_features])
        parts.append(f"Below-average in {feat_str}. ")

    # Add top categorical
    for col, vals in list(cat_profiles.items())[:1]:
        if vals:
            top_cat = list(vals.keys())[0]
            top_pct = list(vals.values())[0]
            parts.append(f"Primarily '{top_cat}' ({top_pct}%) in '{col}'.")

    return "".join(parts)


def _build_radar_data(profiles: Dict, features: List) -> Dict:
    """Build normalized data for radar chart visualization."""
    radar = {"features": features, "clusters": {}}
    for name, profile in profiles.items():
        values = []
        for feat in features:
            stats = profile.get("numeric_stats", {}).get(feat, {})
            z = stats.get("z_score", 0)
            # Normalize z-score to 0-1 range for radar
            normalized = min(max((z + 3) / 6, 0), 1)
            values.append(round(normalized, 3))
        radar["clusters"][name] = values
    return radar
