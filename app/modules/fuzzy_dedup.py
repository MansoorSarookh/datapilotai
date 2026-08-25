"""
DataPilot AI — Fuzzy Deduplication Engine
Near-duplicate detection using text similarity for string columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def find_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.85,
    max_comparisons: int = 50000,
) -> Dict:
    """
    Find near-duplicate rows using text similarity.

    threshold: minimum similarity to flag as duplicate (0.0-1.0)
    Returns groups of near-duplicate row indices.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return {"error": "rapidfuzz not installed. Run: pip install rapidfuzz"}

    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return {"error": "No valid columns selected"}

    # Create composite string for comparison
    df_str = df[valid_cols].fillna("").astype(str)
    composite = df_str.apply(lambda r: " | ".join(r), axis=1)

    n = len(composite)
    if n * (n - 1) / 2 > max_comparisons:
        sample_n = int(np.sqrt(max_comparisons * 2))
        sample_idx = np.random.choice(n, min(sample_n, n), replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n)

    # Find duplicates
    groups = []
    seen = set()

    for i_pos, i in enumerate(sample_idx):
        if i in seen:
            continue
        group = [int(i)]
        for j in sample_idx[i_pos + 1:]:
            if j in seen:
                continue
            try:
                sim = fuzz.ratio(composite.iloc[i], composite.iloc[j]) / 100
                if sim >= threshold:
                    group.append(int(j))
                    seen.add(j)
            except Exception:
                continue

        if len(group) > 1:
            groups.append({
                "indices": group,
                "size": len(group),
                "sample_values": {
                    col: [str(df[col].iloc[idx]) for idx in group[:3]]
                    for col in valid_cols[:3]
                },
            })
        seen.add(i)

    total_fuzzy_dupes = sum(g["size"] - 1 for g in groups)

    return {
        "n_groups": len(groups),
        "total_fuzzy_duplicates": total_fuzzy_dupes,
        "groups": groups[:100],
        "threshold": threshold,
        "columns_compared": valid_cols,
        "rows_scanned": len(sample_idx),
    }


def merge_fuzzy_duplicates(
    df: pd.DataFrame,
    groups: List[Dict],
    strategy: str = "keep_first",
) -> Tuple[pd.DataFrame, int]:
    """
    Merge fuzzy duplicate groups.

    strategy: 'keep_first', 'keep_last', 'remove_all'
    """
    indices_to_remove = set()

    for group in groups:
        indices = group.get("indices", [])
        if len(indices) < 2:
            continue

        if strategy == "keep_first":
            indices_to_remove.update(indices[1:])
        elif strategy == "keep_last":
            indices_to_remove.update(indices[:-1])
        elif strategy == "remove_all":
            indices_to_remove.update(indices)

    valid_indices = [i for i in indices_to_remove if i < len(df)]
    df_clean = df.drop(index=df.index[list(valid_indices)]).reset_index(drop=True)
    return df_clean, len(valid_indices)
