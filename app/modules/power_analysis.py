"""
DataPilot AI — Power Analysis Tool
Compute required sample sizes and statistical power for study planning.
"""

import numpy as np
from typing import Dict, Optional


def compute_power_analysis(
    test_type: str = "t-test",
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    n: Optional[int] = None,
    ratio: float = 1.0,
) -> Dict:
    """
    Compute sample size or power for a given test type.

    test_type: 't-test', 'anova', 'chi-square', 'correlation'
    If n is None, computes required sample size.
    If n is provided, computes achievable power.
    """
    try:
        from statsmodels.stats.power import (
            TTestIndPower,
            FTestAnovaPower,
            GofChisquarePower,
            NormalIndPower,
        )
    except ImportError:
        return {"error": "statsmodels is required for power analysis"}

    result = {
        "test_type": test_type,
        "effect_size": effect_size,
        "alpha": alpha,
        "effect_label": _effect_label(effect_size, test_type),
    }

    try:
        if test_type == "t-test":
            analysis = TTestIndPower()
            if n is None:
                required_n = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                    alternative="two-sided",
                )
                result["required_n_per_group"] = int(np.ceil(required_n))
                result["total_n"] = int(np.ceil(required_n * (1 + ratio)))
                result["power"] = power
            else:
                achieved_power = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs1=n,
                    ratio=ratio,
                    alternative="two-sided",
                )
                result["achieved_power"] = round(float(achieved_power), 4)
                result["n_per_group"] = n
                result["sufficient"] = achieved_power >= 0.80

        elif test_type == "anova":
            analysis = FTestAnovaPower()
            if n is None:
                required_n = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    k_groups=3,
                )
                result["required_n_per_group"] = int(np.ceil(required_n))
                result["power"] = power
            else:
                achieved_power = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=n,
                    k_groups=3,
                )
                result["achieved_power"] = round(float(achieved_power), 4)
                result["sufficient"] = achieved_power >= 0.80

        elif test_type == "chi-square":
            analysis = GofChisquarePower()
            if n is None:
                required_n = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    n_bins=4,
                )
                result["required_n"] = int(np.ceil(required_n))
                result["power"] = power
            else:
                achieved_power = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=n,
                    n_bins=4,
                )
                result["achieved_power"] = round(float(achieved_power), 4)
                result["sufficient"] = achieved_power >= 0.80

        elif test_type == "correlation":
            analysis = NormalIndPower()
            if n is None:
                required_n = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    alternative="two-sided",
                )
                result["required_n"] = int(np.ceil(required_n))
                result["power"] = power
            else:
                achieved_power = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=n,
                    alternative="two-sided",
                )
                result["achieved_power"] = round(float(achieved_power), 4)
                result["sufficient"] = achieved_power >= 0.80

        # Generate power curve data
        result["power_curve"] = _compute_power_curve(test_type, effect_size, alpha)

    except Exception as e:
        result["error"] = str(e)

    return result


def _compute_power_curve(test_type: str, effect_size: float, alpha: float) -> Dict:
    """Compute power as a function of sample size for plotting."""
    try:
        from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
    except ImportError:
        return {"n_values": [], "power_values": []}

    n_values = list(range(10, 510, 10))
    power_values = []

    for n in n_values:
        try:
            if test_type == "t-test":
                p = TTestIndPower().solve_power(
                    effect_size=effect_size, alpha=alpha, nobs1=n, alternative="two-sided"
                )
            elif test_type == "anova":
                p = FTestAnovaPower().solve_power(
                    effect_size=effect_size, alpha=alpha, nobs=n, k_groups=3
                )
            else:
                p = TTestIndPower().solve_power(
                    effect_size=effect_size, alpha=alpha, nobs1=n, alternative="two-sided"
                )
            power_values.append(round(float(p), 4))
        except Exception:
            power_values.append(None)

    return {"n_values": n_values, "power_values": power_values}


def _effect_label(effect_size: float, test_type: str) -> str:
    """Classify effect size as small/medium/large."""
    if test_type in ("t-test", "anova"):
        if effect_size < 0.2:
            return "Negligible"
        elif effect_size < 0.5:
            return "Small"
        elif effect_size < 0.8:
            return "Medium"
        else:
            return "Large"
    elif test_type == "chi-square":
        if effect_size < 0.1:
            return "Negligible"
        elif effect_size < 0.3:
            return "Small"
        elif effect_size < 0.5:
            return "Medium"
        else:
            return "Large"
    elif test_type == "correlation":
        if effect_size < 0.1:
            return "Negligible"
        elif effect_size < 0.3:
            return "Small"
        elif effect_size < 0.5:
            return "Medium"
        else:
            return "Large"
    return "Unknown"


EFFECT_SIZE_PRESETS = {
    "t-test": {"Small": 0.2, "Medium": 0.5, "Large": 0.8},
    "anova": {"Small": 0.1, "Medium": 0.25, "Large": 0.4},
    "chi-square": {"Small": 0.1, "Medium": 0.3, "Large": 0.5},
    "correlation": {"Small": 0.1, "Medium": 0.3, "Large": 0.5},
}
