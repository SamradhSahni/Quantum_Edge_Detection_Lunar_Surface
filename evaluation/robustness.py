"""
Robustness summaries: AUC over degradation-level curves.

Robustness AUC (PROJECT_SUMMARY.md Section 10) = area under the
F1-vs-degradation-level curve, normalised to [0, 1] by the degradation
range, so higher AUC means the method degrades more gracefully.
"""

import numpy as np


def robustness_auc(levels, f1_values) -> float:
    """Normalised area under the F1-vs-level curve.

    levels and f1_values are equal-length 1-D sequences; the area is
    computed by the trapezoid rule and divided by the level span.
    """
    x = np.asarray(levels, dtype=np.float64)
    y = np.asarray(f1_values, dtype=np.float64)
    if len(x) < 2:
        return float(y[0]) if len(y) else 0.0
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = x[-1] - x[0]
    if span <= 0:
        return float(y.mean())
    return float(np.trapezoid(y, x) / span)


def degradation_table(level_grid, results_by_method: dict) -> dict:
    """Compute AUC per method for one degradation family.

    results_by_method: {method: [F1 at each level]}
    Returns {method: auc}.
    """
    return {m: robustness_auc(level_grid, vals) for m, vals in results_by_method.items()}
