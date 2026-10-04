"""
Validation-set threshold selection (never a fixed magic number).

Protocol from PROJECT_SUMMARY.md Section 5.5:

    1. Take a held-out VALIDATION subset (synthetic images with exact
       ground-truth edge masks).
    2. Sweep the threshold over a dense grid.
    3. Select the threshold maximising boundary F1 (1-px tolerance) on
       the validation set, separately for every method.
    4. Freeze the thresholds and apply them to all TEST evaluations.
    5. Full PR curves are still reported (they are threshold-free).
"""

import numpy as np

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def sweep_thresholds(strength_map: np.ndarray, gt_mask: np.ndarray,
                     grid=None, tol: int = 1):
    """Boundary F1 (with tolerance) for every threshold in the grid."""
    from evaluation.metrics import boundary_f1
    if grid is None:
        grid = THRESHOLD_GRID
    smax = float(strength_map.max())
    out = []
    for t in grid:
        if smax > 0:
            binary = (strength_map >= t * smax).astype(np.uint8)
        else:
            binary = np.zeros_like(strength_map, dtype=np.uint8)
        f1 = boundary_f1(binary, gt_mask, tolerance_px=tol)
        out.append((float(t), f1))
    return out


def select_threshold(strength_maps, gt_masks, grid=None, tol: int = 1):
    """Threshold (relative fraction of each map's max) maximising mean
    validation boundary-F1.

    Returns (best_t, curve) where curve is the mean F1 per grid point.
    """
    if grid is None:
        grid = THRESHOLD_GRID
    curves = [np.array([f1 for _, f1 in sweep_thresholds(m, g, grid, tol)])
              for m, g in zip(strength_maps, gt_masks)]
    mean_curve = np.mean(curves, axis=0)
    best_idx = int(np.argmax(mean_curve))
    return float(grid[best_idx]), mean_curve


def apply_threshold(strength_map: np.ndarray, rel_threshold: float) -> np.ndarray:
    """Apply a relative threshold (fraction of the map's maximum)."""
    smax = float(strength_map.max()) if strength_map.size else 0.0
    if smax <= 0:
        return np.zeros(strength_map.shape, dtype=np.uint8)
    return (strength_map >= rel_threshold * smax).astype(np.uint8)
