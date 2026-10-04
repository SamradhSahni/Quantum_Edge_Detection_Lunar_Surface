"""
Shared experiment harness: method execution, trial loops, threshold
protocol, and CSV writing.  Every expN_*.py script uses these helpers so
the protocol is identical across experiments.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from classical.classical_pipeline import run_classical
from classical.robinson_compass import robinson_compass, robinson_direction_strength
from evaluation.metrics import full_metrics, pr_auc
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold, sweep_thresholds


# ---------------------------------------------------------------------------
# method execution
# ---------------------------------------------------------------------------
def run_all_methods(image, gt_mask, shots=config.DEFAULT_SHOTS, seed=config.SEED,
                    noise_name="ideal", quantum_sizes=None):
    """Run every method on one image; return {method: metrics dict}.

    Thresholds are RELATIVE (fraction of each strength map's max).  The
    relative threshold used here is the global validation-selected value
    (selected by select_thresholds in exp1/exp6 and frozen project-wide,
    see results/tables/thresholds.csv).
    """
    image = np.asarray(image, dtype=np.float64)
    gt = (np.asarray(gt_mask) > 0).astype(np.uint8)
    out = {}

    cls = run_classical(image)
    for name in ("sobel", "canny", "robinson"):
        strength = cls[name]["strength"]
        binary = cls[name]["binary"] if name == "canny" else None
        m = _score(strength, gt, binary=binary)
        m["method"] = name
        out[name] = m

    for qname in ("qhed", "neqr"):
        if quantum_sizes and qname not in quantum_sizes:
            continue
        res, rec = run_quantum(image, qname, shots=shots, seed=seed,
                               noise_name=noise_name, timing=True)
        m = _score(res.strength, gt)
        m["method"] = qname
        m.update({k: v for k, v in rec.items() if isinstance(v, (int, float, str))})
        out[qname] = m
    return out, out


def _score(strength, gt, binary=None):
    """Metrics for one method; binary from canny is used as-is, others
    are thresholded by the project-wide frozen relative threshold."""
    if binary is None:
        binary = apply_threshold(strength, FROZEN_REL_T)
    m = full_metrics(binary, gt, strength)
    return m


# ---------------------------------------------------------------------------
# project-wide threshold selection (validation protocol, Section 5.5)
# ---------------------------------------------------------------------------
FROZEN_REL_T = 0.3  # overwritten by select_thresholds() at run time


def select_thresholds(val_images_val_masks, methods=("sobel", "canny", "robinson", "qhed", "neqr"),
                      shots=config.DEFAULT_SHOTS):
    """Sweep thresholds on the VALIDATION set, return {method: best_rel_t}.

    val_images_val_masks: list of (image, gt_mask) pairs.
    """
    from quantum.threshold_selector import THRESHOLD_GRID
    per_method = {m: [] for m in methods}
    for img, gt in val_images_val_masks:
        gt = (np.asarray(gt) > 0).astype(np.uint8)
        maps = strength_maps_for_image(img, methods, shots)
        for m, s in maps.items():
            curve = [f1 for _, f1 in sweep_thresholds(s, gt)]
            per_method[m].append(curve)
    best = {}
    for m, curves in per_method.items():
        mean_curve = np.mean(curves, axis=0)
        best[m] = float(THRESHOLD_GRID[int(np.argmax(mean_curve))])
    return best


def strength_maps_for_image(image, methods, shots=config.DEFAULT_SHOTS):
    """Continuous strength maps for the requested methods on one image."""
    image = np.asarray(image, dtype=np.float64)
    maps = {}
    cls = run_classical(image)
    if "sobel" in methods:
        maps["sobel"] = cls["sobel"]["strength"]
    if "canny" in methods:
        maps["canny"] = cls["canny"]["strength"]
    if "robinson" in methods:
        maps["robinson"] = cls["robinson"]["strength"]
    for qname in ("qhed", "neqr"):
        if qname in methods:
            res, _ = run_quantum(image, qname, shots=shots)
            maps[qname] = res.strength
    return maps


# ---------------------------------------------------------------------------
# trial loops
# ---------------------------------------------------------------------------
def trials_metrics(runner, n_trials=config.N_TRIALS, seed0=config.SEED):
    """Repeat runner(trial_seed) n times; return (mean_metrics, all_rows).

    runner must return a metrics dict.  Quantum runs vary the shot seed;
    classical runs are deterministic (std = 0, as the spec requires).
    """
    rows = []
    for t in range(n_trials):
        m = runner(seed0 + t)
        m["trial"] = t
        rows.append(m)
    df = pd.DataFrame(rows)
    num = df.select_dtypes(include=[np.number])
    mean, std = num.mean(), num.std(ddof=1) if n_trials > 1 else num.std(ddof=0) * 0
    return mean.to_dict(), std.to_dict(), df


def save_table(df: pd.DataFrame, name: str):
    os.makedirs(config.TABLE_DIR, exist_ok=True)
    path = os.path.join(config.TABLE_DIR, name)
    df.to_csv(path, index=False)
    print(f"  saved {path} ({len(df)} rows)")
    return path


def save_figure(fig, name: str):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")
    return path


def fmt_mean_std(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"
