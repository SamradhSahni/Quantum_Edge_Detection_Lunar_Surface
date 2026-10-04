"""
Edge-quality evaluation metrics against independent ground-truth masks.

All metrics compare a BINARY predicted edge map with a BINARY ground-truth
mask.  Tolerance variants dilate the GT by t pixels before matching --
critical at 4x4 and 8x8 where a 1-pixel offset is a large fraction of the
image (PROJECT_SUMMARY.md Section 10).

Provided:  boundary F1 (exact, 1px, 2px), precision, recall, IoU,
Chamfer distance, and the full precision-recall curve + AUC.
"""

import numpy as np
import cv2
from sklearn.metrics import auc


def _as_binary(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) > 0).astype(np.uint8)


def _dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return mask
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _tp_fp_fn(pred: np.ndarray, gt: np.ndarray, tol: int):
    gt_d = _dilate(gt, tol)
    tp = int(np.logical_and(pred, gt_d).sum())
    fp = int(np.logical_and(pred, 1 - gt_d).sum())
    fn = int(np.logical_and(gt, 1 - pred).sum())
    return tp, fp, fn


def precision_recall_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 0):
    """Precision, recall, F1 at a given tolerance."""
    pred, gt = _as_binary(pred), _as_binary(gt)
    tp, fp, fn = _tp_fp_fn(pred, gt, tol)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 0) -> float:
    return precision_recall_f1(pred, gt, tolerance_px)[2]


def iou(pred: np.ndarray, gt: np.ndarray, tol: int = 1) -> float:
    """Jaccard index of pred vs tol-dilated GT."""
    pred, gt = _as_binary(pred), _as_binary(gt)
    gt_d = _dilate(gt, tol)
    inter = np.logical_and(pred, gt_d).sum()
    union = np.logical_or(pred, gt_d).sum()
    return float(inter / union) if union else 1.0


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric mean nearest-edge-pixel distance via distance transforms."""
    pred, gt = _as_binary(pred), _as_binary(gt)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(max(pred.shape))
    dt_to_gt = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 3)
    dt_to_pred = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 3)
    d1 = dt_to_gt[pred > 0].mean() if (pred > 0).any() else 0.0
    d2 = dt_to_pred[gt > 0].mean() if (gt > 0).any() else 0.0
    return float((d1 + d2) / 2.0)


def pr_curve(strength: np.ndarray, gt: np.ndarray, n_points: int = 100):
    """Full precision-recall curve from a continuous strength map."""
    gt = _as_binary(gt)
    s = np.asarray(strength, dtype=np.float64)
    if s.max() <= 0:
        return np.array([0.0]), np.array([0.0]), 0.0
    thresholds = np.linspace(s.min(), s.max(), n_points)
    precisions, recalls = [], []
    gt_d = _dilate(gt, 1)  # 1-px tolerance for curve stability
    total_gt = gt_d.sum()
    for t in thresholds:
        pred = (s >= t).astype(np.uint8)
        tp = np.logical_and(pred, gt_d).sum()
        precisions.append(tp / pred.sum() if pred.sum() else 1.0)
        recalls.append(tp / total_gt if total_gt else 0.0)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    # sort by recall for AUC
    order = np.argsort(recalls)
    r_sorted, p_sorted = recalls[order], precisions[order]
    auc_val = auc(r_sorted, p_sorted) if len(r_sorted) > 1 else float(p_sorted[0] if len(p_sorted) else 0.0)
    return precisions, recalls, float(auc_val)


def pr_auc(strength: np.ndarray, gt: np.ndarray) -> float:
    return pr_curve(strength, gt)[2]


def full_metrics(pred: np.ndarray, gt: np.ndarray, strength: np.ndarray = None) -> dict:
    """Complete metric battery at the standard tolerances."""
    out = {}
    for tol, tag in [(0, "exact"), (1, "1px"), (2, "2px")]:
        p, r, f1 = precision_recall_f1(pred, gt, tol)
        out[f"precision_{tag}"] = p
        out[f"recall_{tag}"] = r
        out[f"f1_{tag}"] = f1
    out["iou"] = iou(pred, gt, 1)
    out["chamfer"] = chamfer_distance(pred, gt)
    if strength is not None:
        out["pr_auc"] = pr_auc(strength, gt)
    else:
        out["pr_auc"] = 0.0
    return out
