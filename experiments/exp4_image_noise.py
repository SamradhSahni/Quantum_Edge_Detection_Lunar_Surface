"""
Experiment 4 -- Image noise and degradation robustness.

4A Gaussian noise (sigma grid) | 4B Gaussian blur | 4C contrast
4D illumination & shadow (lunar-specific)
Measure: F1 per method per level -> robustness AUC
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from classical.classical_pipeline import run_classical
from data.synthetic.generator import BUILDERS
from evaluation.metrics import full_metrics
from evaluation.robustness import robustness_auc
from experiments.exp_common import save_table
from preprocessing.augmentor import (apply_degradation, directional_gradient_illumination,
                                     low_illumination, shadow_region)
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

FAMILIES = {
    "gaussian_noise": config.GAUSS_NOISE_SIGMAS,
    "gaussian_blur": config.BLUR_SIGMAS,
    "contrast": config.CONTRAST_SCALES,
}


def eval_all_methods(image, gt):
    image = np.asarray(image, dtype=np.float64)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    out = {}
    cls = run_classical(image)
    for cname in ("sobel", "canny", "robinson"):
        binary = cls[cname]["binary"] if cname == "canny" else \
            apply_threshold(cls[cname]["strength"], 0.3)
        out[cname] = full_metrics(binary, gt, cls[cname]["strength"])["f1_1px"]
    for qname in ("qhed", "neqr"):
        res, _ = run_quantum(image, qname, shots=config.DEFAULT_SHOTS)
        out[qname] = full_metrics(apply_threshold(res.strength, 0.2), gt,
                                  res.strength)["f1_1px"]
    return out


def main():
    print("=" * 70)
    print("EXPERIMENT 4: Image noise and degradation robustness")
    print("=" * 70)

    img, mask = BUILDERS["circle"](8)
    gt = (mask > 0).astype(np.uint8)
    rows = []

    for family, levels in FAMILIES.items():
        by_method = {m: [] for m in config.METHODS}
        for level in levels:
            degraded = apply_degradation(img, family, level)
            scores = eval_all_methods(degraded, gt)
            for m, f1 in scores.items():
                by_method[m].append(f1)
                rows.append({"family": family, "level": level, "method": m, "f1_1px": f1})
            print(f"  {family}={level}: " +
                  " ".join(f"{m}={f1:.2f}" for m, f1 in scores.items()))
        auc_row = {"family": family, "level": "AUC"}
        for m in config.METHODS:
            auc_row[m] = robustness_auc(levels, by_method[m])
        rows.append(auc_row)

    # 4D illumination conditions
    img4d = {"normal": img,
             "low_illumination": low_illumination(img, 0.3),
             "directional_illumination": directional_gradient_illumination(img, 0.5),
             "shadow": shadow_region(img, 0.7)}
    by_method = {m: [] for m in config.METHODS}
    for cond, im in img4d.items():
        scores = eval_all_methods(im, gt)
        for m, f1 in scores.items():
            by_method[m].append(f1)
            rows.append({"family": "illumination", "level": cond, "method": m,
                         "f1_1px": f1})
        print(f"  illumination/{cond}: " +
              " ".join(f"{m}={f1:.2f}" for m, f1 in scores.items()))
    for m in config.METHODS:
        rows.append({"family": "illumination", "level": "AUC", "method": m,
                     "f1_1px": robustness_auc([0, 1, 2, 3], by_method[m])})

    df = pd.DataFrame(rows)
    save_table(df, "exp4_image_noise.csv")

    # robustness curves
    from evaluation.visualiser import robustness_curves
    illumination_levels = ["normal", "low_illumination",
                           "directional_illumination", "shadow"]
    for family, levels in list(FAMILIES.items()) + [("illumination", illumination_levels)]:
        sub = df[(df.family == family) & (df.level != "AUC")]
        if sub.empty:
            continue
        sub = sub[sub.level.astype(str).isin([str(lv) for lv in levels])]
        by_method = {m: [float(sub[(sub.method == m) & (sub.level.astype(str) == str(lv))].f1_1px.iloc[0])
                         for lv in levels] for m in config.METHODS
                     if len(sub[sub.method == m]) == len(levels)}
        robustness_curves(levels, by_method,
                          xlabel=family, title=f"Experiment 4: {family}",
                          save_path=os.path.join(config.FIG_DIR, f"exp4_{family}.png"))
    print("Experiment 4 COMPLETE")


if __name__ == "__main__":
    main()
