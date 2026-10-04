"""
Experiment 8 -- Real lunar crater benchmark.

Dataset : lunar crater rim patches (SomBench if downloadable, else the
          documented procedural fallback), 8x8 and 16x16
GT      : independent rim masks (annotation-derived, never a detector)
Measure : F1 exact / 1px / 2px, P, R, IoU, Chamfer; grouped also by
          illumination condition via tile metadata if present
Trials  : 10 per patch per quantum method
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from classical.classical_pipeline import run_classical
from evaluation.metrics import full_metrics
from experiments.exp_common import save_table
from preprocessing.patch_extractor import load_patches
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

N_PATCHES = 20


def main():
    print("=" * 70)
    print("EXPERIMENT 8: Real lunar crater benchmark")
    print("=" * 70)

    rows = []
    for size in (8, 16):
        patches = load_patches(size)[:N_PATCHES]
        if not patches:
            print(f"  no {size}x{size} patches found; skipping")
            continue
        for fname, img, mask in patches:
            gt = (mask > 0).astype(np.uint8)
            cls = run_classical(img)
            for cname in ("sobel", "canny", "robinson"):
                binary = cls[cname]["binary"] if cname == "canny" else \
                    apply_threshold(cls[cname]["strength"], 0.3)
                m = full_metrics(binary, gt, cls[cname]["strength"])
                rows.append({"size": size, "patch": fname, "method": cname,
                             "trial": 0, "f1_exact": m["f1_exact"], "f1_1px": m["f1_1px"],
                             "f1_2px": m["f1_2px"], "precision": m["precision_1px"],
                             "recall": m["recall_1px"], "iou": m["iou"],
                             "chamfer": m["chamfer"], "pr_auc": m["pr_auc"]})
            for qname in ("qhed", "neqr"):
                vals = []
                for t in range(config.N_TRIALS):
                    res, _ = run_quantum(img, qname, shots=config.DEFAULT_SHOTS,
                                         seed=config.SEED + t)
                    m = full_metrics(apply_threshold(res.strength, 0.2), gt, res.strength)
                    vals.append((m["f1_exact"], m["f1_1px"], m["f1_2px"], m["precision_1px"],
                                 m["recall_1px"], m["iou"], m["chamfer"], m["pr_auc"]))
                arr = np.array(vals)
                rows.append({"size": size, "patch": fname, "method": qname,
                             "trial": "mean",
                             "f1_exact": arr[:, 0].mean(), "f1_1px": arr[:, 1].mean(),
                             "f1_2px": arr[:, 2].mean(), "precision": arr[:, 3].mean(),
                             "recall": arr[:, 4].mean(), "iou": arr[:, 5].mean(),
                             "chamfer": arr[:, 6].mean(), "pr_auc": arr[:, 7].mean(),
                             "f1_1px_std": arr[:, 1].std(ddof=1)})
            print(f"  {size}px {fname}: done")

    df = pd.DataFrame(rows)
    save_table(df, "exp8_lunar_benchmark.csv")

    summary = df[(df.trial == "mean") |
                 (df.method.isin(["sobel", "canny", "robinson"]))].groupby(
        ["size", "method"]).agg(
        f1_exact=("f1_exact", "mean"), f1_1px=("f1_1px", "mean"),
        f1_2px=("f1_2px", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), iou=("iou", "mean"),
        chamfer=("chamfer", "mean"), pr_auc=("pr_auc", "mean")).reset_index()
    save_table(summary, "exp8_summary.csv")
    print(summary.to_string(index=False))
    print("Experiment 8 COMPLETE")


if __name__ == "__main__":
    main()
