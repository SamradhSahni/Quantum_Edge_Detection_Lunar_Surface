"""
Experiment 7 -- Preprocessing strategy comparison.

Variable : None | Gaussian | Median | CLAHE | Hist-stretch | Gamma 0.5/1.5
Fixed    : 8x8 lunar crater patches, shots=8192, ideal simulator
Measure  : boundary F1 per method and strategy
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
from preprocessing.preprocessor import STRATEGIES, apply_preprocessing
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

N_PATCHES = 8


def main():
    print("=" * 70)
    print("EXPERIMENT 7: Preprocessing strategies (lunar patches)")
    print("=" * 70)

    patches = load_patches(8)[:N_PATCHES]
    rows = []
    for strategy in STRATEGIES:
        per_method = {m: [] for m in config.METHODS}
        for fname, img, mask in patches:
            gt = (mask > 0).astype(np.uint8)
            img_p = apply_preprocessing(img, strategy)
            cls = run_classical(img_p)
            for cname in ("sobel", "canny", "robinson"):
                binary = cls[cname]["binary"] if cname == "canny" else \
                    apply_threshold(cls[cname]["strength"], 0.3)
                per_method[cname].append(full_metrics(binary, gt)["f1_1px"])
            for qname in ("qhed", "neqr"):
                res, _ = run_quantum(img_p, qname)
                per_method[qname].append(full_metrics(apply_threshold(res.strength, 0.2), gt)["f1_1px"])
        for m in config.METHODS:
            rows.append({"strategy": strategy, "method": m,
                         "f1_1px_mean": float(np.mean(per_method[m])),
                         "f1_1px_std": float(np.std(per_method[m], ddof=1)),
                         "n_patches": len(patches)})
        print(f"  {strategy}: " + " ".join(
            f"{m}={np.mean(per_method[m]):.2f}" for m in config.METHODS))

    df = pd.DataFrame(rows)
    save_table(df, "exp7_preprocessing.csv")

    from evaluation.visualiser import robustness_curves
    strategies = STRATEGIES
    by_method = {m: [float(df[(df.method == m) & (df.strategy == s)].f1_1px_mean.iloc[0])
                     for s in strategies] for m in config.METHODS}
    robustness_curves(range(len(strategies)), by_method,
                      xlabel="preprocessing strategy index",
                      title="Experiment 7: F1 by preprocessing strategy",
                      save_path=os.path.join(config.FIG_DIR, "exp7_preprocessing.png"))
    print("Experiment 7 COMPLETE")


if __name__ == "__main__":
    main()
