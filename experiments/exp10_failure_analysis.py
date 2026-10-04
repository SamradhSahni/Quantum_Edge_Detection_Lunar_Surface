"""
Experiment 10 -- Failure analysis and failure taxonomy.

Builds the side-by-side failure gallery (Input | GT | Sobel | Canny |
Robinson | QPIE-QHED | NEQR-Gradient) for each IMAGE-LEVEL and
QUANTUM-SPECIFIC failure condition from the spec, with per-failure
severity classification (mild/moderate/severe) based on the F1 drop
relative to the clean condition.
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
from experiments.exp_common import save_table
from evaluation.visualiser import comparison_grid
from preprocessing.augmentor import (apply_degradation, directional_gradient_illumination,
                                     low_illumination, shadow_region)
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

METHODS = ("sobel", "canny", "robinson", "qhed", "neqr")
LABELS = dict(config.METHOD_LABELS)


def edge_maps(image):
    image = np.asarray(image, dtype=np.float64)
    cls = run_classical(image)
    outs = {"Sobel": cls["sobel"]["binary"], "Canny": cls["canny"]["binary"],
            "Robinson": cls["robinson"]["binary"]}
    for qname, label in (("qhed", "QPIE-QHED"), ("neqr", "NEQR-Grad")):
        res, _ = run_quantum(image, qname)
        outs[label] = apply_threshold(res.strength, 0.2)
    return outs


def severity(f1_clean, f1_degraded):
    drop = f1_clean - f1_degraded
    if drop < 0.1:
        return "none"
    if drop < 0.25:
        return "mild"
    if drop < 0.5:
        return "moderate"
    return "severe"


def main():
    print("=" * 70)
    print("EXPERIMENT 10: Failure analysis and failure taxonomy")
    print("=" * 70)

    img, mask = BUILDERS["circle"](8)
    gt = (mask > 0).astype(np.uint8)

    conditions = {
        "clean": img,
        "low_contrast_0.25": apply_degradation(img, "contrast", 0.25),
        "heavy_blur_2.0": apply_degradation(img, "gaussian_blur", 2.0),
        "strong_noise_0.10": apply_degradation(img, "gaussian_noise", 0.10),
        "directional_illumination": directional_gradient_illumination(img, 0.5),
        "shadow_region": shadow_region(img, 0.7),
        "low_illumination": low_illumination(img, 0.3),
    }
    img4, mask4 = BUILDERS["circle"](4)
    conditions["low_resolution_4x4"] = img4

    rows = []
    gallery_rows = []
    for cond, im in conditions.items():
        g = gt if cond != "low_resolution_4x4" else (mask4 > 0).astype(np.uint8)
        outs = edge_maps(im)
        f1s = {}
        for label, binary in outs.items():
            m = full_metrics(binary, g)
            f1s[label] = m["f1_1px"]
            rows.append({"condition": cond, "method": label,
                         "f1_1px": m["f1_1px"], "precision": m["precision_1px"],
                         "recall": m["recall_1px"]})
        gallery_rows.append((cond, {"Input": im, "GT": g, **outs}))
        print(f"  {cond}: " + " ".join(f"{k}={v:.2f}" for k, v in f1s.items()))

    # severity vs clean
    clean_f1 = {r["method"]: r["f1_1px"] for r in rows if r["condition"] == "clean"}
    for r in rows:
        r["severity"] = severity(clean_f1[r["method"]], r["f1_1px"])

    # quantum-specific conditions (exact density-matrix noisy runs; NEQR
    # on the minimal 2x2 / 4-bit circuit -- see exp5 module docstring)
    from data.synthetic.generator import BUILDERS as _B
    img2, mask2 = _B["vertical_line"](2)
    gt2 = (mask2 > 0).astype(np.uint8)
    for noise in ("depol_0.05", "readout_0.10", "combined"):
        outs = {}
        outs["QPIE-QHED"] = apply_threshold(
            run_quantum(img, "qhed", shots=2048, noise_name=noise,
                        sim_method="density_matrix")[0].strength, 0.2)
        res_n = run_quantum(img2, "neqr", shots=2048, noise_name=noise,
                            sim_method="density_matrix", intensity_bits=4)[0]
        outs["NEQR-Grad (2x2)"] = apply_threshold(res_n.strength, 0.2)
        for label, binary in outs.items():
            g = gt if label == "QPIE-QHED" else gt2
            m = full_metrics(binary, g)
            rows.append({"condition": f"quantum_{noise}", "method": label,
                         "f1_1px": m["f1_1px"], "precision": m["precision_1px"],
                         "recall": m["recall_1px"],
                         "severity": severity(clean_f1.get(label, m["f1_1px"]),
                                              m["f1_1px"])})
        gallery_rows.append((f"quantum_{noise}",
                             {"Input": img2, "GT": gt2, **outs}))
        print(f"  quantum_{noise}: done")

    df = pd.DataFrame(rows)
    save_table(df, "exp10_failure_taxonomy.csv")

    # visual galleries (split into two figures to stay readable)
    os.makedirs(config.FAIL_DIR, exist_ok=True)
    from evaluation.visualiser import failure_grid
    failure_grid(gallery_rows[:4],
                 os.path.join(config.FAIL_DIR, "exp10_gallery_image_level_1.png"))
    failure_grid(gallery_rows[4:8],
                 os.path.join(config.FAIL_DIR, "exp10_gallery_image_level_2.png"))
    failure_grid(gallery_rows[8:],
                 os.path.join(config.FAIL_DIR, "exp10_gallery_quantum.png"))
    print("Experiment 10 COMPLETE")


if __name__ == "__main__":
    main()
