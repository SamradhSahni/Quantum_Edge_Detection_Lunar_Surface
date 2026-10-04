"""
Experiment 1 -- Representation Ablation (core experiment).

Control : same images (8x8), shots=8192, no preprocessing, ideal
          simulator, independent GT, validation-selected thresholds.
Variable: QPIE + QHED   vs   NEQR + Quantum Gradient
Measure : boundary F1 (exact/1px/2px), P, R, IoU, Chamfer, PR-AUC
          + full resource breakdown (qubits, gates, depth, timing)
Dataset : synthetic 8x8 (exact GT) + lunar crater patches
Trials  : 10 per configuration -> mean +/- std
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.synthetic.generator import BUILDERS, generate, validation_split
from evaluation.metrics import full_metrics
from experiments import exp_common
from experiments.exp_common import (fmt_mean_std, save_table,
                                    strength_maps_for_image,
                                    trials_metrics)
from quantum.quantum_pipeline import qubit_counts
from quantum.threshold_selector import sweep_thresholds, apply_threshold

GEO_8 = [k for k in generate(8).keys()]


def threshold_protocol():
    """Validation-selected relative thresholds on 3 held-out geometries."""
    val_imgs = [("horizontal_line", BUILDERS["horizontal_line"](8)),
                ("circle", BUILDERS["circle"](8)),
                ("irregular_curve", BUILDERS["irregular_curve"](8))]
    val_pairs = [(img, (mask > 0).astype(np.uint8)) for _, (img, mask) in val_imgs]
    thresholds = exp_common.select_thresholds(val_pairs)
    save_table(pd.DataFrame([{"method": m, "rel_threshold": t}
                              for m, t in thresholds.items()]),
               "thresholds.csv")
    return thresholds


def main():
    print("=" * 70)
    print("EXPERIMENT 1: Representation ablation (QPIE-QHED vs NEQR-Gradient)")
    print("=" * 70)

    thresholds = threshold_protocol()
    exp_common.FROZEN_REL_T = thresholds.get("qhed", 0.3)
    t_qhed = thresholds["qhed"]
    t_neqr = thresholds["neqr"]
    t_cl = {m: thresholds[m] for m in ("sobel", "canny", "robinson")}
    print(f"validation-selected relative thresholds: {thresholds}")

    # ---------------- synthetic 8x8 geometries ----------------
    rows = []
    res_rows = []
    for geo in GEO_8:
        img, mask = BUILDERS[geo](8)
        gt = (mask > 0).astype(np.uint8)
        cls = __import__("classical.classical_pipeline", fromlist=["run_classical"]).run_classical(img)
        strength_by = {m: cls[m]["strength"] for m in cls}

        # quantum: n=10 trials with different shot seeds
        for qname, trel in (("qhed", t_qhed), ("neqr", t_neqr)):
            def runner(seed, _img=img, _gt=gt, _q=qname, _t=trel):
                from quantum.quantum_pipeline import run_quantum
                res, rec = run_quantum(_img, _q, shots=config.DEFAULT_SHOTS, seed=seed)
                m = full_metrics(apply_threshold(res.strength, _t), _gt, res.strength)
                m["kernel_time_s"] = rec["kernel_time_s"]
                m["e2e_time_s"] = rec["e2e_time_s"]
                return m
            mean, std, _ = trials_metrics(runner)
            rows.append({"dataset": "synthetic", "image": geo, "method": qname,
                         "f1_exact_mean": mean["f1_exact"], "f1_exact_std": std["f1_exact"],
                         "f1_1px_mean": mean["f1_1px"], "f1_1px_std": std["f1_1px"],
                         "f1_2px_mean": mean["f1_2px"], "f1_2px_std": std["f1_2px"],
                         "precision_mean": mean["precision_1px"], "recall_mean": mean["recall_1px"],
                         "iou": mean["iou"], "chamfer": mean["chamfer"],
                         "pr_auc": mean["pr_auc"]})
            qc = qubit_counts(qname, 8)
            res_rows.append({"dataset": "synthetic", "image": geo, "method": qname,
                             "logical_qubits": qc["logical"], "ancilla": qc["ancilla"],
                             "total_qubits": qc["total"],
                             "kernel_time_s": mean["kernel_time_s"],
                             "e2e_time_s": mean["e2e_time_s"]})

        # classical: deterministic
        for cname, trel in t_cl.items():
            binary = (strength_by[cname] >= trel * strength_by[cname].max()).astype(np.uint8) \
                if cname != "canny" else cls["canny"]["binary"]
            m = full_metrics(binary, gt, strength_by[cname])
            rows.append({"dataset": "synthetic", "image": geo, "method": cname,
                         "f1_exact_mean": m["f1_exact"], "f1_exact_std": 0.0,
                         "f1_1px_mean": m["f1_1px"], "f1_1px_std": 0.0,
                         "f1_2px_mean": m["f1_2px"], "f1_2px_std": 0.0,
                         "precision_mean": m["precision_1px"], "recall_mean": m["recall_1px"],
                         "iou": m["iou"], "chamfer": m["chamfer"], "pr_auc": m["pr_auc"]})
        print(f"  {geo}: done")

    # ---------------- lunar crater patches ----------------
    from preprocessing.patch_extractor import load_patches
    patches = load_patches(8)[:10]
    for fname, img, mask in patches:
        gt = (mask > 0).astype(np.uint8)
        cls = __import__("classical.classical_pipeline", fromlist=["run_classical"]).run_classical(img)
        for cname, trel in t_cl.items():
            binary = (cls[cname]["strength"] >= trel * cls[cname]["strength"].max()).astype(np.uint8) \
                if cname != "canny" else cls["canny"]["binary"]
            m = full_metrics(binary, gt, cls[cname]["strength"])
            rows.append({"dataset": "lunar", "image": fname, "method": cname,
                         "f1_exact_mean": m["f1_exact"], "f1_exact_std": 0.0,
                         "f1_1px_mean": m["f1_1px"], "f1_1px_std": 0.0,
                         "f1_2px_mean": m["f1_2px"], "f1_2px_std": 0.0,
                         "precision_mean": m["precision_1px"], "recall_mean": m["recall_1px"],
                         "iou": m["iou"], "chamfer": m["chamfer"], "pr_auc": m["pr_auc"]})
        for qname, trel in (("qhed", t_qhed), ("neqr", t_neqr)):
            def runner(seed, _img=img, _gt=gt, _q=qname, _t=trel):
                from quantum.quantum_pipeline import run_quantum
                res, rec = run_quantum(_img, _q, shots=config.DEFAULT_SHOTS, seed=seed)
                m = full_metrics(apply_threshold(res.strength, _t), _gt, res.strength)
                m["kernel_time_s"] = rec["kernel_time_s"]
                m["e2e_time_s"] = rec["e2e_time_s"]
                return m
            mean, std, _ = trials_metrics(runner)
            rows.append({"dataset": "lunar", "image": fname, "method": qname,
                         "f1_exact_mean": mean["f1_exact"], "f1_exact_std": std["f1_exact"],
                         "f1_1px_mean": mean["f1_1px"], "f1_1px_std": std["f1_1px"],
                         "f1_2px_mean": mean["f1_2px"], "f1_2px_std": std["f1_2px"],
                         "precision_mean": mean["precision_1px"], "recall_mean": mean["recall_1px"],
                         "iou": mean["iou"], "chamfer": mean["chamfer"],
                         "pr_auc": mean["pr_auc"]})
            qc = qubit_counts(qname, 8)
            res_rows.append({"dataset": "lunar", "image": fname, "method": qname,
                             "logical_qubits": qc["logical"], "ancilla": qc["ancilla"],
                             "total_qubits": qc["total"],
                             "kernel_time_s": mean["kernel_time_s"],
                             "e2e_time_s": mean["e2e_time_s"]})
        print(f"  {fname}: done")

    df = pd.DataFrame(rows)
    save_table(df, "exp1_representation_ablation.csv")

    # summary by method x dataset
    summary = df.groupby(["dataset", "method"]).agg(
        f1_1px_mean=("f1_1px_mean", "mean"), f1_1px_std=("f1_1px_mean", "std"),
        pr_auc=("pr_auc", "mean"), iou=("iou", "mean"), chamfer=("chamfer", "mean")
    ).reset_index()
    save_table(summary, "exp1_summary.csv")

    res_df = pd.DataFrame(res_rows).drop_duplicates(subset=["method", "dataset"])
    save_table(res_df, "exp1_resources.csv")

    # ablation table from the spec (Section 8, Experiment 1)
    abl = []
    for qname in ("qhed", "neqr"):
        sub = df[(df.method == qname) & (df.dataset == "synthetic")]
        qc = qubit_counts(qname, 8)
        abl.append({"method": qname,
                    "logical_qubits": qc["logical"], "ancilla": qc["ancilla"],
                    "boundary_f1_1px": fmt_mean_std(sub["f1_1px_mean"].mean(),
                                                    sub["f1_1px_mean"].std()),
                    "pr_auc": round(sub["pr_auc"].mean(), 3)})
    save_table(pd.DataFrame(abl), "exp1_ablation_table.csv")

    # figures for a few representative geometries
    from evaluation.visualiser import comparison_grid
    from quantum.quantum_pipeline import run_quantum
    for geo in ("horizontal_line", "circle", "diagonal_45"):
        img, mask = BUILDERS[geo](8)
        outs = {}
        cls = __import__("classical.classical_pipeline", fromlist=["run_classical"]).run_classical(img)
        outs["Sobel"] = cls["sobel"]["binary"]
        outs["Canny"] = cls["canny"]["binary"]
        for qname, trel, label in (("qhed", t_qhed, "QPIE-QHED"), ("neqr", t_neqr, "NEQR-Grad")):
            res, _ = run_quantum(img, qname)
            outs[label] = apply_threshold(res.strength, trel)
        comparison_grid(img, gt=(mask > 0), method_outputs=outs,
                        title=f"Experiment 1: {geo} (8x8)",
                        save_path=os.path.join(config.FIG_DIR, f"exp1_{geo}.png"))
    print("Experiment 1 COMPLETE")


if __name__ == "__main__":
    main()
