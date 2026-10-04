"""
Experiment 2 -- Resolution scaling (4x4, 8x8, 16x16).

Variable : image size
Fixed    : shots=8192, ideal simulator, same geometry (circle)
Measure  : F1, logical/ancilla/total qubits, raw+transpiled gates,
           depth, state-prep cost, kernel and E2E time
Trials   : 10 per configuration
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.synthetic.generator import BUILDERS
from evaluation.metrics import full_metrics
from evaluation.resource_counter import circuit_resources, transpiled_resources
from experiments.exp_common import save_table, trials_metrics
from quantum.quantum_pipeline import qubit_counts, run_quantum
from quantum.threshold_selector import apply_threshold

GEOMETRIES = ("circle", "ellipse")


def main():
    print("=" * 70)
    print("EXPERIMENT 2: Resolution scaling (4x4, 8x8, 16x16)")
    print("=" * 70)

    rows = []
    for geo in GEOMETRIES:
        for size in config.SIZES:
            img, mask = BUILDERS[geo](size)
            gt = (mask > 0).astype(np.uint8)
            for qname in ("qhed", "neqr"):
                qc = qubit_counts(qname, size)

                # resources from ONE circuit (build once)
                from quantum.qhed_circuit import build_qhed_circuit
                from quantum.quantum_gradient import build_neqr_gradient_circuit
                from quantum.neqr_encoder import neqr_layout
                if qname == "qhed":
                    enc = img / img.max() if img.max() > 0 else img
                    circ = build_qhed_circuit(enc, "horizontal", measure=True)
                else:
                    img255 = np.clip(np.rint(img * 255), 0, 255)
                    circ = build_neqr_gradient_circuit(img255, "horizontal", measure=True)
                raw = circuit_resources(circ)
                tr = transpiled_resources(circ)

                def runner(seed, _img=img, _gt=gt, _q=qname, _s=size):
                    res, rec = run_quantum(_img, _q, shots=config.DEFAULT_SHOTS, seed=seed)
                    m = full_metrics(apply_threshold(res.strength, 0.2), _gt, res.strength)
                    m["kernel_time_s"] = rec["kernel_time_s"]
                    m["e2e_time_s"] = rec["e2e_time_s"]
                    return m

                mean, std, _ = trials_metrics(runner)
                rows.append({
                    "geometry": geo, "size": size, "method": qname,
                    "f1_1px_mean": mean["f1_1px"], "f1_1px_std": std["f1_1px"],
                    "pr_auc": mean["pr_auc"],
                    "logical_qubits": qc["logical"], "ancilla": qc["ancilla"],
                    "total_qubits": qc["total"],
                    "raw_gates": raw.get("raw_gates"), "raw_depth": raw.get("raw_depth"),
                    "transpiled_gates": tr.get("transpiled_gates"),
                    "transpiled_2q_gates": tr.get("transpiled_2q_gates"),
                    "transpiled_depth": tr.get("transpiled_depth"),
                    "statevector_MB": round(raw.get("statevector_bytes", 0) / 1e6, 2),
                    "kernel_time_s": mean["kernel_time_s"],
                    "e2e_time_s": mean["e2e_time_s"],
                })
                print(f"  {geo} {size}x{size} {qname}: F1={mean['f1_1px']:.3f} "
                      f"qubits={qc['total']} gates={raw.get('raw_gates')}")

    df = pd.DataFrame(rows)
    save_table(df, "exp2_resolution_scaling.csv")

    # classical reference row for context
    cls_rows = []
    from classical.classical_pipeline import run_classical
    for geo in GEOMETRIES:
        for size in config.SIZES:
            img, mask = BUILDERS[geo](size)
            gt = (mask > 0).astype(np.uint8)
            cls = run_classical(img)
            t0 = __import__("time").perf_counter()
            cls = run_classical(img)
            dt = __import__("time").perf_counter() - t0
            for cname in ("sobel", "canny"):
                m = full_metrics(cls[cname]["binary"], gt, cls[cname]["strength"])
                cls_rows.append({"geometry": geo, "size": size, "method": cname,
                                 "f1_1px_mean": m["f1_1px"], "f1_1px_std": 0.0,
                                 "pr_auc": m["pr_auc"], "logical_qubits": 0,
                                 "ancilla": 0, "total_qubits": 0, "raw_gates": None,
                                 "raw_depth": None, "transpiled_gates": None,
                                 "transpiled_2q_gates": None, "transpiled_depth": None,
                                 "statevector_MB": 0.0, "kernel_time_s": 0.0,
                                 "e2e_time_s": dt / 2})
    save_table(pd.DataFrame(cls_rows), "exp2_classical_reference.csv")
    print("Experiment 2 COMPLETE")


if __name__ == "__main__":
    main()
