"""
Experiment 9 -- Resource scaling analysis.

Variable : image size 4x4, 8x8, 16x16
Methods  : QPIE-QHED, NEQR-Gradient
Measure  : logical/ancilla/total qubits, raw + transpiled gates,
           1q/2q split, depth pre/post, state-prep vs edge-op split,
           statevector memory, kernel time, E2E time (reported
           SEPARATELY), classical Sobel/Canny E2E for context
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from classical.classical_pipeline import run_classical
from data.synthetic.generator import BUILDERS
from evaluation.resource_counter import circuit_resources, transpiled_resources
from experiments.exp_common import save_table
from quantum.neqr_encoder import build_neqr_image_circuit
from quantum.qhed_circuit import build_qhed_circuit
from quantum.quantum_pipeline import qubit_counts, run_quantum

STATEMENT = (
    "This project evaluates edge detection quality and robustness, not "
    "quantum computational advantage. State preparation, measurement, and "
    "reconstruction dominate total quantum pipeline time at these small scales."
)


def main():
    print("=" * 70)
    print("EXPERIMENT 9: Resource scaling analysis")
    print("=" * 70)

    img, _ = BUILDERS["circle"](16)
    rows = []
    for size in config.SIZES:
        sub = img[:size, :size]
        for qname in ("qhed", "neqr"):
            qc = qubit_counts(qname, size)
            if qname == "qhed":
                enc = sub / sub.max() if sub.max() > 0 else sub
                circ = build_qhed_circuit(enc, "horizontal", measure=True)
            else:
                img255 = np.clip(np.rint(sub * 255), 0, 255)
                t_build0 = time.perf_counter()
                circ = build_neqr_image_circuit(img255, measure=False)
                build_s = time.perf_counter() - t_build0
            raw = circuit_resources(circ)
            t_tr0 = time.perf_counter()
            tr = transpiled_resources(circ)
            transpile_s = time.perf_counter() - t_tr0

            # timing over 5 quantum runs (kernel + e2e)
            kts, ets = [], []
            for t in range(5):
                res, rec = run_quantum(sub, qname, shots=config.DEFAULT_SHOTS,
                                       seed=config.SEED + t)
                kts.append(rec["kernel_time_s"])
                ets.append(rec["e2e_time_s"])

            # classical context
            t0 = time.perf_counter()
            run_classical(sub)
            cls_s = time.perf_counter() - t0

            rows.append({
                "size": size, "method": qname,
                "logical_qubits": qc["logical"], "ancilla": qc["ancilla"],
                "total_qubits": qc["total"],
                "raw_gates": raw["raw_gates"], "raw_1q": raw["raw_1q_gates"],
                "raw_2q": raw["raw_2q_gates"], "raw_depth": raw["raw_depth"],
                "transpiled_gates": tr["transpiled_gates"],
                "transpiled_1q": tr["transpiled_1q_gates"],
                "transpiled_2q": tr["transpiled_2q_gates"],
                "transpiled_depth": tr["transpiled_depth"],
                "statevector_MB": round(raw["statevector_bytes"] / 1e6, 3),
                "circuit_build_s": round(locals().get("build_s", 0.0), 4),
                "transpile_s": round(transpile_s, 4),
                "kernel_time_s": round(float(np.mean(kts)), 4),
                "e2e_time_s": round(float(np.mean(ets)), 4),
                "classical_e2e_s": round(cls_s, 6),
            })
            print(f"  {size}x{size} {qname}: qubits={qc['total']} "
                  f"raw={raw['raw_gates']} transp={tr['transpiled_gates']} "
                  f"kernel={np.mean(kts):.3f}s e2e={np.mean(ets):.3f}s")

    df = pd.DataFrame(rows)
    save_table(df, "exp9_resources.csv")

    with open(os.path.join(config.TABLE_DIR, "exp9_statement.txt"), "w",
              encoding="utf-8") as f:
        f.write(STATEMENT + "\n")

    # circuit diagrams for the report
    from evaluation.visualiser import save_circuit_diagram
    enc4 = img[:4, :4] / max(img[:4, :4].max(), 1e-9)
    save_circuit_diagram(build_qhed_circuit(enc4, "horizontal", measure=True),
                         os.path.join(config.CIRCUIT_DIR, "qhed_4x4.txt"))
    save_circuit_diagram(build_qhed_circuit(enc4, "horizontal", measure=False).decompose(),
                         os.path.join(config.CIRCUIT_DIR, "qhed_4x4_decomposed.txt"), fold=120)
    img255 = np.clip(np.rint(img[:4, :4] * 255), 0, 255).astype(float)
    save_circuit_diagram(build_neqr_image_circuit(img255, measure=False),
                         os.path.join(config.CIRCUIT_DIR, "neqr_4x4.txt"), fold=200)
    print("Experiment 9 COMPLETE")


if __name__ == "__main__":
    main()
