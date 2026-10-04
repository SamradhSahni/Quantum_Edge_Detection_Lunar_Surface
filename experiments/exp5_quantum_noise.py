"""
Experiment 5 -- Quantum hardware noise simulation (AerSimulator).

Method note (documented deviation from naive trajectories):
    Aer's noisy trajectory (statevector) simulation exhibits a severe
    per-shot cost anomaly on this platform (~120x ideal evolution, for
    ANY quantum error in the noise model), making trajectory-based noisy
    NEQR runs computationally intractable.  All noisy runs here instead
    use the EXACT density-matrix method: one simulation yields the full
    noisy output distribution (all shots sampled from it).  This is
    exact -- no sampling approximation in the noise channel itself.

    Sizes:
      QPIE-QHED : 8x8 (7 qubits) and 2x2 (5 qubits)
      NEQR      : 2x2 image with 4-bit intensity registers (12 qubits;
                  16 gray levels).  The H5 depth comparison therefore
                  runs QHED at 2x2 as well (same image, equal rates).

Variable : noise grid (ideal, depolarizing x4, readout x3,
           coherent over-rotation, combined)
Fixed    : shots=2048, n=10 trials per configuration
Measure  : F1, Hellinger circuit fidelity vs ideal, sampling variance
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.synthetic.generator import BUILDERS
from evaluation.metrics import full_metrics
from experiments.exp_common import save_table
from quantum.aer_runner import hellinger_fidelity
from quantum.noise_models import NOISE_GRID
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

SHOTS_NOISY = 2048
DM = "density_matrix"   # exact noisy simulation (see module docstring)
NEQR_NOISY_SIZE = 2     # 2x2 image
NEQR_NOISY_BITS = 4     # 4-bit intensity registers -> 12 qubits total


def main():
    print("=" * 70)
    print("EXPERIMENT 5: Quantum hardware noise (exact density-matrix runs)")
    print("=" * 70)

    img8, mask8 = BUILDERS["circle"](8)
    gt8 = (mask8 > 0).astype(np.uint8)
    # 2x2 test: vertical split (circle crop would have a degenerate mask)
    img2, mask2 = BUILDERS["vertical_line"](2)
    gt2 = (mask2 > 0).astype(np.uint8)

    rows = []
    blocks = [
        ("qhed", 8, img8, gt8, None),
        ("qhed", 2, img2, gt2, None),
        ("neqr", 2, img2, gt2, NEQR_NOISY_BITS),
    ]
    for qname, size, img, gt, bits in blocks:
        # ideal reference distribution for fidelity
        ideal_res, _ = run_quantum(img, qname, shots=SHOTS_NOISY,
                                   seed=config.SEED, intensity_bits=bits or 8)
        ideal_counts = ideal_res.meta.get("counts_h", {})

        for noise_name in NOISE_GRID:
            f1s, fids = [], []
            for t in range(config.N_TRIALS):
                res, _ = run_quantum(img, qname, shots=SHOTS_NOISY,
                                     seed=config.SEED + t,
                                     noise_name=noise_name,
                                     sim_method=DM,
                                     intensity_bits=bits or 8)
                m = full_metrics(apply_threshold(res.strength, 0.2), gt, res.strength)
                f1s.append(m["f1_1px"])
                fid = hellinger_fidelity(ideal_counts, res.meta.get("counts_h", {}))
                fids.append(fid)
            rows.append({"method": qname, "size": size,
                         "intensity_bits": bits or 8, "noise": noise_name,
                         "shots": SHOTS_NOISY,
                         "f1_1px_mean": float(np.mean(f1s)),
                         "f1_1px_std": float(np.std(f1s, ddof=1)),
                         "fidelity_mean": float(np.mean(fids)),
                         "fidelity_std": float(np.std(fids, ddof=1))})
            print(f"  {qname} {size}x{size} {noise_name}: "
                  f"F1={np.mean(f1s):.3f}+/-{np.std(f1s, ddof=1):.3f} "
                  f"fid={np.mean(fids):.3f}")

    df = pd.DataFrame(rows)
    save_table(df, "exp5_quantum_noise.csv")

    # H5 comparison: both representations at 2x2, depolarizing family
    depol = df[df.noise.str.startswith(("depol", "ideal"))]
    h5 = depol[depol["size"] == 2].pivot_table(
        index="noise", columns="method", values="f1_1px_mean")
    if not h5.empty:
        save_table(h5.reset_index(), "exp5_H5_depth_comparison.csv")

    from evaluation.visualiser import robustness_curves
    p_order = ["ideal", "depol_0.001", "depol_0.005", "depol_0.01", "depol_0.05"]
    for (qname, size) in df[["method", "size"]].drop_duplicates().itertuples(index=False):
        sub = df[(df.method == qname) & (df["size"] == size) & (df.noise.isin(p_order))]
        if sub.empty:
            continue
        robustness_curves(
            [n.replace("depol_", "") for n in p_order],
            {qname: list(sub.f1_1px_mean)},
            xlabel="depolarizing p",
            title=f"Experiment 5: {qname} {size}x{size} F1 vs depolarizing noise",
            save_path=os.path.join(config.FIG_DIR, f"exp5_{qname}_{size}.png"))
    print("Experiment 5 COMPLETE")


if __name__ == "__main__":
    main()
