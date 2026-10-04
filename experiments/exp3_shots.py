"""
Experiment 3 -- Measurement shot count sensitivity.

Variable : shots in {512, 1024, 2048, 4096, 8192, 16384, 32768}
Fixed    : 8x8 image, ideal simulator
Measure  : boundary F1 +/- std (n=10), PR-AUC, sampling variance
Report   : F1 vs shots with error bars
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
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold


def main():
    print("=" * 70)
    print("EXPERIMENT 3: Shot count sensitivity (n=10 trials)")
    print("=" * 70)

    img, mask = BUILDERS["circle"](8)
    gt = (mask > 0).astype(np.uint8)

    rows = []
    for qname in ("qhed", "neqr"):
        for shots in config.SHOT_GRID:
            f1s, aucs, strengths = [], [], []
            for t in range(config.N_TRIALS):
                res, _ = run_quantum(img, qname, shots=shots, seed=config.SEED + t)
                m = full_metrics(apply_threshold(res.strength, 0.2), gt, res.strength)
                f1s.append(m["f1_1px"])
                aucs.append(m["pr_auc"])
                strengths.append(res.strength)
            # pixel-wise sampling variance across trials
            var_map = np.var(np.stack(strengths), axis=0).mean()
            rows.append({"method": qname, "shots": shots,
                         "f1_1px_mean": float(np.mean(f1s)),
                         "f1_1px_std": float(np.std(f1s, ddof=1)),
                         "pr_auc_mean": float(np.mean(aucs)),
                         "pr_auc_std": float(np.std(aucs, ddof=1)),
                         "map_variance": float(var_map)})
            print(f"  {qname} shots={shots}: F1={np.mean(f1s):.3f} "
                  f"+/- {np.std(f1s, ddof=1):.3f}")

    df = pd.DataFrame(rows)
    save_table(df, "exp3_shots.csv")

    from evaluation.visualiser import shots_curve
    for qname in ("qhed", "neqr"):
        sub = df[df.method == qname]
        shots_curve(list(sub.shots), list(sub.f1_1px_mean), list(sub.f1_1px_std),
                    qname, title=f"Experiment 3: {qname} F1 vs shots",
                    save_path=os.path.join(config.FIG_DIR, f"exp3_shots_{qname}.png"))
    print("Experiment 3 COMPLETE")


if __name__ == "__main__":
    main()
