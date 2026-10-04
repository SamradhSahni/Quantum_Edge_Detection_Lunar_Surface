"""
Experiment 6 -- Synthetic edge geometry ablation.

Variable : geometry (8 families), ALL else fixed (8x8, same contrast,
           no noise, shots=8192, ideal simulator)
Methods  : Sobel | Canny | Robinson (8-dir) | QPIE-QHED | NEQR-Gradient
Measure  : F1/P/R per geometry; Robinson dominant-direction diagnostics
           to classify failures as QUANTUM-SPECIFIC vs IMAGE-LEVEL
Trials   : 10 per geometry
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from classical.classical_pipeline import run_classical
from classical.robinson_compass import ROBINSON_ANGLES, robinson_direction_strength
from data.synthetic.generator import BUILDERS
from evaluation.metrics import full_metrics
from experiments.exp_common import save_table
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

GEOMETRIES = list(BUILDERS.keys())


def robinson_diagnostics(image):
    """Directional response strengths for the geometry diagnostic."""
    _, _, responses = run_classical(image)["robinson"]["extra"].values() \
        if False else (None, None, None)
    return None


def main():
    print("=" * 70)
    print("EXPERIMENT 6: Synthetic edge geometry ablation")
    print("=" * 70)

    rows = []
    dir_rows = []
    for geo in GEOMETRIES:
        img, mask = BUILDERS[geo](8)
        gt = (mask > 0).astype(np.uint8)

        cls = run_classical(img)
        for cname in ("sobel", "canny", "robinson"):
            binary = cls[cname]["binary"] if cname == "canny" else \
                apply_threshold(cls[cname]["strength"], 0.3)
            m = full_metrics(binary, gt, cls[cname]["strength"])
            rows.append({"geometry": geo, "method": cname, "trial": 0,
                         "f1_1px": m["f1_1px"], "precision": m["precision_1px"],
                         "recall": m["recall_1px"], "pr_auc": m["pr_auc"]})

        # Robinson directional response strengths per compass angle
        r_mag, r_dir, r_resp = cls["robinson"]["extra"]["direction"], None, None
        from classical.robinson_compass import robinson_compass
        r_mag, r_dir, r_resp = robinson_compass(img)
        mean_dir_strength = {a: float(np.mean(robinson_direction_strength(r_resp, a)[gt > 0]))
                             for a in ROBINSON_ANGLES}
        dominant = max(mean_dir_strength, key=mean_dir_strength.get)
        dir_rows.append({"geometry": geo, "dominant_angle_deg": dominant,
                         "dominant_strength": round(mean_dir_strength[dominant], 3),
                         "spread": round(float(np.std(list(mean_dir_strength.values()))), 3)})

        for qname in ("qhed", "neqr"):
            f1s, ps, rs = [], [], []
            for t in range(config.N_TRIALS):
                res, _ = run_quantum(img, qname, shots=config.DEFAULT_SHOTS,
                                     seed=config.SEED + t)
                m = full_metrics(apply_threshold(res.strength, 0.2), gt, res.strength)
                f1s.append(m["f1_1px"]); ps.append(m["precision_1px"]); rs.append(m["recall_1px"])
            rows.append({"geometry": geo, "method": qname, "trial": "mean",
                         "f1_1px": float(np.mean(f1s)),
                         "precision": float(np.mean(ps)), "recall": float(np.mean(rs)),
                         "pr_auc": np.nan})
        print(f"  {geo}: robinson dominant={dominant}deg, " +
              " ".join(f"{m}={df_val}" for m, df_val in
                       [(r['method'], round(r['f1_1px'], 2)) for r in rows[-5:]]))

    df = pd.DataFrame(rows)
    save_table(df, "exp6_geometry.csv")
    save_table(pd.DataFrame(dir_rows), "exp6_robinson_directions.csv")

    # failure classification table (Case A / Case B from the spec)
    cls_rows = []
    mean_by = df.groupby(["geometry", "method"]).f1_1px.mean().unstack()
    for geo in GEOMETRIES:
        d = dir_row = pd.DataFrame(dir_rows).set_index("geometry").loc[geo]
        for qname in ("qhed", "neqr"):
            if qname not in mean_by.columns or geo not in mean_by.index:
                continue
            f1q = mean_by.loc[geo, qname]
            f1c = mean_by.loc[geo, ["sobel", "robinson"]].mean()
            case = "A: quantum-specific" if (f1c > 0.6 and f1q < f1c - 0.15) else \
                   ("B: image-level" if f1c < 0.6 else "no failure")
            cls_rows.append({"geometry": geo, "method": qname, "quantum_f1": round(f1q, 3),
                             "classical_f1": round(f1c, 3),
                             "robinson_dominant_deg": int(d.dominant_angle_deg),
                             "failure_case": case})
    save_table(pd.DataFrame(cls_rows), "exp6_failure_classification.csv")

    # geometry grid figure
    from evaluation.visualiser import comparison_grid
    for geo in ("diagonal_45", "circle", "irregular_curve"):
        img, mask = BUILDERS[geo](8)
        cls = run_classical(img)
        outs = {"Sobel": cls["sobel"]["binary"], "Canny": cls["canny"]["binary"]}
        for qname, label in (("qhed", "QPIE-QHED"), ("neqr", "NEQR-Grad")):
            res, _ = run_quantum(img, qname)
            outs[label] = apply_threshold(res.strength, 0.2)
        comparison_grid(img, gt=(mask > 0), method_outputs=outs,
                        title=f"Experiment 6: {geo}",
                        save_path=os.path.join(config.FIG_DIR, f"exp6_{geo}.png"))
    print("Experiment 6 COMPLETE")


if __name__ == "__main__":
    main()
