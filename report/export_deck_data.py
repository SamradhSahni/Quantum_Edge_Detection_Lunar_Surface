"""
Exports all experiment results to deck_data.json for the presentation
builder (report/build_deck.js).  Run after the experiment suite.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

OUT = os.path.join(config.ROOT, "report", "deck_data.json")


def load(name):
    p = os.path.join(config.TABLE_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    d = {}
    e1 = load("exp1_summary.csv")
    if e1 is not None:
        d["exp1"] = e1.to_dict(orient="records")
    e1res = load("exp1_resources.csv")
    if e1res is not None:
        d["exp1_resources"] = e1res.to_dict(orient="records")
    e2 = load("exp2_resolution_scaling.csv")
    if e2 is not None:
        d["exp2"] = e2.to_dict(orient="records")
    e3 = load("exp3_shots.csv")
    if e3 is not None:
        d["exp3"] = e3.to_dict(orient="records")
    e4 = load("exp4_image_noise.csv")
    if e4 is not None:
        auc = e4[e4.level.astype(str) == "AUC"]
        # normalise the two AUC row shapes into {family, method, auc}
        norm = []
        for _, row in auc.iterrows():
            if "method" in row and pd.notna(row.get("method")):
                norm.append({"family": row["family"], "method": row["method"],
                             "auc": row.get("f1_1px")})
            else:
                for m in ("sobel", "canny", "robinson", "qhed", "neqr"):
                    if m in row and pd.notna(row.get(m)):
                        norm.append({"family": row["family"], "method": m,
                                     "auc": row[m]})
        d["exp4_auc"] = norm
    e5 = load("exp5_quantum_noise.csv")
    if e5 is not None:
        d["exp5"] = e5.to_dict(orient="records")
    h5 = load("exp5_H5_depth_comparison.csv")
    if h5 is not None:
        d["exp5_h5"] = h5.to_dict(orient="records")
    e6 = load("exp6_geometry.csv")
    if e6 is not None:
        piv = e6[e6.trial.astype(str) == "mean"].pivot_table(
            index="geometry", columns="method", values="f1_1px")
        d["exp6"] = piv.reset_index().to_dict(orient="records")
    e6c = load("exp6_failure_classification.csv")
    if e6c is not None:
        d["exp6_cases"] = e6c.to_dict(orient="records")
    e7 = load("exp7_preprocessing.csv")
    if e7 is not None:
        piv = e7.pivot_table(index="strategy", columns="method",
                             values="f1_1px_mean")
        d["exp7"] = piv.reset_index().to_dict(orient="records")
    e8 = load("exp8_summary.csv")
    if e8 is not None:
        d["exp8"] = e8.to_dict(orient="records")
    e9 = load("exp9_resources.csv")
    if e9 is not None:
        d["exp9"] = e9.to_dict(orient="records")
    e10 = load("exp10_failure_taxonomy.csv")
    if e10 is not None:
        piv = e10.pivot_table(index="condition", columns="method", values="f1_1px")
        d["exp10"] = piv.reset_index().to_dict(orient="records")

    def _clean(obj):
        if isinstance(obj, float) and obj != obj:  # NaN
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_clean(d), f, indent=1, default=str, allow_nan=False)
    print(f"wrote {OUT} with sections: {list(d)}")


if __name__ == "__main__":
    main()
