"""
Streamlit research dashboard (Optional Extension A).

Upload any grayscale image (or pick a bundled patch) and run:
  Sobel | Canny | Robinson | QPIE-QHED | NEQR-Gradient
with selectable resolution, preprocessing, shot count, and noise model.

Run:  streamlit run dashboard/app.py
"""

import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from classical.classical_pipeline import run_classical
from evaluation.metrics import full_metrics
from evaluation.visualiser import comparison_grid
from preprocessing.image_loader import load_grayscale, to_power_of_two
from preprocessing.preprocessor import STRATEGIES, apply_preprocessing
from quantum.quantum_pipeline import run_quantum
from quantum.threshold_selector import apply_threshold

st.set_page_config(page_title="Quantum Edge Detection", layout="wide")
st.title("Quantum Edge Detection -- QPIE-QHED vs NEQR-Gradient")
st.caption("Grayscale lunar crater imagery | Qiskit AerSimulator | "
           "QPIE amplitude encoding + Hadamard interference vs NEQR basis-state "
           "encoding + quantum subtractor")

# ---------------------------------------------------------------------------
# image source
# ---------------------------------------------------------------------------
source = st.sidebar.radio("Image source", ("Bundled lunar patch", "Synthetic geometry",
                                           "Upload"))
size = st.sidebar.select_slider("Resolution", options=[4, 8, 16], value=8)

img = None
gt = None
if source == "Upload":
    up = st.sidebar.file_uploader("Grayscale image", type=("png", "jpg", "jpeg", "tif"))
    if up is not None:
        img = to_power_of_two(load_grayscale_from_bytes(up.getvalue()), size)
elif source == "Synthetic geometry":
    from data.synthetic.generator import BUILDERS
    geo = st.sidebar.selectbox("Geometry", list(BUILDERS.keys()))
    img, mask = BUILDERS[geo](size)
    gt = (mask > 0).astype(np.uint8)
else:
    from preprocessing.patch_extractor import load_patches
    patches = load_patches(size)
    if patches:
        idx = st.sidebar.slider("Patch index", 0, len(patches) - 1, 0)
        _, img, m = patches[idx]
        gt = (m > 0).astype(np.uint8)

if img is None:
    st.info("Pick an image source in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
pre = st.sidebar.selectbox("Preprocessing", STRATEGIES, index=0)
shots = st.sidebar.select_slider("Shots", options=[512, 1024, 2048, 4096, 8192], value=2048)
noise = st.sidebar.selectbox("Noise model",
                             ("ideal", "depol_0.001", "depol_0.005", "depol_0.01",
                              "depol_0.05", "readout_0.01", "readout_0.05",
                              "readout_0.10", "overrot", "combined"))
thr = st.sidebar.slider("Relative threshold", 0.01, 0.99, 0.25, 0.01)

img_p = apply_preprocessing(img, pre)

# ---------------------------------------------------------------------------
# run methods
# ---------------------------------------------------------------------------
st.subheader("Edge maps")
colA, colB = st.columns(2)
with colA:
    st.write("**Classical**")
    cls = run_classical(img_p)
    st.image(np.round(cls["sobel"]["binary"] * 255), caption="Sobel",
             use_container_width=True, clamp=True)
    st.image(np.round(cls["canny"]["binary"] * 255), caption="Canny",
             use_container_width=True, clamp=True)
    st.image(np.round(cls["robinson"]["binary"] * 255), caption="Robinson (8-dir)",
             use_container_width=True, clamp=True)
with colB:
    st.write("**Quantum**")
    res_q, rec_q = run_quantum(img_p, "qhed", shots=shots, noise_name=noise)
    res_n, rec_n = run_quantum(img_p, "neqr", shots=shots, noise_name=noise)
    st.image(np.round(apply_threshold(res_q.strength, thr) * 255),
             caption=f"QPIE-QHED ({shots} shots)", use_container_width=True, clamp=True)
    st.image(np.round(apply_threshold(res_n.strength, thr) * 255),
             caption=f"NEQR-Gradient ({shots} shots)", use_container_width=True, clamp=True)
    st.write("**QHED confidence map**")
    st.image(res_q.strength / max(res_q.strength.max(), 1e-9), caption="QHED strength",
             use_container_width=True, clamp=True)

if gt is not None:
    st.subheader("Metrics vs ground truth")
    rows = []
    for label, strength, binary, rec in (
            ("Sobel", cls["sobel"]["strength"], cls["sobel"]["binary"], None),
            ("Canny", cls["canny"]["strength"], cls["canny"]["binary"], None),
            ("Robinson", cls["robinson"]["strength"], cls["robinson"]["binary"], None),
            ("QPIE-QHED", res_q.strength, apply_threshold(res_q.strength, thr), rec_q),
            ("NEQR-Grad", res_n.strength, apply_threshold(res_n.strength, thr), rec_n)):
        m = full_metrics(binary, gt, strength)
        rows.append({"method": label, "F1 (1px)": round(m["f1_1px"], 3),
                     "Precision": round(m["precision_1px"], 3),
                     "Recall": round(m["recall_1px"], 3),
                     "IoU": round(m["iou"], 3), "Chamfer": round(m["chamfer"], 3),
                     "PR-AUC": round(m["pr_auc"], 3),
                     "qubits": rec["total_qubits"] if rec else "",
                     "kernel s": round(rec["kernel_time_s"], 3) if rec else "",
                     "E2E s": round(rec["e2e_time_s"], 3) if rec else ""})
    st.dataframe(rows, use_container_width=True)
    st.caption("Ground truth: analytic rim mask (synthetic) or annotation-derived "
               "rim mask (lunar patches). Classical baselines are references, "
               "never the ground truth.")
else:
    st.caption("No ground truth for this source -- metrics panel hidden.")

st.subheader("Quantum resources")
c1, c2 = st.columns(2)
c1.metric("QPIE-QHED qubits", rec_q["total_qubits"])
c1.metric("QPIE-QHED depth", rec_q["raw_depth"])
c2.metric("NEQR qubits", rec_n["total_qubits"])
c2.metric("NEQR gates", rec_n["raw_gates"])
st.caption("NEQR circuit shown at the chosen resolution; noisy runs transpile to "
           "the hardware basis {rz, sx, x, cx} so gate-level noise applies.")
