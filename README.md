# Quantum Edge Detection Under Changing Image Conditions

A comparative study of **two quantum image representations** — QPIE amplitude encoding with the Quantum Hadamard Edge Detection (QHED) algorithm, and NEQR basis-state encoding with a quantum gradient (subtractor) operator — against classical Sobel / Canny / Robinson-compass baselines, on grayscale lunar crater imagery.

Built with **Python + Qiskit 2.x + Qiskit AerSimulator** (ideal and noisy simulation), OpenCV, scikit-image. Everything runs locally on CPU.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. validate the quantum circuits (statevector checks against analytic targets)
python tests/test_gates.py
python tests/test_qhed.py
python tests/test_neqr.py

# 2. generate datasets (synthetic geometries + procedural lunar craters)
python data/synthetic/generator.py
python preprocessing/lunar_fallback.py
python -c "import sys; sys.path.insert(0,'.'); \
  from preprocessing.patch_extractor import extract_crater_patches; \
  extract_crater_patches(max_patches_per_size=24)"

# 3. run all 10 experiments (≈2–4 h; or run a subset, e.g. `1 3 9`)
python experiments/run_all.py

# 4. results land in results/tables (CSV) and results/figures (PNG)

# 5. optional interactive dashboard
streamlit run dashboard/app.py
```

---

## What is implemented

### Quantum branch A — QPIE + QHED (Yao et al., PRX 2017)
- `quantum/qpie_encoder.py` — amplitude encoding `|ψ⟩ = Σ I(r,c)|r⟩|c⟩ / ‖I‖` via `StatePreparation`.
- `quantum/qhed_circuit.py` — canonical QHED: `H ⊗ dec ⊗ H` on `[ancilla, position]`, with the **exact** analytic output `P(a=1,r,c) = |a(r,c) − a(r,c±1)|²/4`; both gradient directions, Sobel-style combination.
- **Validated to machine precision** against the analytic statevector (`tests/test_qhed.py`).

### Quantum branch B — NEQR + quantum gradient (Zhang et al., 2013; Wang et al., 2015)
- `quantum/neqr_encoder.py` — exact 8-bit intensity register, H⊗ position superposition + X-wrapped multi-controlled X network per pixel; the neighbour image is written into a second register by **inverted-control state preparation** (quantum image translation).
- `quantum/gates.py` — ancilla-free increment/decrement, and the CDKM ripple-carry **quantum subtractor** computing `F − G (mod 2⁸)` exactly.
- **Validated exhaustively**: pixel recovery at 4×4 and gradient payloads at 2×2/4×4 both directions match the analytic target bit-for-bit (`tests/test_neqr.py`).

### Classical baselines
- `classical/sobel_canny.py`, `classical/robinson_compass.py` (8-direction compass with dominant-direction maps used for the failure-mode diagnostics of Experiment 6).

### Experiments (`experiments/`, one script each; master: `run_all.py`)
| # | Experiment | Output |
|---|------------|--------|
| 1 | Representation ablation (QPIE-QHED vs NEQR, all else fixed) | `exp1_*.csv`, grids |
| 2 | Resolution scaling 4/8/16 | `exp2_*.csv` |
| 3 | Shot-count sensitivity (7 levels × 10 trials, error bars) | `exp3_shots.csv` |
| 4 | Image noise: Gaussian / blur / contrast / illumination / shadow | `exp4_*.csv` + AUC |
| 5 | Quantum hardware noise (depolarizing, readout, over-rotation, combined) | `exp5_quantum_noise.csv` + fidelity |
| 6 | Edge-geometry ablation + Robinson direction diagnostics | `exp6_*.csv` + failure cases A/B |
| 7 | Preprocessing strategies (Gaussian, median, CLAHE, stretch, gamma) | `exp7_preprocessing.csv` |
| 8 | Real lunar crater benchmark (rim-mask GT) | `exp8_*.csv` |
| 9 | Resource scaling (qubits/gates/depth/kernel-vs-E2E time) | `exp9_resources.csv` |
| 10 | Failure taxonomy gallery (mild/moderate/severe per condition) | `exp10_*.csv` + galleries |

### Protocol guarantees
- Ground truth is **independent** (analytic masks / annotation-derived rim masks) — Sobel/Canny are baselines, never the reference.
- Thresholds are **validation-selected** per method (`results/tables/thresholds.csv`), never fixed magic numbers; full PR curves and PR-AUC are reported threshold-free.
- n = 10 repeated trials with shot-seed variation for all quantum results (mean ± std); classical results are deterministic (std = 0).
- Kernel time and end-to-end time are **reported separately**; no quantum-advantage claims are made (see `results/tables/exp9_statement.txt`).

### Dataset note
The primary dataset is the NASA-IBM SomBench WAC lunar crater set (HuggingFace: `ibm-nasa-geospatial/sombench`). `preprocessing/sombench_data.py` attempts the download automatically; when the hub is unreachable the pipeline falls back to **procedurally generated analytic crater tiles** (`preprocessing/lunar_fallback.py`, 512×512, exact rim masks) and every result table states the fallback source. MNIST is available for circuit debugging only (`data/mnist_loader.py`) and is used in no scientific comparison.

---

## Directory layout

See `PROJECT_SUMMARY.md` §13 — the implementation follows it exactly. Key entry points:

```
quantum/          qpie_encoder, qhed_circuit, neqr_encoder, quantum_gradient,
                  gates, aer_runner, noise_models, reconstruction,
                  confidence_map, threshold_selector, quantum_pipeline
classical/        sobel_canny, robinson_compass, classical_pipeline
preprocessing/    image_loader, patch_extractor, annotation_rasteriser,
                  augmentor, preprocessor, sombench_data, lunar_fallback
evaluation/       metrics, robustness, resource_counter, visualiser
experiments/      exp1..exp10 + run_all.py
tests/            test_gates, test_qhed, test_neqr  (all must PASS)
dashboard/        Streamlit app
```

## References
Yao et al., *Phys. Rev. X* 7, 031041 (2017) · Zhang et al., *QIP* 12, 2833 (2013) · Wang et al., *QIP* 14, 1589 (2015) · Le et al., *QIP* 10, 63 (2011) · Robinson, *CGIP* 6, 492 (1977) · Canny, *IEEE TPAMI* 8, 679 (1986) · NASA-IBM SomBench (2023).
