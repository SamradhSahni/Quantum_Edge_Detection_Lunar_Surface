"""
Central configuration for the Quantum Edge Detection project.

All experiment scripts import from here so that every parameter listed in
PROJECT_SUMMARY.md Section 8 ("Fixed Parameters") is defined in exactly one
place:

    Bit depth ................ 8-bit (0-255)
    Random seed .............. 42
    Trials per configuration . 10
    Simulator ................ Qiskit AerSimulator (ideal + noisy)
    Threshold ................ validation-set selected (never hard-coded)
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")
SYNTH_DIR = os.path.join(DATA_DIR, "synthetic")
SYNTH_IMG_DIR = os.path.join(SYNTH_DIR, "images")
SYNTH_MASK_DIR = os.path.join(SYNTH_DIR, "masks")

SOMBENCH_DIR = os.path.join(DATA_DIR, "sombench")
SOMBENCH_IMG_DIR = os.path.join(SOMBENCH_DIR, "images")
SOMBENCH_ANN_DIR = os.path.join(SOMBENCH_DIR, "annotations")
SOMBENCH_PATCH_DIR = os.path.join(SOMBENCH_DIR, "patches")
SOMBENCH_MASK_DIR = os.path.join(SOMBENCH_DIR, "masks")
LUNAR_FALLBACK_DIR = os.path.join(DATA_DIR, "lunar_fallback")  # procedural craters

LRO_DIR = os.path.join(DATA_DIR, "nasa_lro")
MNIST_DIR = os.path.join(DATA_DIR, "mnist")

RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
CIRCUIT_DIR = os.path.join(RESULTS_DIR, "circuits")
CONF_DIR = os.path.join(RESULTS_DIR, "confidence_maps")
FAIL_DIR = os.path.join(RESULTS_DIR, "failure_gallery")

for _d in (SYNTH_IMG_DIR, SYNTH_MASK_DIR, SOMBENCH_IMG_DIR, SOMBENCH_ANN_DIR,
           SOMBENCH_PATCH_DIR, SOMBENCH_MASK_DIR, LUNAR_FALLBACK_DIR,
           FIG_DIR, TABLE_DIR, CIRCUIT_DIR, CONF_DIR, FAIL_DIR,
           os.path.join(LUNAR_FALLBACK_DIR, "images"),
           os.path.join(LUNAR_FALLBACK_DIR, "masks"),
           os.path.join(LUNAR_FALLBACK_DIR, "patches")):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed experimental parameters (PROJECT_SUMMARY.md Section 8)
# ---------------------------------------------------------------------------
SEED = 42                      # global reproducibility seed
N_TRIALS = 10                  # repeated trials per configuration (mean +/- std)
DEFAULT_SHOTS = 8192           # canonical shot count
BIT_DEPTH = 8                  # 8-bit grayscale (0-255)
DEFAULT_SIZE = 8               # canonical image resolution (8x8)
SIZES = [4, 8, 16]             # resolution-scaling experiment
SHOT_GRID = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Degradation grids (Experiment 4)
GAUSS_NOISE_SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.10]
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
CONTRAST_SCALES = [0.25, 0.50, 0.75, 1.00, 1.25]

# Quantum noise grids (Experiment 5)
DEPOLARIZING_P = [0.001, 0.005, 0.01, 0.05]
READOUT_P = [0.01, 0.05, 0.10]
OVERROTATION_FRAC = 0.02       # coherent over-rotation: theta -> theta*(1+eps)

# ---------------------------------------------------------------------------
# Method registry (used by every experiment loop)
# ---------------------------------------------------------------------------
METHODS = ["sobel", "canny", "robinson", "qhed", "neqr"]
METHOD_LABELS = {
    "sobel": "Sobel",
    "canny": "Canny",
    "robinson": "Robinson (8-dir)",
    "qhed": "QPIE-QHED",
    "neqr": "NEQR-Gradient",
}
METHOD_COLORS = {
    "sobel": "#1f77b4",
    "canny": "#ff7f0e",
    "robinson": "#2ca02c",
    "qhed": "#d62728",
    "neqr": "#9467bd",
}

# ---------------------------------------------------------------------------
# Simulation strategy notes
# ---------------------------------------------------------------------------
# Qiskit AerSimulator is the execution backend for every quantum result.
# Two execution modes are used, both provided by quantum/aer_runner.py:
#
#   "circuit"   : transpile the real circuit and run it on AerSimulator.
#                 Used for: every noisy run, kernel/E2E timing, circuit
#                 validation, and all QPIE ideal runs (circuits are tiny).
#
#   "statevector": evolve the real circuit once with the exact statevector
#                 method and sample multinomial counts from the final
#                 probability distribution (this is what Aer's statevector
#                 method does internally for repeated shots).  Used for the
#                 ideal NEQR runs at 8x8/16x16 where re-evolving a ~10^5-gate
#                 circuit for each of 10 trials would dominate runtime
#                 without changing the statistics.  Circuits are verified
#                 against analytic target states in tests/.
STATEVECTOR_MODE_NOTE = (
    "Ideal NEQR results use exact statevector sampling of the validated "
    "circuit output distribution; noisy results always execute the real "
    "circuit on AerSimulator."
)
