"""
Unified quantum edge-detection pipelines (QPIE-QHED and NEQR-Gradient).

run_quantum(image, method, ...) is the single entry point used by every
experiment and the dashboard.  It returns, per method:

    strength map (continuous), uncertainty map, binary map under a
    relative threshold, resource record (qubits/depth/gates/timing), and
    raw counts for reproducibility.

Ideal vs noisy
--------------
    noise_model=None (ideal):
        QHED  -> real circuit on AerSimulator (statevector method).
        NEQR  -> exact-statevector multinomial sampling of the validated
                 circuit distribution (see quantum/aer_runner.py); the
                 equivalence is asserted in tests/test_neqr.py.
    noise_model given:
        BOTH branches always execute the REAL (transpiled) circuit on
        AerSimulator with that noise model.
"""

import time

import numpy as np

import config
from classical.sobel_canny import sobel_magnitude  # noqa: F401  (scale reference)
from quantum import reconstruction
from quantum.aer_runner import (run_circuit_counts, sample_statevector,
                                transpile_for_aer)
from quantum.confidence_map import EdgeResult, binomial_uncertainty
from quantum.neqr_encoder import neqr_layout
from quantum.noise_models import build_noise_model
from quantum.qhed_circuit import build_qhed_circuit
from quantum.qpie_encoder import image_to_amplitudes
from quantum.quantum_gradient import (build_neqr_gradient_circuit,
                                      sample_ideal_neqr_gradient)


def _encode_for_qpie(image: np.ndarray) -> np.ndarray:
    """Scale [0,1] floats into the amplitude encoding (max -> 1)."""
    img = np.asarray(image, dtype=np.float64)
    m = img.max()
    return img / m if m > 0 else img


def run_qhed(image: np.ndarray, shots: int = config.DEFAULT_SHOTS, seed: int = config.SEED,
             noise_model=None, timing: bool = False, sim_method: str = None):
    """QPIE + QHED pipeline, both directions.

    sim_method: force an Aer method (e.g. 'density_matrix' for exact
    noisy runs); None -> ideal statevector / noisy trajectories.
    """
    t0 = time.perf_counter()
    img = np.asarray(image, dtype=np.float64)
    size = img.shape[0]
    enc = _encode_for_qpie(img)
    _, norm = image_to_amplitudes(enc)

    counts_h = counts_v = {}
    kernel_s = 0.0
    transpiled = None

    for direction, store in (("horizontal", "h"), ("vertical", "v")):
        qc = build_qhed_circuit(enc, direction=direction, measure=True)
        c, ks = run_circuit_counts(qc, shots=shots, seed=seed,
                                   noise_model=noise_model,
                                   optimization_level=1, method=sim_method)
        if store == "h":
            counts_h, transpiled = c, qc
        else:
            counts_v = c
        kernel_s += ks

    # probability maps and magnitude in |Delta I| units (max scale 1)
    ph = reconstruction.qhed_counts_to_prob_map(counts_h, size)
    pv = reconstruction.qhed_counts_to_prob_map(counts_v, size)
    mag_h = reconstruction.qhed_prob_map_to_magnitude(ph, norm) if norm > 0 else np.zeros((size, size))
    mag_v = reconstruction.qhed_prob_map_to_magnitude(pv, norm) if norm > 0 else np.zeros((size, size))
    strength = np.sqrt(mag_h ** 2 + mag_v ** 2)

    unc_h = binomial_uncertainty(ph, shots)
    unc_v = binomial_uncertainty(pv, shots)
    unc = 2 * norm * 0.5 * np.sqrt(unc_h ** 2 + unc_v ** 2) if norm > 0 else np.zeros_like(strength)

    result = EdgeResult("qhed", size, strength, uncertainty=unc,
                        meta={"p_h": ph, "p_v": pv, "counts_h": counts_h,
                              "counts_v": counts_v, "shots": shots,
                              "image_norm": norm})
    e2e_s = time.perf_counter() - t0

    resources = _resource_record(transpiled, kernel_s, e2e_s, "qhed", size)
    if timing:
        return result, resources
    return result, resources


def run_neqr(image: np.ndarray, shots: int = config.DEFAULT_SHOTS, seed: int = config.SEED,
             noise_model=None, timing: bool = False, intensity_bits: int = 8,
             sim_method: str = None):
    """NEQR + quantum-gradient pipeline, both directions.

    intensity_bits: register width; 8 everywhere except the minimal
    density-matrix noise experiment (Experiment 5 uses a 2x2 image with
    4-bit registers, 12 qubits).
    sim_method: force an Aer method ('density_matrix' for exact noisy
    runs; None -> ideal statevector sampling / noisy trajectories).
    """
    t0 = time.perf_counter()
    # quantize the [0,1] image to the full 0-255 scale first; the
    # encoder and payload then map 0-255 -> register levels (2^w - 1)
    img = np.clip(np.rint(np.asarray(image, dtype=np.float64) * 255.0), 0, 255)
    size = img.shape[0]

    counts_h = counts_v = {}
    kernel_s = 0.0
    transpiled = None

    if noise_model is None and sim_method is None:
        for direction, store in (("horizontal", "h"), ("vertical", "v")):
            ts = time.perf_counter()
            counts = sample_ideal_neqr_gradient(img, direction, shots, seed,
                                                intensity_bits=intensity_bits)
            ks = time.perf_counter() - ts
            if store == "h":
                counts_h = counts
            else:
                counts_v = counts
            kernel_s += ks
        # build the real circuit once for resource accounting (no run)
        transpiled = build_neqr_gradient_circuit(img, "horizontal", measure=True,
                                                 intensity_bits=intensity_bits)
    else:
        for direction, store in (("horizontal", "h"), ("vertical", "v")):
            qc = build_neqr_gradient_circuit(img, direction, measure=True,
                                             intensity_bits=intensity_bits)
            c, ks = run_circuit_counts(qc, shots=shots, seed=seed,
                                       noise_model=noise_model,
                                       optimization_level=1, method=sim_method)
            if store == "h":
                counts_h, transpiled = c, qc
            else:
                counts_v = c
            kernel_s += ks

    strength = reconstruction.reconstruct_neqr_edge_map(counts_h, counts_v, img,
                                                        intensity_bits=intensity_bits)
    per_pixel_shots = shots / (size * size)
    unc = strength / np.sqrt(max(per_pixel_shots, 1.0))

    result = EdgeResult("neqr", size, strength, uncertainty=unc,
                        meta={"counts_h": counts_h, "counts_v": counts_v,
                              "shots": shots, "intensity_bits": intensity_bits})
    e2e_s = time.perf_counter() - t0
    resources = _resource_record(transpiled, kernel_s, e2e_s, "neqr", size)
    if timing:
        return result, resources
    return result, resources


def _resource_record(circuit, kernel_s: float, e2e_s: float, method: str, size: int) -> dict:
    """Qubit/gate/depth breakdown for one method at one size."""
    from evaluation.resource_counter import circuit_resources
    rec = circuit_resources(circuit) if circuit is not None else {}
    rec.update({
        "method": method,
        "image_size": size,
        "kernel_time_s": kernel_s,
        "e2e_time_s": e2e_s,
    })
    return rec


def run_quantum(image, method: str, shots: int = config.DEFAULT_SHOTS,
                seed: int = config.SEED, noise_name: str = "ideal", timing: bool = False,
                sim_method: str = None, intensity_bits: int = 8):
    """Dispatch by method name ('qhed' or 'neqr')."""
    noise_model = build_noise_model(noise_name) if noise_name != "ideal" else None
    if method == "qhed":
        return run_qhed(image, shots=shots, seed=seed, noise_model=noise_model,
                        timing=timing, sim_method=sim_method)
    if method == "neqr":
        return run_neqr(image, shots=shots, seed=seed, noise_model=noise_model,
                        timing=timing, intensity_bits=intensity_bits,
                        sim_method=sim_method)
    raise ValueError(f"unknown quantum method {method}")


def qubit_counts(method: str, size: int, intensity_bits: int = 8) -> dict:
    """Analytic logical/ancilla/total qubit counts (no circuit needed)."""
    k = int(np.log2(size))
    if method == "qhed":
        pos = 2 * k
        return {"logical": pos, "ancilla": 1, "total": pos + 1}
    if method == "neqr":
        lay = neqr_layout(k, intensity_bits)
        logical = 2 * k + 2 * lay["intensity_bits"]  # position + 2 intensity regs
        return {"logical": logical, "ancilla": len(lay["anc"]), "total": lay["n_qubits"]}
    raise ValueError(method)
