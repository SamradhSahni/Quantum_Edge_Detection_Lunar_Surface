"""
AerSimulator execution wrapper (ideal + noisy) for both quantum branches.

Provides three execution paths used by the experiment suite:

  run_circuit_counts()  : transpile + run the REAL circuit on
                          AerSimulator (used for every noisy run, for
                          kernel/E2E timing, and for all QPIE ideal runs).

  sample_statevector()  : evolve the real circuit once with the exact
                          statevector method and draw multinomial shot
                          counts from the final distribution (what Aer's
                          statevector method does internally per shot).
                          Used for ideal NEQR runs at 8x8/16x16 to keep
                          the 10-trial x multi-image matrix tractable;
                          equivalence to run_circuit_counts is asserted
                          in tests/.

  Timing: kernel time is measured strictly around simulator.run();
          end-to-end time is measured around the whole pipeline call.
"""

import time

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

_SIM_CACHE = {}


def get_simulator(noise_model=None, seed=None, method=None):
    """(Cached) AerSimulator instance."""
    key = (id(noise_model) if noise_model is not None else None, method)
    if key not in _SIM_CACHE:
        kwargs = {}
        if method is not None:
            kwargs["method"] = method
        if noise_model is not None:
            kwargs["noise_model"] = noise_model
        _SIM_CACHE[key] = AerSimulator(**kwargs)
    return _SIM_CACHE[key]


HARDWARE_BASIS = ["rz", "sx", "x", "cx"]


def transpile_for_aer(circuit, noise_model=None, optimization_level=1, method=None):
    """Transpile a circuit for AerSimulator.

    Noisy runs AND all density-matrix runs are transpiled to the
    hardware basis {rz, sx, x, cx}: noise must apply to every
    decomposed operation, and the density-matrix method rejects Aer's
    opaque multi-controlled primitives.  Plain ideal runs keep the
    simulator's native target (mcx executed as an efficient primitive).
    """
    if noise_model is not None or method == "density_matrix":
        return transpile(circuit, basis_gates=HARDWARE_BASIS,
                         optimization_level=optimization_level, seed_transpiler=42)
    sim = get_simulator(noise_model=noise_model)
    return transpile(circuit, sim, optimization_level=optimization_level)


def run_circuit_counts(circuit, shots=8192, seed=42, noise_model=None,
                       optimization_level=1, method=None):
    """Execute the real circuit on AerSimulator and return (counts, kernel_s).

    kernel_s is the wall-clock seconds spent inside simulator.run()
    (transpilation excluded -- it is reported separately by the resource
    counter).
    """
    sim = get_simulator(noise_model=noise_model, method=method)
    tq = transpile_for_aer(circuit, noise_model=noise_model,
                           optimization_level=optimization_level, method=method)
    t0 = time.perf_counter()
    result = sim.run(tq, shots=shots, seed_simulator=seed).result()
    kernel_s = time.perf_counter() - t0
    counts = result.get_counts()
    if isinstance(counts, list):
        counts = counts[0]
    return counts, kernel_s


def sample_statevector(circuit, shots=8192, seed=42):
    """Exact-statevector sampling: evolve once, draw multinomial counts.

    Returns (counts, kernel_s) with the same interface as
    run_circuit_counts so experiments can use the two interchangeably.
    """
    t0 = time.perf_counter()
    circ = circuit.remove_final_measurements(inplace=False)
    probs = Statevector(circ).probabilities_dict()
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(shots, p)
    counts = {k: int(c) for k, c in zip(keys, draws) if c > 0}
    kernel_s = time.perf_counter() - t0
    return counts, kernel_s


def counts_to_prob_dict(counts: dict) -> dict:
    """Normalise a counts dict to probabilities."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def hellinger_fidelity(counts_a: dict, counts_b: dict) -> float:
    """Hellinger fidelity between two count distributions (0..1).

    Used as the circuit-fidelity proxy under quantum noise.
    """
    pa = counts_to_prob_dict(counts_a)
    pb = counts_to_prob_dict(counts_b)
    keys = set(pa) | set(pb)
    bc = sum(np.sqrt(pa.get(k, 0.0) * pb.get(k, 0.0)) for k in keys)
    return float(bc ** 2)
