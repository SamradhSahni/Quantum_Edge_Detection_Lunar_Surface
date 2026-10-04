"""
QPIE -- Quantum Probability Image Encoding (Le et al., 2011).

Encodes a grayscale image as probability amplitudes of a pure state:

    |psi> = (1/C) * SUM_{r,c} I(r,c) * |r>|c>,      C = ||I||_2

Intensity lives in the AMPLITUDE, not in a computational-basis register.

Qubit layout for an n x n image (n = 2^k) used across the project:

    q[0]        : ancilla (used later by QHED)
    q[1 .. k]   : row register    (q[1] = row LSB)
    q[k+1..2k]  : column register (q[k+1] = col LSB)

Because Qiskit orders basis states with q[0] as the LEAST significant
bit, the statevector index of |r, c> is  row + col * n.  The helpers
below flatten/reshape images in exactly that convention and are covered
by round-trip tests in tests/test_qhed.py.
"""

import numpy as np
from qiskit import QuantumCircuit


def image_to_amplitudes(image: np.ndarray):
    """Return (amplitudes, norm) for a grayscale image in [0,1].

    amplitudes[i] corresponds to basis state |row, col> with
    i = row + col * n (Qiskit little-endian over the position register).
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("image must be square")
    n = image.shape[0]
    if n & (n - 1):
        raise ValueError("image side length must be a power of two")
    flat = image.T.reshape(-1)  # flat[row + col*n] = image[row, col]
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        return np.zeros_like(flat), 0.0
    return flat / norm, norm


def amplitudes_to_image(amplitudes: np.ndarray) -> np.ndarray:
    """Inverse of image_to_amplitudes (reshape back to a square image)."""
    amplitudes = np.asarray(amplitudes)
    m = amplitudes.size
    n = int(round(np.sqrt(m)))
    return amplitudes.reshape(n, n).T.copy()


def num_position_qubits(image: np.ndarray) -> int:
    """Position register size: 2*log2(n) for an n x n image."""
    n = np.asarray(image).shape[0]
    return 2 * int(np.log2(n))


def build_qpie_circuit(image: np.ndarray, add_ancilla: bool = True) -> QuantumCircuit:
    """Circuit preparing the QPIE state of `image`.

    With add_ancilla=True the circuit has 2*log2(n) + 1 qubits and the
    ancilla q[0] stays |0>, ready for the QHED interference step.  State
    preparation uses Qiskit's StatePreparation (isometry-style
    decomposition), matching the "RY / isometry loading" family in
    PROJECT_SUMMARY.md Section 5.2.
    """
    amplitudes, norm = image_to_amplitudes(image)
    n = np.asarray(image).shape[0]
    k = int(np.log2(n))
    n_pos = 2 * k
    n_qubits = n_pos + 1 if add_ancilla else n_pos
    qc = QuantumCircuit(n_qubits, name=f"QPIE_{n}x{n}")
    if norm == 0.0:
        return qc  # |0...0> encodes the all-black image trivially
    # Qiskit amplitude vector is indexed over ALL circuit qubits with the
    # given qubits as the least-significant positions; with the ancilla
    # at position 0 the position register starts at qubit 1, otherwise
    # at qubit 0 (the |row, col> index mapping is identical either way).
    start = 1 if add_ancilla else 0
    qc.prepare_state(amplitudes, list(range(start, start + n_pos)))
    return qc
