"""
QHED -- Quantum Hadamard Edge Detection (Yao et al., PRX 7, 031041, 2017).

Branch A of the project: edge extraction from a QPIE-encoded image via
Hadamard interference.  The exact state evolution implemented here:

    1. QPIE state        |psi> = SUM_{r,c} a(r,c) |r>|c>,  a = I/||I||

    2. H on ancilla      |psi>(|0>+|1>)/sqrt(2)

    3. (k+1)-qubit DECREMENT over [ancilla, col-register] with the
       ancilla as LEAST-significant bit.  For the joint integer
       Y = 2*col + ancilla this maps
           (col, a=0) -> (col-1, a=1)      and      (col, a=1) -> (col, a=0)
       giving
           SUM_{r,c} a(r,c) (|r,c>|0> + |r,c-1>|1>)/sqrt(2)
         = SUM_{r,c} (a(r,c)|0> + a(r,c+1)|1>) |r,c> / sqrt(2)

    4. H on ancilla      SUM_{r,c} [ (a(r,c)+a(r,c+1))/2 ]|0>|r,c>
                                  + [ (a(r,c)-a(r,c+1))/2 ]|1>|r,c>

    Measuring the ancilla in |1> projects onto the edge basis with

        P(a=1, r, c) = |a(r,c) - a(r,c+1)|^2 / 4

i.e. the squared horizontal intensity difference at pixel (r, c), with
periodic wrap-around inside each row.  The same construction over the
ROW register yields the vertical gradient.  The two probability maps
are combined Sobel-style:  E = sqrt(P_h + P_v).

This module is validated against the analytic target statevector in
tests/test_qhed.py (exact amplitude comparison on random images).
"""

import numpy as np
from qiskit import QuantumCircuit

from quantum.gates import decrement
from quantum.qpie_encoder import build_qpie_circuit, image_to_amplitudes

# Qubit layout (n x n image, n = 2^k):
#   q[0]      ancilla
#   q[1..k]   row    (q[1] = LSB)
#   q[k+1..2k] col   (q[k+1] = LSB)


def build_qhed_circuit(image: np.ndarray, direction: str = "horizontal",
                       measure: bool = True) -> QuantumCircuit:
    """Canonical QHED circuit for one gradient direction.

    Parameters
    ----------
    image      : n x n float array in [0, 1] (or any non-negative scale)
    direction  : 'horizontal' -> differences a(r,c) vs a(r,c+1) (col register)
                 'vertical'   -> differences a(r,c) vs a(r+1,c) (row register)
    measure    : append measurement of all qubits into one register
    """
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    image = np.asarray(image, dtype=np.float64)
    n = image.shape[0]
    k = int(np.log2(n))

    qc = build_qpie_circuit(image, add_ancilla=True)

    qc.h(0)
    if direction == "horizontal":
        shift_register = list(range(k + 1, 2 * k + 1))
    else:
        shift_register = list(range(1, k + 1))
    # ancilla acts as the LSB of the shifted register (Yao et al. trick)
    decrement(qc, [0] + shift_register)
    qc.h(0)

    if measure:
        qc.measure_all()
    return qc


def qhed_analytic_edge_probability(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Exact P(a=1, r, c) map the QHED circuit produces (ideal physics).

    Used for unit-testing the circuit and as a reference for the
    reconstruction pipeline.
    """
    amplitudes, norm = image_to_amplitudes(image)
    n = image.shape[0]
    amp_img = amplitudes.reshape(n, n).T  # amp_img[r, c]
    if direction == "horizontal":
        shifted = np.roll(amp_img, -1, axis=1)
    else:
        shifted = np.roll(amp_img, -1, axis=0)
    diff = amp_img - shifted
    return (diff ** 2) / 4.0


def combine_directions(p_h: np.ndarray, p_v: np.ndarray) -> np.ndarray:
    """Sobel-style combination of the two QHED probability maps."""
    return np.sqrt(np.asarray(p_h) + np.asarray(p_v))
