"""
NEQR -- Novel Enhanced Quantum Representation (Zhang et al., 2013).

Branch B state encoding: the EXACT 8-bit intensity of every pixel is
stored in the computational basis of a dedicated intensity register:

    |Q> = (1/sqrt(N)) * SUM_{r,c} |f(r,c)> |r> |c>

    |f(r,c)> = |q7 q6 q5 q4 q3 q2 q1 q0>   (8-qubit binary intensity)
    |r>,|c>  = position in the computational basis

State preparation is a network of multi-controlled X gates, one group
per pixel (NOT RY rotations / angle mappings -- that is the QPIE
family).  Measuring the register in the computational basis recovers
exact pixel values, which is the deterministic-recovery property
contrasted with QPIE in PROJECT_SUMMARY.md Section 5.4.

Qubit layout (n x n image, n = 2^k):

    q[0  .. k-1 ]  row register    (q[0] = row LSB)
    q[k  .. 2k-1]  column register (q[k] = col LSB)
    q[2k .. 2k+7]  F intensity register (q[2k] = bit 0 of the byte)
    q[2k+8.. 2k+15] G intensity register (loaded with the NEIGHBOUR's
                    value by quantum_gradient.py)
    q[2k+16, 2k+17] two clean ancillas for the CDKM subtractor

A measured bitstring v (int) decodes as
    row =  v  & (n-1)
    col = (v >> k) & (n-1)
    F   = (v >> 2k)   & 0xFF
    G   = (v >> 2k+8) & 0xFF
"""

import numpy as np
from qiskit import QuantumCircuit

from quantum.gates import mcx

INTENSITY_BITS = 8  # default 8-bit grayscale per the fixed parameters


def neqr_layout(k: int, intensity_bits: int = None):
    """Qubit index ranges for an image with 2^k x 2^k pixels.

    intensity_bits defaults to 8 (full grayscale).  The minimal 4-bit
    variant (16 gray levels, 12 qubits at 2x2) is used for the exact
    density-matrix noise experiment (Experiment 5).
    """
    w = INTENSITY_BITS if intensity_bits is None else intensity_bits
    row = list(range(0, k))
    col = list(range(k, 2 * k))
    f_reg = list(range(2 * k, 2 * k + w))
    g_reg = list(range(2 * k + w, 2 * k + 2 * w))
    ancillas = [2 * k + 2 * w, 2 * k + 2 * w + 1]
    return {"row": row, "col": col, "f": f_reg, "g": g_reg, "anc": ancillas,
            "n_qubits": 2 * k + 2 * w + 2, "intensity_bits": w}


def _position_pattern(r: int, c: int, k: int, shift: str = None):
    """Positive/negative control flags for each position qubit.

    shift=None  -> pattern of position (r, c)
    shift='col' -> pattern of (r, (c-1) mod n)
    shift='row' -> pattern of ((r-1) mod n, c)

    Writing pixel (r,c) at (r, c-1) means position p=(r,c') receives
    f(r, c'+1): G(p) = RIGHT/BOTTOM neighbour of p, matching the
    Sobel-style payload d(r,c) = f(r,c) - f(nb(r,c)).  (An LSB flip
    would only swap pixel pairs and diverge from the roll for n > 2.)
    """
    n = 2 ** k
    r_pat, c_pat = r, c
    if shift == "col":
        c_pat = (c - 1) % n
    elif shift == "row":
        r_pat = (r - 1) % n
    bits = []
    for i in range(k):
        bits.append(((r_pat >> i) & 1) == 1)
    for i in range(k):
        bits.append(((c_pat >> i) & 1) == 1)
    return bits


def append_neqr_prep(qc: QuantumCircuit, image: np.ndarray, k: int,
                     register: str = "f", shift: str = None,
                     put_position_in_superposition: bool = None,
                     intensity_bits: int = None) -> None:
    """Append the NEQR state-preparation network for one intensity register.

    NEQR state preparation (Zhang et al., 2013) is a two-step network:

        1. H on every position qubit -> (1/sqrt(N)) SUM_{r,c} |r>|c>|0>
        2. for each pixel, an X-wrapped multi-controlled X group writes
           the pixel byte into the intensity register.

    Step 1 is applied ONLY for the first register prepared (the F
    register).  The G register is prepared while the position register
    is ALREADY in superposition (entangled with F), so re-applying H
    would destroy the state; for that register only the MCX network is
    appended, with `shift` making each pixel's group write at its
    NEIGHBOUR's position: pixel (r, c)'s controls match position
    (r, (c+1) mod n) for shift='col', so position p ends up holding
    G(p) = f(neighbour of p) -- the quantum-image-translation effect
    (Wang et al., 2015) implemented inside state preparation.

    `put_position_in_superposition` defaults to True when `shift` is
    None (fresh F register) and False otherwise.
    """
    if put_position_in_superposition is None:
        put_position_in_superposition = shift is None
    if put_position_in_superposition:
        lay0 = neqr_layout(k)
        for q in lay0["row"] + lay0["col"]:
            qc.h(q)

    w = INTENSITY_BITS if intensity_bits is None else intensity_bits
    levels = 2 ** w - 1
    image = np.clip(np.rint(np.asarray(image, dtype=np.float64)
                            * (levels / 255.0)), 0, levels).astype(np.uint8)
    n = 2 ** k
    lay = neqr_layout(k, w)
    pos = lay["row"] + lay["col"]
    tgt = lay[register]

    for r in range(n):
        for c in range(n):
            f = int(image[r, c])
            if f == 0:
                continue  # |0> targets stay |0>: no gates needed
            pattern = _position_pattern(r, c, k, shift=shift)
            zeros = [q for q, positive in zip(pos, pattern) if not positive]
            for q in zeros:
                qc.x(q)
            for j in range(w):
                if (f >> j) & 1:
                    mcx(qc, pos, tgt[j])
            for q in zeros:
                qc.x(q)


def build_neqr_image_circuit(image: np.ndarray, measure: bool = True,
                             intensity_bits: int = None) -> QuantumCircuit:
    """NEQR encoding circuit (position + F register only, G left |0>).

    Measuring it in the computational basis recovers exact pixel values
    with uniform 1/N probability -- the deterministic-recovery property
    of the representation (validation experiments in tests/test_neqr.py).
    """
    image = np.asarray(image)
    n = image.shape[0]
    k = int(np.log2(n))
    w = INTENSITY_BITS if intensity_bits is None else intensity_bits
    lay = neqr_layout(k, w)
    measured = lay["row"] + lay["col"] + lay["f"]
    qc = QuantumCircuit(lay["n_qubits"], len(measured) if measure else 0,
                        name=f"NEQR_{n}x{n}")
    append_neqr_prep(qc, image, k, register="f", intensity_bits=w)
    if measure:
        for i, q in enumerate(measured):
            qc.measure(q, i)
    return qc


def decode_neqr_counts(counts: dict, k: int, intensity_bits: int = None):
    """Decode counts into per-pixel intensity histograms.

    Returns dict[pos] = {intensity: count} with pos = row + col * n.
    """
    w = INTENSITY_BITS if intensity_bits is None else intensity_bits
    fmask = (1 << w) - 1
    n = 2 ** k
    out = {}
    for key, cnt in counts.items():
        v = int(key.replace(" ", ""), 2)
        pos = v & (2 ** (2 * k) - 1)
        f = (v >> 2 * k) & fmask
        out.setdefault(pos, {})
        out[pos][f] = out[pos].get(f, 0) + cnt
    return out
