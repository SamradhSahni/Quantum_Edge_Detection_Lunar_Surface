"""
Quantum gradient operator on NEQR-encoded images (Branch B edge detector).

Pipeline (Wang et al. 2015 style quantum-image translation + quantum
arithmetic, following Zhang et al. 2013's representation):

    1. NEQR-prepare the F register at position (r, c):
           (1/sqrt(N)) SUM_{r,c} |f(r,c)> |r> |c>

    2. NEQR-prepare the G register at the SHIFTED position (r, c-1)
       (horizontal) or (r-1, c) (vertical) -- "quantum image
       translation" implemented inside state preparation via inverted
       position controls:
           G now holds f(neighbour of (r,c)) entangled with |r>|c>

    3. Quantum subtractor (CDKM ripple-carry network, quantum/gates.py):
           G <- F - G (mod 2^w), with the adder's carry-out qubit
           recording the sign:  carry=0 -> delta = G (>= 0),
                                carry=1 -> delta = G - 2^w (< 0)

    4. Measure position + F + G + carry.  Each shot returns one pixel's
       position, its intensity, and the EXACT difference

           |delta(r,c)| = G          if carry = 0
                          2^w - G    if carry = 1

       which -- unlike the bare modular byte -- recovers the TRUE
       magnitude |f(r,c) - f(nb(r,c))| even for contrasts above half
       the register range (a 204-level step no longer wraps to 52).

The final state is a computational-basis superposition

    (1/sqrt(N)) SUM_{r,c} |r>|c>|f>|delta mod 2^w>|carry>

so the ideal output distribution is known exactly; ideal experiments
sample that distribution (statevector-equivalent, see config.py note)
while all noisy experiments execute the real circuit on AerSimulator.
Circuit equivalence to the analytic payload is asserted in
tests/test_neqr.py.
"""

import numpy as np
from qiskit import QuantumCircuit

from quantum.gates import append_subtractor
from quantum.neqr_encoder import (INTENSITY_BITS, append_neqr_prep,
                                  neqr_layout)


def build_neqr_gradient_circuit(image: np.ndarray, direction: str = "horizontal",
                                measure: bool = True,
                                intensity_bits: int = None) -> QuantumCircuit:
    """Full NEQR + quantum-gradient circuit for one direction.

    intensity_bits=None -> 8-bit registers; a smaller value (e.g. 4)
    quantizes the image to fewer levels and shrinks the circuit (used by
    the exact density-matrix noise experiment).
    """
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    image = np.asarray(image)
    n = image.shape[0]
    k = int(np.log2(n))
    lay = neqr_layout(k, intensity_bits)
    measured = lay["row"] + lay["col"] + lay["f"] + lay["g"]
    # +1 classical bit for the subtractor carry-out (difference sign)
    qc = QuantumCircuit(lay["n_qubits"], len(measured) + 1 if measure else 0,
                        name=f"NEQRgrad_{n}x{n}_{direction[:3]}")

    append_neqr_prep(qc, image, k, register="f", intensity_bits=lay["intensity_bits"])
    append_neqr_prep(qc, image, k, register="g",
                     shift="col" if direction == "horizontal" else "row",
                     intensity_bits=lay["intensity_bits"])
    append_subtractor(qc, lay["f"], lay["g"], lay["anc"])

    if measure:
        for i, q in enumerate(measured):
            qc.measure(q, i)
        qc.measure(lay["anc"][0], len(measured))
    return qc


def neqr_gradient_payload(image: np.ndarray, direction: str = "horizontal",
                          intensity_bits: int = None):
    """Exact per-pixel payload of the ideal gradient circuit.

    Returns (rows, cols, f_values, d_values, carry_values) arrays with
        d     = (f(r,c) - f(nb(r,c))) mod 2^w
        carry = 1 if f(r,c) < f(nb(r,c)) else 0
    so the true magnitude is |f - nb| = d if carry==0 else 2^w - d.
    All N positions occur with equal probability 1/N.
    """
    w = 8 if intensity_bits is None else intensity_bits
    levels = 2 ** w - 1
    img = np.clip(np.rint(np.asarray(image, dtype=np.float64)
                          * (levels / 255.0)), 0, levels).astype(np.int64)
    n = img.shape[0]
    if direction == "horizontal":
        nb = np.roll(img, -1, axis=1)
    else:
        nb = np.roll(img, -1, axis=0)
    delta = img - nb
    d = delta % (2 ** w)
    carry = (delta < 0).astype(np.int64)
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return rr.ravel(), cc.ravel(), img.ravel(), d.ravel(), carry.ravel()


def signed_value(d: int, bits: int = 8) -> int:
    """Interpret a measured difference value as signed two's complement."""
    half = 1 << (bits - 1)
    return d - (1 << bits) if d >= half else d


def signed_byte(d: int) -> int:
    """Back-compat alias: 8-bit signed interpretation."""
    return signed_value(d, 8)


def sample_ideal_neqr_gradient(image: np.ndarray, direction: str,
                               shots: int, seed: int,
                               intensity_bits: int = None):
    """Multinomial sampling of the ideal gradient-circuit distribution.

    Statistically identical to running the real circuit on Aer's exact
    statevector method (uniform position, deterministic payloads);
    validated in tests/test_neqr.py.  Returns a counts dict equivalent
    to get_counts(): key = single classical register bitstring over
    [carry(1), G(w), F(w), col(k), row(k)] with w = intensity_bits
    (classical bit 0 = row, matching the circuit's measure order).
    """
    rng = np.random.default_rng(seed)
    w = 8 if intensity_bits is None else intensity_bits
    rr, cc, ff, dd, ccy = neqr_gradient_payload(image, direction, w)
    k = int(np.log2(np.asarray(image).shape[0]))
    n_pix = rr.size
    draws = rng.multinomial(shots, np.full(n_pix, 1.0 / n_pix))
    counts = {}
    for r, c, f, d, cy, cnt in zip(rr, cc, ff, dd, ccy, draws):
        if cnt == 0:
            continue
        v = ((int(r) & (2 ** k - 1))
             | (int(c) << k)
             | (int(f) << 2 * k)
             | (int(d) << (2 * k + w))
             | (int(cy) << (2 * k + 2 * w)))
        key = format(v, f"0{2 * k + 2 * w + 1}b")
        counts[key] = counts.get(key, 0) + int(cnt)
    return counts
