"""Correctness tests for the NEQR pipeline (pytest or standalone).

The 4x4 statevector checks are the slow ones (~seconds via Aer's C++
statevector); they are kept because they are the core correctness
guarantee of Branch B.
"""

import os
import sys

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum.neqr_encoder import (build_neqr_image_circuit,   # noqa: E402
                                  decode_neqr_counts, neqr_layout)
from quantum.quantum_gradient import (build_neqr_gradient_circuit,  # noqa: E402
                                      neqr_gradient_payload, sample_ideal_neqr_gradient)


def _sv_probs(circuit):
    sim = AerSimulator(method="statevector", precision="double")
    tqc = transpile(circuit, sim, optimization_level=0)
    tqc.save_statevector()
    sv = sim.run(tqc).result().data()["statevector"].data
    return np.abs(sv) ** 2


def test_neqr_image_recovery_4x4():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (4, 4)).astype(float)
    k = 2
    p = _sv_probs(build_neqr_image_circuit(img, measure=False))
    sig = np.where(p > 1e-9)[0]
    assert len(sig) == 16
    for idx in sig:
        pos = int(idx) & 15
        f = (int(idx) >> 4) & 0xFF
        r, c = pos & 3, pos >> 2
        assert abs(p[idx] - 1 / 16) < 1e-9
        assert f == int(img[r, c]), (r, c, f, img[r, c])
    print("test_neqr_image_recovery_4x4 PASS")


def test_neqr_gradient_matches_analytic_4x4():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (4, 4)).astype(float)
    for direction in ["horizontal", "vertical"]:
        p = _sv_probs(build_neqr_gradient_circuit(img, direction, measure=False))
        sig = np.where(p > 1e-9)[0]
        rr, cc, ff, dd, ccy = neqr_gradient_payload(img, direction)
        assert len(sig) == 16
        for idx in sig:
            pos = int(idx) & 15
            F = (int(idx) >> 4) & 0xFF
            G = (int(idx) >> 12) & 0xFF
            carry = (int(idx) >> 20) & 1
            r, c = pos & 3, pos >> 2
            i = r * 4 + c
            assert abs(p[idx] - 1 / 16) < 1e-9
            assert F == ff[i], (direction, r, c)
            assert G == dd[i], (direction, r, c, G, dd[i])
            assert carry == ccy[i], (direction, r, c, carry, ccy[i])
    print("test_neqr_gradient_matches_analytic_4x4 PASS (both directions)")


def test_ideal_sampler_matches_circuit_statistics():
    """sample_ideal_neqr_gradient's payload == analytic payload (per shot)."""
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (4, 4)).astype(float)
    counts = sample_ideal_neqr_gradient(img, "horizontal", shots=200000, seed=1)
    k = 2
    _, _, ff, dd, ccy = neqr_gradient_payload(img, "horizontal")
    # decode straight from the count keys:
    # v = r | c<<k | F<<2k | G<<(2k+8) | carry<<(2k+16)
    hist_by_pos = {}
    for key, cnt in counts.items():
        v = int(key, 2)
        pos = v & 15
        d = (v >> 12) & 0xFF
        carry = (v >> 20) & 1
        hist_by_pos.setdefault(pos, {})
        hist_by_pos[pos][(d, carry)] = hist_by_pos[pos].get((d, carry), 0) + cnt
    for pos, hist in hist_by_pos.items():
        r, c = pos & 3, pos >> 2
        i = r * 4 + c
        assert len(hist) == 1, hist          # deterministic payload per pixel
        (top, top_carry), top_cnt = next(iter(hist.items()))
        assert top == int(dd[i]), (r, c, top, dd[i])
        assert top_carry == int(ccy[i]), (r, c, top_carry, ccy[i])
        assert top_cnt > 0
    print("test_ideal_sampler_matches_circuit_statistics PASS (200k shots)")


if __name__ == "__main__":
    test_neqr_image_recovery_4x4()
    test_neqr_gradient_matches_analytic_4x4()
    test_ideal_sampler_matches_circuit_statistics()
    print("ALL NEQR TESTS PASSED")
