"""Correctness tests for the QHED pipeline (pytest or standalone)."""

import os
import sys

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum.qhed_circuit import (build_qhed_circuit,          # noqa: E402
                                  qhed_analytic_edge_probability)
from quantum.qpie_encoder import (amplitudes_to_image,         # noqa: E402
                                  build_qpie_circuit, image_to_amplitudes)


def test_qpie_roundtrip():
    rng = np.random.default_rng(1)
    img = rng.random((8, 8))
    amps, norm = image_to_amplitudes(img)
    back = amplitudes_to_image(amps)
    # amplitudes are img / L2-norm, so back * norm == img
    assert np.allclose(back * norm, img), "roundtrip"
    # circuit prepares the same state
    qc = build_qpie_circuit(img, add_ancilla=False)
    sv = Statevector(qc).data[: 2 ** 6]  # ancilla-free: 6 position qubits
    assert np.allclose(np.abs(sv - amps), 0, atol=1e-9)
    print("test_qpie_roundtrip PASS")


def test_qpie_zero_image():
    qc = build_qpie_circuit(np.zeros((4, 4)))
    assert abs(Statevector(qc).probabilities_dict()["0" * 5] - 1.0) < 1e-12
    print("test_qpie_zero_image PASS")


def test_qhed_matches_analytic_state():
    """Circuit output distribution == analytic P(a=1) formula (Yao 2017)."""
    rng = np.random.default_rng(7)
    for size in [2, 4, 8]:
        for _ in range(3):
            img = rng.random((size, size))
            for direction in ["horizontal", "vertical"]:
                qc = build_qhed_circuit(img, direction, measure=False)
                probs = Statevector(qc).probabilities_dict()
                k = int(np.log2(size))
                amps, _ = image_to_amplitudes(img)
                a_img = amps.reshape(size, size).T
                shift = (np.roll(a_img, -1, axis=1) if direction == "horizontal"
                         else np.roll(a_img, -1, axis=0))
                pmap = ((a_img - shift) ** 2) / 4
                smap = ((a_img + shift) ** 2) / 4
                err = 0.0
                for key, p in probs.items():
                    v = int(key, 2)
                    a = v & 1
                    pos = v >> 1
                    r, c = pos % size, pos // size
                    expected = pmap[r, c] if a == 1 else smap[r, c]
                    err = max(err, abs(p - expected))
                assert err < 1e-9, (size, direction, err)
    print("test_qhed_matches_analytic_state PASS (sizes 2/4/8, both directions)")


def test_qhed_analytic_zero_on_flat_image():
    flat = np.full((4, 4), 0.5)
    assert np.allclose(qhed_analytic_edge_probability(flat), 0.0)
    print("test_qhed_analytic_zero_on_flat_image PASS")


if __name__ == "__main__":
    test_qpie_roundtrip()
    test_qpie_zero_image()
    test_qhed_matches_analytic_state()
    test_qhed_analytic_zero_on_flat_image()
    print("ALL QHED TESTS PASSED")
