"""Correctness tests for quantum/gates.py primitives.

Runs with pytest or standalone:  python tests/test_gates.py
"""

import os
import random
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum.gates import append_subtractor, decrement, increment  # noqa: E402


def _basis_out(qc: QuantumCircuit, n: int) -> int:
    """Integer value of the (computational-basis) output state."""
    probs = Statevector(qc).probabilities_dict()
    key = [k for k, v in probs.items() if v > 1 - 1e-9][0]
    return int(key, 2)


def test_increment_exhaustive():
    for m in [1, 2, 3, 4]:
        for y in range(2 ** m):
            qc = QuantumCircuit(m)
            for i in range(m):
                if (y >> i) & 1:
                    qc.x(i)
            increment(qc, list(range(m)))
            assert _basis_out(qc, m) == (y + 1) % (2 ** m), (m, y)
    print("test_increment_exhaustive PASS")


def test_decrement_exhaustive():
    for m in [1, 2, 3, 4]:
        for y in range(2 ** m):
            qc = QuantumCircuit(m)
            for i in range(m):
                if (y >> i) & 1:
                    qc.x(i)
            decrement(qc, list(range(m)))
            assert _basis_out(qc, m) == (y - 1) % (2 ** m), (m, y)
    print("test_decrement_exhaustive PASS")


def test_subtractor_random():
    rng = random.Random(42)
    for _ in range(300):
        w = rng.choice([2, 3, 4, 8])
        F, G = rng.randint(0, 2 ** w - 1), rng.randint(0, 2 ** w - 1)
        qc = QuantumCircuit(2 * w + 2)
        for i in range(w):
            if (F >> i) & 1:
                qc.x(i)
            if (G >> i) & 1:
                qc.x(w + i)
        append_subtractor(qc, list(range(w)), list(range(w, 2 * w)), [2 * w, 2 * w + 1])
        probs = Statevector(qc).probabilities_dict()
        key = [k for k, v in probs.items() if v > 1 - 1e-9][0]
        bits = key[::-1]  # bits[i] = qubit i
        Gout = sum(1 << (i - w) for i in range(w, 2 * w) if bits[i] == "1")
        Fout = sum(1 << i for i in range(w) if bits[i] == "1")
        # ancilla q[2w+1] (work) must be clean; q[2w] is the carry-out
        # indicator (F+G overflow) which is legitimately data-dependent.
        assert bits[2 * w + 1] == "0", "work ancilla not restored"
        assert Gout == (F - G) % (2 ** w), (w, F, G, Gout)
        assert Fout == F, (w, F, Fout)
    print("test_subtractor_random PASS (300 cases, widths 2/3/4/8)")


if __name__ == "__main__":
    test_increment_exhaustive()
    test_decrement_exhaustive()
    test_subtractor_random()
    print("ALL GATE TESTS PASSED")
