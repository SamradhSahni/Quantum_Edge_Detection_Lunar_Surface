"""
Shared low-level quantum primitives for both quantum pipelines.

Contents
--------
increment / decrement : exact, ancilla-free (MCX-based) cyclic +1 / -1
                       operators on an n-qubit register.  The decrement is
                       the workhorse of the canonical QHED algorithm
                       (Yao et al., PRX 7, 031041, 2017): applying an
                       (n+1)-qubit decrement to [ancilla, position...]
                       with the ancilla as LEAST-significant bit produces
                       exactly the interference needed for edge detection
                       (see qhed_circuit.py for the derivation).

append_subtractor     : G <- F - G (mod 2^w) via two's-complement addition
                       using the library CDKM ripple-carry adder
                       (Cuccaro 2004 style MAJ/UMA network, 1 carry
                       ancilla).  Used by the NEQR quantum-gradient
                       operator.

All primitives are validated against exhaustive classical references in
tests/test_gates.py and tests/test_neqr.py.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import CDKMRippleCarryAdder

try:  # Qiskit 1.x/2.x safe multi-controlled X
    from qiskit.circuit.library import MCXGate
    _HAS_MCXGATE = True
except ImportError:  # pragma: no cover - very old/new qiskit fallback
    _HAS_MCXGATE = False


def mcx(qc: QuantumCircuit, controls, target) -> None:
    """Version-tolerant multi-controlled X."""
    controls = list(controls)
    if not controls:
        qc.x(target)
        return
    if len(controls) == 1:
        qc.cx(controls[0], target)
        return
    try:
        qc.mcx(controls, target)
    except (AttributeError, TypeError):
        qc.append(MCXGate(len(controls)), controls + [target])


def increment(qc: QuantumCircuit, qubits) -> None:
    """In-place cyclic +1 on an n-qubit register (wraps mod 2^n).

    Bit i flips iff all LESS significant bits (original values) are 1,
    so gates are emitted from the most significant bit down and every
    control still holds its original value when used.
    """
    qubits = list(qubits)
    for i in range(len(qubits) - 1, 0, -1):
        mcx(qc, qubits[:i], qubits[i])
    qc.x(qubits[0])


def decrement(qc: QuantumCircuit, qubits) -> None:
    """In-place cyclic -1 on an n-qubit register (wraps mod 2^n).

    dec(y) = comp(inc(comp(y))) since y - 1 == ~(~y + 1) in two's
    complement arithmetic modulo 2^n.
    """
    qubits = list(qubits)
    for q in qubits:
        qc.x(q)
    increment(qc, qubits)
    for q in qubits:
        qc.x(q)


def append_subtractor(qc: QuantumCircuit, f_qubits, g_qubits, ancilla_qubits) -> None:
    """Quantum subtractor:  G <- (F - G) mod 2^w  (F unchanged).

    Uses the identity   F - G  ==  ~(~F + G)   (mod 2^w):

        1. X on all F wires                -> F holds ~F
        2. CDKM ripple-carry adder         -> G holds ~F + G
        3. X on all G wires                -> G holds ~(~F + G) == F - G
        4. X on all F wires                -> restore F

    ``ancilla_qubits`` must provide the two clean ancillas required by the
    CDKM ripple-carry adder (carry-out + work qubit); both are returned to
    |0> by the adder itself.  All gates are exact boolean networks
    (CNOT/Toffoli) -- no rotation angles are involved, so basis-state
    readout is exact even under finite simulator precision.
    """
    f_qubits = list(f_qubits)
    g_qubits = list(g_qubits)
    w = len(f_qubits)
    if len(g_qubits) != w:
        raise ValueError("F and G registers must have equal width")
    if len(ancilla_qubits) != 2:
        raise ValueError("CDKM subtractor needs exactly 2 clean ancilla qubits")

    for q in f_qubits:
        qc.x(q)

    adder = CDKMRippleCarryAdder(w, kind="half")
    # Qiskit wire order for CDKMRippleCarryAdder: [A(w), B(w), carry(1), work(1)]
    qc.append(adder, f_qubits + g_qubits + list(ancilla_qubits))

    for q in f_qubits:  # restore F operand
        qc.x(q)
    for q in g_qubits:  # complement the sum -> difference
        qc.x(q)
