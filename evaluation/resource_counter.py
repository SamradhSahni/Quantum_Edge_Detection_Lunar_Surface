"""
Quantum resource accounting: qubits, gates, depth, and timing breakdown.

Follows PROJECT_SUMMARY.md Experiment 9: report logical vs ancilla
qubits, raw vs transpiled gate counts, 1q vs 2q gates, pre/post
transpilation depth, state-preparation vs edge-operation gate counts,
peak statevector memory, kernel time, and end-to-end time -- kernel and
E2E ALWAYS separately.
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator


def circuit_resources(circuit) -> dict:
    """Raw (untranspiled) resource record of a circuit."""
    if circuit is None:
        return {}
    ops = dict(circuit.count_ops())
    gate_total = int(sum(ops.values()))
    two_q = int(sum(v for k, v in ops.items() if _num_qubits(k) == 2))
    return {
        "logical_qubits": circuit.num_qubits,
        "raw_gates": gate_total,
        "raw_1q_gates": gate_total - two_q - int(ops.get("measure", 0)),
        "raw_2q_gates": two_q,
        "raw_depth": circuit.depth(),
        "raw_ops": ops,
        "statevector_bytes": float(16 * (2 ** circuit.num_qubits)),
    }


def transpiled_resources(circuit, optimization_level: int = 1) -> dict:
    """Resource record after transpilation to the Aer basis."""
    if circuit is None:
        return {}
    sim = AerSimulator()
    tq = transpile(circuit, sim, optimization_level=optimization_level)
    ops = dict(tq.count_ops())
    gate_total = int(sum(v for k, v in ops.items() if k != "measure"))
    two_q = int(sum(v for k, v in ops.items() if _num_qubits(k) == 2))
    return {
        "transpiled_gates": gate_total,
        "transpiled_1q_gates": gate_total - two_q,
        "transpiled_2q_gates": two_q,
        "transpiled_depth": tq.depth(),
        "transpiled_ops": ops,
    }


def _num_qubits(gate_name: str) -> int:
    one_q = {"u", "u1", "u2", "u3", "h", "x", "y", "z", "s", "sdg", "t", "tdg",
             "rz", "sx", "p", "id", "ry", "rx", "reset"}
    if gate_name in one_q:
        return 1
    if gate_name in {"cx", "cz", "swap", "ecr", "ch", "crx", "cry", "crz", "cu1", "csx"}:
        return 2
    return 0  # measure/barrier and anything unusual


def split_gate_families(circuit) -> dict:
    """State-preparation vs edge-operation gate counts.

    For QHED: state prep = everything before the first H on the ancilla.
    For NEQR-gradient: state prep = NEQR position/intensity MCX networks;
    edge op = the CDKM subtractor.
    """
    if circuit is None:
        return {}
    n_instr = len(circuit.data)
    first_h = next((i for i, inst in enumerate(circuit.data)
                    if inst.operation.name == "h"), n_instr)
    # crude but honest split: gates before the edge-operation phase
    # (the ancilla H for QHED; for NEQR we pass the split index explicitly)
    return {"pre_edge_ops": first_h, "total_ops": n_instr}
