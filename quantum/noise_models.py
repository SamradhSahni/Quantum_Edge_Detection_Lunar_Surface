"""
Quantum hardware noise models for AerSimulator (Experiment 5).

Families modelled, following PROJECT_SUMMARY.md Section 5.6:

  * depolarizing   : p applied to every 1q and 2q gate
  * readout        : symmetric bit-flip on every measured qubit
  * gate           : coherent over-rotation on single-qubit gates
                     (theta -> theta * (1 + eps)), a fully coherent
                     unitary channel -- hardware-honest after
                     transpilation to the {rz, sx, x, cx} basis
  * combined       : all of the above simultaneously

Noisy runs ALWAYS execute the real transpiled circuit on AerSimulator
with the statevector trajectory method (each shot samples the noise
independently), so the noise actually touches every decomposed gate.
"""

import numpy as np
from qiskit.quantum_info import Operator
from qiskit_aer.noise import (NoiseModel, ReadoutError, coherent_unitary_error,
                              depolarizing_error)

ONE_Q_GATES = ["u", "u1", "u2", "u3", "h", "x", "y", "z", "sx", "rz", "s",
               "sdg", "t", "tdg", "ry", "rx", "p"]
TWO_Q_GATES = ["cx", "cz", "ecr", "swap"]
NOISE_BASIS = ["rz", "sx", "x", "cx"]


def _x_overrotation(frac: float):
    """Coherent extra X-axis rotation R_x(frac) as a unitary error."""
    U = np.array([[np.cos(frac / 2), -1j * np.sin(frac / 2)],
                  [-1j * np.sin(frac / 2), np.cos(frac / 2)]], dtype=complex)
    return coherent_unitary_error(U)


def _z_overrotation(frac: float):
    """Coherent extra Z-axis rotation R_z(frac) as a unitary error."""
    U = np.array([[np.exp(-1j * frac / 2), 0],
                  [0, np.exp(1j * frac / 2)]], dtype=complex)
    return coherent_unitary_error(U)


def make_depolarizing(p: float) -> NoiseModel:
    nm = NoiseModel()
    if p <= 0:
        return nm
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ONE_Q_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), TWO_Q_GATES)
    return nm


def make_readout(p: float) -> NoiseModel:
    nm = NoiseModel()
    if p <= 0:
        return nm
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def make_overrotation(frac: float) -> NoiseModel:
    """Coherent over-rotation channel on single-qubit gates.

    Applied to the hardware basis: R_x over-rotation on sx/x (and any
    u-family gate) and R_z over-rotation on rz (the phase gate every
    hardware 1q gate decomposes into).  Deliberately coherent -- NOT a
    Pauli-twirled approximation -- so coherent error amplification is
    visible in the results.
    """
    nm = NoiseModel()
    if frac <= 0:
        return nm
    nm.add_all_qubit_quantum_error(_x_overrotation(frac),
                                   ["u", "u3", "ry", "rx", "sx", "x"])
    nm.add_all_qubit_quantum_error(_z_overrotation(frac),
                                   ["rz", "p", "s", "sdg", "t", "tdg", "u1"])
    return nm


def make_combined(p_depol: float, p_readout: float, overrot_frac: float) -> NoiseModel:
    """Depolarizing + readout + coherent over-rotation simultaneously."""
    nm = make_depolarizing(p_depol)
    if p_readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p_readout, p_readout],
                                                     [p_readout, 1 - p_readout]]))
    if overrot_frac > 0:
        over = make_overrotation(overrot_frac)
        for qubits, errors in over._local_quantum_errors.items():
            for gate, err in errors.items():
                nm.add_all_qubit_quantum_error(err, [gate])
    return nm


def build_noise_model(name: str):
    """Named noise models used across experiments.

    name in:
        ideal
        depol_0.001 | depol_0.005 | depol_0.01 | depol_0.05
        readout_0.01 | readout_0.05 | readout_0.10
        overrot
        combined
    """
    if name == "ideal":
        return None
    if name.startswith("depol_"):
        return make_depolarizing(float(name.split("_")[1]))
    if name.startswith("readout_"):
        return make_readout(float(name.split("_")[1]))
    if name == "overrot":
        from config import OVERROTATION_FRAC
        return make_overrotation(OVERROTATION_FRAC)
    if name == "combined":
        from config import OVERROTATION_FRAC
        return make_combined(0.005, 0.05, OVERROTATION_FRAC)
    raise ValueError(f"unknown noise model {name}")


NOISE_GRID = ["ideal", "depol_0.001", "depol_0.005", "depol_0.01", "depol_0.05",
              "readout_0.01", "readout_0.05", "readout_0.10", "overrot", "combined"]
