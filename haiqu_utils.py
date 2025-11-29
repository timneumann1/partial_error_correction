"""
haiqu_utils.py

Helper utilities for the Haiqu / PushQuantum 2025 challenge.

This module provides:

- ALLOWED_BASE_GATES: the native IBM-like gate set we operate on
- Explicit FT gate classes (XFTGate, SXFTGate, RZFTGate, RXFTGate, RZZFTGate, CZFTGate)
- Naming helpers for FT gates
- to_ft_instruction: convert a base gate Instruction to its FT placeholder
- reference_transform: a simple baseline circuit transformation pass

Participants typically only need to import:
    from haiqu_utils import ALLOWED_BASE_GATES, to_ft_instruction, reference_transform
"""

from __future__ import annotations

from typing import Callable, List, Set

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Instruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector, state_fidelity, hellinger_fidelity

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# --------------------------------------------------------------------
# Gate set and naming conventions
# --------------------------------------------------------------------

# Native IBM-like gate set used in this challenge
ALLOWED_BASE_GATES: Set[str] = {"x", "sx", "rz", "rx", "rzz", "cz"}

# Suffix used for "fault-tolerant" placeholder gates
FT_SUFFIX = "_ft"

# Mapping: base gate name -> FT placeholder name
FT_GATE_MAP = {g: f"{g}{FT_SUFFIX}" for g in ALLOWED_BASE_GATES}

# Inverse mapping: FT placeholder name -> base gate name
BASE_GATE_FROM_FT = {v: k for k, v in FT_GATE_MAP.items()}


def make_ft_name(base_name: str) -> str:
    """
    Given a base gate name like 'rz', return the FT placeholder name 'rz_ft'.
    """
    if base_name not in ALLOWED_BASE_GATES:
        raise ValueError(f"Gate '{base_name}' is not in the allowed base gate set.")
    return FT_GATE_MAP[base_name]


def is_ft_name(name: str) -> bool:
    """Return True if 'name' is one of the FT placeholder gate names, e.g. 'rz_ft'."""
    return name in BASE_GATE_FROM_FT


def base_name_from_ft(name: str) -> str:
    """
    Map FT placeholder name back to its base gate name.

    Examples:
        'rz_ft'  -> 'rz'
        'cz_ft'  -> 'cz'
        'rz'     -> 'rz'   (identity for non-FT names)
    """
    return BASE_GATE_FROM_FT.get(name, name)


def is_ft_gate(inst: Instruction) -> bool:
    """Return True if an Instruction is a fault-tolerant placeholder gate."""
    return is_ft_name(inst.name)


# --------------------------------------------------------------------
# Creating FT placeholder instructions
# --------------------------------------------------------------------

def to_ft_instruction(inst: Instruction) -> Instruction:
    """
    Convert a native gate Instruction (x, sx, rz, rx, rzz, cz) into an FT version
    by wrapping its exact unitary matrix in a UnitaryGate with a custom *_ft name.

    This creates a *custom unitary* gate that:
        - has the exact same unitary action as the base gate,
        - keeps the same qubit and classical bit structure,
        - has a distinct name "<base>_ft" so that the noise model can assign
          different error channels to it.

    This is fully compatible with Aer custom-gate noise injection.
    """
    ft_name = inst.name + "_ft"

    # Extract exact unitary matrix of the original gate
    op = Operator(inst)        # safe for all gates (1q and 2q)

    # Build a UnitaryGate with FT name
    ft_gate = UnitaryGate(op, label=ft_name)

    return ft_gate


# --------------------------------------------------------------------
# Noise model builder
# --------------------------------------------------------------------

def build_noise_model(
    p_1q: float = 1e-3,
    p_2q: float = 5e-3,
    ft_scale: float = 0.0,
) -> NoiseModel:
    """
    Build a Qiskit Aer NoiseModel that:

      * adds depolarizing noise to the native IBM-like gates
            {x, sx, rz, rx, rzz, cz}
      * adds (optionally reduced) depolarizing noise to the corresponding
        fault-tolerant gates, which are represented as custom *unitary*
        instructions with labels "<gate>_ft" (e.g. "rx_ft", "cz_ft").

    Assumptions:
    - Native gates appear with their usual names: "x", "sx", "rz", "rx", "rzz", "cz".
    - FT gates are inserted as unitary instructions with *labels*
      "x_ft", "sx_ft", "rz_ft", "rx_ft", "rzz_ft", "cz_ft".
      (The underlying gate type will typically be "unitary".)

    Parameters
    ----------
    p_1q : float
        Depolarizing error probability for 1-qubit native gates.

    p_2q : float
        Depolarizing error probability for 2-qubit native gates.

    ft_scale : float
        Scaling factor for FT-gate noise:
            0.0 -> FT gates are ideal (no error added),
            0.1 -> FT gates 10x less noisy than native,
            1.0 -> FT gates equally noisy as native (for debugging).

    Returns
    -------
    NoiseModel
        Configured noise model ready to use with AerSimulator.
    """
    one_qubit_gates = {"x", "sx", "rz", "rx"}
    two_qubit_gates = {"rzz", "cz"}

    noise_model = NoiseModel(basis_gates=list(ALLOWED_BASE_GATES) + ["unitary"])

    # --- Native-gate depolarizing noise ---
    if p_1q > 0:
        err_1q = depolarizing_error(p_1q, 1)
        for g in one_qubit_gates:
            noise_model.add_all_qubit_quantum_error(err_1q, g)

    if p_2q > 0:
        err_2q = depolarizing_error(p_2q, 2)
        for g in two_qubit_gates:
            noise_model.add_all_qubit_quantum_error(err_2q, g)

    # --- FT-gate depolarizing noise via labels ---
    # FT gates are represented as unitaries with labels like "rx_ft", "cz_ft".
    if ft_scale > 0.0:
        if p_1q > 0:
            err_1q_ft = depolarizing_error(p_1q * ft_scale, 1)
            for g in one_qubit_gates:
                ft_label = f"{g}_ft"
                noise_model.add_all_qubit_quantum_error(err_1q_ft, ft_label)

        if p_2q > 0:
            err_2q_ft = depolarizing_error(p_2q * ft_scale, 2)
            for g in two_qubit_gates:
                ft_label = f"{g}_ft"
                noise_model.add_all_qubit_quantum_error(err_2q_ft, ft_label)
    # else: ft_scale == 0 -> FT gates are effectively noiseless (no error added)

    # --- Important: tell the noise model that "unitary" is in the basis ---
    # This mirrors the tutorial's step of adding 'unitary' as a basis gate so
    # that circuits containing custom unitary instructions are not decomposed
    # away before noise is applied.
    noise_model.add_basis_gates(["unitary"])

    return noise_model

# --------------------------------------------------------------------
# Verifier: check transformed circuits obey the challenge rules
# --------------------------------------------------------------------


def is_ft_instruction(inst: Instruction) -> bool:
    """
    Return True if this Instruction is an FT-tagged gate.

    In our design, FT gates are custom unitaries whose *label*
    ends with '_ft', e.g. 'rx_ft', 'cz_ft'.
    """
    return getattr(inst, "label", None) is not None and inst.label.endswith("_ft")


def logical_gate_name(inst: Instruction) -> str:
    """
    Return the 'logical' gate name for comparison purposes:

      - For base gates, this is just inst.name (e.g. 'rx', 'cz').
      - For FT unitaries, we derive it from the label, stripping '_ft':
            label='rx_ft' -> 'rx'

    This lets us compare original vs transformed circuits up to FT tagging.
    """
    if is_ft_instruction(inst):
        label = inst.label
        base = label[:-3]  # strip '_ft'
        return base
    else:
        return inst.name


def verify_transformation(circ: QuantumCircuit, circ_ft: QuantumCircuit) -> dict:
    """
    Verify that:
      1) The circuit structure is identical (instructions not moved,
         inserted, or deleted; only replaced by FT versions).
      2) The ideal logical state matches (fidelity > 0.999).
      3) At most one FT gate per DAG layer.

    Returns:
      { "ok": bool, "errors": [...] }
    """

    errors = []

    # ------------------------------------------------------------
    # 1. Structure check: same number of instructions
    # ------------------------------------------------------------
    if len(circ.data) != len(circ_ft.data):
        errors.append(
            f"Instruction count changed: {len(circ.data)} → {len(circ_ft.data)}."
        )
        return {"ok": False, "errors": errors}

    # ------------------------------------------------------------
    # 1b. Gate-by-gate position check
    # ------------------------------------------------------------
    for idx, ((inst_o, q_o, c_o), (inst_t, q_t, c_t)) in enumerate(
        zip(circ.data, circ_ft.data)
    ):

        # qargs must match exactly
        if q_o != q_t:
            errors.append(
                f"Instruction {idx}: qubit operands changed {q_o} → {q_t}."
            )

        # cargs must match exactly
        if c_o != c_t:
            errors.append(
                f"Instruction {idx}: classical operands changed {c_o} → {c_t}."
            )

        # Operation type must be identical *unless* FT-tagged
        base_name = inst_o.name
        if is_ft_instruction(inst_t):   # FT replacement allowed
            # FT label must correspond to same base gate
            if not inst_t.label.startswith(base_name + "_ft"):
                errors.append(
                    f"Instruction {idx}: FT gate label '{inst_t.label}' does not "
                    f"match original gate '{base_name}'."
                )
        else:
            # For non-FT: must be exactly the same op
            if inst_t.name != inst_o.name:
                errors.append(
                    f"Instruction {idx}: op changed from '{inst_o.name}' "
                    f"to '{inst_t.name}'."
                )

    # ------------------------------------------------------------
    # 2. State fidelity check (logical correctness)
    # ------------------------------------------------------------
    circ_clean = circ.remove_final_measurements(inplace=False)
    circ_ft_clean = circ_ft.remove_final_measurements(inplace=False)

    sv_orig = Statevector(circ_clean)
    sv_ft = Statevector(circ_ft_clean)

    F = state_fidelity(sv_orig, sv_ft)
    if F < 0.999:
        errors.append(
            f"Logical output mismatch: state fidelity = {F:.6f} (< 0.999)."
        )

    # ------------------------------------------------------------
    # 3. FT-per-layer check
    # ------------------------------------------------------------
    dag = circuit_to_dag(circ_ft)
    for layer_idx, layer in enumerate(dag.layers()):
        layer_dag = layer["graph"]

        ft_in_layer = [
            node for node in layer_dag.op_nodes()
            if is_ft_instruction(node.op)
        ]

        if len(ft_in_layer) > 1:
            errors.append(
                f"Layer {layer_idx} contains {len(ft_in_layer)} FT gates "
                "(maximum allowed is 1)."
            )

    return {"ok": len(errors) == 0, "errors": errors}

# --------------------------------------------------------------------
# Grader: simulate and compute Hellinger fidelities
# --------------------------------------------------------------------


def grader(
    transform_circuit_fn: Callable[[QuantumCircuit], QuantumCircuit],
    circuits: List[QuantumCircuit],
    noise_model: NoiseModel,
    shots: int = 1_000_000,
    verify: bool = True,
) -> dict:
    """
    Grade a submission by comparing *improvement in fidelity*:

        improvement = F_ft - F_plain

    where F_plain is the fidelity between the ideal distribution and the noisy
    distribution of the unmodified circuit, and F_ft is the fidelity between
    the ideal distribution and the noisy distribution of the transformed circuit.

    Steps:
      1) Compute ideal counts from original circuit.
      2) Simulate original under noise → noisy_original_counts.
      3) Apply transform and verify correctness.
      4) Simulate transformed under noise → noisy_ft_counts.
      5) Compute fidelity improvement per circuit.
      6) Average over all circuits.

    Returns:
      {
        "ok": bool,
        "errors": [...],
        "average_improvement": float,
        "improvements": [...],
        "fidelities_plain": [...],
        "fidelities_ft": [...],
      }
    """

    # Ideal simulator (no noise)
    ideal_sim = AerSimulator()

    # Noisy simulator (with provided noise model)
    noisy_sim = AerSimulator(noise_model=noise_model)

    improvements = []
    fidelities_plain = []
    fidelities_ft = []
    all_errors = []

    for idx, circ in enumerate(circuits):

        # make sure the circuit has measurements
        if not circ.cregs:
            circ.measure_all()

        # --- 1. Compute ideal distribution from original circuit ---
        try:
            ideal_counts = ideal_sim.run(circ.copy(), shots=shots).result().get_counts()
        except Exception as exc:
            all_errors.append(f"Circuit {idx}: ideal simulation failed: {exc}")
            continue

        # --- 2. Noisy simulation of original circuit ---
        try:
            noisy_orig_t = transpile(
                circ.copy(),
                backend=noisy_sim,
                optimization_level=0,
            )
            noisy_orig_counts = noisy_sim.run(noisy_orig_t, shots=shots).result().get_counts()
        except Exception as exc:
            all_errors.append(f"Circuit {idx}: noisy original simulation failed: {exc}")
            continue

        F_plain = hellinger_fidelity(ideal_counts, noisy_orig_counts)
        fidelities_plain.append(F_plain)

        # --- 3. Apply participant transformation ---
        try:
            circ_ft = transform_circuit_fn(circ.copy())
        except Exception as exc:
            all_errors.append(f"Circuit {idx}: transform_circuit threw an exception: {exc}")
            continue

        # --- 4. Verify correctness of FT circuit ---
        if verify:
            report = verify_transformation(circ, circ_ft)
            if not report["ok"]:
                for err in report["errors"]:
                    all_errors.append(f"Circuit {idx}: {err}")
                # We still simulate for debugging purposes but mark submission invalid.

        # --- 5. Noisy simulation of FT circuit ---
        try:
            noisy_ft_t = transpile(
                circ_ft.copy(),
                backend=noisy_sim,
                optimization_level=0,
            )
            noisy_ft_counts = noisy_sim.run(noisy_ft_t, shots=shots).result().get_counts()
        except Exception as exc:
            all_errors.append(f"Circuit {idx}: noisy FT simulation failed: {exc}")
            continue

        F_ft = hellinger_fidelity(ideal_counts, noisy_ft_counts)
        fidelities_ft.append(F_ft)

        # --- 6. Improvement ---
        improvement = F_ft - F_plain
        improvements.append(improvement)

    avg_improvement = (
        sum(improvements) / len(improvements) if improvements else 0.0
    )

    return {
        "ok": len(all_errors) == 0,
        "errors": all_errors,
        "average_improvement": avg_improvement,
        "improvements": improvements,
        "fidelities_plain": fidelities_plain,
        "fidelities_ft": fidelities_ft,
    }
