import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import plot_histogram
from haiqu_utils import (
        ALLOWED_BASE_GATES,
        to_ft_instruction,
    )
from haiqu_utils import build_noise_model
from haiqu_utils import grader

from circuits import TestCircuits


def transform_circuit(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Baseline transformation pass that:

      - Preserves the circuit structure (same ops, same order, same wires),
      - Selects at most one gate per DAG layer to mark as FT,
      - Replaces only those gates by FT-labelled versions via to_ft_instruction.

    Implementation strategy:
      1. Build DAG from the original circuit.
      2. Build a mapping from node_id -> original DAG node.
      3. Iterate over DAG layers and, per layer, pick at most one candidate node
         whose op.name is in ALLOWED_BASE_GATES.
      4. For each chosen node_id, look up the original node and replace node.op
         with the FT version.
      5. Convert the modified DAG back to a QuantumCircuit.

    This is a simple baseline; participants should implement a better strategy
    in their own transform_circuit, but it must obey the same rules.
    """
    dag = circuit_to_dag(circ)
    new_dag = dag.copy_empty_like()

    # 1) Decide which nodes to mark as FT: at most one per layer
    for layer in dag.layers():
        layer_dag = layer["graph"]
        for node in layer_dag.op_nodes():
            if getattr(node.op, "name") in ALLOWED_BASE_GATES:
                layer_dag.substitute_node(
                    node,
                    to_ft_instruction(node.op)
                )
                break  # Only one per layer
        new_dag.compose(layer_dag, inplace=True)
   
    # 3) Convert back to a QuantumCircuit
    transformed = dag_to_circuit(new_dag)
    transformed.name = circ.name + "_ft"
    return transformed


if __name__ == '__main__':

    np.random.seed(42)
    p_1q = 1e-2   # depolarizing error for 1-qubit native gates
    p_2q = 5e-2   # depolarizing error for 2-qubit native gates
    ft_scale = 0.1 # ideal FT gates

    noise_model = build_noise_model(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)

    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)

    benchmarking = TestCircuits(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)
    benchmarking_circuits = benchmarking.get_qft_circuits(n_circuits=5)
    fig = benchmarking_circuits[0].draw(output="mpl")
    fig.savefig("circuit.png", dpi=300, bbox_inches="tight")

    print("Running grader...")

    grade = grader(
        transform_circuit_fn=transform_circuit,
        circuits=benchmarking_circuits,
        noise_model=noise_model,
        shots=200_000,   # reduce for demo speed
    )

    print("Valid submission?      :", grade["ok"])
    print("Average improvement    :", grade["average_improvement"])
    print("Per-circuit improvement:", grade["improvements"])
    print("Plain fidelities       :", grade["fidelities_plain"])
    print("FT fidelities          :", grade["fidelities_ft"])

    if not grade["ok"]:
        print("\nErrors:")
        for err in grade["errors"]:
            print("  -", err)