import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile

from haiqu_utils import (
        ALLOWED_BASE_GATES,
        to_ft_instruction,
    )
from haiqu_utils import build_noise_model
from haiqu_utils import grader

from circuits import TestCircuits

def transform_circuit(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Designed transformation pass that:

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

    """
    dag = circuit_to_dag(circ)
    new_dag = dag.copy_empty_like()   

    layers = list(dag.layers())
    num_qubits = dag.num_qubits()
    scores = np.zeros(num_qubits)
    gates = []

    # For each layer, select the gate with the maximum influence in future computations
    for i, layer in enumerate(reversed(layers)): # follow reversed order
        layer_dag = layer['graph']
        best_node_index = -1
        best_score = -1
        current_scores = scores.copy()
        for j, node in enumerate(layer_dag.op_nodes()):
            if getattr(node.op, "name") in ALLOWED_BASE_GATES:
                score_node = len(node.qargs)
                for qubit in node.qargs:
                    score_node += scores[qubit._index]
                for qubit in node.qargs:
                    scores[qubit._index] = score_node

                if score_node > best_score:
                    best_score = score_node
                    best_node_index = j       
        gates.append(best_node_index)
        
        scores -= 2/3*current_scores

    gates.reverse()

    # Make the estimated most influencial gates fault-tolerant
    for j, layer in enumerate(dag.layers()):
        layer_dag = layer["graph"]

        if gates[j]!=-1:            
            for i, node in enumerate(layer_dag.op_nodes()):
                if getattr(node.op, "name") in ALLOWED_BASE_GATES and i==gates[j]:
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


def baseline_transform(circ: QuantumCircuit) -> QuantumCircuit:
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

    ################################################
    # 0. Define simulation parameters              #
    ################################################

    np.random.seed(42)
    p_1q = 1e-2   # depolarizing error for 1-qubit native gates
    p_2q = 5e-2   # depolarizing error for 2-qubit native gates
    ft_scale = 0.1 # ideal FT gates
    test_circuit_type = 'qpe' # 'random', 'qft' or 'qft'
    n_circuits = 10 # number of test circuits
    n_qubits = None

    ################################################
    # 1. Initialize noise model and simulator      #
    ################################################

    noise_model = build_noise_model(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)

    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)

    ########################################################################
    # 2. Utilize random circuits, QFT and QPE for testing                  #
    ########################################################################

    benchmarking = TestCircuits(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)
    if test_circuit_type == 'qft':
        benchmarking_circuits = benchmarking.get_qft_circuits(n_circuits, n_qubits=n_qubits)
    elif test_circuit_type == 'random':
        benchmarking_circuits = benchmarking.get_random_circuits(n_circuits, n_qubits=n_qubits)
    elif test_circuit_type == 'qpe':
        benchmarking_circuits = benchmarking.get_qpe_circuits(n_circuits, n_qubits=n_qubits)
    
    ########################################################################
    # 2. Transform the first test circuit and visualize the result         #
    ########################################################################

    viz_circuit = benchmarking_circuits[0]
    viz_circuit_ft = transform_circuit(viz_circuit)
    viz_circuit_ft_baseline = baseline_transform((viz_circuit))
    
    if test_circuit_type != 'qpe':
        viz_circuit.measure_all()
    viz_ideal_result = ideal_sim.run(viz_circuit, shots=10000).result().get_counts()
    fig = viz_circuit.draw(output="mpl")
    fig.savefig("plots/circuit.png", dpi=300, bbox_inches="tight")
    
    if test_circuit_type != 'qpe':
        viz_circuit_ft.measure_all()
    viz_ft_result = noisy_sim.run(transpile(viz_circuit_ft, noisy_sim, optimization_level=0), shots=10000).result().get_counts()
    fig2 = viz_circuit_ft.draw(output="mpl")
    fig2.savefig("plots/circuit_ft.png", dpi=300, bbox_inches="tight")
    
    if test_circuit_type != 'qpe':
        viz_circuit_ft_baseline.measure_all()
    viz_ft_baseline_result = noisy_sim.run(transpile(viz_circuit_ft_baseline, noisy_sim, optimization_level=0), shots=10000).result().get_counts()
    fig3 = viz_circuit_ft_baseline.draw(output="mpl")
    fig3.savefig("plots/circuit_ft_baseline.png", dpi=300, bbox_inches="tight")
    
    ############################################################################
    # Compare the measured histogram of the ideal circuit with the partial     #
    # fault-tolerant transformed circuits using the baseline and our algorithm #
    ############################################################################

    def filter_counts(counts, threshold):
        return {k: v for k, v in counts.items() if v >= threshold}
    
    THRESH = 200  # keep only bitstrings with >= 300 counts
    viz_ideal_f = filter_counts(viz_ideal_result, THRESH)
    viz_ft_f = filter_counts(viz_ft_result, THRESH)
    viz_ft_f_base = filter_counts(viz_ft_baseline_result, THRESH)

    fig3 = plot_histogram([viz_ideal_f, viz_ft_f, viz_ft_f_base],
               legend=["Original", "Lookahead", "Transformation Baseline"],
               title="Comparison of Partial QEC with Ideal Simulation Results", bar_labels = False)
    
    fig3.savefig("plots/histogram.png", dpi=300)
      
    ############################################################################
    # Evaluate our partial fault-tolerant circuit transformation with the      #
    # provided grading function and our benchmarking circuits                  #
    ############################################################################

    def print_results(grade, verbose=True):
        print("Valid submission?      :", grade["ok"])
        print("Average improvement    :", grade["average_improvement"])

        if verbose:
            print("Per-circuit improvement:", grade["improvements"])
            print("Plain fidelities       :", grade["fidelities_plain"])
            print("FT fidelities          :", grade["fidelities_ft"])

        if not grade["ok"]:
            print("\nErrors:")
            for err in grade["errors"]:
                print("  -", err)

    # Grade the baseline circuit transformation
    print("\nRunning grader for baseline circuit transformation...")
    grade = grader(
        transform_circuit_fn=baseline_transform,
        circuits=benchmarking_circuits,
        noise_model=noise_model,
        shots=200_000,   # reduce for demo speed
    )
    print_results(grade, verbose=True)
      
    # Grade our custom circuit transformation  
    print("\nRunning grader for our custom circuit transform...")
    grade = grader(
        transform_circuit_fn=transform_circuit,
        circuits=benchmarking_circuits,
        noise_model=noise_model,
        shots=200_000,   # reduce for demo speed
    )
    print_results(grade, verbose=True)

    print("")
    