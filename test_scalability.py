import haiqu_partial_ec
import numpy as np
import time
import matplotlib.pyplot as plt

from haiqu_utils import build_noise_model
from haiqu_utils import grader

from circuits import TestCircuits

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit

range_qubits=[3, 6, 10, 15, 25]
range_depths=[2, 5, 8, 13, 18]

if __name__ == '__main__':

    np.random.seed(42)
    p_1q = 1e-2   # depolarizing error for 1-qubit native gates
    p_2q = 5e-2   # depolarizing error for 2-qubit native gates
    ft_scale = 0.1 # ideal FT gates
    test_circuit_type = 'random' # 'random', 'qft' or 'qft'
    num_trials=5

    noise_model = build_noise_model(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)

    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)

    times_qubit=[]
    times_depth=[]

    for n_qubit in range_qubits:
        depth=6
        benchmarking = TestCircuits(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)
        benchmarking_circuits = benchmarking.get_random_circuits(num_trials, n_qubits=n_qubit, depth=depth)

        start_time=time.time()

        grade = grader(
            transform_circuit_fn=transform_circuit,
            circuits=benchmarking_circuits,
            noise_model=noise_model,
            shots=200_000,   # reduce for demo speed
        )
        end_time=time.time()
        elapsed=end_time-start_time
        avg_time_per_circuit=elapsed/num_trials
        times_qubit.append(avg_time_per_circuit)
    
    for depth in range_depths:
        n_qubit=5
        benchmarking_circuits = benchmarking.get_random_circuits(num_trials, n_qubits=n_qubit, depth=depth)

        start_time=time.time()

        grade = grader(
            transform_circuit_fn=transform_circuit,
            circuits=benchmarking_circuits,
            noise_model=noise_model,
            shots=200_000,   # reduce for demo speed
        )
        end_time=time.time()
        elapsed=end_time-start_time
        avg_time_per_circuit=elapsed/num_trials
        times_depth.append(avg_time_per_circuit)


#Plot: Time vs Number of Qubits (fixed depth=6)
plt.figure()
plt.plot(range_qubits, times_qubit, marker='o')
plt.xlabel('Number of Qubits')
plt.ylabel('Average Time per Circuit (s)')
plt.title('Scalability: Time vs Qubits (Depth=6)')
plt.grid(True)
plt.savefig('scalability_time_vs_qubits_depth6.png')
plt.savefig('scalability_time_vs_qubits_depth6.svg')

# Plot: Time vs Circuit Depth (fixed qubits=5)
plt.figure()
plt.plot(range_depths, times_depth, marker='o')
plt.xlabel('Circuit Depth')
plt.ylabel('Average Time per Circuit (s)')
plt.title('Scalability: Time vs Depth (Qubits=5)')
plt.grid(True)
plt.savefig('scalability_time_vs_depth_qubits5.png')
plt.savefig('scalability_time_vs_depth_qubits5.svg')