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

from haiqu_partial_ec import transform_circuit

range_qubits = np.arange(4, 29)
range_depths = np.arange(5, 55, 2)  #np.arange(4, 25)

np.random.seed(42)
p_1q = 1e-2   # depolarizing error for 1-qubit native gates
p_2q = 5e-2   # depolarizing error for 2-qubit native gates
ft_scale = 0.1 # ideal FT gates
test_circuit_type = 'qpe' # 'random', 'qft' or 'qft'
num_trials = 10

noise_model = build_noise_model(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)

ideal_sim = AerSimulator()
noisy_sim = AerSimulator(noise_model=noise_model)

times_qubit = []
times_std_qubit=[]
times_depth = []
times_std_depth=[]

for n_qubit in range_qubits:
    benchmarking = TestCircuits(p_1q=p_1q, p_2q=p_2q, ft_scale=ft_scale)
    if test_circuit_type == 'random':
        depth = 10
        benchmarking_circuits = benchmarking.get_random_circuits(num_trials, n_qubits=n_qubit, depth=depth)
    elif test_circuit_type == 'qft':
        benchmarking_circuits = benchmarking.get_qft_circuits(num_trials, n_qubits=n_qubit)
    elif test_circuit_type == 'qpe':
        benchmarking_circuits = benchmarking.get_qpe_circuits(num_trials, n_qubits=n_qubit)

    times_intermediate=[]

    for circuit in benchmarking_circuits:
        start_time = time.perf_counter()
        transform_circuit(circuit)
        end_time = time.perf_counter()
        elapsed = end_time-start_time
        times_intermediate.append(elapsed)

    avg_time = np.mean(times_intermediate)
    std_time = np.std(times_intermediate)
    times_qubit.append(avg_time)
    times_std_qubit.append(std_time)

if test_circuit_type == 'random':
    for depth in range_depths:
        n_qubit = 5
        benchmarking_circuits = benchmarking.get_random_circuits(num_trials, n_qubits=n_qubit, depth=depth)

        times_intermediate=[]

        for circuit in benchmarking_circuits:
            start_time = time.perf_counter()
            transform_circuit(circuit)
            end_time = time.perf_counter()
            elapsed = end_time-start_time

        avg_time = np.mean(times_intermediate)
        std_time= np.std(times_intermediate)
        times_depth.append(avg_time)
        times_std_depth.append(std_time)


#Plot: Time vs Number of Qubits (fixed depth=6)
plt.figure()
plt.errorbar(range_qubits, times_qubit, xerr=times_std_qubit, marker='o', capsize=4)
plt.xlabel('Number of Qubits')
plt.ylabel('Average Time per Circuit (s)')
if test_circuit_type == 'qft' or 'qpe':
    plt.title(f'Scalability: Time vs Qubits for {test_circuit_type.upper()}')
elif test_circuit_type == 'random':
    plt.title(f'Scalability: Time vs Qubits for random circuits of depth 10')
plt.grid(True)
if test_circuit_type == 'qft' or 'qpe':
    plt.savefig(f'scalability_time_vs_qubits_{test_circuit_type}.png')
    plt.savefig(f'scalability_time_vs_qubits_{test_circuit_type}.svg')
    
elif test_circuit_type == 'random':
    plt.savefig(f'scalability_time_vs_qubits_random_depth10.png')
    plt.savefig(f'scalability_time_vs_qubits_random_depth10.svg')

if test_circuit_type == 'random':
    # Plot: Time vs Circuit Depth (fixed qubits=5)
    plt.figure()
    plt.errorbar(range_depths, times_depth, yerr=times_std_depth, marker='o')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Average Time per Circuit (s)')
    plt.title('Scalability: Time vs Depth (Qubits=5)')
    plt.grid(True)
    plt.savefig('scalability_time_vs_depth_qubits5.png')
    plt.savefig('scalability_time_vs_depth_qubits5.svg')