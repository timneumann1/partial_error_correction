import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from haiqu_utils import build_noise_model
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, QFT, PhaseEstimation


np.random.seed(42)

class TestCircuits:
    
    def __init__(self, p_1q=1e-2, p_2q=5e-2, ft_scale=0.1):
        self.p_1q = p_1q            # depolarizing error for 1-qubit native gates
        self.p_2q = p_2q            # depolarizing error for 2-qubit native gates
        self.ft_scale = ft_scale    # ideal FT gates

        self.noise_model = build_noise_model(p_1q=self.p_1q, p_2q=self.p_2q, ft_scale=self.ft_scale)

        self.ideal_sim = AerSimulator()
        self.noisy_sim = AerSimulator(noise_model=self.noise_model)    
    
    def get_random_circuits(self, n_circuits):
        random_qubits = np.random.randint(3, 5, size=n_circuits)
        random_depths = np.random.randint(3, 6, size=n_circuits)
        random_circuits = [
        transpile(
            random_circuit(num_qubits=nq, depth=nd, max_operands=2, seed=seed, measure=True),
            self.noisy_sim,
            optimization_level=0,
        )
        for seed, nq, nd in zip(range(n_circuits), random_qubits, random_depths)
        ]
        return random_circuits 
    
    def get_qft_circuits(self, n_circuits):
        
        def qft(nq):
            qc = QuantumCircuit(nq)
            qc.h(range(nq))
            qc.compose(QFT(num_qubits=nq), inplace=True)
            return qc
            
        random_qubits = np.random.randint(7, 11, size=n_circuits)

        qft_circuits = [
        transpile(
            qft(nq),
            self.noisy_sim,
            optimization_level=0,
        )
        for nq in random_qubits
        ]
        
        return qft_circuits
    
    def get_qpe_circuits(self, n_circuits):
            
        def QPE(estimation_wires, target_wires):

            # Helper functions for Quantum Phase Estimation (QPE) circuit
            def U_power_2k(unitary, k):
                """Computes U at a power of 2^k (U^(2^k))"""
                return np.linalg.matrix_power(unitary, 2**k)

            def apply_controlled_powers_of_U(unitary, estimation_wires, target_wires):
                """
                Applies controlled-U^(2^k) gates on the target wires, controlled by estimation wires.
                """
                qc = QuantumCircuit(len(estimation_wires) + len(target_wires))

                t = len(estimation_wires)
                for i in range(t):
                    k = t - 1 - i  # same indexing as your Pennylane code
                    U_k = U_power_2k(unitary, k)

                    # Wrap U_k as a UnitaryGate
                    U_gate = UnitaryGate(U_k, label=f"U^{2**k}")

                    # Add controlled version
                    controlled_gate = U_gate.control(1)

                    qc.append(controlled_gate, [estimation_wires[i]] + target_wires)

                return qc

            # Prepare eigenstate |1> for the unitary gate T on the target qubit 
            def prepare_eigenvector(qc, target_wires):
                qc.x(target_wires[0])  # same as qml.PauliX

            # Compute U^(2^k)
            def U_power_2k(unitary, k):
                return np.linalg.matrix_power(unitary, 2**k)

            # Apply controlled powers of U
            def apply_controlled_powers_of_U(qc, unitary, estimation_wires, target_wires):
                t = len(estimation_wires)
                for i in range(t):
                    k = t - 1 - i
                    U_k = U_power_2k(unitary, k)
                    U_gate = UnitaryGate(U_k)
                    controlled_gate = U_gate.control(1)
                    qc.append(controlled_gate, [estimation_wires[i]] + target_wires)

            # Main Quantum Phase estimation (QPE) circuit
            def qpe(unitary, estimation_wires, target_wires):
                n_total = len(estimation_wires) + len(target_wires)
                qc = QuantumCircuit(n_total, len(estimation_wires))
                
                # Prepare eigenstate
                prepare_eigenvector(qc, target_wires)
                
                # Apply Hadamards to estimation qubits
                for i in estimation_wires:
                    qc.h(i)
                
                # Apply controlled powers of U
                apply_controlled_powers_of_U(qc, unitary, estimation_wires, target_wires)
                
                # Apply inverse QFT
                qc.append(QFT(len(estimation_wires), inverse=True, do_swaps=False), estimation_wires)
                
                # Measure estimation qubits
                for i, wire in enumerate(estimation_wires):
                    qc.measure(wire, i)
                
                return qc

            # Quantum Phase Estimation (QPE) circuit for T operator
            T = np.array([[1, 0],
                        [0, np.exp(1j*np.pi/4)]]) 
        
            return qpe(T, estimation_wires, target_wires)
        
        estimation_wires = np.random.randint(4, 8, size=n_circuits)
        qpe_circuits = [
        transpile(
            QPE(range(est_wires), [est_wires]),
            self.noisy_sim,
            optimization_level=0,
        )
        for est_wires in estimation_wires
        ]
        
        return qpe_circuits