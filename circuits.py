import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from haiqu_utils import build_noise_model
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit

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
        random_qubits = np.random.randint(3, 7, size=n_circuits)
        random_depths = np.random.randint(10, 21, size=n_circuits)
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
    
    