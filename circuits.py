import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from haiqu_utils import build_noise_model
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit

np.random.seed(42)

class circuits:
    
    def __init__(self):
        self.p_1q = 1e-2   # depolarizing error for 1-qubit native gates
        self.p_2q = 5e-2   # depolarizing error for 2-qubit native gates
        self.ft_scale = 0.1 # ideal FT gates

        self.noise_model = build_noise_model(p_1q=self.p_1q, p_2q=self.p_2q, ft_scale=self.ft_scale)

        self.ideal_sim = AerSimulator()
        self.noisy_sim = AerSimulator(noise_model=self.noise_model)    
    
    def random_circuits(self):
        random_qubits = np.random.randint(3, 7, size=5)
        random_depths = np.random.randint(10, 21, size=5)
        random_circuits = [
        transpile(
            random_circuit(num_qubits=nq, depth=nd, max_operands=2, seed=seed, measure=True),
            self.noisy_sim,
            optimization_level=0,
        )
        for seed, nq, nd in zip(range(5), random_qubits, random_depths)
        ]
        return random_circuits 
    
    def qft_circuits(self):
        def qft(nq):
            qc = QuantumCircuit(nq)
            qc.h(range(nq))
            qc.compose(QFT(num_qubits=nq), inplace=True)
            return qc
            
        random_qubits = np.random.randint(3, 7, size=5)

        qft_circuits = [
        transpile(
            qft(nq),
            self.noisy_sim,
            optimization_level=0,
        )
        for nq in random_qubits
        ]
        
        return qft_circuits
    
    