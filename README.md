# Partial Error Correction – PushQuantum Hackathon 2025

## Overview
This repository contains code and resources for the Haiqu Challenge at the PushQuantum Hackathon 2025. The challenge explores partial fault tolerance in quantum circuits, where only one gate per layer can be executed in a fault-tolerant (FT) manner. The goal is to maximize circuit fidelity under noise by strategically choosing which gates to protect.

## Hackathon Team
The Hackathon team consisted of César Hernando, Arturo Castano, Vittorio Macripo and Tim Neumann.

## Challenge Description
Quantum circuits are inherently noisy. Full quantum error correction is not yet practical, so this challenge focuses on early fault-tolerant quantum computing (EFTQC) with partial QEC. You will implement a transformation pass that marks selected gates as FT placeholders, following strict constraints:
- Only one FT gate per layer (DAG slice)
- FT gates use a `_ft` suffix (e.g., `x_ft`, `cz_ft`)
- The noise model treats FT gates as ideal or less noisy

## Repository Structure

- `haiqu_partial_ec.py`: Main script implementing the custom circuit transformation and baseline, grading, and visualization.
- `circuits.py`: Utilities for generating random, QFT, and QPE circuits for benchmarking.
- `test_scalability.py`: Script for benchmarking and plotting scalability (runtime vs qubits/depth).
- `images/`: Folder for challenge and result images.

## Getting Started
1. **Install dependencies**: The code uses Qiskit 2.2.3 and Qiskit Aer 0.17.2. Install with:
	```bash
	pip install qiskit==2.2.3 qiskit-aer==0.17.2 numpy matplotlib
	```
    or install all dependencies directly with 
    ```bash
    pip install requirements.txt
    ```
2. **Run the notebook**: Open `haiqu_challenge_pushquantum_2025.ipynb` for a guided introduction and demo.
3. **Run the main script**: Execute `partial_error_correction.py` to test your transformation and see grading results.
4. **Benchmark scalability**: Use `test_scalability.py` to evaluate runtime as a function of qubits and circuit depth.

## Key Functions
- `transform_circuit(circ: QuantumCircuit) -> QuantumCircuit`: Main deliverable. Marks selected gates as FT according to your strategy.
- `baseline_transform(circ: QuantumCircuit)`: Simple reference implementation.
