#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q NOISE ROBUSTNESS - COMPREHENSIVE BENCHMARK
================================================================================

Test Soft Algebra's noise robustness across:
1. Different qubit counts (2, 4, 6)
2. Different noise levels (low, medium, high)
3. Multiple seeds for statistical significance

Goal: Prove that SA advantage INCREASES with noise level

Requirements:
    pip install qiskit qiskit-aer numpy requests

================================================================================
"""

import numpy as np
import requests
import json
import time
from typing import List, Tuple, Dict

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Install: pip install qiskit qiskit-aer")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MOBIU_API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "YOUR_LICENCE"

# Test configurations
QUBIT_CONFIGS = [2, 4, 6]
NOISE_LEVELS = {
    'ideal': 0.0,
    'low': 0.005,
    'medium': 0.01,
    'high': 0.02,
    'extreme': 0.05
}

N_LAYERS = 2
N_STEPS = 50
N_SEEDS = 5
SHOTS = 512  # Reduced for speed


# ==============================================================================
# CIRCUIT AND HAMILTONIAN
# ==============================================================================

def create_vqe_circuit(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Hardware-efficient VQE ansatz"""
    params = []
    qc = QuantumCircuit(n_qubits)
    
    for layer in range(n_layers):
        for q in range(n_qubits):
            p = Parameter(f'θ_{layer}_{q}')
            params.append(p)
            qc.ry(p, q)
        
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    
    for q in range(n_qubits):
        p = Parameter(f'θ_final_{q}')
        params.append(p)
        qc.ry(p, q)
    
    return qc, params


def create_heisenberg_hamiltonian(n_qubits: int) -> SparsePauliOp:
    """Heisenberg XXZ: H = Σ(XX + YY + ZZ)"""
    pauli_terms = []
    coeffs = []
    
    for i in range(n_qubits - 1):
        for pauli in ['X', 'Y', 'Z']:
            term = ['I'] * n_qubits
            term[i], term[i+1] = pauli, pauli
            pauli_terms.append(''.join(term[::-1]))
            coeffs.append(1.0)
    
    return SparsePauliOp(pauli_terms, coeffs)


def create_noisy_backend(noise_level: float) -> AerSimulator:
    """Create backend with specified depolarizing noise"""
    if noise_level == 0:
        return AerSimulator()
    
    noise_model = NoiseModel()
    
    # Single-qubit gate error
    error_1q = depolarizing_error(noise_level, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'h', 'sdg', 'rz', 'rx'])
    
    # Two-qubit gate error (2x single-qubit)
    error_2q = depolarizing_error(noise_level * 2, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    # Measurement error
    from qiskit_aer.noise import ReadoutError
    p_meas = noise_level * 2
    read_err = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    noise_model.add_all_qubit_readout_error(read_err)
    
    return AerSimulator(noise_model=noise_model)


# ==============================================================================
# ENERGY ESTIMATION
# ==============================================================================

def estimate_energy_spsa(circuit, params, param_values, hamiltonian, backend,
                         epsilon=0.1, shots=512) -> Tuple[float, np.ndarray]:
    """Estimate energy and SPSA gradient"""
    n_params = len(param_values)
    delta = np.random.choice([-1, 1], size=n_params)
    
    def measure_energy(values):
        bound = circuit.assign_parameters({p: v for p, v in zip(params, values)})
        energy = 0.0
        
        for pauli_str, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
            meas = bound.copy()
            for i, p in enumerate(pauli_str[::-1]):
                if p == 'X': meas.h(i)
                elif p == 'Y': meas.sdg(i); meas.h(i)
            meas.measure_all()
            
            transpiled = transpile(meas, backend, optimization_level=0)
            counts = backend.run(transpiled, shots=shots).result().get_counts()
            
            exp_val = 0.0
            for bitstring, count in counts.items():
                parity = 1
                for i, p in enumerate(pauli_str[::-1]):
                    if p != 'I' and bitstring[-(i+1)] == '1':
                        parity *= -1
                exp_val += parity * count / shots
            energy += float(np.real(coeff)) * exp_val
        
        return energy
    
    e_center = measure_energy(param_values)
    e_plus = measure_energy(param_values + epsilon * delta)
    e_minus = measure_energy(param_values - epsilon * delta)
    
    grad = (e_plus - e_minus) / (2 * epsilon) * delta
    return e_center, grad


# ==============================================================================
# OPTIMIZERS
# ==============================================================================

class MomentumOptimizer:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = None
    
    def step(self, params, grad):
        if self.v is None: self.v = np.zeros_like(params)
        self.v = self.momentum * self.v + grad
        return params - self.lr * self.v


class MobiuQOptimizer:
    def __init__(self, license_key, base_lr=0.1):
        self.license_key = license_key
        self.base_lr = base_lr
        self.session_id = None
    
    def reset(self):
        if self.session_id: self.end()
        try:
            r = requests.post(MOBIU_API_URL, json={
                'action': 'start',
                'license_key': self.license_key,
                'method': 'qaoa',
                'mode': 'hardware',
                'base_lr': self.base_lr,
                'base_optimizer': 'Momentum'
            }, timeout=10)
            data = r.json()
            if data.get('success'):
                self.session_id = data['session_id']
        except:
            self.session_id = None
    
    def step(self, params, grad, energy):
        if not self.session_id:
            return params - self.base_lr * grad
        try:
            r = requests.post(MOBIU_API_URL, json={
                'action': 'step',
                'license_key': self.license_key,
                'session_id': self.session_id,
                'params': params.tolist(),
                'gradient': grad.tolist(),
                'energy': float(energy)
            }, timeout=30)
            data = r.json()
            if data.get('success'):
                return np.array(data['new_params'])
        except:
            pass
        return params - self.base_lr * grad
    
    def end(self):
        if self.session_id:
            try:
                requests.post(MOBIU_API_URL, json={
                    'action': 'end',
                    'license_key': self.license_key,
                    'session_id': self.session_id
                }, timeout=5)
            except:
                pass
            self.session_id = None


# ==============================================================================
# SINGLE RUN
# ==============================================================================

def run_optimization(circuit, params, hamiltonian, backend, optimizer,
                     init_values, n_steps, use_mobiu=False):
    """Run VQE and return final energy"""
    values = init_values.copy()
    
    for step in range(n_steps):
        energy, grad = estimate_energy_spsa(circuit, params, values, hamiltonian,
                                            backend, shots=SHOTS)
        if use_mobiu:
            values = optimizer.step(values, grad, energy)
        else:
            values = optimizer.step(values, grad)
    
    # Final measurement (more shots)
    final_energy, _ = estimate_energy_spsa(circuit, params, values, hamiltonian,
                                           backend, shots=SHOTS*2)
    return final_energy


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not QISKIT_AVAILABLE:
        print("Install Qiskit first!")
        return
    
    print("=" * 90)
    print("MOBIU-Q NOISE ROBUSTNESS - COMPREHENSIVE BENCHMARK")
    print("=" * 90)
    print(f"Qubit configs: {QUBIT_CONFIGS}")
    print(f"Noise levels: {list(NOISE_LEVELS.keys())}")
    print(f"Seeds per config: {N_SEEDS}")
    print("=" * 90)
    
    # Results: results[n_qubits][noise_level] = {'momentum': [...], 'mobiu': [...]}
    all_results = {}
    
    for n_qubits in QUBIT_CONFIGS:
        print(f"\n{'='*90}")
        print(f"TESTING {n_qubits} QUBITS")
        print(f"{'='*90}")
        
        # Create circuit and hamiltonian
        circuit, params = create_vqe_circuit(n_qubits, N_LAYERS)
        n_params = len(params)
        hamiltonian = create_heisenberg_hamiltonian(n_qubits)
        
        # Ground state energy
        H_matrix = hamiltonian.to_matrix()
        ground_energy = np.min(np.real(np.linalg.eigvalsh(H_matrix)))
        
        print(f"Parameters: {n_params}, Ground energy: {ground_energy:.4f}")
        
        all_results[n_qubits] = {'ground': ground_energy}
        
        for noise_name, noise_level in NOISE_LEVELS.items():
            print(f"\n  📊 Noise: {noise_name} ({noise_level})")
            
            backend = create_noisy_backend(noise_level)
            
            mom_results = []
            mobiu_results = []
            
            for seed in range(N_SEEDS):
                np.random.seed(seed * 100 + n_qubits)
                init_values = np.random.uniform(-np.pi, np.pi, n_params)
                
                # Momentum
                momentum = MomentumOptimizer(lr=0.1)
                mom_energy = run_optimization(circuit, params, hamiltonian, backend,
                                              momentum, init_values, N_STEPS)
                mom_results.append(mom_energy)
                
                # Mobiu-Q
                mobiu = MobiuQOptimizer(LICENSE_KEY, base_lr=0.1)
                mobiu.reset()
                mobiu_energy = run_optimization(circuit, params, hamiltonian, backend,
                                                mobiu, init_values, N_STEPS, use_mobiu=True)
                mobiu.end()
                mobiu_results.append(mobiu_energy)
                
                print(f"    Seed {seed}: Mom={mom_energy:.3f}, Mobiu={mobiu_energy:.3f}")
            
            all_results[n_qubits][noise_name] = {
                'momentum': mom_results,
                'mobiu': mobiu_results
            }
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: Mean Final Energy (lower = better)")
    print("=" * 90)
    
    print(f"\n{'Qubits':<8} {'Noise':<10} {'Ground':<10} {'Momentum':<12} {'Mobiu-Q':<12} {'Winner':<12} {'SA Gain':<12}")
    print("-" * 90)
    
    sa_wins = 0
    total_tests = 0
    gains_by_noise = {k: [] for k in NOISE_LEVELS.keys()}
    
    for n_qubits in QUBIT_CONFIGS:
        ground = all_results[n_qubits]['ground']
        
        for noise_name in NOISE_LEVELS.keys():
            data = all_results[n_qubits][noise_name]
            mom_mean = np.mean(data['momentum'])
            mobiu_mean = np.mean(data['mobiu'])
            
            # Lower energy = better (closer to ground)
            mom_gap = mom_mean - ground
            mobiu_gap = mobiu_mean - ground
            
            if mobiu_gap < mom_gap:
                winner = "Mobiu ✅"
                sa_wins += 1
                gain = (mom_gap - mobiu_gap) / abs(mom_gap) * 100 if mom_gap != 0 else 0
            else:
                winner = "Momentum"
                gain = -(mobiu_gap - mom_gap) / abs(mobiu_gap) * 100 if mobiu_gap != 0 else 0
            
            gains_by_noise[noise_name].append(gain)
            total_tests += 1
            
            print(f"{n_qubits:<8} {noise_name:<10} {ground:<10.3f} {mom_mean:<12.3f} {mobiu_mean:<12.3f} {winner:<12} {gain:+.1f}%")
    
    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 90)
    print("KEY FINDINGS: SA Advantage vs Noise Level")
    print("=" * 90)
    
    print(f"\n{'Noise Level':<15} {'Avg SA Gain':<15} {'Wins':<10}")
    print("-" * 40)
    
    for noise_name in NOISE_LEVELS.keys():
        gains = gains_by_noise[noise_name]
        avg_gain = np.mean(gains)
        wins = sum(1 for g in gains if g > 0)
        total = len(gains)
        
        marker = "🚀" if avg_gain > 10 else ("✅" if avg_gain > 0 else "")
        print(f"{noise_name:<15} {avg_gain:+.1f}%{' '*5} {marker:<3} {wins}/{total}")
    
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    
    print(f"\nTotal: Mobiu-Q wins {sa_wins}/{total_tests} tests ({sa_wins/total_tests*100:.0f}%)")
    
    # Check if advantage increases with noise
    ideal_gain = np.mean(gains_by_noise['ideal'])
    high_gain = np.mean(gains_by_noise['high'])
    extreme_gain = np.mean(gains_by_noise['extreme'])
    
    print(f"\nNoise scaling:")
    print(f"  Ideal:   {ideal_gain:+.1f}%")
    print(f"  High:    {high_gain:+.1f}%")
    print(f"  Extreme: {extreme_gain:+.1f}%")
    
    if extreme_gain > ideal_gain:
        print(f"\n✅ SA ADVANTAGE INCREASES WITH NOISE!")
        print(f"   From {ideal_gain:+.1f}% (ideal) to {extreme_gain:+.1f}% (extreme)")
        print(f"   Improvement: {extreme_gain - ideal_gain:+.1f} percentage points")
    else:
        print(f"\n⚠️  SA advantage does not clearly increase with noise")
    
    # Save results
    print("\nSaving to noise_robustness_results.json...")
    
    # Convert numpy to python types
    save_results = {}
    for n_q in all_results:
        save_results[n_q] = {}
        for k, v in all_results[n_q].items():
            if isinstance(v, dict):
                save_results[n_q][k] = {
                    'momentum': [float(x) for x in v['momentum']],
                    'mobiu': [float(x) for x in v['mobiu']]
                }
            else:
                save_results[n_q][k] = float(v)
    
    with open('noise_robustness_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
