#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q BENCHMARK - QAOA Max Independent Set on FakeFez
================================================================================
"""

import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend
    except ImportError:
        from qiskit.providers.fake_provider import GenericBackendV2
        FakeBackend = lambda: GenericBackendV2(num_qubits=127)

from mobiu_q import MobiuQCore

LICENSE_KEY = "YOUR_KEY"

N_QUBITS = 5
P = 5
N_STEPS = 100
N_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.1
METHOD = "deep"
PENALTY = 2.0

# Create random graph
np.random.seed(42)
EDGES = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS) if np.random.random() < 0.5]
if not EDGES:
    EDGES = [(0, 1), (1, 2)]

# Compute optimal MIS
from itertools import combinations
def is_independent(nodes, edges):
    return not any(i in nodes and j in nodes for i, j in edges)

optimal_mis_size = max(
    size for size in range(N_QUBITS + 1)
    for nodes in combinations(range(N_QUBITS), size)
    if is_independent(nodes, EDGES)
)

print("=" * 70)
print("MOBIU-Q - QAOA Max Independent Set on FakeFez")
print("=" * 70)
print(f"Graph: {N_QUBITS} nodes, {len(EDGES)} edges")
print(f"Optimal MIS: {optimal_mis_size} -> Target: {-optimal_mis_size}")
print(f"p={P} | Steps: {N_STEPS}")
print("=" * 70)

backend = AerSimulator.from_backend(FakeBackend())
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
n_params = 2 * P


def create_mis_qaoa(params):
    gammas = params[:P]
    betas = params[P:]
    
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc.h(i)
    
    for layer in range(P):
        gamma = gammas[layer]
        for i in range(N_QUBITS):
            qc.rz(gamma, i)
        for i, j in EDGES:
            qc.rzz(PENALTY * gamma / 2, i, j)
        for i in range(N_QUBITS):
            qc.rx(2 * betas[layer], i)
    
    return qc


def get_mis_cost(bitstring):
    selected = sum(bitstring)
    violations = sum(1 for i, j in EDGES if bitstring[i] == 1 and bitstring[j] == 1)
    return -selected + PENALTY * violations


def evaluate_qaoa(params, shots=SHOTS):
    qc = create_mis_qaoa(params)
    qc.measure_all()
    qc_t = pm.run(qc)
    
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    
    total_cost = 0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        cost = get_mis_cost(bits)
        total_cost += cost * count
    
    return total_cost / shots


def spsa_gradient(params, delta):
    e_plus = evaluate_qaoa(params + C_SHIFT * delta)
    e_minus = evaluate_qaoa(params - C_SHIFT * delta)
    e_center = evaluate_qaoa(params)
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    return e_center, grad


def main():
    baseline_results, mobiu_results = [], []
    
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]
        
        # Baseline - SGD
        params = init_params.copy()
        baseline_opt = MobiuQCore(LICENSE_KEY, method=METHOD, mode='hardware', 
                                   use_soft_algebra=False, verbose=False,
                                   base_optimizer='SGD', base_lr=0.1)
        b_best = float('inf')
        for step in range(N_STEPS):
            e, g = spsa_gradient(params, spsa_deltas[step])
            params = baseline_opt.step(params, g, e)
            b_best = min(b_best, e)
        baseline_opt.end()
        
        # Mobiu - SGD
        params = init_params.copy()
        mobiu_opt = MobiuQCore(LICENSE_KEY, method=METHOD, mode='hardware', 
                               use_soft_algebra=True, verbose=False,
                               base_optimizer='SGD', base_lr=0.1)
        m_best = float('inf')
        for step in range(N_STEPS):
            e, g = spsa_gradient(params, spsa_deltas[step])
            params = mobiu_opt.step(params, g, e)
            m_best = min(m_best, e)
        mobiu_opt.end()
        
        winner = "Mobiu" if m_best < b_best else "Baseline"
        print(f"    Baseline: {b_best:.4f} | Mobiu: {m_best:.4f} -> {winner}")
        baseline_results.append(b_best)
        mobiu_results.append(m_best)

    b_mean, m_mean = np.mean(baseline_results), np.mean(mobiu_results)
    imp = (b_mean - m_mean) / abs(b_mean) * 100 if b_mean != 0 else 0
    wins = sum(m < b for m, b in zip(mobiu_results, baseline_results))
    
    print("\n" + "=" * 70)
    print(f"  Optimal: {-optimal_mis_size}")
    print(f"  Baseline: {b_mean:.4f} | Mobiu: {m_mean:.4f}")
    print(f"  Improvement: {imp:+.1f}% | Wins: {wins}/{N_SEEDS}")
    print("=" * 70)
    
    with open('fakefez_mis_results.json', 'w') as f:
        json.dump({'problem': 'MIS', 'improvement': imp, 'wins': wins}, f)

if __name__ == "__main__":
    main()
