#!/usr/bin/env python3
"""
================================================================================
🔬 QAOA Max Independent Set - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimizer (what customer has BEFORE Mobiu-Q)
- Test: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

QAOA for Maximum Independent Set (MIS):
- Find largest set of non-adjacent nodes
- NP-hard combinatorial optimization
- Tests Mobiu-Q on constrained optimization
================================================================================
"""

import numpy as np
from itertools import combinations
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

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

N_QUBITS = 5
P = 5  # QAOA layers
N_STEPS = 100
N_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.1
LR = 0.1
METHOD = "deep"
PENALTY = 2.0

# Create random graph
np.random.seed(42)
EDGES = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS) if np.random.random() < 0.5]
if not EDGES:
    EDGES = [(0, 1), (1, 2)]

# Compute optimal MIS
def is_independent(nodes, edges):
    return not any(i in nodes and j in nodes for i, j in edges)

optimal_mis_size = max(
    size for size in range(N_QUBITS + 1)
    for nodes in combinations(range(N_QUBITS), size)
    if is_independent(nodes, EDGES)
)

# ============================================================
# SETUP
# ============================================================

print("Setting up FakeFez backend...")
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔬 QAOA Max Independent Set - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Graph: {N_QUBITS} nodes, {len(EDGES)} edges")
    print(f"Optimal MIS size: {optimal_mis_size} → Target cost: {-optimal_mis_size}")
    print(f"QAOA layers: p={P} | Steps: {N_STEPS} | Seeds: {N_SEEDS}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure SPSA optimization (NO Mobiu)")
    print("  • Test: SPSA + Mobiu-Q enhancement")
    print("=" * 70)
    
    spsa_results = []
    mobiu_results = []
    
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")
        
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]
        
        # ─────────────────────────────────────────────────────────────────
        # BASELINE: Pure SPSA
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        spsa_best = float('inf')
        for step in range(N_STEPS):
            e, g = spsa_gradient(params, spsa_deltas[step])
            spsa_best = min(spsa_best, e)
            ak = LR / ((step + 1) ** 0.602)
            params = params - ak * g
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            base_lr=LR,
            verbose=False
        )
        mobiu_best = float('inf')
        for step in range(N_STEPS):
            e, g = spsa_gradient(params, spsa_deltas[step])
            mobiu_best = min(mobiu_best, e)
            params = mobiu_opt.step(params, g, e)
        mobiu_opt.end()
        
        winner = "✅ Mobiu wins" if mobiu_best < spsa_best else "❌ SPSA wins"
        print(f"    Pure SPSA: {spsa_best:.4f} | + Mobiu: {mobiu_best:.4f} → {winner}")
        
        spsa_results.append(spsa_best)
        mobiu_results.append(mobiu_best)

    # Summary
    spsa_mean = np.mean(spsa_results)
    mobiu_mean = np.mean(mobiu_results)
    improvement = (spsa_mean - mobiu_mean) / abs(spsa_mean) * 100 if spsa_mean != 0 else 0
    wins = sum(m < s for m, s in zip(mobiu_results, spsa_results))
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Optimal target: {-optimal_mis_size}")
    print(f"Pure SPSA:      {spsa_mean:.4f}")
    print(f"SPSA + Mobiu:   {mobiu_mean:.4f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{N_SEEDS} ({100*wins/N_SEEDS:.0f}%)")
    print("=" * 70)
    
    with open('qaoa_mis_customer_results.json', 'w') as f:
        json.dump({'problem': 'MIS', 'improvement': improvement, 'wins': wins}, f)


if __name__ == "__main__":
    main()
