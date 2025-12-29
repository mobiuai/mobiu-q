#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q BENCHMARK - QAOA MaxCut on FakeFez
================================================================================
Real quantum hardware noise model for QAOA optimization.

Requirements:
    pip install qiskit qiskit-aer qiskit-ibm-runtime

Usage:
    python test_fakefez_qaoa.py
================================================================================
"""

import numpy as np
import requests
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "YOUR_LICENCE"  # Replace with your key

# ==============================================================================
# SETTINGS
# ==============================================================================

N_QUBITS = 5
P = 5  # QAOA layers (matching original benchmark)
N_STEPS = 100
N_SEEDS = 10
SHOTS = 4096
C_SHIFT = 0.1

# Create random MaxCut graph
np.random.seed(42)
EDGES = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS) 
         if np.random.random() < 0.5]
if not EDGES:
    EDGES = [(0, 1)]

print("=" * 80)
print("🔬 MOBIU-Q BENCHMARK - QAOA MaxCut on FakeFez")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Graph: {N_QUBITS} qubits, {len(EDGES)} edges: {EDGES}")
print(f"Settings: p={P}, steps={N_STEPS}, seeds={N_SEEDS}, shots={SHOTS}")
print("=" * 80)

# Setup FakeFez
print("\n🔧 Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeFez())
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

n_params = 2 * P


# ==============================================================================
# QAOA CIRCUIT AND EVALUATION
# ==============================================================================

def create_qaoa_circuit(params, n_qubits, edges, p):
    """Create QAOA circuit for MaxCut"""
    gammas = params[:p]
    betas = params[p:]
    
    qc = QuantumCircuit(n_qubits)
    
    # Initial superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # QAOA layers
    for layer in range(p):
        # Cost layer (ZZ interactions)
        for i, j in edges:
            qc.rzz(2 * gammas[layer], i, j)
        
        # Mixer layer (X rotations)
        for i in range(n_qubits):
            qc.rx(2 * betas[layer], i)
    
    return qc


def get_maxcut_cost(bitstring, edges):
    """Calculate MaxCut cost for a bitstring"""
    cost = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cost += 1
    return cost


def evaluate_qaoa(params, n_qubits, edges, p, shots=SHOTS):
    """Evaluate QAOA circuit on FakeFez backend"""
    qc = create_qaoa_circuit(params, n_qubits, edges, p)
    qc.measure_all()
    
    # Transpile
    qc_transpiled = pm.run(qc)
    
    # Run
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Calculate expectation
    total_cost = 0
    total_shots = 0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]  # Reverse for qiskit convention
        cost = get_maxcut_cost(bits, edges)
        total_cost += cost * count
        total_shots += count
    
    # Return negative (we minimize, but want max cut)
    return -total_cost / total_shots


def spsa_gradient(params, delta):
    """SPSA gradient estimation with pre-generated delta"""
    e_plus = evaluate_qaoa(params + C_SHIFT * delta, N_QUBITS, EDGES, P)
    e_minus = evaluate_qaoa(params - C_SHIFT * delta, N_QUBITS, EDGES, P)
    e_center = evaluate_qaoa(params, N_QUBITS, EDGES, P)
    
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    return e_center, grad


# ==============================================================================
# OPTIMIZER RUNNER
# ==============================================================================

def run_optimizer(name, use_soft_algebra, seed, spsa_deltas):
    """Run optimizer with pre-generated SPSA deltas for fair comparison"""
    
    # Start session - CORRECT API FORMAT
    r = requests.post(API_URL, json={
        'action': 'start',
        'license_key': LICENSE_KEY,
        'method': 'deep',          
        'mode': 'hardware',        
        'use_soft_algebra': use_soft_algebra,
        'base_lr': 0.1             # QAOA default
    }, timeout=10)
    
    data = r.json()
    if not data.get('success'):
        print(f"   ERROR: {data.get('error')}")
        return None
    
    sid = data['session_id']
    
    # Init params
    np.random.seed(seed)
    params = np.random.uniform(-np.pi, np.pi, n_params)
    
    best_energy = float('inf')
    
    for step in range(N_STEPS):
        delta = spsa_deltas[step]
        e, grad = spsa_gradient(params, delta)
        best_energy = min(best_energy, e)
        
        r = requests.post(API_URL, json={
            'action': 'step',
            'license_key': LICENSE_KEY,
            'session_id': sid,
            'params': params.tolist(),
            'gradient': grad.tolist(),
            'energy': e
        }, timeout=10)
        
        if r.json().get('success'):
            params = np.array(r.json()['new_params'])
        
        if step % 20 == 0:
            print(f"\r   {name} Step {step}/{N_STEPS} | E: {e:.4f} | Best: {best_energy:.4f}", end="")
    
    requests.post(API_URL, json={
        'action': 'end',
        'license_key': LICENSE_KEY,
        'session_id': sid
    }, timeout=5)
    
    # Final evaluation with more shots
    final_energy = evaluate_qaoa(params, N_QUBITS, EDGES, P, shots=SHOTS * 4)
    print(f" | Final: {final_energy:.4f}")
    
    return final_energy


# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================

def main():
    adam_results = []
    mobiu_results = []
    
    for seed in range(N_SEEDS):
        print(f"\n{'─'*60}")
        print(f"  SEED {seed}")
        print(f"{'─'*60}")
        
        # Pre-generate SPSA deltas for fair comparison
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]
        
        # Run Adam (use_soft_algebra=False)
        adam_final = run_optimizer("Adam ", False, seed, spsa_deltas)
        if adam_final is None:
            continue
        adam_results.append(adam_final)
        
        # Run Mobiu (use_soft_algebra=True) with SAME deltas
        mobiu_final = run_optimizer("Mobiu", True, seed, spsa_deltas)
        if mobiu_final is None:
            continue
        mobiu_results.append(mobiu_final)
        
        # Per-seed result
        winner = "✅ Mobiu" if mobiu_final < adam_final else "❌ Adam"
        print(f"   Adam: {adam_final:.4f} | Mobiu: {mobiu_final:.4f} → {winner}")
    
    # ===========================================================================
    # FINAL RESULTS
    # ===========================================================================
    
    if not adam_results or not mobiu_results:
        print("\n❌ No results collected. Check API connection.")
        return
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS - FakeFez QAOA MaxCut")
    print("=" * 80)
    
    adam_mean = np.mean(adam_results)
    mobiu_mean = np.mean(mobiu_results)
    
    # For MaxCut, lower (more negative) is better
    improvement = (adam_mean - mobiu_mean) / abs(adam_mean) * 100 if adam_mean != 0 else 0
    wins = sum(m < a for m, a in zip(mobiu_results, adam_results))
    
    print(f"\n{'Optimizer':<15} {'Mean Cost':<15} {'Std':<12} {'Best':<12}")
    print("-" * 55)
    print(f"{'Adam':<15} {adam_mean:<15.4f} {np.std(adam_results):<12.4f} {min(adam_results):<12.4f}")
    print(f"{'Mobiu-Q':<15} {mobiu_mean:<15.4f} {np.std(mobiu_results):<12.4f} {min(mobiu_results):<12.4f}")
    print("-" * 55)
    
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Wins: {wins}/{len(adam_results)}")
    
    # Significance test
    from scipy import stats
    try:
        _, p_val = stats.ttest_rel(mobiu_results, adam_results)
        sig = "✅ Significant" if p_val < 0.05 else "🔶 Not significant"
        print(f"p-value: {p_val:.4f} {sig}")
    except:
        pass
    
    if improvement > 0:
        print(f"\n🎉 MOBIU-Q WINS by {improvement:.1f}%")
    else:
        print(f"\n❌ Adam wins by {-improvement:.1f}%")
    
    print("=" * 80)


if __name__ == "__main__":
    main()