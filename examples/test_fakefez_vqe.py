#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q BENCHMARK - VQE H2 on FakeFez
================================================================================
Exact replica of IBM Fez hardware validation.
Uses Qiskit FakeFez backend with real noise model.

Requirements:
    pip install qiskit qiskit-aer qiskit-ibm-runtime

Usage:
    python test_fakefez_vqe.py
================================================================================
"""

import numpy as np
import requests
from datetime import datetime

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeFez

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "e756ce65-186e-4747-aaaf-5a1fb1473b7e"  # Replace with your key

# ==============================================================================
# SETTINGS (Matching IBM Fez run)
# ==============================================================================

N_STEPS = 60
N_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.12

# H2 molecule Hamiltonian
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -0.4804),
    ("ZZ", 0.3435),
    ("ZI", -0.4347),
    ("IZ", 0.5716),
    ("XX", 0.0910),
    ("YY", 0.0910)
])
GROUND_STATE = -1.846  # H2 ground state (from Hamiltonian)

# ==============================================================================
# SETUP
# ==============================================================================

print("=" * 80)
print("🔬 MOBIU-Q BENCHMARK - VQE H2 on FakeFez")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Backend: FakeFez (127-qubit IBM replica)")
print(f"Molecule: H2 (ground state: {GROUND_STATE} Ha)")
print(f"Settings: steps={N_STEPS}, seeds={N_SEEDS}, shots={SHOTS}")
print("=" * 80)

print("\n🔧 Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeFez())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = SHOTS

# Ansatz
ansatz = EfficientSU2(2, reps=4, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = H2_HAMILTONIAN.apply_layout(isa_ansatz.layout)
n_params = ansatz.num_parameters

print(f"   Ansatz parameters: {n_params}")
print(f"   Shots: {SHOTS}")


# ==============================================================================
# ENERGY AND GRADIENT (SPSA)
# ==============================================================================

def get_energy_and_gradient(params, delta):
    """SPSA gradient estimation with pre-generated delta"""
    job = estimator.run([
        (isa_ansatz, isa_ops, params),
        (isa_ansatz, isa_ops, params + C_SHIFT * delta),
        (isa_ansatz, isa_ops, params - C_SHIFT * delta)
    ])
    res = job.result()
    
    e = float(res[0].data.evs)
    ep = float(res[1].data.evs)
    em = float(res[2].data.evs)
    
    grad = (ep - em) / (2 * C_SHIFT) * delta
    return e, grad


# ==============================================================================
# OPTIMIZER RUNNER
# ==============================================================================

def run_optimizer(name, use_soft_algebra, seed, spsa_deltas):
    """Run optimizer with pre-generated SPSA deltas for fair comparison"""
    
    # Start session - CORRECT API FORMAT
    r = requests.post(API_URL, json={
        'action': 'start',
        'license_key': LICENSE_KEY,
        'method': 'vqe',           # NOT 'problem'
        'mode': 'hardware',        # NOT 'noisy'
        'use_soft_algebra': use_soft_algebra
        # NO base_lr - let server decide for hardware mode
    }, timeout=10)
    
    data = r.json()
    if not data.get('success'):
        print(f"   ERROR: {data.get('error')}")
        return None, None
    
    sid = data['session_id']
    
    # Init params
    np.random.seed(seed)
    params = np.random.uniform(-0.3, 0.3, n_params)
    
    best_energy = float('inf')
    
    for step in range(N_STEPS):
        delta = spsa_deltas[step]
        e, grad = get_energy_and_gradient(params, delta)
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
        
        if step % 10 == 0:
            gap = abs(best_energy - GROUND_STATE)
            print(f"\r   {name} Step {step}/{N_STEPS} | E: {e:.4f} | Best: {best_energy:.4f} | Gap: {gap:.4f}", end="")
    
    requests.post(API_URL, json={
        'action': 'end',
        'license_key': LICENSE_KEY,
        'session_id': sid
    }, timeout=5)
    
    print()
    return best_energy, params


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
        adam_best, _ = run_optimizer("Adam ", False, seed, spsa_deltas)
        if adam_best is None:
            continue
        adam_results.append(adam_best)
        
        # Run Mobiu (use_soft_algebra=True) with SAME deltas
        mobiu_best, _ = run_optimizer("Mobiu", True, seed, spsa_deltas)
        if mobiu_best is None:
            continue
        mobiu_results.append(mobiu_best)
        
        # Per-seed result
        adam_gap = abs(adam_best - GROUND_STATE)
        mobiu_gap = abs(mobiu_best - GROUND_STATE)
        winner = "✅ Mobiu" if mobiu_gap < adam_gap else "❌ Adam"
        print(f"   Adam gap: {adam_gap:.4f} | Mobiu gap: {mobiu_gap:.4f} → {winner}")
    
    # ===========================================================================
    # FINAL RESULTS
    # ===========================================================================
    
    if not adam_results or not mobiu_results:
        print("\n❌ No results collected. Check API connection.")
        return
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS - FakeFez VQE H2")
    print("=" * 80)
    
    adam_gaps = [abs(e - GROUND_STATE) for e in adam_results]
    mobiu_gaps = [abs(e - GROUND_STATE) for e in mobiu_results]
    
    adam_mean = np.mean(adam_gaps)
    mobiu_mean = np.mean(mobiu_gaps)
    
    improvement = (adam_mean - mobiu_mean) / adam_mean * 100 if adam_mean != 0 else 0
    wins = sum(m < a for m, a in zip(mobiu_gaps, adam_gaps))
    
    print(f"\n{'Optimizer':<15} {'Mean Gap (Ha)':<15} {'Std Gap':<12} {'Best Gap':<12}")
    print("-" * 55)
    print(f"{'Adam':<15} {adam_mean:<15.4f} {np.std(adam_gaps):<12.4f} {min(adam_gaps):<12.4f}")
    print(f"{'Mobiu-Q':<15} {mobiu_mean:<15.4f} {np.std(mobiu_gaps):<12.4f} {min(mobiu_gaps):<12.4f}")
    print("-" * 55)
    
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Wins: {wins}/{len(adam_results)}")
    
    # Significance test
    from scipy import stats
    try:
        _, p_val = stats.ttest_rel(mobiu_gaps, adam_gaps)
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