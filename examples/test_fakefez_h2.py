#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q REAL TEST - FakeFez VQE for H₂ (Turbo Batched)
================================================================================
Tests Mobiu-Q on H2 molecule with IBM FakeFez noise model.
Uses BackendEstimatorV2 with EXPLICIT BATCHING for maximum speed (3x faster).

FAIR TEST: 
- SPSA deltas generated locally and applied to both optimizers.
- Batched execution: [params, params+delta, params-delta] run in parallel.

Requirements:
    pip install mobiu-q qiskit qiskit-aer qiskit-ibm-runtime

Usage:
    python test_fakefez_h2.py
================================================================================
"""

import numpy as np
from scipy.stats import wilcoxon
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2

# Robust FakeFez Import
try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend
    except ImportError:
        # Fallback to generic if Fez is missing
        print("⚠️ FakeFez not found, using GenericBackendV2")
        from qiskit.providers.fake_provider import GenericBackendV2
        FakeBackend = lambda: GenericBackendV2(num_qubits=127)

# Mobiu-Q
from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_LICENCE"

NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1  # SPSA shift

# ============================================================
# SETUP BACKEND & ESTIMATOR
# ============================================================

# Use FakeFez via Aer (Much faster than raw FakeBackend)
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS

# Hamiltonian & Ansatz for H2
hamiltonian = SparsePauliOp.from_list([
    ("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347),
    ("IZ", 0.5716), ("XX", 0.0910), ("YY", 0.0910)
])
exact_energy = -1.846

ansatz = EfficientSU2(2, reps=4, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters

# ============================================================
# BATCHED SPSA FUNCTION (The Speed Secret)
# ============================================================

def get_batched_energy_and_gradient(params, delta):
    """
    Runs 3 circuits in ONE job for maximum throughput.
    Returns: (current_energy, gradient_estimate)
    """
    # Create 3 sets of parameters: Current, Plus, Minus
    # We measure current params too to get accurate history logging
    pub_current = (isa_ansatz, isa_ops, params)
    pub_plus = (isa_ansatz, isa_ops, params + C_SHIFT * delta)
    pub_minus = (isa_ansatz, isa_ops, params - C_SHIFT * delta)
    
    # Run all 3 in parallel on the simulator
    job = estimator.run([pub_current, pub_plus, pub_minus])
    results = job.result()
    
    e_current = float(results[0].data.evs)
    e_plus = float(results[1].data.evs)
    e_minus = float(results[2].data.evs)
    
    # Compute gradient
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    
    return e_current, grad

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🧬 MOBIU-Q REAL TEST - FakeFez VQE H₂ (Turbo Batched)")
    print("=" * 70)
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print("Technique: Explicit Job Batching (3 circuits per job)")
    print("=" * 70)
    
    baseline_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
        
        # Initialize
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        
        # Pre-generate SPSA deltas for total fairness
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]
        
        # ─────────────────────────────────────────────────────────────
        # Baseline (use_soft_algebra=False)
        # ─────────────────────────────────────────────────────────────
        params = init_params.copy()
        baseline_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method='standard', mode='hardware', use_soft_algebra=False, verbose=False
        )
        
        baseline_best = float('inf')
        for step in range(NUM_STEPS):
            # 1. Calculate Gradient FAST (Batched)
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            
            # 2. Update via Mobiu Client (Manual Gradient Mode)
            params = baseline_opt.step(params, grad, energy)
            
            baseline_best = min(baseline_best, energy)
        
        baseline_opt.end()
        
        # ─────────────────────────────────────────────────────────────
        # Mobiu-Q (use_soft_algebra=True)
        # ─────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method='standard', mode='hardware', use_soft_algebra=True, verbose=False
        )
        
        mobiu_best = float('inf')
        for step in range(NUM_STEPS):
            # 1. Calculate Gradient FAST (Batched) with SAME delta
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            
            # 2. Update via Mobiu Client
            params = mobiu_opt.step(params, grad, energy)
            
            mobiu_best = min(mobiu_best, energy)
        
        mobiu_opt.end()
        
        # Compare
        baseline_gap = abs(baseline_best - exact_energy)
        mobiu_gap = abs(mobiu_best - exact_energy)
        winner = "✅ Mobiu" if mobiu_gap < baseline_gap else "❌ Baseline"
        print(f"    Baseline: {baseline_best:.4f} (gap={baseline_gap:.4f}) | Mobiu: {mobiu_best:.4f} (gap={mobiu_gap:.4f}) → {winner}")
        
        baseline_results.append(baseline_best)
        mobiu_results.append(mobiu_best)

    # Summary
    baseline_arr = np.array(baseline_results)
    mobiu_arr = np.array(mobiu_results)
    baseline_mean = np.mean(np.abs(baseline_arr - exact_energy))
    mobiu_mean = np.mean(np.abs(mobiu_arr - exact_energy))
    imp = (baseline_mean - mobiu_mean) / baseline_mean * 100
    
    print("\n" + "=" * 70)
    print(f"  Improvement: {imp:+.1f}%")
    print("=" * 70)
    
    with open('fakefez_h2_results.json', 'w') as f:
        json.dump({'molecule': 'H2', 'improvement': imp, 'baseline': baseline_mean, 'mobiu': mobiu_mean}, f)

if __name__ == "__main__":
    main()