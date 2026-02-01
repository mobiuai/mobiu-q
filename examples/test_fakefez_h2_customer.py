#!/usr/bin/env python3
"""
================================================================================
🧬 VQE CUSTOMER TEST: Pure SPSA vs SPSA + Mobiu-Q
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimization (no Mobiu at all)
- Test: SPSA enhanced with Mobiu-Q

NOT using use_soft_algebra flag - testing real customer integration!

Requirements:
    pip install mobiu-q qiskit qiskit-aer qiskit-ibm-runtime
================================================================================
"""

import numpy as np
from scipy.stats import wilcoxon
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
        print("⚠️ FakeFez not found, using GenericBackendV2")
        from qiskit.providers.fake_provider import GenericBackendV2
        FakeBackend = lambda: GenericBackendV2(num_qubits=127)

# Mobiu-Q
from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1  # SPSA shift
LR = 0.02     # Learning rate for pure SPSA
METHOD = "standard"  

# ============================================================
# SETUP BACKEND & ESTIMATOR
# ============================================================

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS

# Hamiltonian & Ansatz for H2
hamiltonian = SparsePauliOp.from_list([
    ("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347),
    ("IZ", 0.5716), ("XX", 0.0910), ("YY", 0.0910)
])
EXACT_ENERGY = -1.846

ansatz = EfficientSU2(2, reps=4, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters

# ============================================================
# BATCHED SPSA FUNCTION
# ============================================================

def get_batched_energy_and_gradient(params, delta):
    """
    Runs 3 circuits in ONE job for maximum throughput.
    Returns: (current_energy, gradient_estimate)
    """
    pub_current = (isa_ansatz, isa_ops, params)
    pub_plus = (isa_ansatz, isa_ops, params + C_SHIFT * delta)
    pub_minus = (isa_ansatz, isa_ops, params - C_SHIFT * delta)
    
    job = estimator.run([pub_current, pub_plus, pub_minus])
    results = job.result()
    
    e_current = float(results[0].data.evs)
    e_plus = float(results[1].data.evs)
    e_minus = float(results[2].data.evs)
    
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    
    return e_current, grad

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🧬 VQE CUSTOMER TEST: Pure SPSA vs SPSA + Mobiu-Q")
    print("=" * 70)
    print(f"Molecule: H₂ | Ground State: {EXACT_ENERGY} Ha")
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print(f"")
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure SPSA optimization (NO Mobiu)")
    print("  • Test: SPSA + Mobiu-Q enhancement")
    print("=" * 70)
    
    spsa_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
        
        # Initialize
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        
        # Pre-generate SPSA deltas for fairness
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]
        
        # ─────────────────────────────────────────────────────────────────
        # BASELINE: Pure SPSA (what customer has BEFORE adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        spsa_best = float('inf')
        
        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            
            # Pure SPSA update: params -= lr * grad
            params = params - LR * grad
            spsa_best = min(spsa_best, energy)
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        
        # Customer adds Mobiu-Q like this:
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD, 
            mode='hardware',
            verbose=False
        )
        
        mobiu_best = float('inf')
        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            
            # Mobiu-Q enhanced update
            params = mobiu_opt.step(params, grad, energy)
            mobiu_best = min(mobiu_best, energy)
        
        mobiu_opt.end()
        
        # Compare
        spsa_gap = abs(spsa_best - EXACT_ENERGY)
        mobiu_gap = abs(mobiu_best - EXACT_ENERGY)
        winner = "✅ Mobiu wins" if mobiu_gap < spsa_gap else "❌ SPSA wins"
        
        print(f"    Pure SPSA: {spsa_best:.4f} (gap={spsa_gap:.4f})")
        print(f"    + Mobiu-Q: {mobiu_best:.4f} (gap={mobiu_gap:.4f}) → {winner}")
        
        spsa_results.append(spsa_best)
        mobiu_results.append(mobiu_best)

    # ============================================================
    # SUMMARY
    # ============================================================
    spsa_arr = np.array(spsa_results)
    mobiu_arr = np.array(mobiu_results)
    
    spsa_gaps = np.abs(spsa_arr - EXACT_ENERGY)
    mobiu_gaps = np.abs(mobiu_arr - EXACT_ENERGY)
    
    spsa_mean_gap = np.mean(spsa_gaps)
    mobiu_mean_gap = np.mean(mobiu_gaps)
    
    improvement = (spsa_mean_gap - mobiu_mean_gap) / spsa_mean_gap * 100
    wins = np.sum(mobiu_gaps < spsa_gaps)
    
    # Statistical test
    try:
        stat, p_value = wilcoxon(spsa_gaps, mobiu_gaps)
    except:
        p_value = 1.0
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Pure SPSA mean gap:    {spsa_mean_gap:.4f} Ha")
    print(f"SPSA + Mobiu-Q gap:    {mobiu_mean_gap:.4f} Ha")
    print(f"")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print(f"p-value: {p_value:.6f}")
    print("=" * 70)
    
    # Also show energy values
    print(f"\nEnergy values:")
    print(f"  Pure SPSA:     {np.mean(spsa_arr):.4f} ± {np.std(spsa_arr):.4f} Ha")
    print(f"  SPSA + Mobiu:  {np.mean(mobiu_arr):.4f} ± {np.std(mobiu_arr):.4f} Ha")
    print(f"  Exact:         {EXACT_ENERGY:.4f} Ha")

if __name__ == "__main__":
    main()
