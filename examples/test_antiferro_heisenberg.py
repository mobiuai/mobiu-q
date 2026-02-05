#!/usr/bin/env python3
"""
================================================================================
🧬 VQE Antiferromagnetic Heisenberg - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimizer (what customer has BEFORE Mobiu-Q)
- Test: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Antiferromagnetic Heisenberg chain:
- 6 qubits (spins)
- Hamiltonian: ∑ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
- Open boundary conditions
- Tests scalability of Mobiu-Q
================================================================================
"""

import numpy as np
from scipy.stats import wilcoxon
import json
import warnings
warnings.filterwarnings('ignore')

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2

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
NUM_STEPS = 150
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1
LR = 0.2
METHOD = "deep"
N_QUBITS = 6

# ============================================================
# SETUP
# ============================================================

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS

# Antiferromagnetic Heisenberg Hamiltonian (6 qubits)
ops = []
for i in range(N_QUBITS - 1):
    # XX term
    label = ['I'] * N_QUBITS
    label[i] = 'X'
    label[i + 1] = 'X'
    ops.append(("".join(label), 1.0))
    
    # YY term
    label[i] = 'Y'
    label[i + 1] = 'Y'
    ops.append(("".join(label), 1.0))
    
    # ZZ term
    label[i] = 'Z'
    label[i + 1] = 'Z'
    ops.append(("".join(label), 1.0))

hamiltonian = SparsePauliOp.from_list(ops)

# Compute exact ground state energy
ham_matrix = hamiltonian.to_matrix()
EXACT_ENERGY = np.min(np.linalg.eigvalsh(ham_matrix))

ansatz = EfficientSU2(N_QUBITS, reps=1, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters


def get_batched_energy_and_gradient(params, delta):
    """Runs 3 circuits in ONE job for maximum throughput"""
    job = estimator.run([
        (isa_ansatz, isa_ops, params),
        (isa_ansatz, isa_ops, params + C_SHIFT * delta),
        (isa_ansatz, isa_ops, params - C_SHIFT * delta)
    ])
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
    print("🧬 VQE Antiferromagnetic Heisenberg - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Model: Antiferromagnetic Heisenberg chain (6 spins) | Ground State: {EXACT_ENERGY:.4f} (computed)")
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print()
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
        init_params = np.random.uniform(-np.pi, np.pi, num_params)
        
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
            spsa_best = min(spsa_best, energy)
            
            # Pure SPSA update
            params = params - LR * grad
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            base_lr=LR,  # Set same LR as SPSA for fair comparison
            verbose=False
        )
        mobiu_best = float('inf')
        
        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            mobiu_best = min(mobiu_best, energy)
            params = mobiu_opt.step(params, grad, energy)
        
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
    
    try:
        stat, p_value = wilcoxon(spsa_gaps, mobiu_gaps)
    except:
        p_value = 1.0
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Pure SPSA mean gap:    {spsa_mean_gap:.4f}")
    print(f"SPSA + Mobiu-Q gap:    {mobiu_mean_gap:.4f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print(f"p-value: {p_value:.6f}")
    print("=" * 70)
    
    print(f"\nEnergy values:")
    print(f"  Pure SPSA:     {np.mean(spsa_arr):.4f} ± {np.std(spsa_arr):.4f}")
    print(f"  SPSA + Mobiu:  {np.mean(mobiu_arr):.4f} ± {np.std(mobiu_arr):.4f}")
    print(f"  Exact:         {EXACT_ENERGY:.4f}")
    
    with open('antiferro_heisenberg_customer_results.json', 'w') as f:
        json.dump({
            'model': 'Antiferro_Heisenberg',
            'improvement': improvement,
            'win_rate': wins/NUM_SEEDS,
            'p_value': p_value
        }, f)


if __name__ == "__main__":
    main()