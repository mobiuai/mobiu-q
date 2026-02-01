#!/usr/bin/env python3
"""
================================================================================
🧬 VQE H₄ Chain - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimizer (what customer has BEFORE Mobiu-Q)
- Test: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

H₄ (Linear hydrogen chain):
- 4 qubits
- Tests strongly correlated systems
- Eigenvalue verified: -2.0096 Ha
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
NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1
LR = 0.02
METHOD = "standard"

# ============================================================
# SETUP
# ============================================================

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS

# H4 Chain Hamiltonian (4 qubits)
hamiltonian = SparsePauliOp.from_list([
    ("IIII", -1.0466),
    ("IIIZ", 0.1799), ("IIZI", -0.1799), ("IZII", 0.1799), ("ZIII", -0.1799),
    ("IIZZ", 0.1209), ("IZZI", 0.1677), ("ZZII", 0.1209), ("ZIIZ", 0.1677),
    ("IZIZ", 0.1744), ("ZIZI", 0.1744),
    ("IIXX", 0.0362), ("IIYY", 0.0362), ("XXII", 0.0362), ("YYII", 0.0362),
])
EXACT_ENERGY = -2.0096

ansatz = EfficientSU2(4, reps=2, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters


def get_batched_energy_and_gradient(params, delta):
    job = estimator.run([
        (isa_ansatz, isa_ops, params),
        (isa_ansatz, isa_ops, params + C_SHIFT * delta),
        (isa_ansatz, isa_ops, params - C_SHIFT * delta)
    ])
    results = job.result()
    grad = (float(results[1].data.evs) - float(results[2].data.evs)) / (2 * C_SHIFT) * delta
    return float(results[0].data.evs), grad


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🧬 VQE H₄ Chain - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Molecule: H₄ Chain (4 qubits) | Ground State: {EXACT_ENERGY} Ha")
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
        
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]
        
        # ─────────────────────────────────────────────────────────────────
        # BASELINE: Pure SPSA
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        spsa_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            spsa_best = min(spsa_best, e)
            params = params - LR * g
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            verbose=False
        )
        mobiu_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            mobiu_best = min(mobiu_best, e)
            params = mobiu_opt.step(params, g, e)
        mobiu_opt.end()
        
        spsa_gap = abs(spsa_best - EXACT_ENERGY)
        mobiu_gap = abs(mobiu_best - EXACT_ENERGY)
        winner = "✅ Mobiu wins" if mobiu_gap < spsa_gap else "❌ SPSA wins"
        
        print(f"    Pure SPSA: {spsa_best:.4f} (gap={spsa_gap:.4f})")
        print(f"    + Mobiu-Q: {mobiu_best:.4f} (gap={mobiu_gap:.4f}) → {winner}")
        
        spsa_results.append(spsa_best)
        mobiu_results.append(mobiu_best)

    # Summary
    spsa_arr = np.array(spsa_results)
    mobiu_arr = np.array(mobiu_results)
    
    spsa_gaps = np.abs(spsa_arr - EXACT_ENERGY)
    mobiu_gaps = np.abs(mobiu_arr - EXACT_ENERGY)
    
    improvement = (np.mean(spsa_gaps) - np.mean(mobiu_gaps)) / np.mean(spsa_gaps) * 100
    wins = np.sum(mobiu_gaps < spsa_gaps)
    
    print("\n" + "=" * 70)
    print(f"📊 Improvement: {improvement:+.1f}% | Win Rate: {wins}/{NUM_SEEDS}")
    print("=" * 70)
    
    with open('h4_customer_results.json', 'w') as f:
        json.dump({'molecule': 'H4', 'improvement': improvement, 'wins': int(wins)}, f)


if __name__ == "__main__":
    main()
