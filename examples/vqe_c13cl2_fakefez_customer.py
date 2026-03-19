#!/usr/bin/env python3
"""
================================================================================
🔬 VQE C₁₃Cl₂ Half-Möbius on FakeFez - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

Uses IBM's FakeFez noise model for realistic quantum simulation.
NO artificial noise added - FakeFez provides real hardware noise profile.

Based on: "A molecule with half-Möbius topology" (Rončević et al., 2025)
          IBM Research / Oxford / Manchester

Requirements:
    pip install mobiu-q qiskit qiskit-aer qiskit-ibm-runtime
================================================================================
"""

import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeFez
from mobiu_q import MobiuQCore

LICENSE_KEY = "YOUR_KEY"

# Settings (matching IBM Fez replica benchmark)
N_STEPS = 60
N_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.12

# ============================================================
# C₁₃Cl₂ MODEL HAMILTONIAN
# ============================================================

# The molecule has:
# - 13 carbons in a ring + 2 Cl substituents
# - Helical π-orbitals (half-Möbius topology GML¹₄)
# - Singlet ground state with ~0.44 eV singlet-triplet gap
# - Multireference character (paper uses CASPT2)

hamiltonian = SparsePauliOp.from_list([
    ("IIII", -8.50),    # Offset
    ("ZZZZ", 0.15),     # 4-body correlation (multireference)
    ("ZZII", 0.25),     # Adjacent correlation
    ("IIZZ", 0.25),
    ("ZIZI", 0.12),     # Helical coupling
    ("IZIZ", 0.12),
    ("XXII", 0.08),     # Exchange
    ("IIXX", 0.08),
    ("YYII", 0.08),
    ("IIYY", 0.08),
    ("ZIIZ", 0.10),     # Ring closure
    ("ZIII", -0.15),    # Cl electronegativity
    ("IIIZ", -0.15),
])

# Exact ground state
H_matrix = hamiltonian.to_matrix()
if hasattr(H_matrix, 'toarray'):
    H_matrix = H_matrix.toarray()
GROUND_STATE = min(np.linalg.eigvalsh(H_matrix))

# ============================================================
# SETUP FAKEFEZ
# ============================================================

print("=" * 70)
print("🔬 VQE C₁₃Cl₂ Half-Möbius on FakeFez - CUSTOMER VIEW TEST")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Backend: FakeFez (real IBM noise profile)")
print(f"Ground state: {GROUND_STATE:.4f}")
print(f"Settings: shots={SHOTS}, steps={N_STEPS}, seeds={N_SEEDS}")
print()
print("This test shows what a CUSTOMER would experience:")
print("  • Baseline: Pure Adam (NO Mobiu)")
print("  • Test: Adam + Mobiu-Q")
print("=" * 70)

print("\n🔧 Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeFez())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = SHOTS
estimator.options.seed_simulator = 42

# Ansatz
ansatz = EfficientSU2(4, reps=2, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
n_params = ansatz.num_parameters

print(f"✅ FakeFez ready | Ansatz: {n_params} parameters")

# ============================================================
# ENERGY AND GRADIENT (SPSA)
# ============================================================

def get_energy_and_gradient(params, delta):
    """SPSA gradient estimation on FakeFez — delta pre-generated for fairness"""
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

# ============================================================
# PURE ADAM OPTIMIZER
# ============================================================

class PureAdam:
    """Standard Adam optimizer - what customer has BEFORE Mobiu-Q"""
    def __init__(self, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grad):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ============================================================
# RUN OPTIMIZER
# ============================================================

def run_adam(seed, spsa_deltas):
    """Run Pure Adam (baseline)"""
    np.random.seed(seed)
    params = np.random.uniform(-0.3, 0.3, n_params)
    adam = PureAdam(lr=0.02)  # standard + hardware default
    
    best_energy = float('inf')
    for step in range(N_STEPS):
        e, grad = get_energy_and_gradient(params, spsa_deltas[step])
        best_energy = min(best_energy, e)
        params = adam.step(params, grad)
        if step % 15 == 0:
            print(f"\r    Adam step {step}/{N_STEPS} | Best: {best_energy:.4f}", end="")
    print()
    return best_energy


def run_mobiu(seed, spsa_deltas):
    """Run Adam + Mobiu-Q — same as all other VQE tests"""
    np.random.seed(seed)
    params = np.random.uniform(-0.3, 0.3, n_params)

    mobiu_opt = MobiuQCore(
        license_key=LICENSE_KEY,
        method="standard",
        mode='hardware',
        base_lr=0.02,   # standard + hardware default
        verbose=False
    )

    best_energy = float('inf')
    for step in range(N_STEPS):
        e, grad = get_energy_and_gradient(params, spsa_deltas[step])
        best_energy = min(best_energy, e)
        params = mobiu_opt.step(params, grad, e)
        if step % 15 == 0:
            print(f"\r    Mobiu step {step}/{N_STEPS} | Best: {best_energy:.4f}", end="")

    mobiu_opt.end()
    print()
    return best_energy

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 Starting Benchmark (this may take 10-15 minutes)")
    print("=" * 70)
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")
        
        # Pre-generate deltas — same for both Adam and Mobiu
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]

        # Run Adam
        print("    Running Pure Adam...")
        adam_best = run_adam(seed, spsa_deltas)
        adam_results.append(adam_best)
        
        # Run Mobiu (same seed, same deltas)
        print("    Running Adam + Mobiu-Q...")
        mobiu_best = run_mobiu(seed, spsa_deltas)
        if mobiu_best is None:
            continue
        mobiu_results.append(mobiu_best)
        
        # Compare
        adam_gap = abs(adam_best - GROUND_STATE)
        mobiu_gap = abs(mobiu_best - GROUND_STATE)
        winner = "✅ Mobiu" if mobiu_gap < adam_gap else "❌ Adam"
        print(f"    Adam: {adam_best:.4f} (gap={adam_gap:.4f}) | "
              f"Mobiu: {mobiu_best:.4f} (gap={mobiu_gap:.4f}) → {winner}")

    # ============================================================
    # SUMMARY
    # ============================================================
    adam_arr = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    adam_gaps = np.abs(adam_arr - GROUND_STATE)
    mobiu_gaps = np.abs(mobiu_arr - GROUND_STATE)
    adam_mean = np.mean(adam_gaps)
    mobiu_mean = np.mean(mobiu_gaps)
    improvement = (adam_mean - mobiu_mean) / adam_mean * 100 if adam_mean != 0 else 0
    wins = sum(1 for a, m in zip(adam_gaps, mobiu_gaps) if m < a)
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"  Ground state:           {GROUND_STATE:.4f}")
    print(f"  Pure Adam mean gap:     {adam_mean*1000:.2f} mHa")
    print(f"  Adam + Mobiu mean gap:  {mobiu_mean*1000:.2f} mHa")
    print(f"\n  📈 Improvement: {improvement:+.1f}%")
    print(f"  🏆 Win rate: {wins}/{len(mobiu_results)} ({100*wins/len(mobiu_results):.0f}%)")
    print("=" * 70)
    
    print("\n📚 Paper Context:")
    print("   • Original: 72-qubit SqDRIFT on ibm_kingston")
    print("   • Active space: (32,36) electrons/orbitals")
    print("   • Singlet-triplet gap: 0.44 eV")
    print("   • This benchmark: 4-qubit model on FakeFez")
    print("=" * 70)
    
    # Save
    filename = f'c13cl2_fakefez_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump({
            'molecule': 'C13Cl2_half_mobius',
            'backend': 'FakeFez',
            'baseline': 'Pure Adam',
            'test': 'Adam + Mobiu-Q (method=vqe)',
            'ground_state': GROUND_STATE,
            'improvement_pct': improvement,
            'adam_mean_mHa': adam_mean * 1000,
            'mobiu_mean_mHa': mobiu_mean * 1000,
            'adam_energies': adam_results,
            'mobiu_energies': mobiu_results,
            'wins': wins,
            'total_seeds': N_SEEDS,
        }, f, indent=2)
    print(f"\n💾 Saved: {filename}")


if __name__ == "__main__":
    main()