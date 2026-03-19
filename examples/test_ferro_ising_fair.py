#!/usr/bin/env python3
"""
================================================================================
🧬 VQE Ferromagnetic Ising - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

Ferromagnetic Ising chain:
- 6 qubits (spins)
- Hamiltonian: -∑ (Z_i Z_{i+1})
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

import torch
from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"
NUM_STEPS = 150
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1
LR = 0.02   # standard + hardware default (same for both)
METHOD   = "standard"
N_QUBITS = 6

# ============================================================
# SETUP
# ============================================================

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS
estimator.options.seed_simulator = 42

# Ferromagnetic Ising Hamiltonian (6 qubits)
ops = []
for i in range(N_QUBITS - 1):
    # ZZ term (with -1 for ferromagnetic)
    label = ['I'] * N_QUBITS
    label[i] = 'Z'
    label[i + 1] = 'Z'
    ops.append(("".join(label), -1.0))

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


class PureAdam:
    """PyTorch Adam — same LR as Mobiu hardware default (0.02)."""
    def __init__(self, params_np, lr=0.02):
        self._tensor = torch.tensor(params_np, dtype=torch.float64, requires_grad=True)
        self._opt    = torch.optim.Adam([self._tensor], lr=lr)

    def step(self, params_np, grad_np):
        self._opt.zero_grad()
        self._tensor.grad = torch.tensor(grad_np, dtype=torch.float64)
        self._opt.step()
        return self._tensor.detach().numpy().copy()

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🧬 VQE Ferromagnetic Ising - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Model: Ferromagnetic Ising chain (6 spins) | Ground State: {EXACT_ENERGY:.4f} (computed)")
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
        
        # Initialize — same params for both
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, num_params)

        # Pre-generate SPSA deltas — identical for Adam and Mobiu (fair comparison)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]

        # ── Baseline: Pure Adam ───────────────────────────────────────────
        print("    Running Pure Adam...", end="", flush=True)
        params   = init_params.copy()
        opt      = PureAdam(params, lr=LR)
        adam_best = float('inf')

        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            adam_best = min(adam_best, energy)
            params    = opt.step(params, grad)
            if step % 50 == 0:
                print(f"\r    Adam step {step:3d}/{NUM_STEPS} | Best: {adam_best:.4f}", end="")
        print()

        # ── Test: Adam + Mobiu-Q ──────────────────────────────────────────
        print("    Running Adam + Mobiu-Q...", end="", flush=True)
        params    = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            base_lr=LR,
            verbose=False
        )
        mobiu_best = float('inf')

        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            mobiu_best = min(mobiu_best, energy)
            params     = mobiu_opt.step(params, grad, energy)
            if step % 50 == 0:
                print(f"\r    Mobiu step {step:3d}/{NUM_STEPS} | Best: {mobiu_best:.4f}", end="")
        print()
        mobiu_opt.end()

        # ── Compare ───────────────────────────────────────────────────────
        adam_gap  = abs(adam_best  - EXACT_ENERGY)
        mobiu_gap = abs(mobiu_best - EXACT_ENERGY)
        winner    = "✅ Mobiu" if mobiu_gap < adam_gap else "❌ Adam"
        print(f"    Pure Adam: {adam_best:.4f} (gap={adam_gap:.4f}) | "
              f"+ Mobiu: {mobiu_best:.4f} (gap={mobiu_gap:.4f}) → {winner}")

        adam_results.append(adam_best)
        mobiu_results.append(mobiu_best)

    # ============================================================
    # SUMMARY
    # ============================================================
    adam_arr = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    
    adam_gaps = np.abs(adam_arr - EXACT_ENERGY)
    mobiu_gaps = np.abs(mobiu_arr - EXACT_ENERGY)
    
    adam_mean_gap = np.mean(adam_gaps)
    mobiu_mean_gap = np.mean(mobiu_gaps)
    
    improvement = (adam_mean_gap - mobiu_mean_gap) / adam_mean_gap * 100
    wins = np.sum(mobiu_gaps < adam_gaps)
    
    try:
        _, p_value = wilcoxon(adam_gaps, mobiu_gaps)
    except:
        p_value = 1.0
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"  Pure Adam mean gap:   {adam_mean_gap:.4f}")
    print(f"  Adam + Mobiu gap:     {mobiu_mean_gap:.4f}")
    print(f"\n  📈 Improvement: {improvement:+.1f}%")
    print(f"  🏆 Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print(f"  p-value: {p_value:.6f}")
    print("=" * 70)
    
    print(f"\n  Energy values:")
    print(f"    Pure Adam:     {adam_arr.mean():.4f} ± {adam_arr.std():.4f} Ha")
    print(f"    Adam + Mobiu:  {mobiu_arr.mean():.4f} ± {mobiu_arr.std():.4f} Ha")
    print(f"    Exact:         {EXACT_ENERGY:.4f} Ha")
    
    with open('ferro_ising_customer_results.json', 'w') as f:
        json.dump({
            'model': 'Ferro_Ising',
            'improvement': improvement,
            'win_rate': wins/NUM_SEEDS,
            'p_value': p_value
        }, f)


if __name__ == "__main__":
    main()