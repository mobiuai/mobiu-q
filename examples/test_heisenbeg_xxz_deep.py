#!/usr/bin/env python3
"""
================================================================================
🧬 VQE Heisenberg XXZ - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

Heisenberg XXZ chain:
- 6 qubits (spins)
- Hamiltonian: ∑ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1}) with Δ=2.0
- Open boundary conditions
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

# ── Config ────────────────────────────────────────────────────────────────────
LICENSE_KEY = "YOUR_KEY"
NUM_STEPS   = 150
NUM_SEEDS   = 5
NUM_SHOTS   = 4096
C_SHIFT     = 0.1
LR          = 0.2
METHOD      = "deep"
N_QUBITS    = 6
DELTA       = 2.0

# ── Setup ─────────────────────────────────────────────────────────────────────
print("Setting up FakeFez backend...")
backend   = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS

# Heisenberg XXZ Hamiltonian
ops = []
for i in range(N_QUBITS - 1):
    label = ['I'] * N_QUBITS
    label[i] = 'X'; label[i + 1] = 'X'
    ops.append(("".join(label), 1.0))
    label[i] = 'Y'; label[i + 1] = 'Y'
    ops.append(("".join(label), 1.0))
    label[i] = 'Z'; label[i + 1] = 'Z'
    ops.append(("".join(label), DELTA))

hamiltonian  = SparsePauliOp.from_list(ops)
EXACT_ENERGY = np.min(np.linalg.eigvalsh(hamiltonian.to_matrix()))

ansatz     = EfficientSU2(N_QUBITS, reps=1, entanglement="linear")
pm         = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops    = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters


def evaluate_energy(params):
    job = estimator.run([(isa_ansatz, isa_ops, params)])
    return float(job.result()[0].data.evs)


def get_batched_energy_and_gradient(params, delta):
    """Runs 3 circuits in ONE job — same delta for Adam and Mobiu"""
    job = estimator.run([
        (isa_ansatz, isa_ops, params),
        (isa_ansatz, isa_ops, params + C_SHIFT * delta),
        (isa_ansatz, isa_ops, params - C_SHIFT * delta)
    ])
    results = job.result()
    e_current = float(results[0].data.evs)
    e_plus    = float(results[1].data.evs)
    e_minus   = float(results[2].data.evs)
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    return e_current, grad


# ── Pure Adam (baseline) ──────────────────────────────────────────────────────

class PureAdam:
    """PyTorch Adam — בדיוק כמו שמגדירים ברידמי."""
    def __init__(self, params_np, lr=0.2):
        self._tensor = torch.tensor(params_np, dtype=torch.float64, requires_grad=True)
        self._opt    = torch.optim.Adam([self._tensor], lr=lr)

    def step(self, params_np, grad_np):
        self._opt.zero_grad()
        self._tensor.grad = torch.tensor(grad_np, dtype=torch.float64)
        self._opt.step()
        return self._tensor.detach().numpy().copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🧬 VQE Heisenberg XXZ — CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Model: Heisenberg XXZ (6 spins, Δ={DELTA}) | Ground: {EXACT_ENERGY:.4f} Ha")
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)

    adam_results, mobiu_results = [], []

    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed+1}/{NUM_SEEDS}")

        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, num_params)

        # Pre-generate SPSA deltas — identical for Adam and Mobiu (fair comparison)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params)
                       for _ in range(NUM_STEPS)]

        # ── Baseline: Pure Adam ───────────────────────────────────────────
        print("    Running Pure Adam...", end=" ", flush=True)
        params = init_params.copy()
        opt    = PureAdam(params, lr=LR)
        best   = float('inf')

        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            best   = min(best, energy)
            params = opt.step(params, grad)
            if step % 50 == 0:
                print(f"\r    Adam step {step:3d}/{NUM_STEPS} | Best: {best:.4f}", end="")
        print()

        # ── Test: Adam + Mobiu-Q ──────────────────────────────────────────
        print("    Running Adam + Mobiu-Q...", end=" ", flush=True)
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
        adam_gap  = abs(best       - EXACT_ENERGY)
        mobiu_gap = abs(mobiu_best - EXACT_ENERGY)
        winner    = "✅ Mobiu" if mobiu_gap < adam_gap else "❌ Adam"
        print(f"    Pure Adam: {best:.4f} (gap={adam_gap:.4f}) | "
              f"+ Mobiu: {mobiu_best:.4f} (gap={mobiu_gap:.4f}) → {winner}")

        adam_results.append(best)
        mobiu_results.append(mobiu_best)

    # ── Summary ───────────────────────────────────────────────────────────
    adam_arr  = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    adam_gaps  = np.abs(adam_arr  - EXACT_ENERGY)
    mobiu_gaps = np.abs(mobiu_arr - EXACT_ENERGY)

    improvement = (adam_gaps.mean() - mobiu_gaps.mean()) / adam_gaps.mean() * 100
    wins = int(np.sum(mobiu_gaps < adam_gaps))

    try:
        _, p_value = wilcoxon(adam_gaps, mobiu_gaps)
    except Exception:
        p_value = 1.0

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"  Ground state:           {EXACT_ENERGY:.4f} Ha")
    print(f"  Pure Adam mean gap:     {adam_gaps.mean():.4f}")
    print(f"  Adam + Mobiu mean gap:  {mobiu_gaps.mean():.4f}")
    print(f"\n  📈 Improvement: {improvement:+.1f}%")
    print(f"  🏆 Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print(f"  p-value: {p_value:.6f}")
    print()
    print(f"  Energy values:")
    print(f"    Pure Adam:     {adam_arr.mean():.4f} ± {adam_arr.std():.4f} Ha")
    print(f"    Adam + Mobiu:  {mobiu_arr.mean():.4f} ± {mobiu_arr.std():.4f} Ha")
    print(f"    Exact:         {EXACT_ENERGY:.4f} Ha")
    print("=" * 70)

    fname = 'heisenberg_xxz_customer_results.json'
    with open(fname, 'w') as f:
        json.dump({
            'model': 'Heisenberg_XXZ',
            'baseline': 'Pure Adam',
            'test': 'Adam + MobiuQCore',
            'improvement': float(improvement),
            'adam_mean_gap': float(adam_gaps.mean()),
            'mobiu_mean_gap': float(mobiu_gaps.mean()),
            'win_rate': wins / NUM_SEEDS,
            'p_value': float(p_value),
        }, f, indent=2)
    print(f"\n💾 Saved: {fname}")


if __name__ == "__main__":
    main()