#!/usr/bin/env python3
"""
================================================================================
🔬 VQE H₂ on FakeFez - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

Uses IBM's FakeFez noise model for realistic quantum simulation.
================================================================================
"""

import numpy as np
from datetime import datetime
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

from mobiu_q import MobiuQCore, Demeasurement

# ── Config ────────────────────────────────────────────────────────────────────
LICENSE_KEY = "YOUR_KEY"
NUM_STEPS   = 60
NUM_SEEDS   = 5
NUM_SHOTS   = 4096
C_SHIFT     = 0.1
LR          = 0.02
METHOD      = "standard"

# ── Setup ─────────────────────────────────────────────────────────────────────
print("Setting up FakeFez backend...")
backend   = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS
estimator.options.seed_simulator = 42

hamiltonian = SparsePauliOp.from_list([
    ("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347),
    ("IZ",  0.5716), ("XX",  0.0910), ("YY",  0.0910)
])
EXACT_ENERGY = -1.846

ansatz     = EfficientSU2(2, reps=4, entanglement="linear")
pm         = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops    = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters


def evaluate_energy(params):
    job = estimator.run([(isa_ansatz, isa_ops, params)])
    return float(job.result()[0].data.evs)


# ── Pure Adam (baseline) ──────────────────────────────────────────────────────

class PureAdam:
    """Adam optimizer for quantum parameters — no Mobiu-Q."""
    def __init__(self, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return params - self.lr * mh / (np.sqrt(vh) + self.eps)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🔬 VQE H₂ on FakeFez — CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print(f"Ground state energy: {EXACT_ENERGY} Ha")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)

    adam_results, mobiu_results = [], []

    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed+1}/{NUM_SEEDS}")

        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)

        # ── Baseline: Pure Adam ───────────────────────────────────────────
        print("    Running Pure Adam...")
        np.random.seed(seed * 1000)
        params = init_params.copy()
        opt    = PureAdam(lr=LR)
        best   = float('inf')

        for step in range(NUM_STEPS):
            grad, energy = Demeasurement.spsa(evaluate_energy, params, c_shift=C_SHIFT)
            best   = min(best, energy)
            params = opt.step(params, grad)
            if step % 15 == 0:
                print(f"\r    Adam step {step:2d}/{NUM_STEPS} | Best: {best:.4f}", end="")
        print()

        # ── Test: Adam + Mobiu-Q ──────────────────────────────────────────
        print("    Running Adam + Mobiu-Q...")
        np.random.seed(seed * 1000)
        params = init_params.copy()

        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            base_lr=LR,
            verbose=False
        )

        mobiu_best = float('inf')
        for step in range(NUM_STEPS):
            grad, energy = Demeasurement.spsa(evaluate_energy, params, c_shift=C_SHIFT)
            mobiu_best = min(mobiu_best, energy)
            params     = mobiu_opt.step(params, grad, energy)
            if step % 15 == 0:
                print(f"\r    Mobiu step {step:2d}/{NUM_STEPS} | Best: {mobiu_best:.4f}", end="")
        print()
        mobiu_opt.end()

        # ── Compare ───────────────────────────────────────────────────────
        adam_gap  = abs(best        - EXACT_ENERGY)
        mobiu_gap = abs(mobiu_best  - EXACT_ENERGY)
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
    adam_mean  = np.mean(adam_gaps)
    mobiu_mean = np.mean(mobiu_gaps)
    improvement = (adam_mean - mobiu_mean) / adam_mean * 100
    wins = sum(1 for a, m in zip(adam_gaps, mobiu_gaps) if m < a)

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"  Ground state:            {EXACT_ENERGY} Ha")
    print(f"  Pure Adam mean gap:      {adam_mean*1000:.2f} mHa")
    print(f"  Adam + Mobiu mean gap:   {mobiu_mean*1000:.2f} mHa")
    print(f"\n  📈 Improvement: {improvement:+.1f}%")
    print(f"  🏆 Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print("=" * 70)

    fname = f'vqe_h2_customer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fname, 'w') as f:
        json.dump({
            'test_type': 'customer_view',
            'baseline': 'Pure Adam',
            'test': 'Adam + MobiuQCore',
            'improvement': float(improvement),
            'adam_mean_mHa': float(adam_mean * 1000),
            'mobiu_mean_mHa': float(mobiu_mean * 1000),
            'wins': wins,
            'seeds': NUM_SEEDS,
        }, f, indent=2)
    print(f"\n💾 Saved: {fname}")


if __name__ == "__main__":
    main()