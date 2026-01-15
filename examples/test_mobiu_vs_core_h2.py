#!/usr/bin/env python3
"""
================================================================================
TEST: Mobiu (New API) vs MobiuQCore (Original) - FakeFez VQE H₂
================================================================================
Tests that the new Mobiu API achieves SAME results as MobiuQCore by using
warmup_only() to pre-learn configuration, then running with Soft Algebra
from step 1.

Key insight: The original MobiuQCore uses Soft Algebra for ALL 60 steps.
With warmup_only(), Mobiu also gets Soft Algebra for ALL 60 steps!

Requirements:
    pip install mobiu-q qiskit qiskit-aer qiskit-ibm-runtime

Usage:
    python test_mobiu_vs_core_h2.py
================================================================================
"""

import numpy as np
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
from mobiu_q import MobiuQCore, Mobiu

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1  # SPSA shift
METHOD = "standard"

# ============================================================
# SETUP BACKEND & ESTIMATOR
# ============================================================

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
# BATCHED SPSA FUNCTION
# ============================================================

def get_batched_energy_and_gradient(params, delta):
    """Runs 3 circuits in ONE job for maximum throughput."""
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
# WARMUP DATA COLLECTION
# ============================================================

def collect_warmup_data(seed=999):
    """Collect warmup data for configuration learning."""
    print("📊 Collecting warmup data...")

    np.random.seed(seed)
    params = np.random.uniform(-0.3, 0.3, num_params)

    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(30)]

    warmup_metrics = []
    for step in range(30):
        energy, _ = get_batched_energy_and_gradient(params, spsa_deltas[step])
        warmup_metrics.append(energy)
        # Don't update params - just collect data

    print(f"   Collected {len(warmup_metrics)} samples")
    print(f"   Energy range: {min(warmup_metrics):.4f} to {max(warmup_metrics):.4f}")

    return warmup_metrics

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🧬 MOBIU vs MOBIUQCORE - FakeFez VQE H₂")
    print("=" * 70)
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print("=" * 70)

    # Phase 1: Collect warmup data ONCE
    warmup_metrics = collect_warmup_data()

    baseline_results = []
    mobiu_old_results = []
    mobiu_new_results = []

    # Initialize NEW Mobiu with warmup_only
    init_params = np.random.uniform(-0.3, 0.3, num_params)
    mobiu_new = Mobiu(
        params=init_params,
        lr=0.02,  # Will be overridden by auto-config
        license_key=LICENSE_KEY,
        mode='hardware',  # Force hardware mode
        verbose=True
    )

    # Pre-learn configuration from warmup data
    print("\n🔧 Learning configuration from warmup data...")
    mobiu_new.warmup_only(warmup_metrics)
    print(f"   Detected: maximize={mobiu_new.config.maximize}, method={mobiu_new.config.method}")
    print(f"   Base LR: {mobiu_new.base_lr}")

    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")

        # Initialize params
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)

        # Pre-generate SPSA deltas
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]

        # ─────────────────────────────────────────────────────────────────
        # Baseline (use_soft_algebra=False)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        baseline_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD, mode='hardware', use_soft_algebra=False, verbose=False
        )

        baseline_best = float('inf')
        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            params = baseline_opt.step(params, grad, energy)
            baseline_best = min(baseline_best, energy)

        baseline_opt.end()

        # ─────────────────────────────────────────────────────────────────
        # MobiuQCore OLD (use_soft_algebra=True) - Reference
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_old_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD, mode='hardware', use_soft_algebra=True, verbose=False
        )

        mobiu_old_best = float('inf')
        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            params = mobiu_old_opt.step(params, grad, energy)
            mobiu_old_best = min(mobiu_old_best, energy)

        mobiu_old_opt.end()

        # ─────────────────────────────────────────────────────────────────
        # Mobiu NEW with warmup_only (Soft Algebra from step 1!)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_new.new_run(params)  # Start fresh run with learned config

        mobiu_new_best = float('inf')
        for step in range(NUM_STEPS):
            energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
            params = mobiu_new.step(energy, gradient=grad)
            if params is not None:
                mobiu_new_best = min(mobiu_new_best, energy)

        # Compare
        baseline_gap = abs(baseline_best - exact_energy)
        old_gap = abs(mobiu_old_best - exact_energy)
        new_gap = abs(mobiu_new_best - exact_energy)

        print(f"    Baseline:     {baseline_best:.4f} (gap={baseline_gap:.4f})")
        print(f"    MobiuQCore:   {mobiu_old_best:.4f} (gap={old_gap:.4f})")
        print(f"    Mobiu (new):  {mobiu_new_best:.4f} (gap={new_gap:.4f})")

        baseline_results.append(baseline_best)
        mobiu_old_results.append(mobiu_old_best)
        mobiu_new_results.append(mobiu_new_best)

    mobiu_new.end()

    # Summary
    baseline_arr = np.array(baseline_results)
    old_arr = np.array(mobiu_old_results)
    new_arr = np.array(mobiu_new_results)

    baseline_mean = np.mean(np.abs(baseline_arr - exact_energy))
    old_mean = np.mean(np.abs(old_arr - exact_energy))
    new_mean = np.mean(np.abs(new_arr - exact_energy))

    old_imp = (baseline_mean - old_mean) / baseline_mean * 100
    new_imp = (baseline_mean - new_mean) / baseline_mean * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline mean gap:     {baseline_mean:.4f}")
    print(f"  MobiuQCore mean gap:   {old_mean:.4f} ({old_imp:+.1f}%)")
    print(f"  Mobiu (new) mean gap:  {new_mean:.4f} ({new_imp:+.1f}%)")
    print("=" * 70)

    # Check if new API matches old API
    if abs(old_imp - new_imp) < 10:  # Within 10% of each other
        print("✅ New Mobiu API achieves similar results to MobiuQCore!")
    else:
        print(f"⚠️  Results differ: MobiuQCore={old_imp:+.1f}%, Mobiu={new_imp:+.1f}%")


if __name__ == "__main__":
    main()
