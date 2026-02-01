#!/usr/bin/env python3
"""
================================================================================
🔬 MOBIU-Q BENCHMARK - QAOA Max Independent Set on FakeFez (v4.2.0)
================================================================================
Customer-facing test using the new architecture:
  • Customer provides their own PyTorch optimizer (SGD)
  • MobiuOptimizer wraps it with Soft Algebra
  • Client optimizer always runs — Mobiu enhances via LR + gradient warp

QAOA for Maximum Independent Set (MIS):
  • Find largest set of non-adjacent nodes
  • NP-hard constrained combinatorial optimization
  • Tests Mobiu-Q on penalized cost landscapes

Requirements:
    pip install qiskit qiskit-aer qiskit-ibm-runtime torch mobiu-q
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import json
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend
    except ImportError:
        from qiskit.providers.fake_provider import GenericBackendV2
        FakeBackend = lambda: GenericBackendV2(num_qubits=127)

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

N_QUBITS = 5
P = 5           # QAOA layers
N_STEPS = 100
N_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.1
LR = 0.1
METHOD = "deep"
PENALTY = 2.0

# Create random graph
np.random.seed(42)
EDGES = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS)
         if np.random.random() < 0.5]
if not EDGES:
    EDGES = [(0, 1), (1, 2)]

# Compute optimal MIS
def is_independent(nodes, edges):
    return not any(i in nodes and j in nodes for i, j in edges)

optimal_mis_size = max(
    size for size in range(N_QUBITS + 1)
    for nodes in combinations(range(N_QUBITS), size)
    if is_independent(nodes, EDGES)
)

# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("🔬 MOBIU-Q BENCHMARK - QAOA MIS on FakeFez (v4.2.0)")
print("=" * 70)
print(f"Graph: {N_QUBITS} nodes, {len(EDGES)} edges")
print(f"Optimal MIS: {optimal_mis_size} → Target cost: {-optimal_mis_size}")
print(f"Settings: p={P}, steps={N_STEPS}, seeds={N_SEEDS}")
print()
print("Architecture: Customer's torch.optim.SGD wrapped by MobiuOptimizer")
print("  • Baseline: SGD + MobiuOptimizer(use_soft_algebra=False)")
print("  • Test:     SGD + MobiuOptimizer(use_soft_algebra=True)")
print("  • Small model (<1000 params) → full_sync mode")
print("  • Client's SGD executes the actual update")
print("=" * 70)

print("\nSetting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
n_params = 2 * P  # 10 params → full_sync mode


# ============================================================
# QAOA CIRCUIT AND EVALUATION
# ============================================================

def create_mis_qaoa(params_np):
    gammas = params_np[:P]
    betas = params_np[P:]

    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc.h(i)

    for layer in range(P):
        gamma = gammas[layer]
        for i in range(N_QUBITS):
            qc.rz(gamma, i)
        for i, j in EDGES:
            qc.rzz(PENALTY * gamma / 2, i, j)
        for i in range(N_QUBITS):
            qc.rx(2 * betas[layer], i)

    return qc


def get_mis_cost(bitstring):
    selected = sum(bitstring)
    violations = sum(1 for i, j in EDGES if bitstring[i] == 1 and bitstring[j] == 1)
    return -selected + PENALTY * violations


def evaluate_qaoa(params_np, shots=SHOTS):
    qc = create_mis_qaoa(params_np)
    qc.measure_all()
    qc_t = pm.run(qc)

    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()

    total_cost = 0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        cost = get_mis_cost(bits)
        total_cost += cost * count

    return total_cost / shots


def spsa_gradient(params_np, delta):
    e_plus = evaluate_qaoa(params_np + C_SHIFT * delta)
    e_minus = evaluate_qaoa(params_np - C_SHIFT * delta)
    e_center = evaluate_qaoa(params_np)
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    return e_center, grad


# ============================================================
# PYTORCH WRAPPER FOR QAOA PARAMS
# ============================================================

class QAOAModel(nn.Module):
    """Thin wrapper: holds QAOA angles as a PyTorch Parameter."""
    def __init__(self, init_values):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    baseline_results = []
    mobiu_results = []

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")

        # Shared init
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)

        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]

        # ─────────────────────────────────────────────────────────────
        # BASELINE: SGD + MobiuOptimizer (SA=OFF)
        # ─────────────────────────────────────────────────────────────
        model = QAOAModel(init_params.copy())
        base_opt = torch.optim.SGD(model.parameters(), lr=LR)
        opt = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            use_soft_algebra=False,
            verbose=False
        )

        b_best = float('inf')
        for step in range(N_STEPS):
            params_np = model.theta.detach().cpu().numpy()
            e, grad_np = spsa_gradient(params_np, spsa_deltas[step])
            b_best = min(b_best, e)

            opt.zero_grad()
            model.theta.grad = torch.tensor(grad_np, dtype=torch.float32)
            opt.step(e)

        opt.end()

        # ─────────────────────────────────────────────────────────────
        # MOBIU: SGD + MobiuOptimizer (SA=ON)
        # ─────────────────────────────────────────────────────────────
        model = QAOAModel(init_params.copy())
        base_opt = torch.optim.SGD(model.parameters(), lr=LR)
        opt = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            use_soft_algebra=True,
            verbose=False
        )

        m_best = float('inf')
        for step in range(N_STEPS):
            params_np = model.theta.detach().cpu().numpy()
            e, grad_np = spsa_gradient(params_np, spsa_deltas[step])
            m_best = min(m_best, e)

            opt.zero_grad()
            model.theta.grad = torch.tensor(grad_np, dtype=torch.float32)
            opt.step(e)

        opt.end()

        # Per-seed result
        winner = "Mobiu" if m_best < b_best else "Baseline"
        print(f"    Baseline: {b_best:.4f} | Mobiu: {m_best:.4f} → {winner}")
        baseline_results.append(b_best)
        mobiu_results.append(m_best)

    # ============================================================
    # SUMMARY
    # ============================================================
    b_mean = np.mean(baseline_results)
    m_mean = np.mean(mobiu_results)
    improvement = (b_mean - m_mean) / abs(b_mean) * 100 if b_mean != 0 else 0
    wins = sum(m < b for m, b in zip(mobiu_results, baseline_results))

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - FakeFez QAOA MIS (v4.2.0)")
    print("=" * 70)
    print(f"  Optimal target: {-optimal_mis_size}")
    print(f"  SGD (baseline): {b_mean:.4f}")
    print(f"  SGD + Mobiu-Q:  {m_mean:.4f}")
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  Win rate: {wins}/{N_SEEDS} ({100*wins/N_SEEDS:.0f}%)")
    print("=" * 70)

    with open('fakefez_mis_results.json', 'w') as f:
        json.dump({'problem': 'MIS', 'improvement': improvement, 'wins': wins}, f)


if __name__ == "__main__":
    main()
