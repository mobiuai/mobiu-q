#!/usr/bin/env python3
"""
================================================================================
🔬 MOBIU-Q BENCHMARK - QAOA MaxCut on FakeFez (v4.2.0)
================================================================================
Customer-facing test using the new architecture:
  • Customer provides their own PyTorch optimizer (SGD)
  • MobiuOptimizer wraps it with Soft Algebra
  • Client optimizer always runs — Mobiu enhances via LR + gradient warp

QAOA MaxCut on FakeFez:
  • Real quantum hardware noise model
  • Combinatorial optimization benchmark
  • Tests Mobiu-Q on rugged landscapes

Requirements:
    pip install qiskit qiskit-aer qiskit-ibm-runtime torch mobiu-q
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
N_SEEDS = 10
SHOTS = 4096
C_SHIFT = 0.1
LR = 0.1
METHOD = "deep"

# Create random MaxCut graph
np.random.seed(42)
EDGES = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS)
         if np.random.random() < 0.5]
if not EDGES:
    EDGES = [(0, 1)]

# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("🔬 MOBIU-Q BENCHMARK - QAOA MaxCut on FakeFez (v4.2.0)")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Graph: {N_QUBITS} qubits, {len(EDGES)} edges: {EDGES}")
print(f"Settings: p={P}, steps={N_STEPS}, seeds={N_SEEDS}, shots={SHOTS}")
print(f"Method: {METHOD}")
print()
print("Architecture: Customer's torch.optim.SGD wrapped by MobiuOptimizer")
print("  • Baseline: SGD + MobiuOptimizer(use_soft_algebra=False)")
print("  • Test:     SGD + MobiuOptimizer(use_soft_algebra=True)")
print("  • Small model (<1000 params) → full_sync mode")
print("  • Server computes SA → returns adaptive_lr + warp_factor")
print("  • Client's SGD executes the actual update")
print("=" * 70)

print("\n🔧 Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
n_params = 2 * P  # 10 params → full_sync mode


# ============================================================
# QAOA CIRCUIT AND EVALUATION
# ============================================================

def create_qaoa_circuit(params_np, n_qubits, edges, p):
    gammas = params_np[:p]
    betas = params_np[p:]

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)

    for layer in range(p):
        for i, j in edges:
            qc.rzz(2 * gammas[layer], i, j)
        for i in range(n_qubits):
            qc.rx(2 * betas[layer], i)

    return qc


def get_maxcut_cost(bitstring, edges):
    cost = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cost += 1
    return cost


def evaluate_qaoa(params_np, shots=SHOTS):
    qc = create_qaoa_circuit(params_np, N_QUBITS, EDGES, P)
    qc.measure_all()
    qc_t = pm.run(qc)

    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()

    total_cost = 0
    total_shots = 0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        cost = get_maxcut_cost(bits, EDGES)
        total_cost += cost * count
        total_shots += count

    return -total_cost / total_shots  # Negative: we minimize


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
        print(f"\n{'─'*60}")
        print(f"  SEED {seed}")
        print(f"{'─'*60}")

        # Shared init
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)

        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]

        # ─────────────────────────────────────────────────────────────
        # BASELINE: SGD wrapped by MobiuOptimizer (SA=OFF)
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

        best_e = float('inf')
        for step in range(N_STEPS):
            params_np = model.theta.detach().cpu().numpy()
            e, grad_np = spsa_gradient(params_np, spsa_deltas[step])
            best_e = min(best_e, e)

            opt.zero_grad()
            model.theta.grad = torch.tensor(grad_np, dtype=torch.float32)
            opt.step(e)

            if step % 20 == 0:
                print(f"\r   SGD   Step {step}/{N_STEPS} | E: {e:.4f} | Best: {best_e:.4f}", end="")

        baseline_final = evaluate_qaoa(model.theta.detach().cpu().numpy(), shots=SHOTS * 4)
        print(f" | Final: {baseline_final:.4f}")
        opt.end()

        # ─────────────────────────────────────────────────────────────
        # MOBIU: SGD wrapped by MobiuOptimizer (SA=ON)
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

        best_e = float('inf')
        for step in range(N_STEPS):
            params_np = model.theta.detach().cpu().numpy()
            e, grad_np = spsa_gradient(params_np, spsa_deltas[step])
            best_e = min(best_e, e)

            opt.zero_grad()
            model.theta.grad = torch.tensor(grad_np, dtype=torch.float32)
            opt.step(e)

            if step % 20 == 0:
                print(f"\r   Mobiu Step {step}/{N_STEPS} | E: {e:.4f} | Best: {best_e:.4f}", end="")

        mobiu_final = evaluate_qaoa(model.theta.detach().cpu().numpy(), shots=SHOTS * 4)
        print(f" | Final: {mobiu_final:.4f}")
        opt.end()

        # Per-seed result
        baseline_results.append(baseline_final)
        mobiu_results.append(mobiu_final)

        winner = "✅ Mobiu" if mobiu_final < baseline_final else "❌ Baseline"
        print(f"   SGD: {baseline_final:.4f} | SGD+Mobiu: {mobiu_final:.4f} → {winner}")

    # ============================================================
    # SUMMARY
    # ============================================================
    b_arr = np.array(baseline_results)
    m_arr = np.array(mobiu_results)

    b_mean = np.mean(b_arr)
    m_mean = np.mean(m_arr)

    improvement = (b_mean - m_mean) / abs(b_mean) * 100 if b_mean != 0 else 0
    wins = sum(m < b for m, b in zip(mobiu_results, baseline_results))

    try:
        _, p_value = stats.ttest_rel(mobiu_results, baseline_results)
    except:
        p_value = 1.0

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - FakeFez QAOA MaxCut (v4.2.0)")
    print("=" * 70)

    print(f"\n{'Optimizer':<20} {'Mean Cost':<15} {'Std':<12} {'Best':<12}")
    print("-" * 60)
    print(f"{'SGD (baseline)':<20} {b_mean:<15.4f} {np.std(b_arr):<12.4f} {min(b_arr):<12.4f}")
    print(f"{'SGD + Mobiu-Q':<20} {m_mean:<15.4f} {np.std(m_arr):<12.4f} {min(m_arr):<12.4f}")
    print("-" * 60)

    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{N_SEEDS} ({100*wins/N_SEEDS:.0f}%)")
    print(f"p-value: {p_value:.4f}")

    if improvement > 0:
        print(f"\n🎉 MOBIU-Q WINS by {improvement:.1f}%")
    else:
        print(f"\n❌ Baseline wins by {-improvement:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
