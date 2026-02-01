#!/usr/bin/env python3
"""
================================================================================
🔬 QAOA MaxCut - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimizer (what customer has BEFORE Mobiu-Q)
- Test: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

QAOA MaxCut on FakeFez:
- Real quantum hardware noise model
- Combinatorial optimization benchmark
- Tests Mobiu-Q on rugged landscapes
================================================================================
"""

import numpy as np
from scipy import stats
from datetime import datetime

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

from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

N_QUBITS = 5
P = 5  # QAOA layers
N_STEPS = 100
N_SEEDS = 10
SHOTS = 4096
C_SHIFT = 0.1
LR = 0.1  # Learning rate for pure SPSA
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

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
n_params = 2 * P


# ============================================================
# QAOA CIRCUIT
# ============================================================

def create_qaoa_circuit(params, n_qubits, edges, p):
    """Create QAOA circuit for MaxCut"""
    gammas = params[:p]
    betas = params[p:]
    
    qc = QuantumCircuit(n_qubits)
    
    # Initial superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # QAOA layers
    for layer in range(p):
        for i, j in edges:
            qc.rzz(2 * gammas[layer], i, j)
        for i in range(n_qubits):
            qc.rx(2 * betas[layer], i)
    
    return qc


def get_maxcut_cost(bitstring, edges):
    """Calculate MaxCut cost for a bitstring"""
    cost = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cost += 1
    return cost


def evaluate_qaoa(params, n_qubits, edges, p, shots=SHOTS):
    """Evaluate QAOA circuit on FakeFez backend"""
    qc = create_qaoa_circuit(params, n_qubits, edges, p)
    qc.measure_all()
    qc_transpiled = pm.run(qc)
    
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    total_cost = 0
    total_shots = 0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        cost = get_maxcut_cost(bits, edges)
        total_cost += cost * count
        total_shots += count
    
    # Return negative (we minimize, but want max cut)
    return -total_cost / total_shots


def spsa_gradient(params, delta):
    """SPSA gradient estimation"""
    e_plus = evaluate_qaoa(params + C_SHIFT * delta, N_QUBITS, EDGES, P)
    e_minus = evaluate_qaoa(params - C_SHIFT * delta, N_QUBITS, EDGES, P)
    e_center = evaluate_qaoa(params, N_QUBITS, EDGES, P)
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    return e_center, grad


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔬 QAOA MaxCut - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Graph: {N_QUBITS} qubits, {len(EDGES)} edges: {EDGES}")
    print(f"QAOA layers: p={P} | Steps: {N_STEPS} | Seeds: {N_SEEDS}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure SPSA optimization (NO Mobiu)")
    print("  • Test: SPSA + Mobiu-Q enhancement")
    print("=" * 70)
    
    spsa_results = []
    mobiu_results = []
    
    for seed in range(N_SEEDS):
        print(f"\n{'─'*60}")
        print(f"  SEED {seed}")
        print(f"{'─'*60}")
        
        # Initialize
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Pre-generate SPSA deltas for fairness
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(N_STEPS)]
        
        # ─────────────────────────────────────────────────────────────────
        # BASELINE: Pure SPSA (what customer has BEFORE adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        spsa_best = float('inf')
        
        for step in range(N_STEPS):
            e, grad = spsa_gradient(params, spsa_deltas[step])
            spsa_best = min(spsa_best, e)
            
            # Pure SPSA update with decay
            ak = LR / ((step + 1) ** 0.602)
            params = params - ak * grad
            
            if step % 20 == 0:
                print(f"\r   SPSA  Step {step}/{N_STEPS} | E: {e:.4f} | Best: {spsa_best:.4f}", end="")
        
        spsa_final = evaluate_qaoa(params, N_QUBITS, EDGES, P, shots=SHOTS * 4)
        print(f" | Final: {spsa_final:.4f}")
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            base_lr=LR,
            base_optimizer='SGD',  # Critical for SPSA!
            verbose=False
        )
        mobiu_best = float('inf')
        
        for step in range(N_STEPS):
            e, grad = spsa_gradient(params, spsa_deltas[step])
            mobiu_best = min(mobiu_best, e)
            params = mobiu_opt.step(params, grad, e)
            
            if step % 20 == 0:
                print(f"\r   Mobiu Step {step}/{N_STEPS} | E: {e:.4f} | Best: {mobiu_best:.4f}", end="")
        
        mobiu_opt.end()
        mobiu_final = evaluate_qaoa(params, N_QUBITS, EDGES, P, shots=SHOTS * 4)
        print(f" | Final: {mobiu_final:.4f}")
        
        spsa_results.append(spsa_final)
        mobiu_results.append(mobiu_final)
        
        # Per-seed result
        winner = "✅ Mobiu" if mobiu_final < spsa_final else "❌ SPSA"
        print(f"   Pure SPSA: {spsa_final:.4f} | + Mobiu: {mobiu_final:.4f} → {winner}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    spsa_arr = np.array(spsa_results)
    mobiu_arr = np.array(mobiu_results)
    
    spsa_mean = np.mean(spsa_arr)
    mobiu_mean = np.mean(mobiu_arr)
    
    # For MaxCut, lower (more negative) is better
    improvement = (spsa_mean - mobiu_mean) / abs(spsa_mean) * 100 if spsa_mean != 0 else 0
    wins = sum(m < s for m, s in zip(mobiu_results, spsa_results))
    
    try:
        _, p_value = stats.ttest_rel(mobiu_results, spsa_results)
    except:
        p_value = 1.0
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - FakeFez QAOA MaxCut")
    print("=" * 70)
    
    print(f"\n{'Optimizer':<15} {'Mean Cost':<15} {'Std':<12} {'Best':<12}")
    print("-" * 55)
    print(f"{'Pure SPSA':<15} {spsa_mean:<15.4f} {np.std(spsa_arr):<12.4f} {min(spsa_arr):<12.4f}")
    print(f"{'SPSA + Mobiu':<15} {mobiu_mean:<15.4f} {np.std(mobiu_arr):<12.4f} {min(mobiu_arr):<12.4f}")
    print("-" * 55)
    
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{N_SEEDS} ({100*wins/N_SEEDS:.0f}%)")
    print(f"p-value: {p_value:.4f}")
    
    if improvement > 0:
        print(f"\n🎉 MOBIU-Q WINS by {improvement:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
