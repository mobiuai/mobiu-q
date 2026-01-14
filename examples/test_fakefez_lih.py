#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q REAL TEST - FakeFez VQE for LiH
================================================================================
"""

import numpy as np
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

LICENSE_KEY = "YOUR_KEY"
NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.005
METHOD = "standard"

backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS

hamiltonian = SparsePauliOp.from_list([
    ("IIII", -7.4983), ("IIIZ", 0.3916), ("IIZI", -0.3916),
    ("IIZZ", 0.1811), ("ZZII", 0.1208), ("XXII", 0.0453), ("YYII", 0.0453),
    ("IIXX", 0.0453), ("IIYY", 0.0453), ("XXXX", 0.0057), ("YYYY", 0.0057)
])
exact_energy = -7.8823

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

def main():
    print("=" * 70)
    print("🧬 MOBIU-Q - FakeFez VQE LiH")
    print("=" * 70)
    
    baseline_results, mobiu_results = [], []
    
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]
        
        params = init_params.copy()
        baseline_opt = MobiuQCore(LICENSE_KEY, method=METHOD, mode='hardware', use_soft_algebra=False, verbose=False)
        b_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            params = baseline_opt.step(params, g, e)
            b_best = min(b_best, e)
        baseline_opt.end()
        
        params = init_params.copy()
        mobiu_opt = MobiuQCore(LICENSE_KEY, method=METHOD, mode='hardware', use_soft_algebra=True, verbose=False)
        m_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            params = mobiu_opt.step(params, g, e)
            m_best = min(m_best, e)
        mobiu_opt.end()
        
        b_gap, m_gap = abs(b_best - exact_energy), abs(m_best - exact_energy)
        winner = "✅ Mobiu" if m_gap < b_gap else "❌ Baseline"
        print(f"    Baseline: {b_best:.4f} (gap={b_gap:.4f}) | Mobiu: {m_best:.4f} (gap={m_gap:.4f}) → {winner}")
        baseline_results.append(b_best)
        mobiu_results.append(m_best)

    b_mean = np.mean(np.abs(np.array(baseline_results) - exact_energy))
    m_mean = np.mean(np.abs(np.array(mobiu_results) - exact_energy))
    imp = (b_mean - m_mean) / b_mean * 100
    print(f"\n  Improvement: {imp:+.1f}%")
    with open('fakefez_lih_results.json', 'w') as f:
        json.dump({'molecule': 'LiH', 'method': METHOD, 'improvement': imp}, f)

if __name__ == "__main__":
    main()
