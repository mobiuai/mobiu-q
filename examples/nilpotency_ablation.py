#!/usr/bin/env python3
"""
================================================================================
THE REAL TEST (MATCHING ibm_fakefez.py EXACTLY)
================================================================================
Same settings as ibm_fakefez.py:
- reps=4, init=[-0.3, 0.3], Demeasurement.spsa
================================================================================
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except:
    from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend

from mobiu_q import MobiuQCore

# ============================================================
# FAKE SOFT ALGEBRA
# ============================================================

@dataclass 
class SoftNumberFake:
    """Regular multiplication (NOT nilpotent)"""
    soft: float
    real: float
    
    def __mul__(self, other):
        if isinstance(other, SoftNumberFake):
            a, b = self.soft, self.real
            c, d = other.soft, other.real
            # REGULAR: includes a*c!
            return SoftNumberFake(soft=a*c + a*d + b*c, real=b*d)
        return SoftNumberFake(self.soft*other, self.real*other)
    
    def __add__(self, other):
        return SoftNumberFake(self.soft + other.soft, self.real + other.real)
    
    def __rmul__(self, other):
        return self.__mul__(other)


class FakeSAOnMobiuAdam:
    """Fake SA on top of Mobiu-Q's Adam"""
    def __init__(self, license_key):
        self.mobiu_adam = MobiuQCore(
            license_key=license_key,
            method='standard',
            mode='hardware',
            use_soft_algebra=False,
            verbose=False
        )
        self.sa_state = SoftNumberFake(0.0, 0.0)
        self.energy_history = []
        self.soft_history = []
        
    def _get_signals(self):
        if len(self.energy_history) < 3:
            return 0.0, 0.0
        E = self.energy_history[-1]
        E1 = self.energy_history[-2]
        E2 = self.energy_history[-3]
        curv = abs(E - 2*E1 + E2)
        mean_E = abs(np.mean(self.energy_history[-3:]))
        a_t = curv / (curv + mean_E + 1e-9)
        a_t = max(0.0, min(10.0, a_t))
        b_t = (E1 - E) / (abs(E1) + 1e-9)
        b_t = max(0.0, min(1.0, b_t))
        return a_t, b_t
    
    def _compute_trust(self):
        denom = abs(self.sa_state.soft) + abs(self.sa_state.real) + 1e-9
        return (abs(self.sa_state.real) + 1e-9) / denom
    
    def _compute_soft_factor(self):
        factor = 1.0 + 0.1 * self.sa_state.soft
        return max(0.9, min(1.1, factor))
    
    def step(self, params, grad, energy):
        self.energy_history.append(energy)
        a_t, b_t = self._get_signals()
        delta = SoftNumberFake(a_t, b_t)
        gamma = 0.9
        self.sa_state = (gamma * self.sa_state) * delta + delta
        self.soft_history.append(self.sa_state.soft)
        
        trust = self._compute_trust()
        scale = max(0.5, min(2.0, 1.0 + 1.0 * trust))
        soft_factor = self._compute_soft_factor()
        
        warped_grad = grad * soft_factor
        scaled_grad = warped_grad * scale
        
        return self.mobiu_adam.step(params, scaled_grad, energy)
    
    def end(self):
        self.mobiu_adam.end()


# ============================================================
# SETUP - EXACTLY LIKE ibm_fakefez.py
# ============================================================

LICENSE_KEY = "YOUR_KEY"

NUM_STEPS = 60
NUM_SEEDS = 20
NUM_SHOTS = 4096
C_SHIFT = 0.1

H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -0.4804), 
    ("ZZ", 0.3435), 
    ("ZI", -0.4347),
    ("IZ", 0.5716), 
    ("XX", 0.0910), 
    ("YY", 0.0910)
])
GROUND_STATE_ENERGY = -1.846

print("=" * 70)
print("🧪 THE REAL TEST (MATCHING ibm_fakefez.py)")
print("=" * 70)
print("Settings: reps=4, init=[-0.3,0.3], Demeasurement.spsa")
print("=" * 70)

backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS
estimator.options.seed_simulator = 42

# ANSATZ - reps=4 like ibm_fakefez.py!
ansatz = EfficientSU2(2, reps=4, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = H2_HAMILTONIAN.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters

print(f"🔧 H2 VQE, FakeFez noise, {num_params} params (reps=4)")

# ============================================================
# ENERGY FUNCTION FOR DEMEASUREMENT.SPSA
# ============================================================

def evaluate_energy(params):
    job = estimator.run([(isa_ansatz, isa_ops, params)])
    return float(job.result()[0].data.evs)


def evaluate_energy(params):
    job = estimator.run([(isa_ansatz, isa_ops, params)])
    return float(job.result()[0].data.evs)


def get_batched_energy_and_gradient(params, delta):
    """Same delta for all three methods — fair comparison."""
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

# ============================================================
# EXPERIMENT
# ============================================================

print(f"\n🚀 Running {NUM_SEEDS} seeds x 3 methods...\n")

results = {
    'mobiu_sa_on': [],
    'mobiu_sa_off': [],
    'fake_sa': [],
}
soft_histories = []

for seed in range(NUM_SEEDS):
    print(f"Seed {seed + 1}/{NUM_SEEDS}: ", end="", flush=True)
    
    # Init — same params and deltas for all three methods
    np.random.seed(seed)
    init_params = np.random.uniform(-0.3, 0.3, num_params)

    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]
    
    # ─────────────────────────────────────────────────────────────────
    # Mobiu-Q (SA ON)
    # ─────────────────────────────────────────────────────────────────
    params = init_params.copy()
    opt = MobiuQCore(
        license_key=LICENSE_KEY,
        method='standard',
        mode='hardware',
        use_soft_algebra=True,
        verbose=False
    )
    best = float('inf')
    for step in range(NUM_STEPS):
        energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
        params = opt.step(params, grad, energy)
        best = min(best, energy)
    opt.end()
    e_sa_on = best
    print(f"SA_ON={e_sa_on:.3f} ", end="", flush=True)
    
    # ─────────────────────────────────────────────────────────────────
    # Baseline (SA OFF)
    # ─────────────────────────────────────────────────────────────────
    params = init_params.copy()
    opt = MobiuQCore(
        license_key=LICENSE_KEY,
        method='standard',
        mode='hardware',
        use_soft_algebra=False,
        verbose=False
    )
    best = float('inf')
    for step in range(NUM_STEPS):
        energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
        params = opt.step(params, grad, energy)
        best = min(best, energy)
    opt.end()
    e_sa_off = best
    print(f"SA_OFF={e_sa_off:.3f} ", end="", flush=True)
    
    # ─────────────────────────────────────────────────────────────────
    # Fake SA
    # ─────────────────────────────────────────────────────────────────
    params = init_params.copy()
    opt = FakeSAOnMobiuAdam(license_key=LICENSE_KEY)
    best = float('inf')
    for step in range(NUM_STEPS):
        energy, grad = get_batched_energy_and_gradient(params, spsa_deltas[step])
        params = opt.step(params, grad, energy)
        best = min(best, energy)
    opt.end()
    e_fake = best
    print(f"FAKE={e_fake:.3f}")
    
    results['mobiu_sa_on'].append(e_sa_on)
    results['mobiu_sa_off'].append(e_sa_off)
    results['fake_sa'].append(e_fake)
    soft_histories.append(opt.soft_history)

# ============================================================
# ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("📊 RESULTS")
print("=" * 70)

sa_on_mean = np.mean(results['mobiu_sa_on'])
sa_off_mean = np.mean(results['mobiu_sa_off'])
fake_mean = np.mean(results['fake_sa'])

sa_on_gap = abs(sa_on_mean - GROUND_STATE_ENERGY)
sa_off_gap = abs(sa_off_mean - GROUND_STATE_ENERGY)
fake_gap = abs(fake_mean - GROUND_STATE_ENERGY)

print(f"\n{'Method':<25} | {'Mean Energy':>12} | {'Gap (mHa)':>12}")
print("-" * 55)
print(f"{'Mobiu-Q (SA ON)':<25} | {sa_on_mean:>12.4f} | {sa_on_gap*1000:>12.2f}")
print(f"{'Baseline (SA OFF)':<25} | {sa_off_mean:>12.4f} | {sa_off_gap*1000:>12.2f}")
print(f"{'Fake SA (regular mult)':<25} | {fake_mean:>12.4f} | {fake_gap*1000:>12.2f}")
print(f"\n   Ground State: {GROUND_STATE_ENERGY:.4f}")

# Improvements
sa_imp = (sa_off_gap - sa_on_gap) / sa_off_gap * 100
fake_imp = (sa_off_gap - fake_gap) / sa_off_gap * 100

print(f"\n📈 Improvement vs Baseline:")
print(f"   Mobiu-Q (SA ON):   {sa_imp:+.1f}%")
print(f"   Fake SA:           {fake_imp:+.1f}%")

# Soft component
print("\n📊 SOFT COMPONENT (Fake SA):")
for i, hist in enumerate(soft_histories):
    if hist:
        print(f"   Seed {i+1}: max|soft|={max(abs(s) for s in hist):.3f}")

# Head-to-head
print("\n" + "=" * 70)
print("🥊 HEAD-TO-HEAD")
print("=" * 70)

sa_on_vs_off = sum(1 for on, off in zip(results['mobiu_sa_on'], results['mobiu_sa_off']) if on < off)
sa_on_vs_fake = sum(1 for on, fake in zip(results['mobiu_sa_on'], results['fake_sa']) if on < fake)
fake_vs_off = sum(1 for fake, off in zip(results['fake_sa'], results['mobiu_sa_off']) if fake < off)

print(f"\n   Mobiu-Q (SA ON) vs Baseline:  {sa_on_vs_off}/{NUM_SEEDS}")
print(f"   Mobiu-Q (SA ON) vs Fake SA:   {sa_on_vs_fake}/{NUM_SEEDS}")
print(f"   Fake SA vs Baseline:          {fake_vs_off}/{NUM_SEEDS}")

# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("💡 CONCLUSION")
print("=" * 70)

if sa_imp > 20 and fake_imp < 10:
    print(f"""
   ✅ NILPOTENT MATH (ε²=0) IS ESSENTIAL!
   
   Mobiu-Q (SA ON):   {sa_imp:+.1f}% improvement
   Fake SA (regular): {fake_imp:+.1f}% improvement
   
   Real SA works, Fake SA doesn't!
""")
elif sa_imp > fake_imp + 15:
    print(f"""
   ✅ NILPOTENT PROVIDES SIGNIFICANT ADVANTAGE!
   
   Mobiu-Q: {sa_imp:+.1f}% | Fake SA: {fake_imp:+.1f}%
   Difference: {sa_imp - fake_imp:.1f}%
""")
elif abs(sa_imp - fake_imp) < 10:
    print(f"""
   ⚠️ SIMILAR PERFORMANCE
   
   Mobiu-Q: {sa_imp:+.1f}% | Fake SA: {fake_imp:+.1f}%
""")
else:
    print(f"""
   🤔 MIXED RESULTS
   
   Mobiu-Q: {sa_imp:+.1f}% | Fake SA: {fake_imp:+.1f}%
""")

print("=" * 70)