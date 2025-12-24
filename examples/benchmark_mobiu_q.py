"""
Mobiu-Q Benchmark (v2.4)
========================
Validated benchmarks with exact ground states.

Usage:
    python benchmark_mobiu_q.py                  # 10 seeds
    python benchmark_mobiu_q.py --seeds 50       # 50 seeds
    python benchmark_mobiu_q.py --quick          # 3 seeds
"""

import numpy as np
import argparse
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
import time

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# PAULI MATRICES & GROUND STATE
# ============================================================================

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def pauli_tensor(pauli_str):
    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    result = paulis[pauli_str[0]]
    for p in pauli_str[1:]:
        result = np.kron(result, paulis[p])
    return result

def compute_ground_state(pauli_list):
    n_qubits = len(pauli_list[0][0])
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for pauli_str, coef in pauli_list:
        H += coef * pauli_tensor(pauli_str)
    return np.linalg.eigvalsh(H)[0].real

# ============================================================================
# PROBLEM DEFINITIONS
# ============================================================================

@dataclass
class Problem:
    name: str
    category: str
    energy_fn: Callable
    pauli_list: List[Tuple] = None
    n_params: int = 2
    n_steps: int = 60
    noise: float = 0.02
    init_range: float = np.pi
    
    def __post_init__(self):
        if self.pauli_list:
            self.ground_state = compute_ground_state(self.pauli_list)
        else:
            self.ground_state = 0.0

# Quantum Chemistry
H2_PAULI = [("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347), 
            ("IZ", 0.5716), ("XX", 0.0910), ("YY", 0.0910)]

def h2_energy(params, noise=0.0):
    coeffs = [-0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx + coeffs[5]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

LIH_PAULI = [("II", -0.25), ("ZZ", 0.17), ("ZI", -0.22), ("IZ", 0.12), ("XX", 0.08)]

def lih_energy(params, noise=0.0):
    coeffs = [-0.25, 0.17, -0.22, 0.12, 0.08]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

BEH2_PAULI = [("II", -0.35), ("ZZ", 0.25), ("ZI", -0.30), ("IZ", 0.20), ("XX", 0.12), ("YY", 0.12)]

def beh2_energy(params, noise=0.0):
    coeffs = [-0.35, 0.25, -0.30, 0.20, 0.12, 0.12]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx + coeffs[5]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

H3_PAULI = [("II", -0.40), ("ZZ", 0.30), ("ZI", -0.25), ("IZ", 0.35), ("XX", 0.15)]

def h3_energy(params, noise=0.0):
    coeffs = [-0.40, 0.30, -0.25, 0.35, 0.15]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

HE_PAULI = [("II", -2.0), ("ZZ", 0.5), ("ZI", -0.3), ("IZ", -0.3)]

def he_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    energy = -2.0 + 0.5*zz - 0.3*zi - 0.3*iz
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

# Condensed Matter
HEISENBERG_PAULI = [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]

def heisenberg_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    xx = 4 * c0 * s0 * c1 * s1
    yy = 4 * c0 * s0 * c1 * s1
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    energy = xx + yy + zz
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

TRANSVERSE_ISING_PAULI = [("ZZ", -1.0), ("XI", -0.5), ("IX", -0.5)]

def transverse_ising_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    x0 = 2 * c0 * s0
    x1 = 2 * c1 * s1
    energy = -zz - 0.5 * (x0 + x1)
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

XY_PAULI = [("XX", 1.0), ("YY", 1.0)]

def xy_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    xx = 4 * c0 * s0 * c1 * s1
    yy = 4 * c0 * s0 * c1 * s1
    energy = xx + yy
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

FERRO_PAULI = [("ZZ", -1.0)]

def ferro_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    energy = -zz
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

# Classical
def rosenbrock_energy(params, noise=0.0):
    x, y = params[0], params[1]
    energy = (1 - x)**2 + 100 * (y - x**2)**2
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

def rastrigin_energy(params, noise=0.0):
    n = len(params)
    energy = 10*n + np.sum(params**2 - 10*np.cos(2*np.pi*params))
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

def ackley_energy(params, noise=0.0):
    n = len(params)
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(2*np.pi*params))
    energy = -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

# Build catalog
def build_catalog() -> List[Problem]:
    return [
        # Quantum Chemistry
        Problem("H2 Molecule", "Quantum Chemistry", h2_energy, H2_PAULI),
        Problem("LiH Molecule", "Quantum Chemistry", lih_energy, LIH_PAULI),
        Problem("BeH2 Molecule", "Quantum Chemistry", beh2_energy, BEH2_PAULI),
        Problem("H3+ Chain", "Quantum Chemistry", h3_energy, H3_PAULI),
        Problem("He Atom", "Quantum Chemistry", he_energy, HE_PAULI),
        
        # Condensed Matter
        Problem("Heisenberg XXZ", "Condensed Matter", heisenberg_energy, HEISENBERG_PAULI),
        Problem("Transverse Ising", "Condensed Matter", transverse_ising_energy, TRANSVERSE_ISING_PAULI),
        Problem("XY Model", "Condensed Matter", xy_energy, XY_PAULI),
        Problem("Ferro Ising", "Condensed Matter", ferro_energy, FERRO_PAULI),
        
        # Classical
        Problem("Rosenbrock", "Classical", rosenbrock_energy, None, init_range=2.0, n_steps=100),
        Problem("Rastrigin", "Classical", rastrigin_energy, None, init_range=5.0, n_steps=100, noise=0.1),
        Problem("Ackley", "Classical", ackley_energy, None, init_range=5.0, n_steps=100),
    ]

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(problem: Problem, n_seeds: int, verbose: bool = True) -> Dict:
    from mobiu_q import MobiuQCore, Demeasurement
    
    if verbose:
        print(f"\n  {problem.name} (ground={problem.ground_state:.4f})...", end=" ", flush=True)
    
    adam_gaps = []
    mobiu_gaps = []
    
    for seed in range(n_seeds):
        # Pre-generate random values for fair comparison
        np.random.seed(seed * 1000)
        init_params = np.random.uniform(-problem.init_range, problem.init_range, problem.n_params)
        noise_vals = np.random.normal(0, 1, problem.n_steps * 4)
        
        # Adam (no Soft Algebra)
        np.random.seed(seed * 1000)
        adam = MobiuQCore(license_key="YOUR_LICENCE", method="vqe", use_soft_algebra=False)
        params = init_params.copy()
        clean_energies = []
        
        for step in range(problem.n_steps):
            grad = Demeasurement.finite_difference(
                lambda p: problem.energy_fn(p, 0), params  # Clean gradient!
            )
            energy = problem.energy_fn(params, problem.noise)
            clean_energies.append(problem.energy_fn(params, 0))
            params = adam.step(params, grad, energy)
        
        adam.end()
        adam_final = np.mean(clean_energies[-10:])
        adam_gaps.append(adam_final - problem.ground_state)
        
        # Mobiu (with Soft Algebra)
        np.random.seed(seed * 1000)
        mobiu = MobiuQCore(license_key="YOUR_LICENCE", method="vqe", use_soft_algebra=True)
        params = init_params.copy()
        clean_energies = []
        
        for step in range(problem.n_steps):
            grad = Demeasurement.finite_difference(
                lambda p: problem.energy_fn(p, 0), params  # Clean gradient!
            )
            energy = problem.energy_fn(params, problem.noise)
            clean_energies.append(problem.energy_fn(params, 0))
            params = mobiu.step(params, grad, energy)
        
        mobiu.end()
        mobiu_final = np.mean(clean_energies[-10:])
        mobiu_gaps.append(mobiu_final - problem.ground_state)
    
    # Statistics
    adam_mean = np.mean(adam_gaps)
    mobiu_mean = np.mean(mobiu_gaps)
    improvement = (adam_mean - mobiu_mean) / abs(adam_mean) * 100 if adam_mean != 0 else 0
    wins = sum(m < a for m, a in zip(mobiu_gaps, adam_gaps))
    
    if HAS_SCIPY and n_seeds >= 3:
        _, p_val = stats.ttest_rel(mobiu_gaps, adam_gaps)
    else:
        p_val = 1.0
    
    if verbose:
        sig = "✅" if p_val < 0.05 and improvement > 0 else "❌" if p_val < 0.05 and improvement < 0 else "🔶"
        print(f"{improvement:+.1f}% ({wins}/{n_seeds}) {sig}")
    
    return {
        'name': problem.name,
        'category': problem.category,
        'ground_state': problem.ground_state,
        'adam_gap': adam_mean,
        'mobiu_gap': mobiu_mean,
        'improvement': improvement,
        'wins': wins,
        'total': n_seeds,
        'p_val': p_val,
        'significant': p_val < 0.05
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Mobiu-Q Benchmark')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    n_seeds = 3 if args.quick else args.seeds
    
    print("\n" + "🔬 MOBIU-Q BENCHMARK 🔬".center(70))
    print("=" * 70)
    print(f"Configuration: {n_seeds} seeds per problem")
    print("=" * 70)
    
    problems = build_catalog()
    results = []
    
    # Group by category
    categories = {}
    for p in problems:
        if p.category not in categories:
            categories[p.category] = []
        categories[p.category].append(p)
    
    for category, cat_problems in categories.items():
        print(f"\n📌 {category}")
        print("-" * 50)
        
        for problem in cat_problems:
            try:
                result = run_benchmark(problem, n_seeds)
                results.append(result)
            except Exception as e:
                print(f"\n  ❌ {problem.name} failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY".center(70))
    print("=" * 70)
    
    print(f"\n{'Problem':<25} {'Ground':<10} {'Adam Gap':<12} {'Mobiu Gap':<12} {'Improve':<10} {'Wins'}")
    print("-" * 85)
    
    for r in results:
        sig = "✅" if r['significant'] and r['improvement'] > 0 else ""
        print(f"{r['name']:<25} {r['ground_state']:<10.4f} {r['adam_gap']:<12.4f} "
              f"{r['mobiu_gap']:<12.4f} {r['improvement']:>+8.1f}% {r['wins']:>3}/{r['total']:<3} {sig}")
    
    # Overall
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_win_rate = np.mean([r['wins']/r['total']*100 for r in results])
    n_sig_wins = sum(1 for r in results if r['significant'] and r['improvement'] > 0)
    n_sig_losses = sum(1 for r in results if r['significant'] and r['improvement'] < 0)
    
    print("-" * 85)
    print(f"\nAverage Improvement: {avg_improvement:+.1f}%")
    print(f"Average Win Rate:    {avg_win_rate:.0f}%")
    print(f"Significant Wins:    {n_sig_wins}/{len(results)}")
    print(f"Significant Losses:  {n_sig_losses}/{len(results)}")
    
    print("\n" + "=" * 70)
    if n_sig_wins > n_sig_losses:
        print("🎉 SOFT ALGEBRA WINS!")
    elif n_sig_losses > n_sig_wins:
        print("❌ Plain optimizer wins")
    else:
        print("🔶 Results mixed - increase seeds for clarity")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()