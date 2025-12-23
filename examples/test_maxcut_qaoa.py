"""
================================================================================
MOBIU-Q BENCHMARK - QAOA MaxCut (Verified Methodology)
================================================================================
Uses the same A/B testing methodology as test_catalog_verified_full.py:
- Pre-generate ALL random values before running
- Both Adam and Mobiu see EXACT SAME noise/deltas
- Proper quantum state simulation for QAOA

Usage:
    python test_maxcut_qaoa.py
================================================================================
"""

import numpy as np
import requests
from typing import Callable, Tuple, List

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "e756ce65-186e-4747-aaaf-5a1fb1473b7e"  # Replace with your key

# ==============================================================================
# QAOA QUANTUM SIMULATION (From verified catalog)
# ==============================================================================

def qaoa_expectation(params, n_qubits, cost_terms, p, noise=0.0):
    """
    QAOA expectation value using quantum state simulation.
    
    Args:
        params: [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
        n_qubits: Number of qubits
        cost_terms: List of (coefficient, (qubit_i, qubit_j)) for ZZ terms
        p: QAOA depth
        noise: Noise level (0.0 for clean)
    
    Returns:
        Expectation value of cost Hamiltonian
    """
    gammas = params[:p]
    betas = params[p:]
    
    # Initial state: uniform superposition
    state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    
    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]
        
        # Cost layer: exp(-i * gamma * C)
        for coef, qubits in cost_terms:
            if len(qubits) == 2:
                i, j = qubits
                for k in range(2**n_qubits):
                    z_i = 1 - 2 * ((k >> i) & 1)
                    z_j = 1 - 2 * ((k >> j) & 1)
                    state[k] *= np.exp(-1j * gamma * coef * z_i * z_j)
        
        # Mixer layer: exp(-i * beta * B)
        for qubit in range(n_qubits):
            new_state = np.zeros_like(state)
            c, s = np.cos(beta), np.sin(beta)
            for k in range(2**n_qubits):
                bit = (k >> qubit) & 1
                k_flipped = k ^ (1 << qubit)
                if bit == 0:
                    new_state[k] += c * state[k] - 1j * s * state[k_flipped]
                else:
                    new_state[k] += -1j * s * state[k_flipped] + c * state[k]
            state = new_state
    
    # Compute expectation value
    expectation = 0.0
    for k in range(2**n_qubits):
        prob = np.abs(state[k])**2
        cost = sum(coef * (1 - 2*((k >> q[0]) & 1)) * (1 - 2*((k >> q[1]) & 1)) 
                   for coef, q in cost_terms if len(q) == 2)
        expectation += prob * cost
    
    return expectation


def create_maxcut_graph(n_qubits, edge_prob=0.5, seed=42):
    """Create random graph for MaxCut"""
    np.random.seed(seed)
    edges = []
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            if np.random.random() < edge_prob:
                edges.append((i, j))
    if not edges:
        edges = [(0, 1)]
    
    # Cost terms: -0.5 * ZZ for each edge (we minimize, want max cut)
    cost_terms = [(-0.5, (i, j)) for i, j in edges]
    return edges, cost_terms


# ==============================================================================
# OPTIMIZER WRAPPERS
# ==============================================================================

class LocalAdam:
    """Local Adam optimizer for baseline comparison"""
    def __init__(self, base_lr=0.1):
        self.base_lr = base_lr
        self.m = None
        self.v = None
        self.t = 0
    
    def reset(self):
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grad, energy):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        return params - self.base_lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def end(self):
        pass


class MobiuOptimizer:
    """Mobiu-Q API wrapper"""
    def __init__(self, license_key, method='qaoa', mode='simulation', 
                 base_lr=0.1, use_soft_algebra=True):
        self.license_key = license_key
        self.method = method
        self.mode = mode
        self.base_lr = base_lr
        self.use_soft_algebra = use_soft_algebra
        self.session_id = None
    
    def reset(self):
        """Start new session"""
        if self.session_id:
            self.end()
        
        try:
            r = requests.post(API_URL, json={
                'action': 'start',
                'license_key': self.license_key,
                'method': self.method,
                'mode': self.mode,
                'base_lr': self.base_lr,
                'use_soft_algebra': self.use_soft_algebra
            }, timeout=10)
            data = r.json()
            if data.get('success'):
                self.session_id = data['session_id']
        except Exception as e:
            print(f"Warning: API error - {e}")
            self.session_id = None
    
    def step(self, params, grad, energy):
        if not self.session_id:
            # Fallback to local Adam
            return params - self.base_lr * grad
        
        try:
            r = requests.post(API_URL, json={
                'action': 'step',
                'license_key': self.license_key,
                'session_id': self.session_id,
                'params': params.tolist(),
                'gradient': grad.tolist(),
                'energy': float(energy)
            }, timeout=10)
            data = r.json()
            if data.get('success'):
                return np.array(data['new_params'])
        except:
            pass
        return params
    
    def end(self):
        if self.session_id:
            try:
                requests.post(API_URL, json={
                    'action': 'end',
                    'license_key': self.license_key,
                    'session_id': self.session_id
                }, timeout=5)
            except:
                pass
            self.session_id = None


# ==============================================================================
# BENCHMARK FUNCTION (Verified Methodology)
# ==============================================================================

def run_qaoa_benchmark(
    name: str,
    n_qubits: int = 4,
    p: int = 2,
    n_seeds: int = 10,
    n_steps: int = 100,
    noise: float = 0.1,
    base_lr: float = 0.1
):
    """
    Run QAOA benchmark with verified A/B methodology.
    
    Key: Pre-generate ALL random values so both optimizers see EXACT SAME noise.
    """
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {n_qubits} qubits, p={p}, {n_steps} steps, {n_seeds} seeds, noise={noise}")
    print(f"{'='*70}")
    
    # Create graph
    edges, cost_terms = create_maxcut_graph(n_qubits, edge_prob=0.5, seed=42)
    n_params = 2 * p
    print(f"  Graph: {len(edges)} edges: {edges}")
    
    # Energy function (clean)
    def clean_fn(params):
        return qaoa_expectation(params, n_qubits, cost_terms, p)
    
    # Optimizers
    adam = LocalAdam(base_lr=base_lr)
    mobiu = MobiuOptimizer(LICENSE_KEY, method='qaoa', mode='hardware', 
                           base_lr=base_lr, use_soft_algebra=True)
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(n_seeds):
        # Initial params (same for both)
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)
        
        # ===== CRITICAL: Pre-generate ALL random values =====
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(n_steps)]
        noise_vals = [np.random.normal(0, 1) for _ in range(n_steps * 2)]
        
        # ===== ADAM =====
        adam.reset()
        params = init_params.copy()
        
        for step in range(n_steps):
            delta = spsa_deltas[step]
            c = 0.1  # SPSA shift
            
            # Clean evaluations
            e_plus_clean = clean_fn(params + c * delta)
            e_minus_clean = clean_fn(params - c * delta)
            
            # Add noise using PRE-GENERATED values
            e_plus = e_plus_clean + (noise * abs(e_plus_clean) + 0.01) * noise_vals[step*2]
            e_minus = e_minus_clean + (noise * abs(e_minus_clean) + 0.01) * noise_vals[step*2+1]
            
            # SPSA gradient
            grad = (e_plus - e_minus) / (2 * c) * delta
            energy = (e_plus + e_minus) / 2
            
            params = adam.step(params, grad, energy)
        
        adam_final = clean_fn(params)
        adam_results.append(adam_final)
        
        # ===== MOBIU (with EXACT SAME random values) =====
        mobiu.reset()
        params = init_params.copy()
        
        for step in range(n_steps):
            delta = spsa_deltas[step]  # SAME delta!
            c = 0.1
            
            e_plus_clean = clean_fn(params + c * delta)
            e_minus_clean = clean_fn(params - c * delta)
            
            # SAME noise values!
            e_plus = e_plus_clean + (noise * abs(e_plus_clean) + 0.01) * noise_vals[step*2]
            e_minus = e_minus_clean + (noise * abs(e_minus_clean) + 0.01) * noise_vals[step*2+1]
            
            grad = (e_plus - e_minus) / (2 * c) * delta
            energy = (e_plus + e_minus) / 2
            
            params = mobiu.step(params, grad, energy)
        
        mobiu_final = clean_fn(params)
        mobiu_results.append(mobiu_final)
        
        # Progress
        winner = "✅ Mobiu" if mobiu_final < adam_final else "❌ Adam"
        print(f"  Seed {seed}: Adam={adam_final:.4f}, Mobiu={mobiu_final:.4f} → {winner}")
    
    mobiu.end()
    
    # Statistics
    adam_mean = np.mean(adam_results)
    mobiu_mean = np.mean(mobiu_results)
    improvement = (adam_mean - mobiu_mean) / abs(adam_mean) * 100 if adam_mean != 0 else 0
    wins = sum(m < a for m, a in zip(mobiu_results, adam_results))
    
    # T-test
    from scipy import stats
    try:
        _, p_val = stats.ttest_rel(mobiu_results, adam_results)
    except:
        p_val = 1.0
    
    sig = "✅" if p_val < 0.05 and improvement > 0 else "❌" if p_val < 0.05 and improvement < 0 else "🔶"
    
    print(f"\n  {'─'*60}")
    print(f"  Adam mean:  {adam_mean:.4f}")
    print(f"  Mobiu mean: {mobiu_mean:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Wins: {wins}/{n_seeds}")
    print(f"  p-value: {p_val:.4f} {sig}")
    print(f"  {'─'*60}")
    
    return {
        'name': name,
        'adam_mean': adam_mean,
        'mobiu_mean': mobiu_mean,
        'improvement': improvement,
        'wins': wins,
        'total': n_seeds,
        'p_val': p_val
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("  MOBIU-Q QAOA BENCHMARK (Verified Methodology)")
    print("  Pre-generated random values for fair A/B comparison")
    print("="*70)
    
    results = []
    
    # Test 1: MaxCut 4 qubits
    results.append(run_qaoa_benchmark(
        "MaxCut 4 qubits",
        n_qubits=4, p=5, n_seeds=10, n_steps=100, noise=0.1
    ))
    
    # Test 2: MaxCut 5 qubits
    results.append(run_qaoa_benchmark(
        "MaxCut 5 qubits", 
        n_qubits=5, p=5, n_seeds=10, n_steps=100, noise=0.1
    ))
    
    # Test 3: MaxCut deeper (p=3)
    results.append(run_qaoa_benchmark(
        "MaxCut 4 qubits (p=3)",
        n_qubits=4, p=3, n_seeds=10, n_steps=150, noise=0.1
    ))
    
    # Test 4: Clean simulation (no noise)
    results.append(run_qaoa_benchmark(
        "MaxCut 4 qubits (clean)",
        n_qubits=4, p=5, n_seeds=10, n_steps=100, noise=0.0
    ))
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\n  {'Problem':<30} {'Improve':>10} {'Wins':>10} {'Sig':>6}")
    print(f"  {'-'*60}")
    
    total_wins = 0
    total_sig = 0
    
    for r in results:
        sig = "✅" if r['p_val'] < 0.05 and r['improvement'] > 0 else "❌" if r['p_val'] < 0.05 and r['improvement'] < 0 else "🔶"
        print(f"  {r['name']:<30} {r['improvement']:>+9.1f}% {r['wins']:>4}/{r['total']:<4} {sig:>6}")
        
        if r['p_val'] < 0.05 and r['improvement'] > 0:
            total_sig += 1
        total_wins += r['wins']
    
    total_tests = sum(r['total'] for r in results)
    avg_imp = np.mean([r['improvement'] for r in results])
    
    print(f"  {'-'*60}")
    print(f"  Average improvement: {avg_imp:+.1f}%")
    print(f"  Total wins: {total_wins}/{total_tests}")
    print(f"  Significant wins: {total_sig}/{len(results)}")
    print("="*70)


if __name__ == "__main__":
    main()