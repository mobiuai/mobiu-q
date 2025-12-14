"""
MOBIU-Q CUSTOMER VALIDATION
============================

This script tests Mobiu-Q vs Adam on real quantum problems.
Run this to see how much Mobiu-Q saves you in hardware costs.

Requirements:
    pip install mobiu-q numpy

Usage:
    python customer_validation.py
"""

import numpy as np
from mobiu_q import MobiuQCore, Demeasurement

# ==============================================================================
# 👇 PUT YOUR LICENSE KEY HERE 👇
# ==============================================================================
LICENSE_KEY = "YOUR_LICENSE_KEY_HERE"
# ==============================================================================


# ==============================================================================
# LOCAL ADAM (for fair comparison)
# ==============================================================================

class Adam:
    """Pure Adam optimizer (local, no API)"""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.t = 0
        self.m = None
        self.v = None
    
    def reset(self):
        self.t = 0
        self.m = None
        self.v = None
    
    def step(self, params, gradient, energy):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * (gradient ** 2)
        
        m_hat = self.m / (1 - 0.9 ** self.t)
        v_hat = self.v / (1 - 0.999 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    def end(self):
        pass


# ==============================================================================
# TEST PROBLEMS
# ==============================================================================

def create_h2_vqe():
    """H2 molecule VQE - standard quantum chemistry benchmark"""
    g = [-0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    H = np.real(
        g[0] * np.kron(I, I) + g[1] * np.kron(I, Z) + g[2] * np.kron(Z, I) +
        g[3] * np.kron(Z, Z) + g[4] * np.kron(X, X) + g[5] * np.kron(Y, Y)
    )
    ground = np.linalg.eigvalsh(H)[0]
    
    def energy_fn(params):
        def Ry(t):
            c, s = np.cos(t/2), np.sin(t/2)
            return np.array([[c, -s], [s, c]])
        
        state = np.array([1, 0, 0, 0], dtype=complex)
        state = np.kron(Ry(params[0]), Ry(params[1])) @ state
        CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
        state = CNOT @ state
        state = np.kron(Ry(params[2]), Ry(params[3])) @ state
        return np.real(state.conj() @ H @ state)
    
    return energy_fn, ground, 4


def qaoa_expectation_ising(params, n_qubits, cost_terms, p, noise_level=0.0):
    """QAOA with Ising formulation - exact original implementation"""
    gammas = params[:p]
    betas = params[p:]
    state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        for coef, qubits in cost_terms:
            if len(qubits) == 2:
                i, j = qubits
                for k in range(2**n_qubits):
                    bit_i = (k >> i) & 1
                    bit_j = (k >> j) & 1
                    z_i = 1 - 2 * bit_i
                    z_j = 1 - 2 * bit_j
                    state[k] *= np.exp(-1j * gamma * coef * z_i * z_j)
            elif len(qubits) == 1:
                i = qubits[0]
                for k in range(2**n_qubits):
                    bit_i = (k >> i) & 1
                    z_i = 1 - 2 * bit_i
                    state[k] *= np.exp(-1j * gamma * coef * z_i)
        
        for qubit in range(n_qubits):
            new_state = np.zeros_like(state)
            c = np.cos(beta)
            s = np.sin(beta)
            for k in range(2**n_qubits):
                bit = (k >> qubit) & 1
                k_flipped = k ^ (1 << qubit)
                if bit == 0:
                    new_state[k] += c * state[k] - 1j * s * state[k_flipped]
                else:
                    new_state[k] += -1j * s * state[k_flipped] + c * state[k]
            state = new_state
    
    expectation = 0.0
    for k in range(2**n_qubits):
        prob = np.abs(state[k])**2
        cost = 0.0
        for coef, qubits in cost_terms:
            if len(qubits) == 2:
                i, j = qubits
                bit_i = (k >> i) & 1
                bit_j = (k >> j) & 1
                z_i = 1 - 2 * bit_i
                z_j = 1 - 2 * bit_j
                cost += coef * z_i * z_j
            elif len(qubits) == 1:
                i = qubits[0]
                bit_i = (k >> i) & 1
                z_i = 1 - 2 * bit_i
                cost += coef * z_i
        expectation += prob * cost
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * abs(expectation) + 0.01)
        expectation += noise
    
    return expectation


def create_maxcut_problem(n_qubits, edge_prob=0.5, seed=None):
    """Create MaxCut problem in Ising formulation"""
    if seed is not None:
        np.random.seed(seed)
    
    edges = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if np.random.random() < edge_prob:
                edges.append((i, j))
    
    if len(edges) == 0:
        edges = [(0, 1)]
    
    cost_terms = []
    for i, j in edges:
        cost_terms.append((-0.5, (i, j)))
    
    return cost_terms, edges


# ==============================================================================
# TEST 1: VQE - H2 Molecule (60 steps)
# ==============================================================================

def test_vqe(license_key, n_seeds=10):
    """Test: What accuracy after 60 iterations?"""
    
    print("=" * 70)
    print("TEST 1: VQE - H2 Molecule (60 steps)")
    print("=" * 70)
    print("\nGoal: Best accuracy with 60 iterations")
    print("Ground state: -1.8512 Ha\n")
    
    energy_fn, ground, n_params = create_h2_vqe()
    n_steps = 60
    
    adam_gaps = []
    mobiu_gaps = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        init = np.random.uniform(-np.pi, np.pi, n_params)
        
        # --- ADAM (local) ---
        opt_adam = Adam(lr=0.01)
        
        params = init.copy()
        for step in range(n_steps):
            energy = energy_fn(params)
            gradient = Demeasurement.finite_difference(energy_fn, params)
            params = opt_adam.step(params, gradient, energy)
        
        adam_gaps.append(energy_fn(params) - ground)
        
        # --- MOBIU-Q (API) ---
        opt_mobiu = MobiuQCore(
            license_key=license_key,
            mode="standard",
            problem="vqe"
        )
        
        params = init.copy()
        for step in range(n_steps):
            energy = energy_fn(params)
            gradient = Demeasurement.finite_difference(energy_fn, params)
            params = opt_mobiu.step(params, gradient, energy)
        
        opt_mobiu.end()
        mobiu_gaps.append(energy_fn(params) - ground)
        
        print(f"  Seed {seed+1}/{n_seeds}: Adam gap={adam_gaps[-1]:.6f}, Mobiu-Q gap={mobiu_gaps[-1]:.6f}")
    
    adam_gaps = np.array(adam_gaps)
    mobiu_gaps = np.array(mobiu_gaps)
    
    print(f"\n{'─' * 50}")
    print(f"{'Optimizer':<12} {'Mean Gap':<12} {'Best':<12} {'Std':<10}")
    print(f"{'─' * 50}")
    print(f"{'Adam':<12} {adam_gaps.mean():<12.6f} {adam_gaps.min():<12.6f} {adam_gaps.std():<10.6f}")
    print(f"{'Mobiu-Q':<12} {mobiu_gaps.mean():<12.6f} {mobiu_gaps.min():<12.6f} {mobiu_gaps.std():<10.6f}")
    
    improvement = (adam_gaps.mean() - mobiu_gaps.mean()) / adam_gaps.mean() * 100
    
    print(f"\n📊 IMPROVEMENT: {improvement:.1f}% better accuracy with Mobiu-Q")
    
    return adam_gaps, mobiu_gaps


# ==============================================================================
# TEST 2: QAOA - MaxCut Ising (ORIGINAL SETTINGS)
# ==============================================================================

def test_qaoa(license_key, n_seeds=10):
    """Test: QAOA MaxCut with Ising formulation (original settings)"""
    
    print("\n" + "=" * 70)
    print("TEST 2: QAOA - MaxCut Ising (ORIGINAL SETTINGS)")
    print("=" * 70)
    print("\n5 qubits, p=5, 150 steps, noise=10%, lr=0.1\n")
    
    N_QUBITS = 5
    P = 5
    N_STEPS = 150
    NOISE = 0.1
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(n_seeds):
        cost_terms, edges = create_maxcut_problem(N_QUBITS, edge_prob=0.5, seed=seed * 100 + 42)
        
        # Energy function WITH noise (for optimization)
        def energy_fn(params):
            return qaoa_expectation_ising(params, N_QUBITS, cost_terms, P, NOISE)
        
        # Energy function WITHOUT noise (for final evaluation)
        def energy_fn_clean(params):
            return qaoa_expectation_ising(params, N_QUBITS, cost_terms, P, 0.0)
        
        np.random.seed(seed)
        init = np.random.uniform(-np.pi, np.pi, 2 * P)
        
        # --- ADAM (local, lr=0.1) ---
        np.random.seed(seed * 1000 + 1)  # Consistent SPSA seed for Adam
        opt_adam = Adam(lr=0.1)
        params = init.copy()
        for _ in range(N_STEPS):
            # SPSA gradient - same as original
            delta = np.random.choice([-1, 1], size=len(params))
            E_plus = energy_fn(params + 0.1 * delta)
            E_minus = energy_fn(params - 0.1 * delta)
            grad = (E_plus - E_minus) / 0.2 * delta
            energy = (E_plus + E_minus) / 2
            params = opt_adam.step(params, grad, energy)
        adam_results.append(energy_fn_clean(params))
        
        # --- MOBIU-Q (API, lr=0.1) ---
        np.random.seed(seed * 1000 + 1)  # SAME SPSA seed as Adam!
        opt_mobiu = MobiuQCore(
            license_key=license_key,
            mode="noisy",
            problem="qaoa",
            base_lr=0.1
        )
        
        params = init.copy()
        for step in range(N_STEPS):
            # SPSA gradient - same as Adam
            delta = np.random.choice([-1, 1], size=len(params))
            E_plus = energy_fn(params + 0.1 * delta)
            E_minus = energy_fn(params - 0.1 * delta)
            grad = (E_plus - E_minus) / 0.2 * delta
            energy = (E_plus + E_minus) / 2
            params = opt_mobiu.step(params, grad, energy)
        
        opt_mobiu.end()
        mobiu_results.append(energy_fn_clean(params))
        
        print(f"  Seed {seed+1}/{n_seeds}: Adam={adam_results[-1]:.4f}, Mobiu-Q={mobiu_results[-1]:.4f}")
    
    adam_results = np.array(adam_results)
    mobiu_results = np.array(mobiu_results)
    
    adam_mean = np.mean(adam_results)
    mobiu_mean = np.mean(mobiu_results)
    
    # Lower is better for QAOA energy
    improvement = (adam_mean - mobiu_mean) / abs(adam_mean) * 100
    wins = np.sum(mobiu_results < adam_results)
    
    print(f"\n{'─' * 50}")
    print(f"{'Optimizer':<12} {'Mean':<12} {'Best':<12} {'Std':<10}")
    print(f"{'─' * 50}")
    print(f"{'Adam':<12} {adam_mean:<12.4f} {adam_results.min():<12.4f} {adam_results.std():<10.4f}")
    print(f"{'Mobiu-Q':<12} {mobiu_mean:<12.4f} {mobiu_results.min():<12.4f} {mobiu_results.std():<10.4f}")
    
    print(f"\n📊 IMPROVEMENT: {improvement:.1f}% (Mobiu-Q wins {wins}/{n_seeds} seeds)")
    
    return adam_results, mobiu_results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("MOBIU-Q CUSTOMER VALIDATION")
    print("🔬 " * 20 + "\n")
    
    if LICENSE_KEY == "YOUR_LICENSE_KEY_HERE":
        print("❌ Please set your LICENSE_KEY at the top of the file!")
        exit(1)
    
    test_vqe(LICENSE_KEY, n_seeds=10)
    test_qaoa(LICENSE_KEY, n_seeds=10)
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
