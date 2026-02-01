#!/usr/bin/env python3
"""
================================================================================
🌐 FEDERATED LEARNING - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Federated Learning bias source:
- Multiple clients with non-IID data distributions
- Aggregated gradients have systematic bias from client heterogeneity
================================================================================
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass

# Mobiu-Q
from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

# Test parameters
N_SEEDS = 10
N_STEPS = 80
DIM = 20
N_CLIENTS = 10
NON_IID_STRENGTH = 0.8  # How different each client's data is
LR = 0.01
METHOD = "standard"
BASE_OPTIMIZER = "Adam"


# ============================================================
# SIMPLE ADAM IMPLEMENTATION (for baseline)
# ============================================================

class SimpleAdam:
    """Vanilla Adam optimizer for baseline comparison"""
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, gradient):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# FEDERATED LEARNING PROBLEM
# ============================================================

@dataclass
class FederatedProblem:
    """
    Simulates Federated Learning with non-IID client data.
    
    The bias comes from:
    - Each client has a slightly different view of the true target
    - When we aggregate gradients from a subset of clients,
      we get systematic bias depending on which clients we sample
    """
    dim: int
    n_clients: int
    non_iid: float
    w_true: np.ndarray = None
    client_biases: list = None
    
    def __post_init__(self):
        # Fixed true target (same for all runs)
        np.random.seed(999)
        self.w_true = np.random.randn(self.dim)
        
        # Each client has systematic bias
        self.client_biases = [
            np.random.randn(self.dim) * self.non_iid 
            for _ in range(self.n_clients)
        ]
    
    def init_params(self, seed: int) -> np.ndarray:
        """Initialize parameters - deterministic per seed"""
        np.random.seed(seed)
        return np.random.randn(self.dim) * 0.5
    
    def global_loss(self, w: np.ndarray) -> float:
        """True global loss (what we want to minimize)"""
        return float(np.mean((w - self.w_true) ** 2))
    
    def federated_gradient(self, w: np.ndarray, rng_state: int) -> np.ndarray:
        """
        Aggregate gradients from random subset of clients.
        Uses rng_state for deterministic client sampling.
        """
        np.random.seed(rng_state)
        n_sampled = 5
        sampled_clients = np.random.choice(self.n_clients, n_sampled, replace=False)
        
        grads = []
        for c in sampled_clients:
            # Each client's gradient is biased toward their local optimum
            target = self.w_true + self.client_biases[c]
            grad = 2 * (w - target) / self.dim
            grads.append(grad)
        
        return np.mean(grads, axis=0)


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("🌐 FEDERATED LEARNING - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Clients: {N_CLIENTS} | Non-IID: {NON_IID_STRENGTH} | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS} | LR: {LR}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = FederatedProblem(
        dim=DIM,
        n_clients=N_CLIENTS,
        non_iid=NON_IID_STRENGTH
    )
    
    baseline_losses = []
    mobiu_losses = []
    
    print(f"\n{'Seed':<8} {'Pure Adam':<12} {'Adam+Mobiu':<12} {'Δ%':<10} {'Winner'}")
    print("-" * 55)
    
    for i, seed in enumerate([42 + s * 17 for s in range(N_SEEDS)]):
        
        # ─────────────────────────────────────────────────────────────────
        # BASELINE: Pure Adam (what customer has BEFORE adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = problem.init_params(seed)
        adam = SimpleAdam(lr=LR)
        baseline_best = float('inf')
        
        for step in range(N_STEPS):
            energy = problem.global_loss(params)
            gradient = problem.federated_gradient(params, seed * 10000 + step)
            baseline_best = min(baseline_best, energy)
            params = adam.step(params, gradient)
        
        baseline_losses.append(baseline_best)
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = problem.init_params(seed)
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='simulation',
            base_optimizer=BASE_OPTIMIZER,
            base_lr=LR,
            verbose=False
        )
        mobiu_best = float('inf')
        
        for step in range(N_STEPS):
            energy = problem.global_loss(params)
            gradient = problem.federated_gradient(params, seed * 10000 + step)
            mobiu_best = min(mobiu_best, energy)
            params = mobiu_opt.step(params, gradient, energy)
        
        mobiu_opt.end()
        mobiu_losses.append(mobiu_best)
        
        # Calculate improvement
        if baseline_best > 0:
            improvement = (baseline_best - mobiu_best) / baseline_best * 100
        else:
            improvement = 0
        
        winner = "✅ Mobiu" if mobiu_best < baseline_best else "❌ Adam"
        print(f"{seed:<8} {baseline_best:<12.6f} {mobiu_best:<12.6f} {improvement:>+8.1f}%  {winner}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    baseline_arr = np.array(baseline_losses)
    mobiu_arr = np.array(mobiu_losses)
    
    baseline_avg = np.mean(baseline_arr)
    mobiu_avg = np.mean(mobiu_arr)
    avg_improvement = (baseline_avg - mobiu_avg) / baseline_avg * 100
    
    wins_mobiu = sum(m < b for m, b in zip(mobiu_losses, baseline_losses))
    
    print(f"\nPure Adam avg:    {baseline_avg:.6f} ± {np.std(baseline_arr):.6f}")
    print(f"Adam + Mobiu avg: {mobiu_avg:.6f} ± {np.std(mobiu_arr):.6f}")
    print(f"\nImprovement: {avg_improvement:+.1f}%")
    print(f"Win rate: {wins_mobiu}/{N_SEEDS} ({100*wins_mobiu/N_SEEDS:.0f}%)")
    
    # Statistical significance
    t_stat, p_value = stats.ttest_rel(baseline_losses, mobiu_losses)
    print(f"p-value: {p_value:.6f}")
    
    if p_value < 0.05 and mobiu_avg < baseline_avg:
        print("✅ SIGNIFICANT: Mobiu-Q is better (p < 0.05)")
    elif p_value < 0.05:
        print("⚠️ SIGNIFICANT: Baseline is better (p < 0.05)")
    else:
        print("⚪ NOT SIGNIFICANT (p >= 0.05)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
