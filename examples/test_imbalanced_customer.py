#!/usr/bin/env python3
"""
================================================================================
⚖️ IMBALANCED DATA - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Imbalanced Data bias source:
- Class distribution highly skewed (e.g., 90% class A, 10% class B)
- Gradient dominated by majority class
- Model biased toward predicting majority
- Common in fraud detection, medical diagnosis, anomaly detection
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
N_CLASSES = 4
IMBALANCE_RATIO = 0.9  # 90% samples from class 0
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
# IMBALANCED DATA PROBLEM
# ============================================================

@dataclass
class ImbalancedDataProblem:
    """
    Simulates classification with highly imbalanced classes.
    
    The bias comes from:
    - Most samples from majority class
    - Gradient dominated by majority class direction
    - Minority classes underrepresented in updates
    """
    dim: int
    n_classes: int
    imbalance_ratio: float
    
    # True class centers (balanced optimum)
    true_centers: np.ndarray = None
    # Class probabilities (imbalanced)
    class_probs: np.ndarray = None
    
    def __post_init__(self):
        # Fixed true class centers
        np.random.seed(666)
        self.true_centers = np.random.randn(self.n_classes, self.dim)
        
        # Imbalanced class distribution
        # Class 0 has imbalance_ratio, rest share equally
        minority_prob = (1 - self.imbalance_ratio) / (self.n_classes - 1)
        self.class_probs = np.array(
            [self.imbalance_ratio] + [minority_prob] * (self.n_classes - 1)
        )
    
    def init_params(self, seed: int) -> np.ndarray:
        """Initialize classifier weights - deterministic per seed"""
        np.random.seed(seed)
        return np.random.randn(self.n_classes, self.dim) * 0.5
    
    def balanced_loss(self, weights: np.ndarray) -> float:
        """
        Balanced loss (what we WANT to optimize).
        Equal weight to all classes.
        """
        return float(np.mean((weights - self.true_centers) ** 2))
    
    def imbalanced_gradient(self, weights: np.ndarray, rng_state: int) -> np.ndarray:
        """
        Gradient from imbalanced batch.
        Majority class dominates the gradient.
        """
        np.random.seed(rng_state)
        
        batch_size = 32
        grad = np.zeros_like(weights)
        
        # Sample classes according to imbalanced distribution
        classes = np.random.choice(
            self.n_classes, 
            size=batch_size, 
            p=self.class_probs
        )
        
        for c in classes:
            # Gradient for this sample
            grad[c] += 2 * (weights[c] - self.true_centers[c]) / self.dim
        
        return grad / batch_size


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("⚖️ IMBALANCED DATA - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Classes: {N_CLASSES} | Imbalance: {IMBALANCE_RATIO*100:.0f}% majority | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS} | LR: {LR}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("  • Class 0 dominates with 90% of samples")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = ImbalancedDataProblem(
        dim=DIM,
        n_classes=N_CLASSES,
        imbalance_ratio=IMBALANCE_RATIO
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
            energy = problem.balanced_loss(params)
            gradient = problem.imbalanced_gradient(params, seed * 10000 + step)
            baseline_best = min(baseline_best, energy)
            
            # Flatten for Adam
            params_flat = params.flatten()
            gradient_flat = gradient.flatten()
            params_flat = adam.step(params_flat, gradient_flat)
            params = params_flat.reshape(N_CLASSES, DIM)
        
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
            energy = problem.balanced_loss(params)
            gradient = problem.imbalanced_gradient(params, seed * 10000 + step)
            mobiu_best = min(mobiu_best, energy)
            
            # Flatten for API
            params_flat = params.flatten()
            gradient_flat = gradient.flatten()
            params_flat = mobiu_opt.step(params_flat, gradient_flat, energy)
            params = np.array(params_flat).reshape(N_CLASSES, DIM)
        
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
