#!/usr/bin/env python3
"""
================================================================================
🏷️ NOISY LABELS - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Noisy Labels bias source:
- Systematic label errors (not random!)
- Certain classes consistently confused with others
- Gradient points toward wrong targets
- Common in real data: OCR confuses 3↔8, medical images, crowdsourced labels
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
N_CLASSES = 5
NOISE_RATE = 0.3  # 30% of labels are wrong
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
# NOISY LABELS PROBLEM
# ============================================================

@dataclass
class NoisyLabelsProblem:
    """
    Simulates classification with systematic label noise.
    
    The bias comes from:
    - Certain classes are systematically confused (not random!)
    - e.g., class 0 often mislabeled as class 1
    - Gradient consistently points toward wrong direction
    """
    dim: int
    n_classes: int
    noise_rate: float
    
    # True class centers (what we want to learn)
    true_centers: np.ndarray = None
    # Confusion matrix: confusion[i] = class that i gets mislabeled as
    confusion_map: dict = None
    
    def __post_init__(self):
        # Fixed true class centers
        np.random.seed(888)
        self.true_centers = np.random.randn(self.n_classes, self.dim)
        
        # Systematic confusion pattern (not random!)
        # Class i often confused with class (i+1) % n_classes
        self.confusion_map = {
            i: (i + 1) % self.n_classes 
            for i in range(self.n_classes)
        }
    
    def init_params(self, seed: int) -> np.ndarray:
        """Initialize classifier weights - deterministic per seed"""
        np.random.seed(seed)
        return np.random.randn(self.n_classes, self.dim) * 0.5
    
    def true_loss(self, weights: np.ndarray) -> float:
        """Loss against TRUE labels (what we want to minimize)"""
        # MSE between learned weights and true centers
        return float(np.mean((weights - self.true_centers) ** 2))
    
    def noisy_gradient(self, weights: np.ndarray, rng_state: int) -> np.ndarray:
        """
        Gradient with systematic label noise.
        Some samples have wrong labels → gradient points wrong way.
        """
        np.random.seed(rng_state)
        
        batch_size = 32
        grad = np.zeros_like(weights)
        
        for _ in range(batch_size):
            # Pick a random true class
            true_class = np.random.randint(self.n_classes)
            
            # With noise_rate probability, use wrong label
            if np.random.rand() < self.noise_rate:
                # Systematic confusion (not random!)
                label = self.confusion_map[true_class]
            else:
                label = true_class
            
            # Gradient pushes toward the (possibly wrong) label's center
            target = self.true_centers[label]
            
            # Only update the row for this class
            grad[true_class] += 2 * (weights[true_class] - target) / self.dim
        
        return grad / batch_size


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("🏷️ NOISY LABELS - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Classes: {N_CLASSES} | Noise Rate: {NOISE_RATE*100:.0f}% | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS} | LR: {LR}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("  • Systematic label confusion: class i → class (i+1)")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = NoisyLabelsProblem(
        dim=DIM,
        n_classes=N_CLASSES,
        noise_rate=NOISE_RATE
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
            energy = problem.true_loss(params)
            gradient = problem.noisy_gradient(params, seed * 10000 + step)
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
            energy = problem.true_loss(params)
            gradient = problem.noisy_gradient(params, seed * 10000 + step)
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
