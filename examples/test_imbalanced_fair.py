#!/usr/bin/env python3
"""
================================================================================
FAIR TEST: Imbalanced Data with Mobiu-Q AUTO Mode
================================================================================
Same methodology as test_fakefez_vqe.py:
- Both use REAL API
- Both see SAME energy and gradient
- Only difference: use_soft_algebra=True vs False
- Deterministic: same seed = same results

Imbalanced Data bias source:
- Class distribution highly skewed (e.g., 90% class A, 10% class B)
- Gradient dominated by majority class
- Model biased toward predicting majority
- Common in fraud detection, medical diagnosis, anomaly detection

AUTO MODE: Automatically selects best Soft Algebra mode (boost/dampen/off)
================================================================================
"""

import numpy as np
import requests
from dataclasses import dataclass

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "YOUR_LICENSE_HERE"

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
# API RUNNER
# ============================================================

def run_optimizer(use_soft_algebra: bool, problem: ImbalancedDataProblem, 
                  seed: int, n_steps: int = N_STEPS) -> float:
    """
    Run optimizer via REAL API.
    Both see SAME energy and gradient. Only use_soft_algebra differs.
    """
    
    # Start session
    r = requests.post(API_URL, json={
        'action': 'start',
        'license_key': LICENSE_KEY,
        'method': METHOD,
        'mode': 'simulation',
        'use_soft_algebra': use_soft_algebra,
        'base_optimizer': BASE_OPTIMIZER,
        'base_lr': LR
    }, timeout=15)
    
    data = r.json()
    if not data.get('success'):
        print(f"Error starting session: {data}")
        return float('inf')
    
    session_id = data['session_id']
    
    # Initialize params (deterministic)
    params = problem.init_params(seed)
    best_loss = float('inf')
    
    # Run optimization
    for step in range(n_steps):
        # Energy = balanced loss, Gradient = imbalanced
        energy = problem.balanced_loss(params)
        gradient = problem.imbalanced_gradient(params, seed * 10000 + step)
        
        best_loss = min(best_loss, energy)
        
        # Flatten for API
        params_flat = params.flatten()
        gradient_flat = gradient.flatten()
        
        # Send to API
        r = requests.post(API_URL, json={
            'action': 'step',
            'license_key': LICENSE_KEY,
            'session_id': session_id,
            'params': params_flat.tolist(),
            'gradient': gradient_flat.tolist(),
            'energy': energy
        }, timeout=15)
        
        step_data = r.json()
        if step_data.get('success'):
            params = np.array(step_data['new_params']).reshape(problem.n_classes, problem.dim)
        else:
            print(f"Step {step} failed: {step_data}")
            break
    
    # End session
    requests.post(API_URL, json={
        'action': 'end',
        'license_key': LICENSE_KEY,
        'session_id': session_id
    }, timeout=5)
    
    return best_loss


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("⚖️  IMBALANCED DATA - FAIR A/B TEST (AUTO MODE)")
    print("=" * 70)
    print(f"Method: {METHOD} | Optimizer: {BASE_OPTIMIZER} | LR: {LR}")
    print(f"Classes: {N_CLASSES} | Imbalance: {IMBALANCE_RATIO*100:.0f}% majority | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS}")
    print()
    print("Class 0 dominates with 90% of samples.")
    print("Energy = balanced loss, Gradient = imbalanced.")
    print("Both optimizers see SAME energy and SAME gradient.")
    print("Only difference: use_soft_algebra = True vs False")
    print("AUTO mode selects best strategy (boost/dampen/off)")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = ImbalancedDataProblem(
        dim=DIM,
        n_classes=N_CLASSES,
        imbalance_ratio=IMBALANCE_RATIO
    )
    
    baseline_losses = []
    mobiu_losses = []
    
    print(f"\n{'Seed':<8} {'Baseline':<12} {'Mobiu-Q':<12} {'Δ%':<10} {'Winner'}")
    print("-" * 55)
    
    for i, seed in enumerate([42 + s * 17 for s in range(N_SEEDS)]):
        # Run baseline (use_soft_algebra=False)
        baseline = run_optimizer(False, problem, seed)
        baseline_losses.append(baseline)
        
        # Run Mobiu-Q (use_soft_algebra=True)
        mobiu = run_optimizer(True, problem, seed)
        mobiu_losses.append(mobiu)
        
        # Calculate improvement
        if baseline > 0:
            improvement = (baseline - mobiu) / baseline * 100
        else:
            improvement = 0
        
        winner = "🏆 Mobiu" if mobiu < baseline else "❌ Base"
        print(f"{seed:<8} {baseline:<12.6f} {mobiu:<12.6f} {improvement:>+8.1f}%  {winner}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    baseline_avg = np.mean(baseline_losses)
    mobiu_avg = np.mean(mobiu_losses)
    avg_improvement = (baseline_avg - mobiu_avg) / baseline_avg * 100
    
    wins_mobiu = sum(m < b for m, b in zip(mobiu_losses, baseline_losses))
    wins_baseline = N_SEEDS - wins_mobiu
    
    print(f"\nBaseline avg: {baseline_avg:.6f}")
    print(f"Mobiu-Q avg:  {mobiu_avg:.6f}")
    print(f"Improvement:  {avg_improvement:+.1f}%")
    print(f"\nWins: Mobiu {wins_mobiu}/{N_SEEDS} | Baseline {wins_baseline}/{N_SEEDS}")
    
    # Statistical significance (paired t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(baseline_losses, mobiu_losses)
    
    print(f"\nStatistical significance:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.05 and mobiu_avg < baseline_avg:
        print("  ✅ SIGNIFICANT: Mobiu-Q is better (p < 0.05)")
    elif p_value < 0.05:
        print("  ⚠️ SIGNIFICANT: Baseline is better (p < 0.05)")
    else:
        print("  ⚪ NOT SIGNIFICANT (p >= 0.05)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
