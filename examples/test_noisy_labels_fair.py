#!/usr/bin/env python3
"""
================================================================================
FAIR TEST: Noisy Labels with Mobiu-Q API
================================================================================
Same methodology as test_fakefez_vqe.py:
- Both use REAL API
- Both see SAME energy and gradient
- Only difference: use_soft_algebra=True vs False
- Deterministic: same seed = same results

Noisy Labels bias source:
- Systematic label errors (not random!)
- Certain classes consistently confused with others
- Gradient points toward wrong targets
- Common in real data: OCR confuses 3↔8, medical images, crowdsourced labels
================================================================================
"""

import numpy as np
import requests
from dataclasses import dataclass

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "YOUR_KEY_HERE"

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
# API RUNNER
# ============================================================

def run_optimizer(use_soft_algebra: bool, problem: NoisyLabelsProblem, 
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
        # Compute energy (true loss) and gradient (noisy)
        energy = problem.true_loss(params)
        gradient = problem.noisy_gradient(params, seed * 10000 + step)
        
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
    print("🏷️  NOISY LABELS - FAIR A/B TEST")
    print("=" * 70)
    print(f"Method: {METHOD} | Optimizer: {BASE_OPTIMIZER} | LR: {LR}")
    print(f"Classes: {N_CLASSES} | Noise Rate: {NOISE_RATE*100:.0f}% | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS}")
    print()
    print("Systematic label confusion: class i → class (i+1)")
    print("Both optimizers see SAME energy and SAME gradient.")
    print("Only difference: use_soft_algebra = True vs False")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = NoisyLabelsProblem(
        dim=DIM,
        n_classes=N_CLASSES,
        noise_rate=NOISE_RATE
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
