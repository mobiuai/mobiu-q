#!/usr/bin/env python3
"""
================================================================================
FAIR TEST: Federated Learning with Mobiu-Q API
================================================================================
Same methodology as test_fakefez_vqe.py:
- Both use REAL API
- Both see SAME energy and gradient
- Only difference: use_soft_algebra=True vs False
- Deterministic: same seed = same results

Federated Learning bias source:
- Multiple clients with non-IID data distributions
- Aggregated gradients have systematic bias from client heterogeneity
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
N_CLIENTS = 10
NON_IID_STRENGTH = 0.8  # How different each client's data is
LR = 0.01  # Same for both!
METHOD = "standard"
BASE_OPTIMIZER = "Adam"


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
# API RUNNER (same pattern as test_fakefez_vqe.py)
# ============================================================

def run_optimizer(use_soft_algebra: bool, problem: FederatedProblem, 
                  seed: int, n_steps: int = N_STEPS) -> float:
    """
    Run optimizer via REAL API.
    
    CRITICAL: Both use_soft_algebra=True and False see the SAME:
    - Initial params (from seed)
    - Energy values
    - Gradient values
    
    The ONLY difference is the use_soft_algebra flag.
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
        # Compute energy and gradient (deterministic per step)
        energy = problem.global_loss(params)
        gradient = problem.federated_gradient(params, seed * 10000 + step)
        
        best_loss = min(best_loss, energy)
        
        # Send to API
        r = requests.post(API_URL, json={
            'action': 'step',
            'license_key': LICENSE_KEY,
            'session_id': session_id,
            'params': params.tolist(),
            'gradient': gradient.tolist(),
            'energy': energy
        }, timeout=15)
        
        step_data = r.json()
        if step_data.get('success'):
            params = np.array(step_data['new_params'])
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
    print("🌐 FEDERATED LEARNING - FAIR A/B TEST")
    print("=" * 70)
    print(f"Method: {METHOD} | Optimizer: {BASE_OPTIMIZER} | LR: {LR}")
    print(f"Clients: {N_CLIENTS} | Non-IID: {NON_IID_STRENGTH} | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS}")
    print()
    print("Both optimizers see SAME energy and SAME gradient.")
    print("Only difference: use_soft_algebra = True vs False")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = FederatedProblem(
        dim=DIM,
        n_clients=N_CLIENTS,
        non_iid=NON_IID_STRENGTH
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
