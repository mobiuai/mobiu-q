#!/usr/bin/env python3
"""
================================================================================
FAIR TEST: Sim-to-Real with Mobiu-Q AUTO Mode
================================================================================
Same methodology as test_fakefez_vqe.py:
- Both use REAL API
- Both see SAME energy and gradient
- Only difference: use_soft_algebra=True vs False
- Deterministic: same seed = same results

Sim-to-Real bias source:
- Training in simulator, deploying in reality
- Simulator has systematic errors (friction, mass, dynamics)
- Gradient from simulator consistently biased vs real world
- Common in robotics, autonomous vehicles, game AI

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
SIM_BIAS_STRENGTH = 0.4  # How wrong the simulator is
LR = 0.01
METHOD = "standard"  
BASE_OPTIMIZER = "Adam"


# ============================================================
# SIM-TO-REAL PROBLEM
# ============================================================

@dataclass
class SimToRealProblem:
    """
    Simulates the Sim-to-Real gap in robotics/control.
    
    The bias comes from:
    - Simulator has wrong physics parameters
    - e.g., friction coefficient off by 20%
    - Gradient from simulator systematically wrong
    - Real world behaves differently
    """
    dim: int
    sim_bias_strength: float
    
    # Real world optimal parameters
    real_optimum: np.ndarray = None
    # Simulator's systematic bias (wrong physics)
    sim_bias: np.ndarray = None
    
    def __post_init__(self):
        # Fixed real-world optimum
        np.random.seed(777)
        self.real_optimum = np.random.randn(self.dim)
        
        # Simulator has consistent bias (not random noise!)
        # e.g., always underestimates friction
        np.random.seed(778)
        self.sim_bias = np.random.randn(self.dim) * self.sim_bias_strength
    
    def init_params(self, seed: int) -> np.ndarray:
        """Initialize control parameters - deterministic per seed"""
        np.random.seed(seed)
        return np.random.randn(self.dim) * 0.5
    
    def real_loss(self, params: np.ndarray) -> float:
        """Loss in REAL world (what we actually care about)"""
        return float(np.mean((params - self.real_optimum) ** 2))
    
    def simulator_gradient(self, params: np.ndarray, rng_state: int) -> np.ndarray:
        """
        Gradient from SIMULATOR (biased!).
        Simulator thinks optimum is at real_optimum + sim_bias.
        Plus some random noise from stochastic simulation.
        """
        np.random.seed(rng_state)
        
        # Simulator's wrong target
        sim_target = self.real_optimum + self.sim_bias
        
        # Base gradient toward wrong target
        grad = 2 * (params - sim_target) / self.dim
        
        # Small stochastic noise from simulation
        noise = np.random.randn(self.dim) * 0.05
        
        return grad + noise


# ============================================================
# API RUNNER
# ============================================================

def run_optimizer(use_soft_algebra: bool, problem: SimToRealProblem, 
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
        # Energy = real loss, Gradient = from simulator (biased!)
        energy = problem.real_loss(params)
        gradient = problem.simulator_gradient(params, seed * 10000 + step)
        
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
    print("🤖 SIM-TO-REAL - FAIR A/B TEST (AUTO MODE)")
    print("=" * 70)
    print(f"Method: {METHOD} | Optimizer: {BASE_OPTIMIZER} | LR: {LR}")
    print(f"Sim Bias: {SIM_BIAS_STRENGTH} | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS}")
    print()
    print("Simulator has systematic bias (wrong physics parameters).")
    print("Energy = real loss, Gradient = simulator (biased).")
    print("Both optimizers see SAME energy and SAME gradient.")
    print("Only difference: use_soft_algebra = True vs False")
    print("AUTO mode selects best strategy (boost/dampen/off)")
    print("=" * 70)
    
    # Create problem (fixed structure)
    problem = SimToRealProblem(
        dim=DIM,
        sim_bias_strength=SIM_BIAS_STRENGTH
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
