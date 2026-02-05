#!/usr/bin/env python3
"""
================================================================================
📦 BLACK-BOX OPTIMIZATION - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimizer (what customer has BEFORE Mobiu-Q)
- Test: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Mimics the structure of QAOA/VQE:
- SPSA gradient estimation (not backprop)
- Shot noise (discrete sampling)
- Rugged landscape with periodic structure

Target domains:
- Quantum computing (QAOA, VQE)
- Robotics sensor calibration
- Hyperparameter optimization
- Simulation-based optimization
================================================================================
"""

import numpy as np
from scipy import stats
from typing import Callable

# Mobiu-Q
from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

# Test parameters
N_SEEDS = 10
N_STEPS = 100
N_PARAMS = 10
SHOTS = 1024
NOISE_SCALE = 0.1
C_SHIFT = 0.1
LR = 0.2
BOUNDS = (-5.0, 5.0)
FUNCTION = "portfolio"  # portfolio, credit_risk, option_pricing
METHOD = "standard"  # deep works better for rugged landscapes
MAXIMIZE = True  # For profit maximization


# ============================================================
# LANDSCAPE FUNCTIONS (Rugged + Periodic)
# ============================================================

class LandscapeFunctions:
    """Test functions with properties similar to QAOA energy landscapes"""
    
    @staticmethod
    def portfolio(x: np.ndarray) -> float:
        """Portfolio optimization objective: maximize return - 0.5 * risk"""
        n = len(x)
        mu = np.linspace(0.01, 0.1, n)  # expected returns
        cov = np.eye(n) * 0.05 + np.random.rand(n, n) * 0.01
        cov = (cov + cov.T) / 2  # symmetric
        risk = x.T @ cov @ x
        returns = mu @ x
        return returns - 0.5 * risk  # to maximize

# ============================================================
# SHOT NOISE EVALUATOR
# ============================================================

class ShotNoiseEvaluator:
    """Simulates shot noise like in quantum computing"""
    
    def __init__(self, base_function: Callable, shots: int = 1024, noise_scale: float = 0.1):
        self.base_function = base_function
        self.shots = shots
        self.noise_scale = noise_scale
    
    def evaluate(self, x: np.ndarray, shots: int = None) -> float:
        """Evaluate with shot noise (like quantum measurement)"""
        if shots is None:
            shots = self.shots
        
        true_value = self.base_function(x)
        shot_values = true_value + self.noise_scale * np.abs(true_value + 1) * np.random.randn(shots)
        return np.mean(shot_values)
    
    def evaluate_true(self, x: np.ndarray) -> float:
        """Get true value without noise"""
        return self.base_function(x)


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("📦 BLACK-BOX OPTIMIZATION - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Function: {FUNCTION} | Params: {N_PARAMS}")
    print(f"Shots: {SHOTS} | Noise: {NOISE_SCALE} | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS} | LR: {LR}")
    print(f"Maximize: {MAXIMIZE}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure SPSA optimizer (NO Mobiu)")
    print("  • Test: SPSA + Mobiu-Q enhancement")
    print("  • Mimics QAOA/VQE structure (SPSA + shot noise)")
    print("=" * 70)
    
    # Select function
    functions = {
        'portfolio': LandscapeFunctions.portfolio,
    }
    base_func = functions[FUNCTION]
    
    baseline_results = []
    mobiu_results = []
    
    print(f"\n{'Seed':<8} {'Pure SPSA':<12} {'SPSA+Mobiu':<12} {'Δ%':<10} {'Winner'}")
    print("-" * 55)
    
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        
        # Create evaluator with shot noise
        evaluator = ShotNoiseEvaluator(
            base_function=base_func,
            shots=SHOTS,
            noise_scale=NOISE_SCALE
        )
        
        # Initialize parameters
        init_params = np.random.uniform(BOUNDS[0], BOUNDS[1], N_PARAMS)
        
        # Pre-generate SPSA deltas for fairness
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=N_PARAMS).astype(float) 
                       for _ in range(N_STEPS)]
        
        # ─────────────────────────────────────────────────────────────────
        # BASELINE: Pure SPSA (what customer has BEFORE adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        baseline_best = float('-inf') if MAXIMIZE else float('inf')
        
        for step in range(N_STEPS):
            delta = spsa_deltas[step]
            ck = C_SHIFT / ((step + 1) ** 0.101)
            ak = LR / ((step + 1) ** 0.602)
            
            # SPSA gradient estimation
            e_plus = evaluator.evaluate(params + ck * delta)
            e_minus = evaluator.evaluate(params - ck * delta)
            e_center = evaluator.evaluate(params)
            
            if MAXIMIZE:
                e_plus, e_minus, e_center = -e_plus, -e_minus, -e_center
            
            gradient = (e_plus - e_minus) / (2 * ck) * delta
            
            # Track true energy
            true_energy = evaluator.evaluate_true(params)
            baseline_best = max(baseline_best, true_energy) if MAXIMIZE else min(baseline_best, true_energy)
            
            # Pure SPSA update
            params = params - ak * gradient
            params = np.clip(params, BOUNDS[0], BOUNDS[1])
        
        baseline_results.append(baseline_best)
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',  # Important for deep!
            base_lr=LR,
            maximize=MAXIMIZE,
            verbose=False
        )
        mobiu_best = float('-inf') if MAXIMIZE else float('inf')
        
        for step in range(N_STEPS):
            delta = spsa_deltas[step]
            ck = C_SHIFT / ((step + 1) ** 0.101)
            
            # SPSA gradient estimation (same as baseline)
            e_plus = evaluator.evaluate(params + ck * delta)
            e_minus = evaluator.evaluate(params - ck * delta)
            e_center = evaluator.evaluate(params)
            
            if MAXIMIZE:
                e_plus, e_minus, e_center = -e_plus, -e_minus, -e_center
            
            gradient = (e_plus - e_minus) / (2 * ck) * delta
            
            # Track true energy
            true_energy = evaluator.evaluate_true(params)
            mobiu_best = max(mobiu_best, true_energy) if MAXIMIZE else min(mobiu_best, true_energy)
            
            # Mobiu-Q enhanced update
            params = mobiu_opt.step(params, gradient, e_center)
            params = np.clip(params, BOUNDS[0], BOUNDS[1])
        
        mobiu_opt.end()
        mobiu_results.append(mobiu_best)
        
        # Calculate improvement
        if MAXIMIZE:
            improvement = (mobiu_best - baseline_best) / abs(baseline_best) * 100 if baseline_best != 0 else 0
            winner = "✅ Mobiu" if mobiu_best > baseline_best else "❌ SPSA"
        else:
            improvement = (baseline_best - mobiu_best) / abs(baseline_best) * 100 if baseline_best != 0 else 0
            winner = "✅ Mobiu" if mobiu_best < baseline_best else "❌ SPSA"
        
        print(f"{seed:<8} {baseline_best:<12.4f} {mobiu_best:<12.4f} {improvement:>+8.1f}%  {winner}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    baseline_arr = np.array(baseline_results)
    mobiu_arr = np.array(mobiu_results)
    
    baseline_avg = np.mean(baseline_arr)
    mobiu_avg = np.mean(mobiu_arr)
    
    if MAXIMIZE:
        avg_improvement = (mobiu_avg - baseline_avg) / abs(baseline_avg) * 100 if baseline_avg != 0 else 0
    else:
        avg_improvement = (baseline_avg - mobiu_avg) / abs(baseline_avg) * 100 if baseline_avg != 0 else 0
    
    wins_mobiu = sum(m > b for m, b in zip(mobiu_results, baseline_results)) if MAXIMIZE else sum(m < b for m, b in zip(mobiu_results, baseline_results))
    
    print(f"\nPure SPSA avg:    {baseline_avg:.4f} ± {np.std(baseline_arr):.4f}")
    print(f"SPSA + Mobiu avg: {mobiu_avg:.4f} ± {np.std(mobiu_arr):.4f}")
    print(f"\nImprovement: {avg_improvement:+.1f}%")
    print(f"Win rate: {wins_mobiu}/{N_SEEDS} ({100*wins_mobiu/N_SEEDS:.0f}%)")
    
    # Statistical significance
    if MAXIMIZE:
        t_stat, p_value = stats.ttest_rel(mobiu_results, baseline_results)  # For higher better, ttest mobiu - baseline >0
    else:
        t_stat, p_value = stats.ttest_rel(baseline_results, mobiu_results)
    print(f"p-value: {p_value:.6f}")
    
    if p_value < 0.05 and ((mobiu_avg > baseline_avg) if MAXIMIZE else (mobiu_avg < baseline_avg)):
        print("✅ SIGNIFICANT: Mobiu-Q is better (p < 0.05)")
    elif p_value < 0.05:
        print("⚠️ SIGNIFICANT: Baseline is better (p < 0.05)")
    else:
        print("⚪ NOT SIGNIFICANT (p >= 0.05)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()