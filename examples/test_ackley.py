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
from dataclasses import dataclass
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
FUNCTION = "ackley"  # rosenbrock, sphere, ackley, beale
METHOD = "adaptive"  # deep works better for rugged landscapes


# ============================================================
# LANDSCAPE FUNCTIONS (Rugged + Periodic)
# ============================================================

class LandscapeFunctions:
    """Test functions with properties similar to QAOA energy landscapes"""
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin: highly multimodal with periodic structure"""
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley: multimodal with periodic structure"""
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e
    
    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """Schwefel: deceptive with global minimum far from local minima"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank: many local minima with periodic structure"""
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        return sum_sq - prod_cos + 1
    
    @staticmethod
    def levy(x: np.ndarray) -> float:
        """Levy: multimodal with sinusoidal components"""
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3

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
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure SPSA optimizer (NO Mobiu)")
    print("  • Test: SPSA + Mobiu-Q enhancement")
    print("  • Mimics QAOA/VQE structure (SPSA + shot noise)")
    print("=" * 70)
    
    # Select function
    functions = {
        'rastrigin': LandscapeFunctions.rastrigin,
        'ackley': LandscapeFunctions.ackley,
        'schwefel': LandscapeFunctions.schwefel,
        'griewank': LandscapeFunctions.griewank,
        'levy': LandscapeFunctions.levy,
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
        baseline_best = float('inf')
        
        for step in range(N_STEPS):
            delta = spsa_deltas[step]
            ck = C_SHIFT / ((step + 1) ** 0.101)
            ak = LR / ((step + 1) ** 0.602)
            
            # SPSA gradient estimation
            e_plus = evaluator.evaluate(params + ck * delta)
            e_minus = evaluator.evaluate(params - ck * delta)
            e_center = evaluator.evaluate(params)
            gradient = (e_plus - e_minus) / (2 * ck) * delta
            
            # Track true energy
            true_energy = evaluator.evaluate_true(params)
            baseline_best = min(baseline_best, true_energy)
            
            # Pure SPSA update
            params = params - ak * gradient
            params = np.clip(params, BOUNDS[0], BOUNDS[1])
        
        baseline_final = evaluator.evaluate_true(params)
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
            verbose=False
        )
        mobiu_best = float('inf')
        
        for step in range(N_STEPS):
            delta = spsa_deltas[step]
            ck = C_SHIFT / ((step + 1) ** 0.101)
            
            # SPSA gradient estimation (same as baseline)
            e_plus = evaluator.evaluate(params + ck * delta)
            e_minus = evaluator.evaluate(params - ck * delta)
            e_center = evaluator.evaluate(params)
            gradient = (e_plus - e_minus) / (2 * ck) * delta
            
            # Track true energy
            true_energy = evaluator.evaluate_true(params)
            mobiu_best = min(mobiu_best, true_energy)
            
            # Mobiu-Q enhanced update
            params = mobiu_opt.step(params, gradient, e_center)
            params = np.clip(params, BOUNDS[0], BOUNDS[1])
        
        mobiu_opt.end()
        mobiu_final = evaluator.evaluate_true(params)
        mobiu_results.append(mobiu_best)
        
        # Calculate improvement (lower is better)
        if baseline_best > 0:
            improvement = (baseline_best - mobiu_best) / baseline_best * 100
        else:
            improvement = 0
        
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
    avg_improvement = (baseline_avg - mobiu_avg) / abs(baseline_avg) * 100 if baseline_avg != 0 else 0
    
    wins_mobiu = sum(m < b for m, b in zip(mobiu_results, baseline_results))
    
    print(f"\nPure SPSA avg:    {baseline_avg:.4f} ± {np.std(baseline_arr):.4f}")
    print(f"SPSA + Mobiu avg: {mobiu_avg:.4f} ± {np.std(mobiu_arr):.4f}")
    print(f"\nImprovement: {avg_improvement:+.1f}%")
    print(f"Win rate: {wins_mobiu}/{N_SEEDS} ({100*wins_mobiu/N_SEEDS:.0f}%)")
    
    # Statistical significance
    t_stat, p_value = stats.ttest_rel(baseline_results, mobiu_results)
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