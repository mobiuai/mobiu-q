#!/usr/bin/env python3
"""
================================================================================
📡 ANTENNA ARRAY OPTIMIZATION - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure SPSA optimizer (what customer has BEFORE Mobiu-Q)
- Test: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Optimize phase shifts in antenna array to maximize signal in desired direction.

This is a REAL problem in:
- 5G/6G communications ($50B+ market)
- Radar systems
- Satellite communications
- WiFi beamforming

Why Mobiu-Q should work:
- Phases are INHERENTLY PERIODIC (0 to 2π)
- Cost function has sinusoidal structure
- Multiple local optima due to phase wrapping
- Very similar to QAOA parameter optimization!
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
N_STEPS = 50
N_ANTENNAS = 16
TARGET_ANGLE = 30.0  # degrees
NOISE_LEVEL = 0.05
SHOTS = 512
C_SHIFT = 0.1
LR = 0.1
METHOD = "deep"  # deep for periodic/rugged landscape


# ============================================================
# ANTENNA ARRAY PROBLEM
# ============================================================

class AntennaArrayProblem:
    """
    Uniform Linear Array (ULA) beamforming optimization.
    
    Goal: Find phase shifts that maximize gain in target direction
    while minimizing sidelobes.
    """
    
    def __init__(self, n_antennas: int, target_angle: float, 
                 null_angles: list = None, noise_level: float = 0.1, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.n_antennas = n_antennas
        self.target_angle = np.deg2rad(target_angle)
        self.null_angles = [np.deg2rad(a) for a in (null_angles or [60, -45])]
        self.noise_level = noise_level
        
        # Array parameters
        self.d = 0.5  # Element spacing in wavelengths
        self.k = 2 * np.pi  # Wavenumber
        self.positions = np.arange(n_antennas) * self.d
    
    def array_factor(self, phases: np.ndarray, theta: float) -> complex:
        """Compute array factor at angle theta"""
        steering = self.k * self.positions * np.sin(theta)
        return np.sum(np.exp(1j * (steering + phases)))
    
    def evaluate(self, phases: np.ndarray, shots: int = 1024) -> float:
        """Evaluate with shot noise"""
        # Gain at target
        target_gain = np.abs(self.array_factor(phases, self.target_angle))**2
        target_gain_normalized = target_gain / (self.n_antennas**2)
        
        # Null penalty
        null_penalty = 0
        for null_angle in self.null_angles:
            null_gain = np.abs(self.array_factor(phases, null_angle))**2
            null_penalty += null_gain / (self.n_antennas**2)
        
        # Sidelobe penalty
        sidelobe_penalty = 0
        for angle in np.linspace(-np.pi/3, np.pi/3, 10):
            if abs(angle - self.target_angle) > 0.2:
                sl_gain = np.abs(self.array_factor(phases, angle))**2
                sidelobe_penalty += max(0, sl_gain / (self.n_antennas**2) - 0.1)
        
        # Combined cost (lower = better)
        cost = -target_gain_normalized + 0.5 * null_penalty + 0.3 * sidelobe_penalty
        
        # Add shot noise
        shot_noise = self.noise_level * np.random.randn(shots)
        noisy_cost = cost + np.mean(shot_noise) * np.abs(cost + 1)
        
        return noisy_cost
    
    def evaluate_true(self, phases: np.ndarray) -> float:
        """True cost without noise"""
        target_gain = np.abs(self.array_factor(phases, self.target_angle))**2
        target_gain_normalized = target_gain / (self.n_antennas**2)
        
        null_penalty = 0
        for null_angle in self.null_angles:
            null_gain = np.abs(self.array_factor(phases, null_angle))**2
            null_penalty += null_gain / (self.n_antennas**2)
        
        sidelobe_penalty = 0
        for angle in np.linspace(-np.pi/3, np.pi/3, 10):
            if abs(angle - self.target_angle) > 0.2:
                sl_gain = np.abs(self.array_factor(phases, angle))**2
                sidelobe_penalty += max(0, sl_gain / (self.n_antennas**2) - 0.1)
        
        return -target_gain_normalized + 0.5 * null_penalty + 0.3 * sidelobe_penalty


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("📡 ANTENNA ARRAY OPTIMIZATION - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Antennas: {N_ANTENNAS} | Target: {TARGET_ANGLE}° | Steps: {N_STEPS}")
    print(f"Seeds: {N_SEEDS} | LR: {LR}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure SPSA optimizer (NO Mobiu)")
    print("  • Test: SPSA + Mobiu-Q enhancement")
    print("  • Real 5G/6G beamforming problem!")
    print("=" * 70)
    
    baseline_results = []
    mobiu_results = []
    
    print(f"\n{'Seed':<8} {'Pure SPSA':<12} {'SPSA+Mobiu':<12} {'Δ%':<10} {'Winner'}")
    print("-" * 55)
    
    for seed in range(N_SEEDS):
        # Create problem
        problem = AntennaArrayProblem(
            n_antennas=N_ANTENNAS,
            target_angle=TARGET_ANGLE,
            noise_level=NOISE_LEVEL,
            seed=seed
        )
        
        # Initialize phases randomly in [-π, π]
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, N_ANTENNAS)
        
        # Pre-generate SPSA deltas
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=N_ANTENNAS).astype(float) 
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
            
            # SPSA gradient
            e_plus = problem.evaluate(params + ck * delta, SHOTS)
            e_minus = problem.evaluate(params - ck * delta, SHOTS)
            e_center = problem.evaluate(params, SHOTS)
            gradient = (e_plus - e_minus) / (2 * ck) * delta
            
            true_cost = problem.evaluate_true(params)
            baseline_best = min(baseline_best, true_cost)
            
            # Pure SPSA update
            params = params - ak * gradient
            
            # Wrap phases to [-π, π]
            params = np.mod(params + np.pi, 2*np.pi) - np.pi
        
        baseline_results.append(baseline_best)
        
        # ─────────────────────────────────────────────────────────────────
        # MOBIU-Q: SPSA + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
        # ─────────────────────────────────────────────────────────────────
        params = init_params.copy()
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=METHOD,
            mode='hardware',
            base_lr=LR,
            verbose=False
        )
        mobiu_best = float('inf')
        
        for step in range(N_STEPS):
            delta = spsa_deltas[step]
            ck = C_SHIFT / ((step + 1) ** 0.101)
            
            # SPSA gradient (same as baseline)
            e_plus = problem.evaluate(params + ck * delta, SHOTS)
            e_minus = problem.evaluate(params - ck * delta, SHOTS)
            e_center = problem.evaluate(params, SHOTS)
            gradient = (e_plus - e_minus) / (2 * ck) * delta
            
            true_cost = problem.evaluate_true(params)
            mobiu_best = min(mobiu_best, true_cost)
            
            # Mobiu-Q enhanced update
            params = mobiu_opt.step(params, gradient, e_center)
            
            # Wrap phases to [-π, π]
            params = np.mod(params + np.pi, 2*np.pi) - np.pi
        
        mobiu_opt.end()
        mobiu_results.append(mobiu_best)
        
        # Calculate improvement (lower is better for cost)
        if abs(baseline_best) > 1e-10:
            improvement = (baseline_best - mobiu_best) / abs(baseline_best) * 100
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
    
    print(f"\nCost (lower = better):")
    print(f"Pure SPSA avg:    {baseline_avg:.4f} ± {np.std(baseline_arr):.4f}")
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
