#!/usr/bin/env python3
"""
================================================================================
ANTENNA ARRAY OPTIMIZATION (Beamforming)
================================================================================
Optimize phase shifts in antenna array to maximize signal in desired direction.

This is a REAL problem in:
- 5G/6G communications ($50B+ market)
- Radar systems
- Satellite communications
- WiFi beamforming

Why deep should work:
- Phases are INHERENTLY PERIODIC (0 to 2π)
- Cost function has sinusoidal structure
- Multiple local optima due to phase wrapping
- Very similar to QAOA parameter optimization!
================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
import requests

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "YOUR_LICENSE_HERE"  # Replace with your key

# ============================================================================
# ANTENNA ARRAY PROBLEM
# ============================================================================

class AntennaArrayProblem:
    """
    Uniform Linear Array (ULA) beamforming optimization.
    
    Goal: Find phase shifts that maximize gain in target direction
    while minimizing sidelobes.
    
    The array factor is:
    AF(θ) = Σ exp(j * (k*d*n*sin(θ) + φ_n))
    
    This has PERIODIC structure due to the complex exponentials!
    """
    
    def __init__(self, n_antennas: int = 8, target_angle: float = 30.0,
                 null_angles: list = None, noise_level: float = 0.1, seed: int = None):
        """
        Args:
            n_antennas: Number of antenna elements
            target_angle: Desired beam direction (degrees)
            null_angles: Directions to suppress (degrees)
            noise_level: Measurement noise
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_antennas = n_antennas
        self.target_angle = np.deg2rad(target_angle)
        self.null_angles = [np.deg2rad(a) for a in (null_angles or [60, -45])]
        self.noise_level = noise_level
        
        # Array parameters
        self.d = 0.5  # Element spacing in wavelengths
        self.k = 2 * np.pi  # Wavenumber
        
        # Element positions
        self.positions = np.arange(n_antennas) * self.d
    
    def array_factor(self, phases: np.ndarray, theta: float) -> complex:
        """
        Compute array factor at angle theta.
        
        AF = Σ exp(j * (k*d*n*sin(θ) + φ_n))
        """
        steering = self.k * self.positions * np.sin(theta)
        return np.sum(np.exp(1j * (steering + phases)))
    
    def compute_pattern(self, phases: np.ndarray, angles: np.ndarray = None) -> np.ndarray:
        """Compute radiation pattern over range of angles"""
        if angles is None:
            angles = np.linspace(-np.pi/2, np.pi/2, 181)
        
        pattern = np.array([np.abs(self.array_factor(phases, a))**2 for a in angles])
        return pattern / np.max(pattern)  # Normalize
    
    def evaluate(self, phases: np.ndarray, shots: int = 1024) -> float:
        """
        Evaluate antenna configuration with shot noise.
        
        Cost = -gain_at_target + penalty_for_sidelobes + penalty_for_nulls
        
        We MINIMIZE this, so good configurations have LOW cost.
        """
        # Gain at target (we want to MAXIMIZE this, so negate)
        target_gain = np.abs(self.array_factor(phases, self.target_angle))**2
        target_gain_normalized = target_gain / (self.n_antennas**2)
        
        # Penalty for not having nulls at specified angles
        null_penalty = 0
        for null_angle in self.null_angles:
            null_gain = np.abs(self.array_factor(phases, null_angle))**2
            null_penalty += null_gain / (self.n_antennas**2)
        
        # Sidelobe level (simplified: check a few angles)
        sidelobe_penalty = 0
        for angle in np.linspace(-np.pi/3, np.pi/3, 10):
            if abs(angle - self.target_angle) > 0.2:  # Not near main beam
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
    
    def get_optimal_phases(self) -> np.ndarray:
        """Analytical optimal for simple beam steering (no nulls)"""
        return -self.k * self.positions * np.sin(self.target_angle)


# ============================================================================
# BENCHMARK
# ============================================================================

@dataclass  
class AntennaConfig:
    n_antennas: int = 8
    target_angle: float = 30.0
    noise_level: float = 0.1
    n_steps: int = 100
    shots: int = 1024
    c_shift: float = 0.1
    learning_rate: float = 0.1
    num_seeds: int = 10


def run_single_experiment(
    config: AntennaConfig,
    use_soft_algebra: bool,
    seed: int
) -> Dict:
    """Run single antenna optimization experiment"""
    
    problem = AntennaArrayProblem(
        n_antennas=config.n_antennas,
        target_angle=config.target_angle,
        noise_level=config.noise_level,
        seed=seed
    )
    
    n_params = config.n_antennas
    
    # Initialize phases randomly in [-π, π]
    np.random.seed(seed)
    params = np.random.uniform(-np.pi, np.pi, n_params)
    
    # Pre-generate SPSA deltas
    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=n_params).astype(float) 
                   for _ in range(config.n_steps)]
    
    if use_soft_algebra:
        try:
            r = requests.post(API_URL, json={
                'action': 'start',
                'license_key': LICENSE_KEY,
                'method': 'deep',
                'mode': 'hardware',
                'use_soft_algebra': True,
                'base_lr': config.learning_rate
            }, timeout=10)
            
            data = r.json()
            if not data.get('success'):
                print(f"   API Error: {data.get('error')}")
                return None
            
            session_id = data['session_id']
            use_api = True
        except Exception as e:
            print(f"   API Error: {e}")
            return None
    else:
        use_api = False
        session_id = None
    
    best_true = float('inf')
    
    for step in range(config.n_steps):
        delta = spsa_deltas[step]
        ck = config.c_shift / ((step + 1) ** 0.101)
        
        # SPSA gradient
        e_plus = problem.evaluate(params + ck * delta, config.shots)
        e_minus = problem.evaluate(params - ck * delta, config.shots)
        e_center = problem.evaluate(params, config.shots)
        gradient = (e_plus - e_minus) / (2 * ck) * delta
        
        true_cost = problem.evaluate_true(params)
        best_true = min(best_true, true_cost)
        
        if use_api:
            try:
                r = requests.post(API_URL, json={
                    'action': 'step',
                    'license_key': LICENSE_KEY,
                    'session_id': session_id,
                    'params': params.tolist(),
                    'gradient': gradient.tolist(),
                    'energy': e_center
                }, timeout=10)
                
                resp = r.json()
                if resp.get('success'):
                    params = np.array(resp['new_params'])
                else:
                    ak = config.learning_rate / ((step + 1) ** 0.602)
                    params = params - ak * gradient
            except:
                ak = config.learning_rate / ((step + 1) ** 0.602)
                params = params - ak * gradient
        else:
            ak = config.learning_rate / ((step + 1) ** 0.602)
            params = params - ak * gradient
        
        # Wrap phases to [-π, π]
        params = np.mod(params + np.pi, 2*np.pi) - np.pi
        
        if step % 20 == 0:
            print(f"    Step {step}: cost={true_cost:.4f}, best={best_true:.4f}")
    
    if use_api and session_id:
        try:
            requests.post(API_URL, json={
                'action': 'end',
                'license_key': LICENSE_KEY,
                'session_id': session_id
            }, timeout=5)
        except:
            pass
    
    final_true = problem.evaluate_true(params)
    
    return {
        'final_cost': final_true,
        'best_cost': best_true
    }


def run_full_benchmark(config: AntennaConfig):
    """Run full A/B benchmark"""
    
    print("=" * 60)
    print("ANTENNA ARRAY OPTIMIZATION (Beamforming)")
    print(f"Antennas: {config.n_antennas}, Target: {config.target_angle}°")
    print(f"Method: deep, LR: {config.learning_rate}")
    print("=" * 60)
    
    results_off = []
    results_on = []
    
    for seed in range(config.num_seeds):
        print(f"\n--- Seed {seed + 1}/{config.num_seeds} ---")
        
        print("  Soft Algebra OFF:")
        result_off = run_single_experiment(config, use_soft_algebra=False, seed=seed)
        if result_off:
            results_off.append(result_off)
        
        print("  Soft Algebra ON (deep):")
        result_on = run_single_experiment(config, use_soft_algebra=True, seed=seed)
        if result_on:
            results_on.append(result_on)
    
    if not results_off or not results_on:
        print("\nNo results!")
        return None
    
    cost_off = [r['final_cost'] for r in results_off]
    cost_on = [r['final_cost'] for r in results_on]
    
    mean_off = np.mean(cost_off)
    mean_on = np.mean(cost_on)
    
    wins = sum(1 for off, on in zip(cost_off, cost_on) if on < off)
    win_rate = wins / len(cost_off) * 100
    
    improvement = (mean_off - mean_on) / abs(mean_off) * 100 if mean_off != 0 else 0
    
    from scipy import stats
    _, p_value = stats.ttest_rel(cost_off, cost_on)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Cost (lower = better):")
    print(f"  Soft Algebra OFF: {mean_off:.4f} ± {np.std(cost_off):.4f}")
    print(f"  Soft Algebra ON:  {mean_on:.4f} ± {np.std(cost_on):.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Win Rate: {win_rate:.0f}% ({wins}/{len(cost_off)})")
    print(f"p-value: {p_value:.6f}")
    print("=" * 60)
    
    return {
        'mean_off': mean_off,
        'mean_on': mean_on,
        'improvement': improvement,
        'win_rate': win_rate,
        'p_value': p_value
    }


if __name__ == "__main__":
    config = AntennaConfig(
        n_antennas=16,          # היה 8
        target_angle=30.0,
        noise_level=0.05,       # פחות רעש - יותר קשה להראות יתרון
        n_steps=50,             # היה 100 - פחות זמן
        shots=512,              # היה 1024 - פחות דיוק
        c_shift=0.1,
        learning_rate=0.1,
        num_seeds=10
    )
    
    run_full_benchmark(config)
