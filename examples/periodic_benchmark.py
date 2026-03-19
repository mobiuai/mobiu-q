#!/usr/bin/env python3
"""
Noisy Periodic Optimization Benchmark
Designed specifically for Mobiu-Q deep method

Uses DIRECT API calls (not PyTorch client) for reliable results.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
import requests

API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
LICENSE_KEY = "YOUR_KEY"

# ============================================================================
# Noisy Periodic Loss Landscape
# ============================================================================

class NoisyPeriodicProblem:
    """
    A synthetic optimization problem with:
    - Base quadratic loss (convex)
    - Periodic perturbations (creates local minima)
    - Shot-like noise (discrete sampling)
    """
    
    def __init__(self, dim: int = 10, num_frequencies: int = 5,
                 amplitude: float = 0.5, shot_noise_level: float = 0.1,
                 num_shots: int = 100, seed: int = None):
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.amplitude = amplitude
        self.shot_noise_level = shot_noise_level
        self.num_shots = num_shots
        
        if seed is not None:
            np.random.seed(seed)
        
        # Random frequencies for periodic structure
        self.frequencies = np.random.randn(num_frequencies, dim) * 2.0
        self.phases = np.random.rand(num_frequencies) * 2 * np.pi
        
        # True optimum
        self.true_optimum = np.zeros(dim)
    
    def true_loss(self, params: np.ndarray) -> float:
        """True underlying loss (without noise)"""
        quadratic = 0.5 * np.sum(params ** 2)
        
        periodic = 0.0
        for i in range(self.num_frequencies):
            projection = np.dot(params, self.frequencies[i])
            periodic += np.sin(projection + self.phases[i])
        
        periodic = self.amplitude * periodic / self.num_frequencies
        return quadratic + periodic
    
    def noisy_loss(self, params: np.ndarray) -> float:
        """Loss with shot-like noise"""
        true_val = self.true_loss(params)
        
        if self.num_shots > 0:
            noise_std = self.shot_noise_level / np.sqrt(self.num_shots)
            noise = np.random.randn() * noise_std * (1 + abs(true_val))
        else:
            noise = 0
        
        return true_val + noise
    
    def spsa_gradient(self, params: np.ndarray, delta: np.ndarray, c: float) -> tuple:
        """SPSA gradient estimation"""
        e_plus = self.noisy_loss(params + c * delta)
        e_minus = self.noisy_loss(params - c * delta)
        e_center = self.noisy_loss(params)
        
        grad = (e_plus - e_minus) / (2 * c) * delta
        return grad, e_center


# ============================================================================
# BENCHMARK
# ============================================================================

@dataclass
class PeriodicConfig:
    dim: int = 10
    num_frequencies: int = 5
    amplitude: float = 0.5
    shot_noise_level: float = 0.2
    num_shots: int = 100
    n_steps: int = 100
    c_shift: float = 0.1
    learning_rate: float = 0.1
    num_seeds: int = 10


def run_single_experiment(config: PeriodicConfig, use_soft_algebra: bool, seed: int) -> Dict:
    """Run single experiment using direct API"""
    
    problem = NoisyPeriodicProblem(
        dim=config.dim,
        num_frequencies=config.num_frequencies,
        amplitude=config.amplitude,
        shot_noise_level=config.shot_noise_level,
        num_shots=config.num_shots,
        seed=seed + 1000
    )
    
    # Initialize params
    np.random.seed(seed)
    params = np.random.randn(config.dim) * 0.5
    
    # Pre-generate SPSA deltas
    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=config.dim).astype(float) 
                   for _ in range(config.n_steps)]
    
    # Start API session
    if use_soft_algebra:
        try:
            r = requests.post(API_URL, json={
                'action': 'start',
                'license_key': LICENSE_KEY,
                'method': 'deep',
                'mode': 'simulation',
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
        
        grad, e_center = problem.spsa_gradient(params, delta, ck)
        true_loss = problem.true_loss(params)
        best_true = min(best_true, true_loss)
        
        if use_api:
            try:
                r = requests.post(API_URL, json={
                    'action': 'step',
                    'license_key': LICENSE_KEY,
                    'session_id': session_id,
                    'params': params.tolist(),
                    'gradient': grad.tolist(),
                    'energy': e_center
                }, timeout=10)
                
                resp = r.json()
                if resp.get('success'):
                    params = np.array(resp['new_params'])
                else:
                    ak = config.learning_rate / ((step + 1) ** 0.602)
                    params = params - ak * grad
            except:
                ak = config.learning_rate / ((step + 1) ** 0.602)
                params = params - ak * grad
        else:
            ak = config.learning_rate / ((step + 1) ** 0.602)
            params = params - ak * grad
        
        if step % 20 == 0:
            print(f"    Step {step}: true_loss={true_loss:.4f}")
    
    if use_api and session_id:
        try:
            requests.post(API_URL, json={
                'action': 'end',
                'license_key': LICENSE_KEY,
                'session_id': session_id
            }, timeout=5)
        except:
            pass
    
    final_true = problem.true_loss(params)
    
    return {
        'final_loss': final_true,
        'best_loss': best_true
    }


def run_full_benchmark(config: PeriodicConfig):
    """Run full A/B benchmark"""
    
    print("=" * 60)
    print("NOISY PERIODIC OPTIMIZATION (Direct API)")
    print(f"Dim: {config.dim}, Frequencies: {config.num_frequencies}")
    print(f"Method: deep, LR: {config.learning_rate}")
    print("=" * 60)
    
    results_off = []
    results_on = []
    
    for seed in range(config.num_seeds):
        print(f"\n--- Seed {seed + 1}/{config.num_seeds} ---")
        
        np_state = np.random.get_state()  # save shot noise RNG state

        print("  Soft Algebra OFF:")
        result_off = run_single_experiment(config, use_soft_algebra=False, seed=seed)
        if result_off:
            results_off.append(result_off)

        np.random.set_state(np_state)  # restore for fair comparison

        print("  Soft Algebra ON (deep):")
        result_on = run_single_experiment(config, use_soft_algebra=True, seed=seed)
        if result_on:
            results_on.append(result_on)
    
    if not results_off or not results_on:
        print("\nNo results!")
        return None
    
    losses_off = [r['final_loss'] for r in results_off]
    losses_on = [r['final_loss'] for r in results_on]
    
    mean_off = np.mean(losses_off)
    mean_on = np.mean(losses_on)
    
    wins = sum(1 for off, on in zip(losses_off, losses_on) if on < off)
    win_rate = wins / len(losses_off) * 100
    
    improvement = (mean_off - mean_on) / abs(mean_off) * 100 if mean_off != 0 else 0
    
    from scipy import stats
    _, p_value = stats.ttest_rel(losses_off, losses_on)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"True Loss (lower = better):")
    print(f"  Soft Algebra OFF: {mean_off:.4f} ± {np.std(losses_off):.4f}")
    print(f"  Soft Algebra ON:  {mean_on:.4f} ± {np.std(losses_on):.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Win Rate: {win_rate:.0f}% ({wins}/{len(losses_off)})")
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
    config = PeriodicConfig(
        dim=10,
        num_frequencies=5,
        amplitude=0.5,
        shot_noise_level=0.2,
        num_shots=100,
        n_steps=100,
        c_shift=0.1,
        learning_rate=0.1,
        num_seeds=10
    )
    
    run_full_benchmark(config)