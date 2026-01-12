#!/usr/bin/env python3
"""
================================================================================
BLACK-BOX OPTIMIZATION WITH SHOT NOISE
================================================================================
Mimics the structure of QAOA that works with deep method:
- SPSA gradient estimation (not backprop)
- Shot noise (discrete sampling)
- Rugged landscape with periodic structure

This is essentially "classical QAOA" - same optimization structure without quantum.

Target domains:
- Robotics sensor calibration (noisy measurements)
- Hyperparameter optimization (expensive evaluations)
- Simulation-based optimization (stochastic simulators)
================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
import time

# ============================================================================
# LANDSCAPE FUNCTIONS (Rugged + Periodic)
# ============================================================================

class LandscapeFunctions:
    """
    Test functions with properties similar to QAOA energy landscapes:
    - Multiple local minima
    - Periodic structure
    - Noise sensitivity
    """
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Rastrigin function - highly multimodal with periodic structure.
        Global minimum at x=0 with f(x)=0.
        """
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """
        Ackley function - multimodal with periodic structure.
        Global minimum at x=0 with f(x)=0.
        """
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e
    
    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """
        Schwefel function - deceptive with global minimum far from local minima.
        Similar to QAOA's rugged landscape.
        """
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """
        Griewank function - many local minima with periodic structure.
        """
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        return sum_sq - prod_cos + 1
    
    @staticmethod
    def levy(x: np.ndarray) -> float:
        """
        Levy function - multimodal with sinusoidal components.
        """
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3


# ============================================================================
# SHOT NOISE SIMULATOR
# ============================================================================

class ShotNoiseEvaluator:
    """
    Simulates shot noise like in quantum computing.
    
    Instead of getting exact function value, we "sample" multiple times
    and average - just like measuring a quantum state multiple times.
    """
    
    def __init__(self, 
                 base_function: Callable,
                 shots: int = 1024,
                 noise_scale: float = 0.1):
        """
        Args:
            base_function: The underlying function to optimize
            shots: Number of "measurements" (like quantum shots)
            noise_scale: Scale of per-shot noise
        """
        self.base_function = base_function
        self.shots = shots
        self.noise_scale = noise_scale
    
    def evaluate(self, x: np.ndarray, shots: int = None) -> float:
        """
        Evaluate with shot noise.
        
        Simulates quantum-like measurement:
        - Each "shot" gives a noisy estimate
        - Final value is average of all shots
        - Variance decreases as 1/sqrt(shots)
        """
        if shots is None:
            shots = self.shots
        
        true_value = self.base_function(x)
        
        # Simulate shot noise (like quantum measurement variance)
        # Each shot has noise proportional to |true_value|
        shot_values = true_value + self.noise_scale * np.abs(true_value + 1) * \
                      np.random.randn(shots)
        
        return np.mean(shot_values)
    
    def evaluate_true(self, x: np.ndarray) -> float:
        """Get true value without noise (for evaluation only)"""
        return self.base_function(x)


# ============================================================================
# SPSA OPTIMIZER (Same as QAOA uses)
# ============================================================================

class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation.
    
    This is the same gradient estimation method used in QAOA.
    Key difference from backprop: only needs function evaluations, not gradients.
    """
    
    def __init__(self,
                 evaluator: ShotNoiseEvaluator,
                 n_params: int,
                 c: float = 0.1,        # Perturbation size
                 a: float = 0.1,        # Learning rate
                 alpha: float = 0.602,  # LR decay exponent
                 gamma: float = 0.101): # Perturbation decay exponent
        
        self.evaluator = evaluator
        self.n_params = n_params
        self.c = c
        self.a = a
        self.alpha = alpha
        self.gamma = gamma
        self.step_count = 0
    
    def get_perturbation(self) -> np.ndarray:
        """Generate Bernoulli perturbation vector"""
        return np.random.choice([-1, 1], size=self.n_params).astype(float)
    
    def estimate_gradient(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        SPSA gradient estimation.
        
        Returns:
            energy: Function value at current params
            gradient: Estimated gradient
        """
        k = self.step_count + 1
        ck = self.c / (k ** self.gamma)
        
        delta = self.get_perturbation()
        
        # Two-sided evaluation (like QAOA)
        e_plus = self.evaluator.evaluate(params + ck * delta)
        e_minus = self.evaluator.evaluate(params - ck * delta)
        e_center = self.evaluator.evaluate(params)
        
        # SPSA gradient estimate
        gradient = (e_plus - e_minus) / (2 * ck) * delta
        
        return e_center, gradient
    
    def get_learning_rate(self) -> float:
        """Decaying learning rate"""
        k = self.step_count + 1
        return self.a / (k ** self.alpha)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

@dataclass
class BlackBoxConfig:
    """Configuration for black-box optimization benchmark"""
    n_params: int = 10
    n_steps: int = 100
    shots: int = 1024
    noise_scale: float = 0.1
    c_shift: float = 0.1
    learning_rate: float = 0.1  # deep method default
    function: str = "rastrigin"  # rastrigin, ackley, schwefel, griewank, levy
    num_seeds: int = 10
    bounds: Tuple[float, float] = (-5.0, 5.0)


def run_single_experiment(
    config: BlackBoxConfig,
    use_soft_algebra: bool,
    seed: int
) -> Dict:
    """Run single black-box optimization experiment"""
    
    np.random.seed(seed)
    
    # Select function
    functions = {
        'rastrigin': LandscapeFunctions.rastrigin,
        'ackley': LandscapeFunctions.ackley,
        'schwefel': LandscapeFunctions.schwefel,
        'griewank': LandscapeFunctions.griewank,
        'levy': LandscapeFunctions.levy,
    }
    base_func = functions[config.function]
    
    # Create evaluator with shot noise
    evaluator = ShotNoiseEvaluator(
        base_function=base_func,
        shots=config.shots,
        noise_scale=config.noise_scale
    )
    
    # Initialize parameters
    params = np.random.uniform(config.bounds[0], config.bounds[1], config.n_params)
    
    # Pre-generate SPSA deltas for fair comparison
    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=config.n_params).astype(float) 
                   for _ in range(config.n_steps)]
    
    # Reset seed for reproducibility
    np.random.seed(seed)
    
    # Setup optimizer (Mobiu-Q or baseline)
    if use_soft_algebra:
        try:
            import requests
            API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
            LICENSE_KEY = "YOUR_LICENSE_HERE"
            
            r = requests.post(API_URL, json={
                'action': 'start',
                'license_key': LICENSE_KEY,
                'method': 'deep',
                'mode': 'hardware',  # Important for deep!
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
        # Baseline: simple gradient descent
        use_api = False
        session_id = None
    
    # Optimization loop
    history = []
    best_energy = float('inf')
    best_true = float('inf')
    
    for step in range(config.n_steps):
        delta = spsa_deltas[step]
        ck = config.c_shift / ((step + 1) ** 0.101)
        
        # SPSA gradient estimation
        e_plus = evaluator.evaluate(params + ck * delta)
        e_minus = evaluator.evaluate(params - ck * delta)
        e_center = evaluator.evaluate(params)
        gradient = (e_plus - e_minus) / (2 * ck) * delta
        
        # True energy (for evaluation)
        true_energy = evaluator.evaluate_true(params)
        best_energy = min(best_energy, e_center)
        best_true = min(best_true, true_energy)
        
        history.append({
            'step': step,
            'noisy_energy': e_center,
            'true_energy': true_energy,
            'best_true': best_true
        })
        
        if use_api:
            # Use Mobiu-Q API
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
                    # Fallback
                    ak = config.learning_rate / ((step + 1) ** 0.602)
                    params = params - ak * gradient
                    
            except Exception as e:
                ak = config.learning_rate / ((step + 1) ** 0.602)
                params = params - ak * gradient
        else:
            # Baseline: vanilla SPSA update
            ak = config.learning_rate / ((step + 1) ** 0.602)
            params = params - ak * gradient
        
        # Clip to bounds
        params = np.clip(params, config.bounds[0], config.bounds[1])
        
        if step % 20 == 0:
            print(f"    Step {step}: true={true_energy:.4f}, best={best_true:.4f}")
    
    # Cleanup API session
    if use_api and session_id:
        try:
            requests.post(API_URL, json={
                'action': 'end',
                'license_key': LICENSE_KEY,
                'session_id': session_id
            }, timeout=5)
        except:
            pass
    
    # Final evaluation with more shots
    final_energy = evaluator.evaluate(params, shots=config.shots * 4)
    final_true = evaluator.evaluate_true(params)
    
    return {
        'final_noisy': final_energy,
        'final_true': final_true,
        'best_true': best_true,
        'history': history
    }


def run_full_benchmark(config: BlackBoxConfig):
    """Run full A/B benchmark"""
    
    print("=" * 60)
    print("BLACK-BOX OPTIMIZATION WITH SHOT NOISE")
    print(f"Function: {config.function}, Params: {config.n_params}")
    print(f"Shots: {config.shots}, Noise: {config.noise_scale}")
    print(f"Method: deep, LR: {config.learning_rate}")
    print("=" * 60)
    
    results_off = []
    results_on = []
    
    for seed in range(config.num_seeds):
        print(f"\n--- Seed {seed + 1}/{config.num_seeds} ---")
        
        # Soft Algebra OFF (baseline SPSA)
        print("  Soft Algebra OFF:")
        result_off = run_single_experiment(config, use_soft_algebra=False, seed=seed)
        if result_off:
            results_off.append(result_off)
        
        # Soft Algebra ON (deep method)
        print("  Soft Algebra ON (deep):")
        result_on = run_single_experiment(config, use_soft_algebra=True, seed=seed)
        if result_on:
            results_on.append(result_on)
    
    if not results_off or not results_on:
        print("\nNo results collected!")
        return None
    
    # Aggregate using true energy (lower = better)
    energy_off = [r['final_true'] for r in results_off]
    energy_on = [r['final_true'] for r in results_on]
    
    mean_off = np.mean(energy_off)
    mean_on = np.mean(energy_on)
    std_off = np.std(energy_off)
    std_on = np.std(energy_on)
    
    # Win rate (lower energy = better)
    wins = sum(1 for off, on in zip(energy_off, energy_on) if on < off)
    win_rate = wins / len(energy_off) * 100
    
    # Improvement
    improvement = (mean_off - mean_on) / abs(mean_off) * 100 if mean_off != 0 else 0
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(energy_off, energy_on)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Function: {config.function}")
    print(f"Soft Algebra OFF: {mean_off:.4f} ± {std_off:.4f}")
    print(f"Soft Algebra ON:  {mean_on:.4f} ± {std_on:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Win Rate: {win_rate:.0f}% ({wins}/{len(energy_off)})")
    print(f"p-value: {p_value:.6f}")
    print("=" * 60)
    
    return {
        'results_off': results_off,
        'results_on': results_on,
        'mean_off': mean_off,
        'mean_on': mean_on,
        'improvement': improvement,
        'win_rate': win_rate,
        'p_value': p_value
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = BlackBoxConfig(
        n_params=10,
        n_steps=100,
        shots=1024,
        noise_scale=0.1,
        c_shift=0.1,
        learning_rate=0.1,  # deep method default
        function="rastrigin",  # Start with Rastrigin (periodic + multimodal)
        num_seeds=10,
        bounds=(-5.0, 5.0)
    )
    
    results = run_full_benchmark(config)
