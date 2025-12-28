"""
Mobiu-Q — Soft Algebra Optimizer for Quantum, RL, LLM & Complex Optimization
=============================================================================

A next-generation optimizer built on Soft Algebra and Demeasurement theory,
enabling stable and efficient optimization in noisy, stochastic environments.

Version: 2.7 - New Method Names + Noise Robustness + LLM Support

What's New in v2.5:
- New method names: 'standard', 'deep', 'adaptive' (legacy names still work!)
- 80% win rate across all quantum noise levels
- +32.5% more robust to quantum hardware noise
- +18% improvement on LLM soft prompt tuning
- Full backward compatibility

Methods:
    | Method   | Legacy | Use Case                                    |
    |----------|--------|---------------------------------------------|
    | standard | vqe    | Smooth landscapes, chemistry, physics       |
    | deep     | qaoa   | Deep circuits, noisy hardware, complex opt  |
    | adaptive | rl     | RL, LLM fine-tuning, high-variance problems |

Quick Start (Quantum VQE):
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", method="standard")
    
    for step in range(100):
        E = energy_fn(params)
        grad = Demeasurement.finite_difference(energy_fn, params)
        params = opt.step(params, grad, E)
    
    opt.end()

For Deep circuits / Noisy hardware:
    opt = MobiuQCore(method="deep", mode="hardware")
    
    for step in range(150):
        grad, E = Demeasurement.spsa(energy_fn, params)
        params = opt.step(params, grad, E)
    
    opt.end()

For RL / LLM fine-tuning:
    opt = MobiuQCore(method="adaptive")
    
    for episode in range(1000):
        episode_return = run_episode(policy)
        gradient = compute_policy_gradient()
        params = opt.step(params, gradient, episode_return)
    
    opt.end()

Method & Mode:
    | Method   | Mode       | Use Case                      | Default LR |
    |----------|------------|-------------------------------|------------|
    | standard | simulation | Chemistry, physics (clean)    | 0.01       |
    | standard | hardware   | VQE on quantum hardware       | 0.02       |
    | deep     | simulation | Combinatorial (simulator)     | 0.1        |
    | deep     | hardware   | QAOA on quantum hardware      | 0.1        |
    | adaptive | (any)      | RL, LLM fine-tuning           | 0.0003     |

Optimizers:
    Default: Adam (recommended - works best across all methods)
    Available: Adam, NAdam, AMSGrad, SGD, Momentum, LAMB
    
    Example: MobiuQCore(method="deep", base_optimizer="NAdam")

Benchmark Results:
    - Quantum: 80% win rate, +5% to +65% improvement
    - Noise Robustness: +32.5% more robust than standard optimizers
    - LLM: +18% improvement on soft prompt tuning
    - RL: +129% on LunarLander, +118% on MuJoCo

License:
    Free tier: 20 runs/month
    Pro tier: Unlimited - https://app.mobiu.ai
"""

__version__ = "2.5.0"
__author__ = "Mobiu Technologies"

# Core optimizer
from .core import (
    MobiuQCore, 
    Demeasurement, 
    get_default_lr,
    AVAILABLE_OPTIMIZERS,
    DEFAULT_OPTIMIZER,
    METHOD_ALIASES,
    VALID_METHODS
)

# CLI utilities
from .core import activate_license, check_status

# Problem catalog (optional - for built-in problems)
try:
    from .catalog import (
        PROBLEM_CATALOG,
        get_energy_function,
        get_ground_state_energy,
        list_problems,
        get_method,
        Ansatz
    )
except ImportError:
    # Catalog not installed
    pass

__all__ = [
    # Core
    "MobiuQCore",
    "Demeasurement",
    "get_default_lr",
    "AVAILABLE_OPTIMIZERS",
    "DEFAULT_OPTIMIZER",
    "METHOD_ALIASES",
    "VALID_METHODS",
    # CLI
    "activate_license",
    "check_status",
    # Optional catalog exports
    "PROBLEM_CATALOG",
    "get_energy_function",
    "get_ground_state_energy",
    "list_problems",
    "get_method",
    "Ansatz"
]