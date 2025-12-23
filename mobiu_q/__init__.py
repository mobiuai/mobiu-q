"""
Mobiu-Q — Soft Algebra Optimizer for Quantum, RL & Complex Optimization
========================================================================

A next-generation optimizer built on Soft Algebra and Demeasurement theory,
enabling stable and efficient optimization in noisy, stochastic environments.

Version: 2.4.2

What's New in v2.4:
- Reinforcement Learning support (method="rl")
- Multi-optimizer support (Adam, NAdam, AMSGrad, SGD, Momentum, LAMB)
- +129% improvement on LunarLander, +118% on MuJoCo, +53% on VQE

Quick Start (VQE - Chemistry):
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", method="vqe")
    
    for step in range(100):
        E = energy_fn(params)
        grad = Demeasurement.finite_difference(energy_fn, params)
        params = opt.step(params, grad, E)
    
    opt.end()

For QAOA (Combinatorial Optimization):
    opt = MobiuQCore(method="qaoa", mode="hardware")
    
    for step in range(150):
        grad, E = Demeasurement.spsa(energy_fn, params)
        params = opt.step(params, grad, E)
    
    opt.end()

For RL (Reinforcement Learning) - NEW in v2.4:
    opt = MobiuQCore(method="rl")
    
    for episode in range(1000):
        episode_return = run_episode(policy)
        gradient = compute_policy_gradient()
        params = opt.step(params, gradient, episode_return)
    
    opt.end()

Method & Mode:
    | Method | Mode       | Use Case                    | Default LR |
    |--------|------------|-----------------------------+------------|
    | vqe    | simulation | Chemistry, physics (clean)  | 0.01       |
    | vqe    | hardware   | VQE on quantum hardware     | 0.02       |
    | qaoa   | simulation | Combinatorial (simulator)   | 0.1        |
    | qaoa   | hardware   | QAOA on quantum hardware    | 0.1        |
    | rl     | (any)      | Reinforcement learning      | 0.0003     |

Optimizers (NEW in v2.4):
    Default: Adam (recommended - works best across all methods)
    Available: Adam, NAdam, AMSGrad, SGD, Momentum, LAMB
    
    Example: MobiuQCore(method="qaoa", base_optimizer="NAdam")

License:
    Free tier: 20 runs/month
    Pro tier: Unlimited - https://app.mobiu.ai
"""

__version__ = "2.4.0"
__author__ = "Mobiu Technologies"

# Core optimizer
from .core import (
    MobiuQCore, 
    Demeasurement, 
    get_default_lr,
    AVAILABLE_OPTIMIZERS,
    DEFAULT_OPTIMIZER
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
