"""
Mobiu-Q — Soft Algebra Optimizer for Quantum, RL, LLM & Complex Optimization
=============================================================================
A next-generation optimizer built on Soft Algebra and Demeasurement theory,
enabling stable and efficient optimization in noisy, stochastic environments.

Version: 2.7.3 - MobiuOptimizer + Hybrid Mode for PyTorch

What's New in v2.7:
- MobiuOptimizer: Universal wrapper that auto-detects PyTorch optimizers!
- Hybrid Mode: Cloud intelligence + local PyTorch performance
- Zero Friction: One-line integration for PyTorch users
- Full backward compatibility with all existing code

What's New in v2.6:
- New method names: 'standard', 'deep', 'adaptive' (legacy names still work!)
- 80% win rate across all quantum noise levels
- +32.5% more robust to quantum hardware noise
- +18% improvement on LLM soft prompt tuning

Classes:
    | Class          | Use Case                                   |
    |----------------|---------------------------------------------|
    | MobiuOptimizer | PyTorch (RL, LLM, Deep Learning) - NEW!    |
    | MobiuQCore     | Quantum (VQE, QAOA) & NumPy optimization   |

Methods:
    | Method   | Legacy | Use Case                                    |
    |----------|--------|---------------------------------------------|
    | standard | vqe    | Smooth landscapes, chemistry, physics       |
    | deep     | qaoa   | Deep circuits, noisy hardware, complex opt  |
    | adaptive | rl     | RL, LLM fine-tuning, high-variance problems |

Quick Start (PyTorch - NEW in v2.7!):
    import torch
    from mobiu_q import MobiuOptimizer
    
    model = MyModel()
    base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    opt = MobiuOptimizer(base_opt, method="adaptive")
    
    for epoch in range(100):
        loss = criterion(model(x), y)
        loss.backward()
        opt.step(loss.item())  # Pass loss for Soft Algebra
        opt.zero_grad()
    
    opt.end()

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

For RL / LLM fine-tuning (using MobiuOptimizer):
    base_opt = torch.optim.Adam(policy.parameters(), lr=0.0003)
    opt = MobiuOptimizer(base_opt, method="adaptive")
    
    for episode in range(1000):
        # ... run episode, compute policy gradient ...
        loss.backward()
        opt.step(episode_return)  # Pass return for Soft Algebra
        opt.zero_grad()
    
    opt.end()

A/B Testing (Fair Comparison):
    # With Soft Algebra
    opt = MobiuOptimizer(base_opt, method="adaptive", use_soft_algebra=True)
    
    # Without Soft Algebra (baseline)
    opt = MobiuOptimizer(base_opt, method="adaptive", use_soft_algebra=False)

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
    Available: Adam, AdamW, NAdam, AMSGrad, SGD, Momentum, LAMB
    
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

__version__ = "2.7.0"
__author__ = "Mobiu Technologies"

# Core optimizer classes
from .core import (
    # NEW in v2.7 - Universal wrapper
    MobiuOptimizer,
    # Quantum/NumPy optimizer
    MobiuQCore, 
    # Gradient estimation
    Demeasurement, 
    # Utilities
    get_default_lr,
    get_license_key,
    save_license_key,
    # Constants
    AVAILABLE_OPTIMIZERS,
    DEFAULT_OPTIMIZER,
    METHOD_ALIASES,
    VALID_METHODS,
    API_ENDPOINT,
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
    # NEW in v2.7
    "MobiuOptimizer",
    # Core
    "MobiuQCore",
    "Demeasurement",
    "get_default_lr",
    "get_license_key",
    "save_license_key",
    # Constants
    "AVAILABLE_OPTIMIZERS",
    "DEFAULT_OPTIMIZER",
    "METHOD_ALIASES",
    "VALID_METHODS",
    "API_ENDPOINT",
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