"""
Mobiu-Q — Soft Algebra Optimizer for Quantum, RL, LLM & Complex Optimization
=============================================================================
A next-generation optimizer built on Soft Algebra and Demeasurement theory,
enabling stable and efficient optimization in noisy, stochastic environments.

Version: 2.9.0 - The "Frustration Engine" Update

Classes:
    | Class                      | Use Case                                   |
    |----------------------------|---------------------------------------------|
    | MobiuOptimizer             | PyTorch (RL, LLM, Deep Learning)           |
    | MobiuQCore                 | Quantum (VQE, QAOA) & NumPy optimization   |
    | UniversalFrustrationEngine | Stagnation detection & LR boosting         |

Methods:
    | Method   | Legacy | Use Case                                    |
    |----------|--------|---------------------------------------------|
    | standard | vqe    | Smooth landscapes, chemistry, physics       |
    | deep     | qaoa   | Deep circuits, noisy hardware, complex opt  |
    | adaptive | rl     | RL, LLM fine-tuning, high-variance problems |

Quick Start (PyTorch):
    import torch
    from mobiu_q import MobiuOptimizer
    
    model = MyModel()
    base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    opt = MobiuOptimizer(base_opt, method="adaptive")
    
    for epoch in range(100):
        loss = criterion(model(x), y)
        loss.backward()
        opt.step(loss.item())
        opt.zero_grad()
    
    opt.end()

Quick Start (RL with Frustration Engine):
    from mobiu_q import MobiuOptimizer
    
    base_opt = torch.optim.Adam(policy.parameters(), lr=0.0003)
    opt = MobiuOptimizer(base_opt, method="adaptive", maximize=True)
    
    for episode in range(1000):
        reward = run_episode(policy)
        loss.backward()
        opt.step(reward)  # Frustration Engine auto-detects stagnation
        opt.zero_grad()
    
    opt.end()

Quick Start (Quantum VQE):
    from mobiu_q import MobiuQCore
    
    opt = MobiuQCore(license_key="your-key", method="standard")
    
    for step in range(100):
        params = opt.step(params, energy_fn)
    
    opt.end()

License:
    Free tier: 20 runs/month
    Pro tier: Unlimited - https://app.mobiu.ai
"""

__version__ = "2.9.0"
__author__ = "Mobiu Technologies"

from .core import (
    # Universal wrapper for PyTorch
    MobiuOptimizer,
    # Frustration Engine (v2.9)
    UniversalFrustrationEngine,
    # Quantum/NumPy optimizer
    MobiuQCore, 
    # Gradient estimation
    Demeasurement, 
    # Utilities
    get_default_lr,
    get_license_key,
    save_license_key,
    activate_license,
    check_status,
    # Constants
    AVAILABLE_OPTIMIZERS,
    DEFAULT_OPTIMIZER,
    METHOD_ALIASES,
    VALID_METHODS,
    API_ENDPOINT,
)

__all__ = [
    # Optimizers
    "MobiuOptimizer",
    "UniversalFrustrationEngine",
    "MobiuQCore",
    "Demeasurement",
    # Utilities
    "get_default_lr",
    "get_license_key",
    "save_license_key",
    "activate_license",
    "check_status",
    # Constants
    "AVAILABLE_OPTIMIZERS",
    "DEFAULT_OPTIMIZER",
    "METHOD_ALIASES",
    "VALID_METHODS",
    "API_ENDPOINT",
]