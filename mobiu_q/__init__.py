"""
Mobiu-Q — Soft Algebra Optimizer for Quantum & Complex Optimization
====================================================================

A next-generation optimizer built on Soft Algebra and Demeasurement theory,
enabling stable and efficient optimization in quantum variational algorithms.

Quick Start:
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", mode="standard")
    
    for step in range(100):
        E = energy_fn(params)
        grad = Demeasurement.finite_difference(energy_fn, params)
        params = opt.step(params, grad, E)
    
    opt.end()

For noisy quantum hardware:
    opt = MobiuQCore(mode="noisy")
    
    for step in range(100):
        grad, E = Demeasurement.spsa(energy_fn, params)
        params = opt.step(params, grad, E)
    
    opt.end()

License:
    Free tier: 5 runs/month
    Pro tier: Unlimited - https://mobiu-q.com/pricing
"""

__version__ = "1.0.0"
__author__ = "Mobiu Technologies"

# Core optimizer
from .core import MobiuQCore, Demeasurement

# CLI utilities
from .core import activate_license, check_status

# Problem catalog (optional - for built-in problems)
try:
    from .catalog import (
        PROBLEM_CATALOG,
        get_energy_function,
        get_ground_state_energy,
        list_problems,
        Ansatz
    )
except ImportError:
    # Catalog not installed
    pass

__all__ = [
    "MobiuQCore",
    "Demeasurement",
    "activate_license",
    "check_status",
    # Optional catalog exports
    "PROBLEM_CATALOG",
    "get_energy_function",
    "get_ground_state_energy",
    "list_problems",
    "Ansatz"
]
