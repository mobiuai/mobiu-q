"""
Mobiu-Q Client - Soft Algebra Optimizer
========================================
Cloud-connected optimizer for quantum, RL, and LLM applications.

Version: 4.0.1 - Frustration Engine for Quantum

NEW in v2.7:
- MobiuOptimizer: Universal wrapper that auto-detects PyTorch optimizers
- Hybrid mode: Uses cloud for Soft Algebra intelligence, local PyTorch for updates
- Zero friction: Same API for quantum and deep learning

Method names:
- method='standard' (was 'vqe'): For smooth landscapes, chemistry, physics
- method='deep' (was 'qaoa'): For deep circuits, noisy hardware, complex optimization
- method='adaptive' (was 'rl'): For RL, LLM fine-tuning, high-variance problems

Backward compatible: 'vqe', 'qaoa', 'rl' still work!

Usage (PyTorch - NEW!):
    import torch
    from mobiu_q import MobiuOptimizer
    
    base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    opt = MobiuOptimizer(base_opt, license_key="your-key", method="adaptive")
    
    for epoch in range(100):
        loss = compute_loss(model, batch)
        loss.backward()
        opt.step(loss.item())  # Mobiu-Q adjusts LR, PyTorch updates weights
        opt.zero_grad()
    
    opt.end()

Usage (Quantum - unchanged):
    from mobiu_q import MobiuQCore
    
    opt = MobiuQCore(license_key="your-key", method="standard")
    
    for step in range(100):
        params = opt.step(params, energy_fn)
    
    opt.end()

NEW in v2.7.3:
- sync_interval: Contact cloud every N steps (default: 50 for Deep Learning)
- Reduces latency overhead from 1200% to ~5-10%
- Smoothed loss signal improves Soft Algebra accuracy
"""

import numpy as np
import requests
from typing import Optional, Tuple, List, Union
import os
import json
import warnings
import time
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_ENDPOINT = os.environ.get(
    "MOBIU_Q_API_ENDPOINT",
    "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
)

LICENSE_KEY_FILE = os.path.expanduser("~/.mobiu_q_license")

# Available optimizers (must match server's AVAILABLE_OPTIMIZERS)
AVAILABLE_OPTIMIZERS = ["Adam", "AdamW", "NAdam", "AMSGrad", "SGD", "Momentum", "LAMB"]
DEFAULT_OPTIMIZER = "Adam"

# Method name mapping (new names + legacy support)
METHOD_ALIASES = {
    # New names (v2.5+)
    "standard": "standard",
    "deep": "deep", 
    "adaptive": "adaptive",
    # Legacy names (backward compatibility)
    "vqe": "standard",
    "qaoa": "deep",
    "rl": "adaptive",
}

VALID_METHODS = list(METHOD_ALIASES.keys())


def get_license_key() -> Optional[str]:
    """Get license key from environment or file."""
    key = os.environ.get("MOBIU_Q_LICENSE_KEY")
    if key:
        return key
    
    if os.path.exists(LICENSE_KEY_FILE):
        with open(LICENSE_KEY_FILE, "r") as f:
            return f.read().strip()
    
    return None


def save_license_key(key: str):
    """Save license key to file."""
    with open(LICENSE_KEY_FILE, "w") as f:
        f.write(key)
    print(f"✅ License key saved to {LICENSE_KEY_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT LEARNING RATE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def get_default_lr(method: str, mode: str) -> float:
    """
    Get default learning rate based on method and mode.
    
    | Method    | Mode       | Default LR |
    |-----------|------------|------------|
    | standard  | simulation | 0.01       |
    | standard  | hardware   | 0.02       |
    | deep      | simulation | 0.1        |
    | deep      | hardware   | 0.1        |
    | adaptive  | any        | 0.0003     |
    
    Legacy names (vqe, qaoa, rl) are automatically mapped.
    """
    method = METHOD_ALIASES.get(method, method)
    
    if method == 'adaptive':
        return 0.0003
    elif method == 'deep':
        return 0.1
    elif mode == 'hardware':
        return 0.02
    else:  # standard + simulation
        return 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FRUSTRATION ENGINE (CLIENT-SIDE)
# ═══════════════════════════════════════════════════════════════════════════════

class UniversalFrustrationEngine:
    """
    Logic Injection Engine that detects stagnation and boosts Learning Rate.
    Works entirely client-side for zero latency.
    """
    def __init__(self, base_lr: float, sensitivity: float = 0.05):
        self.base_lr = base_lr
        self.history = deque(maxlen=50)
        self.cooldown = 0
        self.best_metric = -float('inf')
        self.stagnation_counter = 0
        self.sensitivity = sensitivity 

    def get_lr_factor(self, current_metric: float) -> float:
        """
        Returns a multiplier factor (e.g. 1.0, 2.0, 3.0) for the LR.
        Assumes 'current_metric' is something we want to MAXIMIZE.
        """
        self.history.append(current_metric)
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if self.cooldown > 0:
            self.cooldown -= 1
            decay = 0.95 ** (30 - self.cooldown)
            return 1.0 + (2.0 * decay)

        if len(self.history) >= 20:
            recent_avg = np.mean(list(self.history)[-10:])
            old_avg = np.mean(list(self.history)[:10])
            is_stuck = (recent_avg < old_avg + abs(old_avg) * self.sensitivity)
            
            if is_stuck and self.stagnation_counter > 20:
                self.cooldown = 30
                self.history.clear()
                self.stagnation_counter = 0
                return 3.0

        return 1.0
    
    def reset(self):
        """Reset engine state for new run."""
        self.history.clear()
        self.cooldown = 0
        self.best_metric = -float('inf')
        self.stagnation_counter = 0

# ═══════════════════════════════════════════════════════════════════════════════
# MOBIU OPTIMIZER - UNIVERSAL WRAPPER (NEW in v2.7!)
# ═══════════════════════════════════════════════════════════════════════════════

class MobiuOptimizer:
    """
    Universal Mobiu-Q Optimizer - wraps any optimizer with Soft Algebra intelligence.
    
    Auto-detects PyTorch optimizers and uses hybrid mode:
    - Cloud computes adaptive learning rate (Soft Algebra + Super-Equation)
    - Local PyTorch handles weight updates (fast, precise, GPU-accelerated)
    
    For non-PyTorch usage (quantum), delegates to MobiuQCore.
    
    Args:
        optimizer_or_params: Either:
            - torch.optim.Optimizer: PyTorch optimizer to wrap (hybrid mode)
            - np.ndarray or list: Initial parameters (quantum mode, uses MobiuQCore)
        license_key: Your Mobiu-Q license key
        method: "standard", "deep", or "adaptive" (legacy: "vqe", "qaoa", "rl")
        mode: "simulation" (clean, uses finite difference) or "hardware" (noisy, uses SPSA)
        verbose: Print status messages
        **kwargs: Additional arguments passed to MobiuQCore (quantum mode only)
    
    Example (PyTorch - Recommended for RL/LLM):
        import torch
        from mobiu_q import MobiuOptimizer
        
        model = MyModel()
        base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
        opt = MobiuOptimizer(base_opt, method="adaptive")
        
        for epoch in range(100):
            loss = criterion(model(x), y)
            loss.backward()
            opt.step(loss.item())  # Pass loss value for Soft Algebra
            opt.zero_grad()
        
        opt.end()
    
    Example (RL with episode returns):
        opt = MobiuOptimizer(base_opt, method="adaptive")
        
        for episode in range(1000):
            # ... run episode, compute policy gradient ...
            loss.backward()
            opt.step(episode_return)  # Pass return for Soft Algebra
            opt.zero_grad()
        
        opt.end()
    
    Example (Quantum - delegates to MobiuQCore):
        params = np.random.randn(10)
        opt = MobiuOptimizer(params, method="standard")
        
        for step in range(100):
            params = opt.step(params, gradient, energy)
        
        opt.end()
    """
    
    def __init__(
        self,
        optimizer_or_params,
        license_key: Optional[str] = None,
        method: str = "adaptive",
        mode: str = "simulation",  # להוסיף!
        use_soft_algebra: bool = True,
        sync_interval: Optional[int] = None,
        maximize: bool = False,
        verbose: bool = True,
        problem: Optional[str] = None,
        **kwargs
    ):
        self.license_key = license_key or get_license_key()
        if not self.license_key:
            raise ValueError(
                "License key required. Set MOBIU_Q_LICENSE_KEY environment variable, "
                "or pass license_key parameter, or run: mobiu-q activate YOUR_KEY"
            )
        
        # Handle deprecated 'problem' parameter
        if problem is not None:
            warnings.warn(
                "Parameter 'problem' is deprecated, use 'method' instead",
                DeprecationWarning,
                stacklevel=2
            )
            if method == "adaptive":  # Only override if method wasn't explicitly set
                method = problem
        
        # Validate method
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
        
        self.method = METHOD_ALIASES.get(method, method)
        self._original_method = method
        self.verbose = verbose
        self.use_soft_algebra = use_soft_algebra
        
        # Auto-detect: Is this a PyTorch optimizer?
        self._is_pytorch = (
            hasattr(optimizer_or_params, 'step') and 
            hasattr(optimizer_or_params, 'param_groups') and
            hasattr(optimizer_or_params, 'zero_grad')
        )
        
        if self._is_pytorch:
            # Smart default for sync_interval
            if sync_interval is None:
                sync_interval = 50  # Default for Deep Learning
    
            # Hybrid mode: PyTorch optimizer + Cloud LR
            self._backend = _MobiuPyTorchBackend(
                optimizer_or_params, 
                self.license_key, 
                self.method,
                base_lr=kwargs.get('base_lr'),
                use_soft_algebra=use_soft_algebra,
                sync_interval=sync_interval,
                maximize=maximize,
                verbose=verbose
            )
        else:
            # Quantum mode: Full cloud optimization
            self._backend = MobiuQCore(
                license_key=self.license_key,
                method=method,
                mode=mode,
                use_soft_algebra=use_soft_algebra,
                verbose=verbose,
                **kwargs
            )
    
    @property
    def problem(self) -> str:
        """Deprecated: Use 'method' instead."""
        warnings.warn(
            "Attribute 'problem' is deprecated, use 'method' instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.method
    
    def step(self, *args, **kwargs):
        """
        Perform optimization step.
        
        For PyTorch (hybrid mode):
            opt.step(loss_value)  # Pass scalar loss/return
            opt.step()            # Use last loss value
        
        For Quantum (MobiuQCore mode):
            params = opt.step(params, gradient, energy)
            params = opt.step(params, energy_fn)  # Auto-gradient
        """
        return self._backend.step(*args, **kwargs)
    
    def zero_grad(self):
        """Zero gradients (PyTorch mode only)."""
        if hasattr(self._backend, 'zero_grad'):
            self._backend.zero_grad()
    
    def set_metric(self, metric: float):
        """
        Store metric for next step() call.
    
        Use this for frameworks that call step() without arguments (e.g., Stable-Baselines3).
    
        Example:
            # In callback when episode ends:
            optimizer.set_metric(episode_return)
        
            # Framework calls step() without args - uses stored metric
        """
        if hasattr(self._backend, 'set_metric'):
            self._backend.set_metric(metric)

    def end(self):
        """End optimization session."""
        self._backend.end()
    
    def new_run(self):
        """Reset for new optimization run (multi-seed experiments)."""
        if hasattr(self._backend, 'new_run'):
            self._backend.new_run()
    
    def reset(self):
        """
        DEPRECATED: Use new_run() for multi-seed experiments.
        """
        warnings.warn(
            "reset() is deprecated and counts each call as a separate run. "
            "Use new_run() for multi-seed experiments (counts as 1 run total).",
            DeprecationWarning,
            stacklevel=2
        )
        if hasattr(self._backend, 'reset'):
            self._backend.reset()
        else:
            self.end()
            self._backend._start_session()
    
    def check_usage(self) -> dict:
        """Check current usage without affecting quota."""
        if hasattr(self._backend, 'check_usage'):
            return self._backend.check_usage()
        return {}
    
    def get_server_info(self) -> dict:
        """Get server information including available methods and optimizers."""
        if hasattr(self._backend, 'get_server_info'):
            return self._backend.get_server_info()
        return {
            "available_optimizers": AVAILABLE_OPTIMIZERS,
            "default_optimizer": DEFAULT_OPTIMIZER,
            "methods": ["standard", "deep", "adaptive"],
            "legacy_methods": ["vqe", "qaoa", "rl"]
        }
    
    @property
    def is_pytorch_mode(self) -> bool:
        """True if using hybrid PyTorch mode."""
        return self._is_pytorch
    
    @property
    def energy_history(self) -> List[float]:
        """Energy/loss history."""
        return self._backend.energy_history
    
    @property
    def lr_history(self) -> List[float]:
        """Learning rate history."""
        return self._backend.lr_history

    @property
    def warp_history(self) -> List[float]:
        """Gradient warp factor history."""
        return getattr(self._backend, 'warp_history', [])

    @property
    def sync_interval(self) -> Optional[int]:
        """Get current sync interval (PyTorch mode only)."""
        if hasattr(self._backend, 'sync_interval'):
            return self._backend.sync_interval
        return None

    @sync_interval.setter
    def sync_interval(self, value: int):
        """Set sync interval (PyTorch mode only)."""
        if hasattr(self._backend, 'sync_interval'):
            self._backend.sync_interval = value
    
    @property
    def remaining_runs(self) -> Optional[int]:
        """Get remaining runs (None if unknown or unlimited)."""
        if hasattr(self._backend, 'remaining_runs'):
            return self._backend.remaining_runs
        return None
    
    @property
    def available_optimizers(self) -> List[str]:
        """List of available optimizers."""
        if hasattr(self._backend, 'available_optimizers'):
            return self._backend.available_optimizers
        return AVAILABLE_OPTIMIZERS
    
    @property
    def param_groups(self):
        """Expose param_groups for framework compatibility (e.g., SB3)."""
        if hasattr(self._backend, 'optimizer'):
            return self._backend.optimizer.param_groups
        return []
    
    @property
    def state(self):
        """Expose optimizer state for framework compatibility."""
        if hasattr(self._backend, 'optimizer'):
            return self._backend.optimizer.state
        return {}

    def __del__(self):
        try:
            self.end()
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH BACKEND (INTERNAL)
# ═══════════════════════════════════════════════════════════════════════════════

class _MobiuPyTorchBackend:
    """
    Internal backend for PyTorch hybrid mode.
    
    - Sends energy/loss to cloud
    - Receives adaptive_lr from cloud (Soft Algebra intelligence)
    - Updates local PyTorch optimizer's LR
    - Executes step locally (fast, precise)
    """
    
    def __init__(self, optimizer, license_key: str, method: str, 
                base_lr: Optional[float] = None,
                use_soft_algebra: bool = True, sync_interval: int = 50,
                maximize: bool = False, verbose: bool = True):
        self.optimizer = optimizer
        self.license_key = license_key
        self.method = method
        self.use_soft_algebra = use_soft_algebra
        self.verbose = verbose
        self.session_id = None
        self.api_endpoint = API_ENDPOINT
        
        # Get LR: explicit base_lr > optimizer default logic
        if base_lr is not None:
            self.base_lr = base_lr
        else:
            optimizer_lr = optimizer.param_groups[0]['lr']
            default_lrs = {"standard": 0.01, "deep": 0.1, "adaptive": 0.0003}
            if optimizer_lr == 0.001:  # PyTorch Adam default
                self.base_lr = default_lrs.get(method, 0.01)
            else:
                self.base_lr = optimizer_lr
        
        # Frustration Engine
        self.maximize = maximize
        self.frustration_engine = UniversalFrustrationEngine(base_lr=self.base_lr) if use_soft_algebra else None
        
        # Tracking
        self.energy_history = []
        self.lr_history = []
        self.warp_history = []
        self._last_energy = None
        self._usage_info = None
        self._available_optimizers = AVAILABLE_OPTIMIZERS
        self.sync_interval = sync_interval
        self._local_step_count = 0
        self._accumulated_metric = 0.0
        self._metric_count = 0
        self._stored_metric = None
        
        # Start session
        self._start_session()
    
    def _start_session(self):
        """Initialize cloud session."""
        try:
            r = requests.post(self.api_endpoint, json={
                'action': 'start',
                'license_key': self.license_key,
                'method': self.method,
                'mode': 'simulation',
                'base_lr': self.base_lr,
                'base_optimizer': 'Adam',
                'use_soft_algebra': self.use_soft_algebra,
                'maximize': self.maximize
            }, timeout=10)
            
            data = r.json()
            
            if data.get('success'):
                self.session_id = data['session_id']
                self._usage_info = data.get('usage', {})
                self._available_optimizers = data.get('available_optimizers', AVAILABLE_OPTIMIZERS)
                
                if self.verbose:
                    remaining = self._usage_info.get('remaining', 'unknown')
                    tier = self._usage_info.get('tier', 'unknown')
                    
                    mode_str = f"method={self.method}, base_lr={self.base_lr}"
                    if self.sync_interval > 1:
                        mode_str += f", sync={self.sync_interval}"
                    if not self.use_soft_algebra:
                        mode_str += ", SA=off"
                    
                    if remaining == 'unlimited':
                        print(f"🚀 Mobiu-Q Hybrid session started (Pro tier) [{mode_str}]")
                    elif isinstance(remaining, int):
                        if remaining <= 2:
                            print(f"⚠️  Mobiu-Q Hybrid session started - LOW QUOTA: {remaining} runs remaining!")
                        else:
                            print(f"🚀 Mobiu-Q Hybrid session started ({remaining} runs remaining) [{mode_str}]")
                    else:
                        print(f"🚀 Mobiu-Q Hybrid session started [{mode_str}]")
            else:
                if self.verbose:
                    error = data.get('error', 'Unknown error')
                    if "limit" in error.lower() or "quota" in error.lower():
                        print(f"❌ {error}")
                        print("   Upgrade at: https://app.mobiu.ai")
                    else:
                        print(f"⚠️  API start failed: {error}. Using constant LR.")
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Cannot connect to Mobiu-Q: {e}. Using constant LR.")
    
    def set_metric(self, metric: float):
        """Store metric for next step() call (for frameworks like SB3)."""
        self._stored_metric = metric

    def step(self, metric: float = None):
        """
        metric: The Loss (minimize) or Reward (maximize).
        """
        # Use stored metric if none provided (for SB3 compatibility)
        if metric is None:
            metric = self._stored_metric
    
        self._local_step_count += 1
        
        # 1. FRUSTRATION ENGINE (Client-Side Logic)
        if self.frustration_engine and metric is not None:
            # Engine always wants "Higher is Better" for its internal logic
            score = metric if self.maximize else -metric
            
            # Get Boost Factor (1.0, 2.0, or 3.0)
            factor = self.frustration_engine.get_lr_factor(score)
            
            # If Engine detects stagnation, apply boost immediately
            if factor > 1.0:
                new_lr = self.base_lr * factor
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                # Log only when actual change happens
                self.lr_history.append(new_lr)

        # 2. CLOUD SYNC (Soft Algebra)
        if metric is not None:
            self._accumulated_metric += metric
            self._metric_count += 1

        should_sync = (
            self.use_soft_algebra and  # הוספה!
            self.session_id and self._metric_count > 0 and
            (self._local_step_count % self.sync_interval == 0)
        )

        if should_sync:
            avg_metric = self._accumulated_metric / self._metric_count
            
            # --- FIX: Direction Correction ---
            # Cloud assumes Physics/Energy (Lower = Better).
            # If we are Maximizing (Reward/Sharpe), we flip sign so Cloud sees "Energy dropping".
            energy_to_send = avg_metric
            
            try:
                # Send to cloud for Soft Algebra analysis
                r = requests.post(self.api_endpoint, json={
                    'action': 'step',
                    'license_key': self.license_key,
                    'session_id': self.session_id,
                    'params': [0.0], 
                    'gradient': [0.0],
                    'energy': energy_to_send
                }, timeout=1.0)
                
                data = r.json()
                if data.get('success'):
                    # Update base LR from Soft Algebra
                    if 'adaptive_lr' in data:
                        self.base_lr = data['adaptive_lr']
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = data['adaptive_lr']
                        self.lr_history.append(data['adaptive_lr'])
                    
                    # NEW: Apply gradient warping from server
                    warp_factor = data.get('warp_factor', 1.0)
                    self.warp_history.append(warp_factor)
                    
                    if warp_factor != 1.0:
                        for pg in self.optimizer.param_groups:
                            for param in pg['params']:
                                if param.grad is not None:
                                    param.grad.data.mul_(warp_factor)
                    
            except: pass
            
            self._accumulated_metric = 0.0
            self._metric_count = 0

        # 3. PyTorch Step (Execute weights update)
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def new_run(self):
        """Reset for new run."""
        self.energy_history.clear()
        self.lr_history.clear()
        self.warp_history.clear()  # NEW
        self._last_energy = None
    
        # Reset sync counters
        self._local_step_count = 0
        self._accumulated_metric = 0.0
        self._metric_count = 0
        
        # Reset Frustration Engine
        if self.frustration_engine:
            self.frustration_engine.reset()
    
        # Reset optimizer state (momentum, etc.)
        self.optimizer.state.clear()
        
        # Reset LR to base
        for group in self.optimizer.param_groups:
            group['lr'] = self.base_lr
        
        # Reset cloud session state
        if self.session_id:
            try:
                requests.post(self.api_endpoint, json={
                    'action': 'reset',
                    'license_key': self.license_key,
                    'session_id': self.session_id
                }, timeout=5)
            except:
                pass
    
    def check_usage(self) -> dict:
        """Check current usage without affecting quota."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "action": "usage"
                },
                timeout=10
            )
            data = response.json()
            if data.get("success"):
                self._usage_info = data.get("usage", {})
                return self._usage_info
        except:
            pass
        return {}
    
    def get_server_info(self) -> dict:
        """Get server information."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "action": "info"
                },
                timeout=10
            )
            data = response.json()
            if data.get("success"):
                return data
        except:
            pass
        return {
            "available_optimizers": AVAILABLE_OPTIMIZERS,
            "default_optimizer": DEFAULT_OPTIMIZER,
            "methods": ["standard", "deep", "adaptive"],
            "legacy_methods": ["vqe", "qaoa", "rl"]
        }
    
    @property
    def remaining_runs(self) -> Optional[int]:
        """Get remaining runs."""
        if self._usage_info:
            remaining = self._usage_info.get('remaining')
            if remaining == 'unlimited':
                return None
            return remaining
        return None
    
    @property
    def available_optimizers(self) -> List[str]:
        """List of available optimizers."""
        return self._available_optimizers
    
    def end(self):
        """End session."""
        if self.session_id:
            try:
                response = requests.post(self.api_endpoint, json={
                    'action': 'end',
                    'license_key': self.license_key,
                    'session_id': self.session_id
                }, timeout=5)
                
                data = response.json()
                self._usage_info = data.get('usage', {})
                
                if self.verbose:
                    remaining = self._usage_info.get('remaining', 'unknown')
                    
                    if remaining == 'unlimited':
                        print(f"✅ Session ended (Pro tier)")
                    elif remaining == 0:
                        print(f"✅ Session ended")
                        print(f"❌ Quota exhausted! Upgrade at: https://app.mobiu.ai")
                    elif isinstance(remaining, int) and remaining <= 2:
                        print(f"✅ Session ended")
                        print(f"⚠️  Low quota warning: {remaining} runs remaining")
                    else:
                        print(f"✅ Session ended ({remaining} runs remaining)")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not cleanly end session: {e}")
            
            self.session_id = None


# ═══════════════════════════════════════════════════════════════════════════════
# MOBIU-Q CORE (Cloud Client) - FULL BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class MobiuQCore:
    """
    Mobiu-Q Optimizer - Cloud-connected version for quantum optimization.
    
    For PyTorch users, consider using MobiuOptimizer instead for better performance.
    
    Args:
        license_key: Your Mobiu-Q license key (or set MOBIU_Q_LICENSE_KEY env var)
        method: Optimization method:
            - "standard" (or legacy "vqe"): For smooth landscapes, chemistry, physics
            - "deep" (or legacy "qaoa"): For deep circuits, noisy hardware
            - "adaptive" (or legacy "rl"): For RL, LLM fine-tuning, high-variance
        mode: "simulation" (clean) or "hardware" (noisy quantum hardware)
        base_lr: Learning rate (default: computed from method+mode)
        base_optimizer: Optimizer: "Adam" (default), "AdamW", "NAdam", "AMSGrad", "SGD", "Momentum", "LAMB"
        use_soft_algebra: Enable Soft Algebra enhancement (default: True)
        offline_fallback: If True, use local Adam when API unavailable
    
    Default Learning Rates:
        | Method    | Mode       | Default LR |
        |-----------|------------|------------|
        | standard  | simulation | 0.01       |
        | standard  | hardware   | 0.02       |
        | deep      | simulation | 0.1        |
        | deep      | hardware   | 0.1        |
        | adaptive  | any        | 0.0003     |
    
    Example (Quantum VQE):
        opt = MobiuQCore(license_key="xxx", method="standard")
        
        for step in range(100):
            grad = Demeasurement.finite_difference(energy_fn, params)
            params = opt.step(params, grad, energy_fn(params))
        
        opt.end()
    
    Example (Auto gradient - recommended):
        opt = MobiuQCore(license_key="xxx", method="standard")
        
        for step in range(100):
            params = opt.step(params, energy_fn)  # Gradient auto-computed!
        
        opt.end()
    
    Example (multi-seed, counts as 1 run):
        opt = MobiuQCore(license_key="xxx")
        
        for seed in range(10):
            opt.new_run()  # Reset optimizer state, keep session
            params = init_params(seed)
            for step in range(100):
                params = opt.step(params, grad, energy)
        
        opt.end()  # Counts as 1 run total
    """
    
    def __init__(
        self,
        license_key: Optional[str] = None,
        method: str = "standard",
        mode: str = "simulation",
        base_lr: Optional[float] = None,
        base_optimizer: str = DEFAULT_OPTIMIZER,
        use_soft_algebra: bool = True,
        maximize: bool = False,  # NEW
        offline_fallback: bool = True,
        verbose: bool = True,
        problem: Optional[str] = None,
    ):
        self.license_key = license_key or get_license_key()
        if not self.license_key:
            raise ValueError(
                "License key required. Set MOBIU_Q_LICENSE_KEY environment variable, "
                "or pass license_key parameter, or run: mobiu-q activate YOUR_KEY"
            )
        
        # Handle deprecated 'problem' parameter
        if problem is not None:
            warnings.warn(
                "Parameter 'problem' is deprecated, use 'method' instead",
                DeprecationWarning,
                stacklevel=2
            )
            if method == "standard":  # Only override if method wasn't explicitly set
                method = problem
        
        # Normalize mode (backward compatibility)
        if mode == "standard":
            mode = "simulation"
        elif mode == "noisy":
            mode = "hardware"
        
        # Validate method (accept both new and legacy names)
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
        
        # Map to internal name
        internal_method = METHOD_ALIASES.get(method, method)
        
        # Validate mode
        if mode not in ("simulation", "hardware"):
            raise ValueError(f"mode must be 'simulation' or 'hardware', got '{mode}'")
        
        # Validate optimizer
        if base_optimizer not in AVAILABLE_OPTIMIZERS:
            raise ValueError(
                f"base_optimizer must be one of {AVAILABLE_OPTIMIZERS}, got '{base_optimizer}'"
            )
        
        self.method = internal_method  # Store internal name
        self._original_method = method  # Store what user passed (for display)
        self.mode = mode
        self.base_lr = base_lr if base_lr is not None else get_default_lr(internal_method, mode)
        self.base_optimizer = base_optimizer
        self.use_soft_algebra = use_soft_algebra
        self.offline_fallback = offline_fallback
        self.verbose = verbose
        self.session_id = None
        self.api_endpoint = API_ENDPOINT

        # Frustration Engine (NEW)
        self.frustration_engine = UniversalFrustrationEngine(base_lr=self.base_lr) if use_soft_algebra else None
        self._current_lr = self.base_lr
        self.maximize = maximize
        
        # Local state (for offline fallback)
        self._offline_mode = False
        self._local_m = None
        self._local_v = None
        self._local_t = 0
        
        # History (local tracking)
        self.energy_history = []
        self.lr_history = []
        
        # Track number of runs in this session
        self._run_count = 0

        # Usage tracking
        self._usage_info = None
        
        # Server info
        self._available_optimizers = AVAILABLE_OPTIMIZERS
        
        # Start session
        self._start_session()
    
    @property
    def problem(self) -> str:
        """Deprecated: Use 'method' instead."""
        warnings.warn(
            "Attribute 'problem' is deprecated, use 'method' instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.method
    
    def _start_session(self):
        """Initialize optimization session with server."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "action": "start",
                    "method": self.method,
                    "mode": self.mode,
                    "base_lr": self.base_lr,
                    "base_optimizer": self.base_optimizer,
                    "use_soft_algebra": self.use_soft_algebra,
                    "maximize": self.maximize  # <-- להוסיף!
                },
                timeout=10
            )
            
            data = response.json()
            
            if not data.get("success"):
                error = data.get("error", "Unknown error")
                if "limit" in error.lower() or "quota" in error.lower():
                    print(f"❌ {error}")
                    print("   Upgrade at: https://app.mobiu.ai")
                raise RuntimeError(f"Failed to start session: {error}")
            
            self.session_id = data["session_id"]
            self._usage_info = data.get("usage", {})
            
            # Server may return computed values
            server_method = data.get("method", self.method)
            server_mode = data.get("mode", self.mode)
            server_lr = data.get("base_lr", self.base_lr)
            server_optimizer = data.get("base_optimizer", self.base_optimizer)
            self._available_optimizers = data.get("available_optimizers", AVAILABLE_OPTIMIZERS)
            
            if self.verbose:
                remaining = self._usage_info.get('remaining', 'unknown')
                tier = self._usage_info.get('tier', 'unknown')
                
                mode_str = f"method={server_method}, mode={server_mode}, lr={server_lr}"
                if server_optimizer != DEFAULT_OPTIMIZER:
                    mode_str += f", optimizer={server_optimizer}"
                if not self.use_soft_algebra:
                    mode_str += ", SA=off"

                if self.maximize:
                    mode_str += ", maximize=True"
                
                if remaining == 'unlimited':
                    print(f"🚀 Mobiu-Q session started (Pro tier) [{mode_str}]")
                elif isinstance(remaining, int):
                    if remaining <= 2:
                        print(f"⚠️  Mobiu-Q session started - LOW QUOTA: {remaining} runs remaining!")
                    else:
                        print(f"🚀 Mobiu-Q session started ({remaining} runs remaining) [{mode_str}]")
                else:
                    print(f"🚀 Mobiu-Q session started [{mode_str}]")
                
        except requests.exceptions.RequestException as e:
            if self.offline_fallback:
                if self.verbose:
                    print(f"⚠️  Cannot connect to Mobiu-Q API: {e}")
                    print("   Running in offline fallback mode (plain Adam)")
                self._offline_mode = True
            else:
                raise RuntimeError(f"Cannot connect to Mobiu-Q API: {e}")
    
    def new_run(self):
        """
        Start a new optimization run within the same session.
        
        Use this for multi-seed experiments - all runs count as 1 session.
        Resets optimizer state (momentum, etc.) but keeps the session open.
        
        Example:
            opt = MobiuQCore(license_key="xxx")
            
            for seed in range(10):
                opt.new_run()  # Reset state for new seed
                params = init_params(seed)
                for step in range(100):
                    params = opt.step(params, grad, energy)
            
            opt.end()  # All 10 seeds count as 1 run
        """
        self._run_count += 1
        
        # Reset local tracking
        self.energy_history.clear()
        self.lr_history.clear()
        self._local_m = None
        self._local_v = None
        self._local_t = 0
        
        if self._offline_mode or not self.session_id:
            return
        
        # Call server to reset optimizer state
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "session_id": self.session_id,
                    "action": "reset"
                },
                timeout=10
            )
            
            data = response.json()
            if not data.get("success"):
                if self.verbose:
                    print(f"⚠️  Could not reset server state: {data.get('error')}")
                    
        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"⚠️  Could not reset server state: {e}")
    
    def step(
        self, 
        params: np.ndarray, 
        gradient_or_fn, 
        energy: float = None
    ) -> np.ndarray:
        """
        Perform one optimization step.
    
        Args:
            params: Current parameter values
            gradient_or_fn: Either:
                - np.ndarray: Gradient (backward compatible)
                - Callable: Energy function - gradient computed automatically based on method
            energy: Current objective value. Required if gradient_or_fn is array.
                    Auto-computed if gradient_or_fn is callable.
    
        Returns:
            Updated parameters
    
        Examples:
            # Auto gradient (recommended):
            params = opt.step(params, energy_fn)
        
            # Manual gradient (backward compatible):
            grad = my_custom_gradient(params)
            params = opt.step(params, grad, energy)
    
        Gradient methods by mode:
            - simulation: finite_difference (2N evaluations, exact)
            - hardware: SPSA (2 evaluations, noisy-resilient)
        """
        # Auto-compute gradient if function provided
        if callable(gradient_or_fn):
            energy_fn = gradient_or_fn
    
            # Mode determines gradient method (not method!)
            # hardware = noisy environment → SPSA
            # simulation = clean environment → finite difference
            if self.mode == "hardware":
                gradient, energy = Demeasurement.spsa(energy_fn, params)
            else:  # simulation
                gradient = Demeasurement.finite_difference(energy_fn, params)
                energy = energy_fn(params)
        else:
            gradient = gradient_or_fn
            if energy is None:
                raise ValueError("energy is required when providing gradient array")
    
        self.energy_history.append(energy)

        # === FRUSTRATION ENGINE ===
        if self.frustration_engine:
            score = energy if self.maximize else -energy
            factor = self.frustration_engine.get_lr_factor(score)
    
            if factor > 1.0:
                self._current_lr = self.base_lr * factor
                self.lr_history.append(self._current_lr)
            else:
                self._current_lr = self.base_lr
        
        if self._offline_mode:
            return self._offline_step(params, gradient)
        
        try:
            # Retry loop for rate limiting
            energy_to_send = energy
            for attempt in range(3):
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "license_key": self.license_key,
                        "session_id": self.session_id,
                        "action": "step",
                        "params": params.tolist(),
                        "gradient": gradient.tolist(),
                        "energy": float(energy_to_send)
                    },
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limited
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    else:
                        raise RuntimeError("Rate limit exceeded. Please slow down requests.")
                break
            
            data = response.json()
            
            if not data.get("success"):
                error = data.get("error", "Unknown error")
                if self.offline_fallback:
                    if self.verbose:
                        print(f"⚠️  API error: {error}. Switching to offline mode.")
                    self._offline_mode = True
                    return self._offline_step(params, gradient)
                raise RuntimeError(f"Optimization step failed: {error}")
            
            new_params = np.array(data["new_params"])
            
            # Track LR for diagnostics
            if "adaptive_lr" in data:
                self.lr_history.append(data["adaptive_lr"])
            
            return new_params
            
        except requests.exceptions.RequestException as e:
            if self.offline_fallback:
                if self.verbose:
                    print(f"⚠️  API connection lost: {e}. Switching to offline mode.")
                self._offline_mode = True
                return self._offline_step(params, gradient)
            raise
    
    def _offline_step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Fallback: plain Adam optimizer."""
        self._local_t += 1
        
        if self._local_m is None:
            self._local_m = np.zeros_like(gradient)
            self._local_v = np.zeros_like(gradient)
        
        lr = self._current_lr
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        self._local_m = beta1 * self._local_m + (1 - beta1) * gradient
        self._local_v = beta2 * self._local_v + (1 - beta2) * (gradient ** 2)
        
        m_hat = self._local_m / (1 - beta1 ** self._local_t)
        v_hat = self._local_v / (1 - beta2 ** self._local_t)
        
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return params - update
    
    def end(self):
        """
        End the optimization session.
        
        Call this when optimization is complete!
        This is when the run is counted against your quota.
        """
        if self._offline_mode or not self.session_id:
            return
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "session_id": self.session_id,
                    "action": "end"
                },
                timeout=10
            )
            
            data = response.json()
            self._usage_info = data.get("usage", {})
            
            if self.verbose:
                remaining = self._usage_info.get('remaining', 'unknown')
                used = self._usage_info.get('used', 'unknown')
                
                if remaining == 'unlimited':
                    print(f"✅ Session ended (Pro tier)")
                elif remaining == 0:
                    print(f"✅ Session ended")
                    print(f"❌ Quota exhausted! Upgrade at: https://app.mobiu.ai")
                elif isinstance(remaining, int) and remaining <= 2:
                    print(f"✅ Session ended")
                    print(f"⚠️  Low quota warning: {remaining} runs remaining")
                else:
                    print(f"✅ Session ended ({remaining} runs remaining)")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not cleanly end session: {e}")
        
        self.session_id = None
    
    def reset(self):
        """
        DEPRECATED: Use new_run() for multi-seed experiments.
        """
        warnings.warn(
            "reset() is deprecated and counts each call as a separate run. "
            "Use new_run() for multi-seed experiments (counts as 1 run total).",
            DeprecationWarning,
            stacklevel=2
        )
        self.end()
        self.energy_history.clear()
        self.lr_history.clear()
        self._start_session()
    
    def check_usage(self) -> dict:
        """Check current usage without affecting quota."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "action": "usage"
                },
                timeout=10
            )
            data = response.json()
            if data.get("success"):
                self._usage_info = data.get("usage", {})
                return self._usage_info
        except:
            pass
        return {}
    
    def get_server_info(self) -> dict:
        """Get server information including available methods and optimizers."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "action": "info"
                },
                timeout=10
            )
            data = response.json()
            if data.get("success"):
                return data
        except:
            pass
        return {
            "available_optimizers": AVAILABLE_OPTIMIZERS,
            "default_optimizer": DEFAULT_OPTIMIZER,
            "methods": ["standard", "deep", "adaptive"],
            "legacy_methods": ["vqe", "qaoa", "rl"]
        }
    
    @property
    def available_optimizers(self) -> List[str]:
        """List of available optimizers."""
        return self._available_optimizers
    
    @property
    def remaining_runs(self) -> Optional[int]:
        """Get remaining runs (None if unknown or unlimited)"""
        if self._usage_info:
            remaining = self._usage_info.get('remaining')
            if remaining == 'unlimited':
                return None
            return remaining
        return None

    def __del__(self):
        """Auto-end session on garbage collection."""
        try:
            self.end()
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# DEMEASUREMENT (Gradient Estimation) - Runs Locally
# ═══════════════════════════════════════════════════════════════════════════════

class Demeasurement:
    """
    Gradient estimation methods for quantum circuits.
    
    These run locally - no API call needed.
    
    Choose based on your problem:
    - Standard (smooth landscapes): finite_difference() or parameter_shift()
    - Deep (rugged landscapes): spsa()
    - Hardware (noisy): spsa()
    - RL/LLM: Use your framework's gradient computation (e.g., PyTorch autograd)
    """
    
    @staticmethod
    def parameter_shift(
        circuit_fn, 
        params: np.ndarray, 
        shift: float = np.pi/2
    ) -> np.ndarray:
        """
        Parameter-shift rule gradient estimation.
        Requires 2N circuit evaluations.
        Best for: Clean simulations, exact gradients.
        """
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            grad[i] = (circuit_fn(params_plus) - circuit_fn(params_minus)) / 2.0
        return grad
    
    @staticmethod
    def finite_difference(
        circuit_fn, 
        params: np.ndarray,
        epsilon: float = 1e-3
    ) -> np.ndarray:
        """
        Finite difference gradient estimation.
        Requires 2N circuit evaluations.
        Best for: Clean simulations, approximate gradients.
        """
        grad = np.zeros_like(params)
        base_energy = circuit_fn(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            grad[i] = (circuit_fn(params_plus) - base_energy) / epsilon
        return grad
    
    @staticmethod
    def spsa(
        circuit_fn, 
        params: np.ndarray,
        c_shift: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Simultaneous Perturbation Stochastic Approximation (SPSA).
        Requires only 2 circuit evaluations regardless of parameter count!
        Best for: Noisy quantum hardware, NISQ devices, deep circuits.
        
        Returns:
            (gradient_estimate, estimated_energy)
        """
        delta = np.random.choice([-1, 1], size=params.shape)
        
        params_plus = params + c_shift * delta
        params_minus = params - c_shift * delta
        
        energy_plus = circuit_fn(params_plus)
        energy_minus = circuit_fn(params_minus)
        
        grad = (energy_plus - energy_minus) / (2 * c_shift) * delta
        avg_energy = (energy_plus + energy_minus) / 2.0
        
        return grad, avg_energy


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def activate_license(key: str):
    """Activate and save license key."""
    save_license_key(key)
    
    try:
        opt = MobiuQCore(license_key=key, verbose=False)
        opt.end()
        print("✅ License activated successfully!")
    except Exception as e:
        print(f"❌ License activation failed: {e}")


def check_status():
    """Check license status and remaining runs."""
    key = get_license_key()
    if not key:
        print("❌ No license key found")
        print("   Run: mobiu-q activate YOUR_KEY")
        return
    
    try:
        opt = MobiuQCore(license_key=key, verbose=False)
        usage = opt.check_usage()
        info = opt.get_server_info()
        opt.end()
        
        print("✅ License is active")
        if usage:
            print(f"   Tier: {usage.get('tier', 'unknown')}")
            print(f"   Used this month: {usage.get('used', 'unknown')}")
            print(f"   Remaining: {usage.get('remaining', 'unknown')}")
        if info:
            print(f"   Server version: {info.get('version', 'unknown')}")
            print(f"   Methods: {', '.join(info.get('methods', []))}")
            print(f"   Available optimizers: {', '.join(info.get('available_optimizers', []))}")
    except Exception as e:
        print(f"❌ License check failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "4.0.1"
__all__ = [
    # New universal optimizer (v2.7)
    "MobiuOptimizer",
    # Frustration Engine (v2.9)
    "UniversalFrustrationEngine",
    # Legacy (still fully supported)
    "MobiuQCore",
    "Demeasurement",
    # Utilities
    "activate_license",
    "check_status",
    "get_default_lr",
    "get_license_key",
    "save_license_key",
    # Constants
    "AVAILABLE_OPTIMIZERS",
    "DEFAULT_OPTIMIZER",
    "METHOD_ALIASES",
    "VALID_METHODS",
    "API_ENDPOINT",
]