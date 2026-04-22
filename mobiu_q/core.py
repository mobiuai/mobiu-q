"""
Mobiu-Q Client - Soft Algebra Optimizer
========================================
Cloud-connected optimizer for quantum, RL, and LLM applications.

Version: 5.0 - Pure-library mode (LocalSAEngine)

New in v5.0:
- Pure-library mode: set sa_backend="local" for in-process Soft Algebra
  with zero network calls, zero license key, zero round-trips. Numerically
  identical to cloud mode (verified by parity tests).
- LocalSAEngine class: the in-process Soft Algebra engine that mirrors the
  server's optimization_step() behavior.
- Default remains sa_backend="cloud" — existing code is unaffected.

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

# Default optimizer for Quantum mode (PyTorch mode: user provides any optimizer)
AVAILABLE_OPTIMIZERS = ["Adam"]
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

# Boost presets
BOOST_PRESETS = {
    "none":       {"warmup_factor": 1.0, "boost_factor": 1.0, "check_after": 999,
                   "stagnation_window": 20, "stagnation_threshold": 0.02, "cooldown": 30},
    "normal":     {"warmup_factor": 1.5, "boost_factor": 1.5, "check_after": 8,
                   "stagnation_window": 20, "stagnation_threshold": 0.02, "cooldown": 30},
    "aggressive": {"warmup_factor": 3.0, "boost_factor": 3.0, "check_after": 3,
                   "stagnation_window": 15, "stagnation_threshold": 0.01, "cooldown": 20},
}

class UniversalFrustrationEngine:
    """
    LR Boost Engine — warmup at start + stagnation boost during training.

    Controlled by the 'boost' parameter:
        "none"       — no warmup, no stagnation boost (default)
        "normal"     — gentle warmup + gentle stagnation boost
        "aggressive" — strong warmup + strong stagnation boost

    _is_improving() acts as a smart brake:
        - Ends warmup early if training is already going well
        - Skips stagnation boost if training is still improving

    Always prints what it's doing (when verbose=True).
    """
    def __init__(self, base_lr: float,
                 boost: str = "none",
                 verbose: bool = True,
                 warmup_steps: int = 30,
                 update_interval: int = 320,
                 # Legacy params
                 sensitivity: float = 0.05,
                 flip_on_fire: bool = False,
                 session_id: str = None,
                 license_key: str = None,
                 api_endpoint: str = None):
        preset = BOOST_PRESETS.get(boost, BOOST_PRESETS["none"])
        self.base_lr              = base_lr
        self.boost                = boost
        self.verbose              = verbose
        self.warmup_factor        = preset["warmup_factor"]
        self.boost_factor         = preset["boost_factor"]
        self.check_after          = preset["check_after"]
        self.stagnation_window    = preset["stagnation_window"]
        self.stagnation_threshold = preset["stagnation_threshold"]
        self.cooldown_steps       = preset["cooldown"]
        self.warmup_steps         = warmup_steps
        self.update_interval      = update_interval
        self._call_count          = 0
        self._update_count        = 0
        self._metric_history      = []
        self._warmup_active       = None   # None=undecided, True=active, False=done
        self._cooldown            = 0
        self._current_factor      = 1.0
        self.fire_count           = 0
        self._warmup_attempts     = 0
        self._warmup_cancelled    = 0
        self._stagnation_attempts = 0
        # Legacy
        self.flip_on_fire  = flip_on_fire
        self.session_id    = session_id
        self.license_key   = license_key
        self.api_endpoint  = api_endpoint

    def _is_improving(self, window: int = None) -> bool:
        """Returns True if metric improved meaningfully over the given window."""
        w = window or self.stagnation_window
        history = self._metric_history[-w:] if len(self._metric_history) >= w \
                  else self._metric_history
        if len(history) < 2:
            return False
        first, last = history[0], history[-1]
        if abs(first) < 1e-8:
            return last < first
        return (first - last) / abs(first) > self.stagnation_threshold

    def get_lr_factor(self, current_metric: float = 0.0) -> float:
        self._call_count += 1

        # Between update boundaries: return current factor
        if self._call_count % self.update_interval != 0:
            return self._current_factor

        # At update boundary
        self._update_count += 1
        self._metric_history.append(current_metric)

        # ── BOOST=NONE: do nothing ──
        if self.boost == "none":
            self._current_factor = 1.0
            return 1.0

        # ── PHASE 1: WARMUP ──
        if self._warmup_active is None:
            self._warmup_active = True
            self._warmup_attempts += 1

        if self._warmup_active:
            warmup_update = self._update_count
            if self._update_count >= self.check_after and \
               self._is_improving(window=self.check_after):
                self._warmup_active = False
                self._warmup_cancelled += 1
                self._current_factor = 1.0
                return 1.0
            if warmup_update <= self.warmup_steps:
                progress = (warmup_update - 1) / self.warmup_steps
                factor = self.warmup_factor - (self.warmup_factor - 1.0) * progress
                self._current_factor = max(factor, 1.0)
                return self._current_factor
            else:
                self._warmup_active = False
                self._current_factor = 1.0

        # ── PHASE 2: STAGNATION BOOST ──
        if self._cooldown > 0:
            self._cooldown -= 1
            self._current_factor = 1.0
            return 1.0

        if len(self._metric_history) >= self.stagnation_window:
            if not self._is_improving():
                self._cooldown = self.cooldown_steps
                self.fire_count += 1
                self._stagnation_attempts += 1
                self._current_factor = self.boost_factor
                return self._current_factor

        self._current_factor = 1.0
        return 1.0

    def summary(self) -> str:
        """Returns a one-line summary of boost activity."""
        if self.boost == "none":
            return ""
        parts = []
        if self._warmup_attempts > 0:
            cancelled = f", {self._warmup_cancelled} cancelled" if self._warmup_cancelled else ""
            parts.append(f"warmup: {self._warmup_attempts} attempt{cancelled}")
        if self._stagnation_attempts > 0:
            parts.append(f"stagnation spikes: {self._stagnation_attempts}")
        if not parts:
            return f"   💡 Boost ({self.boost}): no action needed — training improved on its own"
        return f"   ⚡ Boost ({self.boost}): {' | '.join(parts)}"

    def reset(self):
        """Reset engine state for new run."""
        self._call_count          = 0
        self._update_count        = 0
        self._metric_history      = []
        self._warmup_active       = None
        self._cooldown            = 0
        self._current_factor      = 1.0
        self.fire_count           = 0
        self._warmup_attempts     = 0
        self._warmup_cancelled    = 0
        self._stagnation_attempts = 0

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
        mode: str = "simulation",
        use_soft_algebra: bool = True,
        sync_interval: Optional[int] = None,
        boost: str = "none",
        update_interval: int = 320,
        verbose: bool = True,
        problem: Optional[str] = None,
        sa_backend: str = "cloud",
        **kwargs
    ):
        # Validate sa_backend — new in v5.0
        if sa_backend not in ("cloud", "local"):
            raise ValueError(
                f"sa_backend must be 'cloud' or 'local', got {sa_backend!r}. "
                "Use sa_backend='local' for pure-library mode (no network)."
            )
        self.sa_backend = sa_backend

        # License key is REQUIRED for cloud backend, OPTIONAL for local
        if sa_backend == "cloud":
            self.license_key = license_key or get_license_key()
            if not self.license_key:
                raise ValueError(
                    "License key required for cloud backend. "
                    "Either set MOBIU_Q_LICENSE_KEY environment variable, "
                    "pass license_key= parameter, run: mobiu-q activate YOUR_KEY, "
                    "or use sa_backend='local' to run without a license (v5.0+)."
                )
        else:
            # Local mode — license key is ignored if provided
            if license_key and verbose:
                print("ℹ️  sa_backend='local' — license_key ignored (no cloud calls will be made)")
            self.license_key = None
        
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
            # Auto-detect: small models get full Soft Algebra via server
            total_params = sum(
                p.numel() for pg in optimizer_or_params.param_groups
                for p in pg['params']
            )
            full_sync = total_params < 1000

            if sync_interval is None:
                # In local mode, every step is free — no point in batching
                sync_interval = 1 if (full_sync or sa_backend == "local") else 50

            self._backend = _MobiuPyTorchBackend(
                optimizer_or_params, 
                self.license_key, 
                self.method,
                base_lr=kwargs.get('base_lr'),
                use_soft_algebra=use_soft_algebra,
                sync_interval=sync_interval,
                boost=boost,
                update_interval=update_interval,
                verbose=verbose,
                full_sync=full_sync,
                mode=mode or 'simulation',
                sa_backend=sa_backend,
                maximize=kwargs.get('maximize', False)
            )
        else:
            # Quantum mode (MobiuQCore) — local backend not yet supported here
            if sa_backend == "local":
                raise NotImplementedError(
                    "sa_backend='local' is not yet supported for quantum/NumPy "
                    "mode (MobiuQCore). PyTorch hybrid mode only in v5.0."
                )
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
                boost: str = "none",
                update_interval: int = 320,
                verbose: bool = True,
                full_sync: bool = False, mode: str = 'simulation',
                sa_backend: str = 'cloud',
                maximize: bool = False):
        self.optimizer = optimizer
        self.license_key = license_key
        self.method = method
        self.use_soft_algebra = use_soft_algebra
        self.verbose = verbose
        self.session_id = None
        self.api_endpoint = API_ENDPOINT
        self.full_sync = full_sync
        self.mode = mode
        self.sa_backend = sa_backend
        self.maximize = maximize
        
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
        
        # Frustration Engine (not needed in full_sync — server handles SA)
        self.frustration_engine = (
            UniversalFrustrationEngine(base_lr=self.base_lr, boost=boost)
            if use_soft_algebra and not full_sync
            else None
        )
        
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

        # Warning state — track silent-failure and no-op conditions (v4.6)
        self._cloud_sync_failures = 0       # count of failed cloud syncs during step()
        self._cloud_sync_successes = 0      # count of successful cloud syncs
        self._cloud_warned_once = False     # have we already told the user cloud is down
        self._no_op_check_done = False      # have we already checked/warned about no-op regime

        # Local SA engine — v5.0. Instantiated lazily only in local mode.
        self._local_engine = None
        if self.sa_backend == 'local':
            from .local import LocalSAEngine
            self._local_engine = LocalSAEngine(
                method=self.method,
                base_lr=self.base_lr,
                use_soft_algebra=self.use_soft_algebra,
                maximize=self.maximize,
            )
            if self.verbose:
                sa_str = "SA=on" if self.use_soft_algebra else "SA=off"
                print(f"🚀 Mobiu-Q Local session started (no cloud, no license) "
                      f"[method={self.method}, base_lr={self.base_lr}, {sa_str}]")
        else:
            # Cloud mode — existing path
            self._start_session()
    
    def _start_session(self):
        """Initialize cloud session."""
        try:
            r = requests.post(self.api_endpoint, json={
                'action': 'start',
                'license_key': self.license_key,
                'method': self.method,
                'mode': self.mode,
                'base_lr': self.base_lr,
                'base_optimizer': 'Adam',
                'use_soft_algebra': self.use_soft_algebra
            }, timeout=10)
            
            data = r.json()
            
            if data.get('success'):
                self.session_id = data['session_id']
                if self.frustration_engine:
                    self.frustration_engine.session_id   = self.session_id
                    self.frustration_engine.license_key  = self.license_key
                    self.frustration_engine.api_endpoint = self.api_endpoint
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
        # Use stored metric if none provided (for SB3 compatibility)
        if metric is None:
            metric = self._stored_metric

        self._local_step_count += 1

        # ── LOCAL BACKEND (v5.0): in-process SA, no network ──
        if self.sa_backend == 'local':
            import torch
            energy = metric if metric is not None else 0.0
            self.energy_history.append(energy)

            if not self.use_soft_algebra:
                self.optimizer.step()
                return

            # Apply LR boost (frustration engine) — runs in local mode too
            if self.frustration_engine and metric is not None:
                factor = self.frustration_engine.get_lr_factor(metric)
                if factor > 1.0:
                    boosted_lr = self.base_lr * factor
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = boosted_lr
                    self.lr_history.append(boosted_lr)

            # Step the local engine
            result = self._local_engine.step(float(energy))

            if result["converged"]:
                # True minimum detected — don't modify LR, let user decide to halt
                self.lr_history.append(result["adaptive_lr"])
                self.warp_history.append(1.0)
                # Don't call optimizer.step() — we're at the minimum
                return

            adaptive_lr = result["adaptive_lr"]
            warp = result["warp_factor"]

            # Apply LR
            for pg in self.optimizer.param_groups:
                pg['lr'] = adaptive_lr
            self.lr_history.append(adaptive_lr)
            self.warp_history.append(warp)

            # Apply gradient warp (in-place)
            if warp != 1.0:
                for pg in self.optimizer.param_groups:
                    for p in pg['params']:
                        if p.grad is not None:
                            p.grad.data.mul_(warp)

            # Client's own optimizer runs the actual update
            self.optimizer.step()
            return

        # ── FULL SYNC: small models → server computes SA, client runs optimizer ──
        if self.full_sync:
            import torch
            energy = metric if metric is not None else 0.0
            self.energy_history.append(energy)

            # If no session or SA disabled, just run client optimizer as-is
            if not self.session_id or not self.use_soft_algebra:
                self.optimizer.step()
                return

            # Extract real params and gradients from PyTorch
            all_params, all_grads = [], []
            for pg in self.optimizer.param_groups:
                for p in pg['params']:
                    all_params.append(p.data.detach().cpu().flatten())
                    all_grads.append(
                        p.grad.detach().cpu().flatten() if p.grad is not None
                        else torch.zeros(p.numel())
                    )

            params_np = torch.cat(all_params).numpy().tolist()
            grads_np = torch.cat(all_grads).numpy().tolist()

            try:
                r = requests.post(self.api_endpoint, json={
                    'action': 'step',
                    'license_key': self.license_key,
                    'session_id': self.session_id,
                    'params': params_np,
                    'gradient': grads_np,
                    'energy': float(energy),
                    'return_adjustments': True  # Don't run server optimizer
                }, timeout=30)

                data = r.json()
                if data.get('success'):
                    adaptive_lr = data.get('adaptive_lr', self.base_lr)
                    warp = data.get('warp_factor', 1.0)

                    # Warp gradients in-place
                    if warp != 1.0:
                        for pg in self.optimizer.param_groups:
                            for p in pg['params']:
                                if p.grad is not None:
                                    p.grad.data.mul_(warp)

                    # Set SA-adjusted learning rate
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = adaptive_lr

                    self.lr_history.append(adaptive_lr)
                    self.warp_history.append(warp)

                    # CLIENT'S OWN OPTIMIZER runs the actual update
                    self.optimizer.step()
                    return

            except Exception:
                pass

            # Fallback: run client optimizer without SA
            self.optimizer.step()
            return

        # ── HYBRID: periodic LR sync for large models ──

        # 1. FRUSTRATION ENGINE (Client-Side Logic)
        if self.frustration_engine and metric is not None:
            factor = self.frustration_engine.get_lr_factor(metric)
            
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
                    self._cloud_sync_successes += 1
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
                else:
                    # Server returned success=False (rate limit, quota, etc.)
                    self._cloud_sync_failures += 1
                    if self.verbose and not self._cloud_warned_once:
                        err = data.get('error', 'unknown error')
                        print(f"⚠️  Mobiu-Q cloud returned error: {err}")
                        print(f"   Training continues with base Adam only (SA disabled until cloud recovers).")
                        self._cloud_warned_once = True

            except Exception as e:
                # Network error / timeout / JSON decode failure
                self._cloud_sync_failures += 1
                if self.verbose and not self._cloud_warned_once:
                    print(f"⚠️  Mobiu-Q cloud unreachable ({type(e).__name__}): {str(e)[:80]}")
                    print(f"   Training continues with base Adam only (SA disabled for this run).")
                    print(f"   Subsequent failures in this run will be silent.")
                    self._cloud_warned_once = True
            
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

        # Reset local SA engine state (v5.0)
        if self._local_engine is not None:
            self._local_engine.reset()
    
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
        # --- Local mode (v5.0): no session to close, just report ---
        if self.sa_backend == 'local':
            if self.verbose:
                if self._local_engine is not None:
                    print(f"✅ Local session ended ({self._local_engine.t} steps processed)")
                else:
                    print(f"✅ Local session ended")
                if self.frustration_engine:
                    summary = self.frustration_engine.summary()
                    if summary:
                        print(summary)
            return

        # --- Pre-close diagnostic: was SA actually active? (v4.6 no-op detector) ---
        # If SA was meant to be on but the cloud never synced enough points to
        # compute curvature (needs ≥3 energy points server-side), warn the user
        # that they effectively ran plain Adam the whole time. Common cause:
        # total_steps < 3 * sync_interval.
        if (self.verbose
                and self.use_soft_algebra
                and not self.full_sync
                and self.session_id is not None
                and not self._no_op_check_done):
            self._no_op_check_done = True
            # Consider SA "active" only if we got at least 3 successful syncs
            # (server-side curvature formula needs 3 energy points).
            if self._cloud_sync_successes < 3 and self._cloud_sync_failures == 0:
                projected_min = 3 * self.sync_interval
                print(f"⚠️  Mobiu-Q SA was enabled but did not activate meaningfully in this run.")
                print(f"   Only {self._cloud_sync_successes} cloud sync(s) completed (need ≥3 to compute curvature).")
                print(f"   Reason: only {self._local_step_count} step() calls with sync_interval={self.sync_interval}.")
                print(f"   Fix: run at least {projected_min} steps, or lower sync_interval "
                      f"(e.g. sync_interval={max(1, self._local_step_count // 4)}).")
                print(f"   This run effectively used base Adam only — benchmarks may not reflect SA performance.")

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

                    if self.frustration_engine:
                        summary = self.frustration_engine.summary()
                        if summary:
                            print(summary)
                    
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
        base_optimizer: Optimizer: "Adam" (default)
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
                    "use_soft_algebra": self.use_soft_algebra
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
            if self.frustration_engine:
                self.frustration_engine.session_id   = self.session_id
                self.frustration_engine.license_key  = self.license_key
                self.frustration_engine.api_endpoint = self.api_endpoint
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
            factor = self.frustration_engine.get_lr_factor(energy)
    
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

__version__ = "5.0"
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