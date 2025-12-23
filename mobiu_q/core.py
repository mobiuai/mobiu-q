"""
Mobiu-Q Client - Soft Algebra Optimizer
========================================
Cloud-connected optimizer for quantum variational algorithms and RL.

Version: 2.4.3 - Multi-Optimizer + RL Support
- method='vqe'/'qaoa'/'rl'
- mode='simulation'/'hardware'
- base_optimizer='Adam'/'NAdam'/'AMSGrad'/'SGD'/'Momentum'/'LAMB'
- Default optimizer: Adam (works best across all methods)
- Default LR by method+mode

Usage (VQE - Chemistry):
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", method="vqe")
    
    for step in range(100):
        grad = Demeasurement.finite_difference(energy_fn, params)
        params = opt.step(params, grad, energy_fn(params))
    
    opt.end()

Usage (QAOA - Combinatorial, on hardware):
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", method="qaoa", mode="hardware")
    
    for step in range(150):
        grad, energy = Demeasurement.spsa(energy_fn, params)
        params = opt.step(params, grad, energy)
    
    opt.end()

Usage (RL - Reinforcement Learning):
    from mobiu_q import MobiuQCore
    
    opt = MobiuQCore(license_key="your-key", method="rl")
    
    for episode in range(1000):
        # ... run episode, compute policy gradient ...
        opt.step(policy_params, gradient, episode_return)
    
    opt.end()

Multi-seed usage (counts as 1 run):
    opt = MobiuQCore(license_key="your-key")
    
    for seed in range(10):
        opt.new_run()  # Reset state, same session
        params = init_params(seed)
        for step in range(100):
            params = opt.step(params, grad, energy)
    
    opt.end()  # Only here it counts as 1 run
"""

import numpy as np
import requests
from typing import Optional, Tuple, List
import os
import json
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API endpoint
API_ENDPOINT = os.environ.get(
    "MOBIU_Q_API_ENDPOINT",
    "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
)

LICENSE_KEY_FILE = os.path.expanduser("~/.mobiu_q_license")

# Available optimizers
AVAILABLE_OPTIMIZERS = ["Adam", "NAdam", "AMSGrad", "SGD", "Momentum", "LAMB"]
DEFAULT_OPTIMIZER = "Adam"


def get_license_key() -> Optional[str]:
    """Get license key from environment or file."""
    # 1. Environment variable
    key = os.environ.get("MOBIU_Q_LICENSE_KEY")
    if key:
        return key
    
    # 2. License file
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
    
    | Method | Mode       | Default LR |
    |--------|------------|------------|
    | vqe    | simulation | 0.01       |
    | vqe    | hardware   | 0.02       |
    | qaoa   | simulation | 0.1        |
    | qaoa   | hardware   | 0.1        |
    | rl     | any        | 0.0003     |
    """
    if method == 'rl':
        return 0.0003
    elif method == 'qaoa':
        return 0.1
    elif mode == 'hardware':
        return 0.02
    else:  # vqe + simulation
        return 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# MOBIU-Q CORE (Cloud Client)
# ═══════════════════════════════════════════════════════════════════════════════

class MobiuQCore:
    """
    Mobiu-Q Optimizer - Cloud-connected version.
    
    The optimization logic runs on Mobiu's secure servers.
    This client handles communication and local state.
    
    Args:
        license_key: Your Mobiu-Q license key (or set MOBIU_Q_LICENSE_KEY env var)
        method: "vqe" (chemistry/physics), "qaoa" (combinatorial), or "rl" (reinforcement learning)
        mode: "simulation" (clean simulations) or "hardware" (quantum hardware/noisy)
        base_lr: Learning rate (default: computed from method+mode)
        base_optimizer: Optimizer to use: "Adam" (default), "NAdam", "AMSGrad", "SGD", "Momentum", "LAMB"
        use_soft_algebra: Enable Soft Algebra enhancement (default: True)
        offline_fallback: If True, use local Adam when API unavailable
        
        # Deprecated parameters (still supported for backward compatibility):
        problem: Use 'method' instead
    
    Default Learning Rates:
        | Method | Mode       | Default LR |
        |--------|------------|------------|
        | vqe    | simulation | 0.01       |
        | vqe    | hardware   | 0.02       |
        | qaoa   | simulation | 0.1        |
        | qaoa   | hardware   | 0.1        |
        | rl     | any        | 0.0003     |
    
    Optimizers:
        Default: Adam (recommended - works best across all methods)
        
        Alternatives (for experimentation):
        - NAdam: Strong on QAOA problems
        - AMSGrad: May outperform on VQE simulation
        - LAMB: High improvement potential, less stable
        - SGD/Momentum: Simple baselines
    
    Example (VQE - Chemistry):
        opt = MobiuQCore(license_key="xxx", method="vqe")
        
        for step in range(100):
            grad = Demeasurement.finite_difference(energy_fn, params)
            params = opt.step(params, grad, energy_fn(params))
        
        opt.end()
    
    Example (QAOA - Combinatorial on hardware):
        opt = MobiuQCore(license_key="xxx", method="qaoa", mode="hardware")
        
        for step in range(150):
            grad, energy = Demeasurement.spsa(energy_fn, params)
            params = opt.step(params, grad, energy)
        
        opt.end()
    
    Example (RL - Reinforcement Learning):
        opt = MobiuQCore(license_key="xxx", method="rl")
        
        for episode in range(1000):
            # Run episode, compute policy gradient
            params = opt.step(params, policy_gradient, episode_return)
        
        opt.end()
    
    Example (multi-seed, counts as 1 run):
        opt = MobiuQCore(license_key="xxx")
        
        for seed in range(10):
            opt.new_run()  # Reset optimizer state, keep session
            params = init_params(seed)
            for step in range(100):
                params = opt.step(params, grad, energy)
        
        opt.end()  # Counts as 1 run total
    
    Example (custom optimizer):
        opt = MobiuQCore(
            license_key="xxx", 
            method="qaoa",
            base_optimizer="NAdam"  # Try NAdam for QAOA
        )
    """
    
    def __init__(
        self,
        license_key: Optional[str] = None,
        method: str = "vqe",           # "vqe", "qaoa", or "rl"
        mode: str = "simulation",       # "simulation" or "hardware"
        base_lr: Optional[float] = None,
        base_optimizer: str = DEFAULT_OPTIMIZER,  # NEW in v2.4
        use_soft_algebra: bool = True,  # NEW in v2.4
        offline_fallback: bool = True,
        verbose: bool = True,
        # Deprecated parameters (backward compatibility)
        problem: Optional[str] = None,  # Use 'method' instead
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
            if method == "vqe":  # Only override if method wasn't explicitly set
                method = problem
        
        # Normalize mode (backward compatibility)
        if mode == "standard":
            mode = "simulation"
        elif mode == "noisy":
            mode = "hardware"
        
        # Validate method
        if method not in ("vqe", "qaoa", "rl"):
            raise ValueError(f"method must be 'vqe', 'qaoa', or 'rl', got '{method}'")
        
        # Validate mode
        if mode not in ("simulation", "hardware"):
            raise ValueError(f"mode must be 'simulation' or 'hardware', got '{mode}'")
        
        # Validate optimizer
        if base_optimizer not in AVAILABLE_OPTIMIZERS:
            raise ValueError(
                f"base_optimizer must be one of {AVAILABLE_OPTIMIZERS}, got '{base_optimizer}'"
            )
        
        self.method = method
        self.mode = mode
        self.base_lr = base_lr if base_lr is not None else get_default_lr(method, mode)
        self.base_optimizer = base_optimizer
        self.use_soft_algebra = use_soft_algebra
        self.offline_fallback = offline_fallback
        self.verbose = verbose
        self.session_id = None
        self.api_endpoint = API_ENDPOINT
        
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
                    "base_optimizer": self.base_optimizer,  # NEW in v2.4
                    "use_soft_algebra": self.use_soft_algebra  # NEW in v2.4
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
        gradient: np.ndarray, 
        energy: float
    ) -> np.ndarray:
        """
        Perform one optimization step.
        
        Args:
            params: Current parameter values
            gradient: Gradient of the energy w.r.t. params
            energy: Current energy value (or episode return for RL)
        
        Returns:
            Updated parameters
        
        Note for RL:
            For reinforcement learning (method='rl'), pass the episode return
            as the 'energy' parameter. Higher returns are better (maximization).
        """
        self.energy_history.append(energy)
        
        if self._offline_mode:
            return self._offline_step(params, gradient)
        
        try:
            # Retry loop for rate limiting
            for attempt in range(3):
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "license_key": self.license_key,
                        "session_id": self.session_id,
                        "action": "step",
                        "params": params.tolist(),
                        "gradient": gradient.tolist(),
                        "energy": float(energy)
                    },
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limited
                    if attempt < 2:
                        import time
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
        
        lr = self.base_lr
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
        
        This method ends the current session and starts a new one,
        which counts as a separate run. Use new_run() instead to
        keep multiple optimization runs in a single session.
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
        """
        Check current usage without affecting quota.
        
        Returns:
            dict with: tier, used, limit, remaining
        """
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
        """
        Get server information including available optimizers.
        
        Returns:
            dict with: version, default_optimizer, available_optimizers, methods, modes
        """
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
            "default_optimizer": DEFAULT_OPTIMIZER
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
    - VQE (smooth landscapes): finite_difference() or parameter_shift()
    - QAOA (rugged landscapes): spsa()
    - Hardware (noisy): spsa()
    - RL: Use your framework's gradient computation (e.g., PyTorch autograd)
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
        Best for: Noisy quantum hardware, NISQ devices, QAOA.
        
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
    
    # Verify it works
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
            print(f"   Available optimizers: {', '.join(info.get('available_optimizers', []))}")
    except Exception as e:
        print(f"❌ License check failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "2.4.0"
__all__ = [
    "MobiuQCore",
    "Demeasurement",
    "activate_license",
    "check_status",
    "get_default_lr",
    "AVAILABLE_OPTIMIZERS",
    "DEFAULT_OPTIMIZER"
]
