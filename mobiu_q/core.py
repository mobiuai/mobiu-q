"""
Mobiu-Q Client - Soft Algebra Optimizer
========================================
Cloud-connected optimizer for quantum variational algorithms.

Usage (VQE - Chemistry):
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", problem="vqe")
    
    for step in range(100):
        grad = Demeasurement.finite_difference(energy_fn, params)
        params = opt.step(params, grad, energy_fn(params))
    
    opt.end()

Usage (QAOA - Combinatorial):
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key="your-key", problem="qaoa", mode="noisy")
    
    for step in range(150):
        grad, energy = Demeasurement.spsa(energy_fn, params)
        params = opt.step(params, grad, energy)
    
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
from typing import Optional, Tuple
import os
import json

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API endpoint
API_ENDPOINT = os.environ.get(
    "MOBIU_Q_API_ENDPOINT",
    "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
)

LICENSE_KEY_FILE = os.path.expanduser("~/.mobiu_q_license")


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
# MOBIU-Q CORE (Cloud Client)
# ═══════════════════════════════════════════════════════════════════════════════

class MobiuQCore:
    """
    Mobiu-Q Optimizer - Cloud-connected version.
    
    The optimization logic runs on Mobiu's secure servers.
    This client handles communication and local state.
    
    Args:
        license_key: Your Mobiu-Q license key (or set MOBIU_Q_LICENSE_KEY env var)
        mode: "standard" (clean simulations) or "noisy" (quantum hardware)
        problem: "vqe" (chemistry, default) or "qaoa" (combinatorial)
        base_lr: Learning rate (default: 0.01 for standard, 0.02 for noisy)
        offline_fallback: If True, use local Adam when API unavailable
    
    Example (VQE - Chemistry):
        opt = MobiuQCore(license_key="xxx", problem="vqe")
        
        for step in range(100):
            grad = Demeasurement.finite_difference(energy_fn, params)
            params = opt.step(params, grad, energy_fn(params))
        
        opt.end()
    
    Example (QAOA - Combinatorial):
        opt = MobiuQCore(license_key="xxx", problem="qaoa", mode="noisy")
        
        for step in range(150):
            grad, energy = Demeasurement.spsa(energy_fn, params)
            params = opt.step(params, grad, energy)
        
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
        mode: str = "standard",
        problem: str = "vqe",  # NEW: "vqe" or "qaoa"
        base_lr: Optional[float] = None,
        offline_fallback: bool = True,
        verbose: bool = True
    ):
        self.license_key = license_key or get_license_key()
        if not self.license_key:
            raise ValueError(
                "License key required. Set MOBIU_Q_LICENSE_KEY environment variable, "
                "or pass license_key parameter, or run: mobiu-q activate YOUR_KEY"
            )
        
        # Validate problem type
        if problem not in ("vqe", "qaoa"):
            raise ValueError(f"problem must be 'vqe' or 'qaoa', got '{problem}'")
        
        self.mode = mode
        self.problem = problem  # NEW
        self.base_lr = base_lr
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
        
        # Start session
        self._start_session()
    
    def _start_session(self):
        """Initialize optimization session with server."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "license_key": self.license_key,
                    "action": "start",
                    "mode": self.mode,
                    "problem": self.problem,  # NEW
                    "base_lr": self.base_lr
                },
                timeout=10
            )
            
            data = response.json()
            
            if not data.get("success"):
                error = data.get("error", "Unknown error")
                if "limit" in error.lower():
                    print(f"⚠️  {error}")
                    print("   Upgrade at: https://app.mobiu.ai")
                raise RuntimeError(f"Failed to start session: {error}")
            
            self.session_id = data["session_id"]
            runs_remaining = data.get("runs_remaining")
            problem_type = data.get("problem", self.problem)
            
            if runs_remaining is not None and runs_remaining >= 0:
                if self.verbose:
                    print(f"🚀 Mobiu-Q session started ({runs_remaining} runs remaining) [problem={problem_type}]")
            else:
                if self.verbose:
                    print(f"🚀 Mobiu-Q session started (Pro tier) [problem={problem_type}]")
                
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
            energy: Current energy value
        
        Returns:
            Updated parameters
        """
        self.energy_history.append(energy)
        
        if self._offline_mode:
            return self._offline_step(params, gradient)
        
        try:
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
        
        lr = self.base_lr or (0.02 if self.mode == "noisy" else 0.01)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        self._local_m = beta1 * self._local_m + (1 - beta1) * gradient
        self._local_v = beta2 * self._local_v + (1 - beta2) * (gradient ** 2)
        
        m_hat = self._local_m / (1 - beta1 ** self._local_t)
        v_hat = self._local_v / (1 - beta2 ** self._local_t)
        
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return params - update
    
    def end(self, force_count: bool = False):
        """
        End the optimization session.
        
        Call this when optimization is complete!
        Sessions ended within 60 seconds with <5 steps don't count against quota.
        
        Args:
            force_count: If True, always count as a run (even within grace period)
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
            if self.verbose:
                if data.get("counted_as_run"):
                    print("✅ Session ended (counted as 1 run)")
                else:
                    print("✅ Session ended (not counted - within grace period)")
                
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
        import warnings
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
        opt = MobiuQCore(license_key=key)
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
        opt = MobiuQCore(license_key=key)
        opt.end()
        print("✅ License is active")
    except Exception as e:
        print(f"❌ License check failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "1.1.0"
__all__ = [
    "MobiuQCore",
    "Demeasurement",
    "activate_license",
    "check_status"
]