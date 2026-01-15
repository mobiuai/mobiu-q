"""
Mobiu - Adaptive Optimizer with Simple API (v3.6.6)
==========================================
A plug-and-play optimizer that automatically detects and adapts to your problem.

Usage:
    from mobiu_q import Mobiu

    opt = Mobiu(model.parameters(), lr=0.001)

    for batch in data:
        loss = model(batch)
        loss.backward()
        opt.step(loss.item())

That's it! Mobiu automatically:
- Detects if you're maximizing (reward) or minimizing (loss)
- Selects the best optimization strategy
- Adapts learning rate via Cloud Soft Algebra
- Falls back to standard Adam without license
"""

import numpy as np
from typing import Optional, List, Union
import warnings

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .warmup import WarmupPhaseManager
from .auto_config import AutoConfigEngine, MobiuConfig
from .core import (
    UniversalFrustrationEngine,
    get_license_key,
    API_ENDPOINT,
    AVAILABLE_OPTIMIZERS,
)


class Mobiu:
    """
    Adaptive Mobiu Optimizer - Simple API, automatic configuration.

    Designed to work like Adam but with intelligent adaptation:
    - Auto-detects maximize vs minimize
    - Auto-selects optimization strategy (standard/deep/adaptive)
    - Auto-selects learning rate
    - Uses Cloud Soft Algebra for best results

    Args:
        params: Model parameters (PyTorch) or numpy array (Quantum)
        lr: Initial learning rate hint (default: 0.001) - will be auto-adjusted
        license_key: License key for Cloud Soft Algebra
        warmup_steps: Steps to collect before auto-configuration (default: 30)
        verbose: Print status messages (default: True)
        use_soft_algebra: Enable Soft Algebra optimization (default: True)

    Example (PyTorch):
        model = MyModel()
        opt = Mobiu(model.parameters())

        for batch in dataloader:
            loss = criterion(model(batch))
            loss.backward()
            opt.step(loss.item())

    Example (RL):
        policy = PolicyNetwork()
        opt = Mobiu(policy.parameters())

        for episode in range(1000):
            # ... run episode ...
            opt.step(episode_reward)  # Mobiu detects this is reward
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        license_key: Optional[str] = None,
        warmup_steps: int = 30,
        verbose: bool = True,
        use_soft_algebra: bool = True,
        mode: Optional[str] = None,
        method: Optional[str] = None,
    ):
        self.initial_lr = lr
        self.base_lr = lr
        self.verbose = verbose
        self.use_soft_algebra = use_soft_algebra
        self._forced_mode = mode  # 'hardware' or 'simulation' - None for auto
        self._forced_method = method  # 'standard', 'deep', 'adaptive' - None for auto

        # License handling - REQUIRED for Mobiu to work
        self.license_key = license_key or get_license_key()
        self._has_license = bool(self.license_key)

        if not self._has_license:
            raise ValueError(
                "❌ Mobiu requires a license key to work.\n"
                "   Get your FREE key at https://app.mobiu.ai (20 sessions/month free)\n"
                "   Then use: Mobiu(params, license_key='your-key')\n"
                "   Or set MOBIU_LICENSE_KEY environment variable"
            )

        # Detect PyTorch vs NumPy
        self._is_pytorch = self._detect_pytorch(params)

        # Initialize base optimizer
        if self._is_pytorch:
            self._init_pytorch_optimizer(params, lr)
        else:
            self._init_numpy_mode(params, lr)

        # Phase management
        self.warmup = WarmupPhaseManager(warmup_steps)
        self._has_connection = self._check_connection() if self._has_license else False
        self.auto_config = AutoConfigEngine(
            has_license=self._has_license,
            has_connection=self._has_connection
        )
        self.is_configured = False

        # Components (initialized after warmup)
        self._config: Optional[MobiuConfig] = None
        self._frustration_engine: Optional[UniversalFrustrationEngine] = None

        # Cloud API state
        self._cloud_session_id: Optional[str] = None
        self._current_params: Optional[np.ndarray] = None
        self._current_gradient: Optional[np.ndarray] = None

        # State tracking
        self._step_count = 0
        self.energy_history: List[float] = []
        self.lr_history: List[float] = []

    def _detect_pytorch(self, params) -> bool:
        """Detect if params are PyTorch parameters."""
        if not HAS_TORCH:
            return False

        # Convert generator to list to avoid consuming it
        if hasattr(params, '__iter__') and not isinstance(params, list):
            self._params_list = list(params)
        else:
            self._params_list = params if isinstance(params, list) else [params]

        if len(self._params_list) == 0:
            return False

        first = self._params_list[0]
        return isinstance(first, torch.nn.Parameter) or isinstance(first, torch.Tensor)

    def _init_pytorch_optimizer(self, params, lr: float):
        """Initialize PyTorch optimizer."""
        # Use pre-converted list from _detect_pytorch
        self._params = self._params_list
        self._base_optimizer = torch.optim.Adam(self._params, lr=lr)

    def _init_numpy_mode(self, params, lr: float):
        """Initialize for NumPy/Quantum mode."""
        self._params = np.array(params)
        self._base_optimizer = None

        # Local Adam state
        self._m = np.zeros_like(self._params)
        self._v = np.zeros_like(self._params)
        self._t = 0

    def _check_connection(self) -> bool:
        """Check if cloud is reachable."""
        if not self._has_license or not HAS_REQUESTS:
            return False

        try:
            # Use POST with empty body - HEAD returns 500 on Cloud Functions
            response = requests.post(API_ENDPOINT, json={}, timeout=3)
            # API returns 400 for missing params, which means it's reachable
            return response.status_code < 500
        except:
            return False

    def step(self, metric: Optional[float] = None, gradient: Optional[np.ndarray] = None):
        """
        Perform one optimization step.

        Args:
            metric: Loss value (for minimization) or reward (for maximization).
                    Required during warmup.
            gradient: Optional gradient for NumPy mode (required for Cloud API)

        During warmup (first 30 steps):
            - Collects metrics to analyze your problem
            - Uses constant learning rate

        After warmup:
            - Uses Cloud Soft Algebra (with license)
            - Falls back to Frustration Engine (without license)
        """
        self._step_count += 1

        # Store gradient for Cloud API
        if gradient is not None:
            self._current_gradient = gradient

        # Phase 1: Warmup
        if not self.is_configured:
            return self._warmup_step(metric)

        # Phase 2: Adaptive operation
        return self._adaptive_step(metric)

    def _warmup_step(self, metric: Optional[float]):
        """Handle step during warmup phase."""
        if metric is not None:
            self.energy_history.append(metric)

            # Compute gradient norm for analysis
            grad_norm = self._compute_grad_norm() if self._is_pytorch else 0.0

            # Record and check if warmup complete
            if self.warmup.record(metric, grad_norm):
                self._configure_from_warmup()

        # Execute base step
        if self._is_pytorch:
            self._base_optimizer.step()
        else:
            # NumPy mode: apply local Adam step and return new params
            if self._current_gradient is not None:
                self._t += 1
                self._m = 0.9 * self._m + 0.1 * self._current_gradient
                self._v = 0.999 * self._v + 0.001 * (self._current_gradient ** 2)
                m_hat = self._m / (1 - 0.9 ** self._t)
                v_hat = self._v / (1 - 0.999 ** self._t)
                self._params = self._params - self.base_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            return self._params

    def _configure_from_warmup(self):
        """Configure optimizer based on warmup analysis."""
        analysis = self.warmup.analyze()
        self._config = self.auto_config.configure(analysis)

        # Update base_lr to auto-selected value
        self.base_lr = self._config.base_lr

        # Update base optimizer LR
        if self._is_pytorch and self._base_optimizer:
            for pg in self._base_optimizer.param_groups:
                pg['lr'] = self.base_lr

        # Initialize Frustration Engine (works locally)
        self._frustration_engine = UniversalFrustrationEngine(
            base_lr=self.base_lr,
            sensitivity=0.05
        )

        # Try to start Cloud session if beneficial
        if self._config.use_cloud and self._has_license and self._has_connection:
            self._start_cloud_session()

        self.is_configured = True

        if self.verbose:
            self._print_config()

    def _start_cloud_session(self):
        """Start a Cloud API session for Soft Algebra."""
        if not HAS_REQUESTS:
            return

        # Use forced mode/method if provided, otherwise use auto-detected
        mode = self._forced_mode or self._config.mode
        method = self._forced_method or self._config.method

        try:
            response = requests.post(API_ENDPOINT, json={
                'action': 'start',
                'license_key': self.license_key,
                'method': method,
                'mode': mode,
                'use_soft_algebra': self.use_soft_algebra,
                'base_optimizer': 'Adam',
                'base_lr': self.base_lr
            }, timeout=15)

            data = response.json()
            if data.get('success'):
                self._cloud_session_id = data['session_id']
                if self.verbose:
                    print(f"   Cloud session started: {self._cloud_session_id[:8]}...")
            else:
                if self.verbose:
                    print(f"   Cloud session failed: {data.get('error', 'Unknown')}")
                self._config.use_cloud = False

        except Exception as e:
            if self.verbose:
                print(f"   Cloud connection error: {str(e)[:50]}")
            self._config.use_cloud = False

    def _print_config(self):
        """Print user-friendly configuration message."""
        if self._config is None:
            return

        # Determine mode description
        mode = "optimizing reward" if self._config.maximize else "optimizing loss"

        # Format LR nicely
        lr_str = f"{self.base_lr:.6f}".rstrip('0').rstrip('.')

        if self._cloud_session_id:
            print(f"🚀 Mobiu ready ({mode}, Cloud Soft Algebra, lr={lr_str})")
        else:
            print(f"⚠️  Mobiu: Cloud connection failed - running with reduced performance")
            print(f"   ({mode}, lr={lr_str})")

    def _adaptive_step(self, metric: Optional[float]):
        """Handle step during adaptive phase."""
        if metric is not None:
            self.energy_history.append(metric)

        # Route to Cloud or Local based on configuration
        if self._cloud_session_id and metric is not None:
            return self._cloud_step(metric)
        else:
            return self._local_step(metric)

    def _cloud_step(self, metric: float):
        """Execute step via Cloud Soft Algebra API.

        CRITICAL: Send REAL params and gradients to get Soft Algebra benefits!
        The cloud computes new_params using Soft Algebra math.
        """
        if not HAS_REQUESTS:
            return self._local_step(metric)

        # Cloud assumes "lower is better" (energy/loss)
        # If maximizing, flip sign so cloud sees improvement
        energy_to_send = metric if not self._config.maximize else -metric

        # Get REAL params and gradients for Soft Algebra
        params_flat = self._get_params_flat()
        gradient_flat = self._get_gradient_flat()

        try:
            response = requests.post(API_ENDPOINT, json={
                'action': 'step',
                'license_key': self.license_key,
                'session_id': self._cloud_session_id,
                'params': params_flat.tolist(),      # REAL params!
                'gradient': gradient_flat.tolist(),  # REAL gradients!
                'energy': energy_to_send
            }, timeout=10)

            data = response.json()
            if data.get('success'):
                # Cloud returns new_params computed with Soft Algebra
                if 'new_params' in data:
                    new_params = np.array(data['new_params'])
                    self._set_params_from_flat(new_params)

                    # Track LR if provided
                    if 'adaptive_lr' in data:
                        self.lr_history.append(data['adaptive_lr'])
                else:
                    # Fallback: use adaptive_lr if no new_params
                    if 'adaptive_lr' in data:
                        adaptive_lr = data['adaptive_lr']
                        self.lr_history.append(adaptive_lr)
                        if self._is_pytorch and self._base_optimizer:
                            for pg in self._base_optimizer.param_groups:
                                pg['lr'] = adaptive_lr

                    # Run base optimizer if cloud didn't return new_params
                    if self._is_pytorch:
                        self._base_optimizer.step()

        except Exception as e:
            # On error, fall back to local step
            if self._is_pytorch:
                self._base_optimizer.step()

        # Return params for NumPy mode
        if not self._is_pytorch:
            return self._params

    def _local_step(self, metric: Optional[float]):
        """Fallback step when Cloud is temporarily unavailable."""
        # This should rarely happen - Cloud is required for Soft Algebra
        # Use Frustration Engine as basic fallback
        if self._frustration_engine and metric is not None:
            score = metric if self._config and self._config.maximize else -metric
            factor = self._frustration_engine.get_lr_factor(score)

            if factor != 1.0:
                self._apply_lr_factor(factor)
                self.lr_history.append(self.base_lr * factor)

        # Execute base optimizer step
        if self._is_pytorch:
            self._base_optimizer.step()
        else:
            # NumPy mode: apply local Adam step
            if self._current_gradient is not None:
                self._t += 1
                self._m = 0.9 * self._m + 0.1 * self._current_gradient
                self._v = 0.999 * self._v + 0.001 * (self._current_gradient ** 2)
                m_hat = self._m / (1 - 0.9 ** self._t)
                v_hat = self._v / (1 - 0.999 ** self._t)
                self._params = self._params - self.base_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            return self._params

    def _get_params_flat(self) -> np.ndarray:
        """Get flattened parameters."""
        if not self._is_pytorch:
            # NumPy mode - return params directly
            return self._params.flatten()

        params_list = []
        for p in self._params:
            params_list.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params_list)

    def _get_gradient_flat(self) -> np.ndarray:
        """Get flattened gradients."""
        if not self._is_pytorch:
            # NumPy mode - return stored gradient
            if self._current_gradient is not None:
                return self._current_gradient.flatten()
            return np.zeros_like(self._params).flatten()

        grads_list = []
        for p in self._params:
            if p.grad is not None:
                grads_list.append(p.grad.data.cpu().numpy().flatten())
            else:
                grads_list.append(np.zeros(p.numel()))
        return np.concatenate(grads_list)

    def _set_params_from_flat(self, flat_params: np.ndarray):
        """Set parameters from flattened array (PyTorch)."""
        if not self._is_pytorch:
            self._params = flat_params.reshape(self._params.shape)
            return

        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(torch.tensor(
                flat_params[offset:offset + numel].reshape(p.shape),
                dtype=p.dtype,
                device=p.device
            ))
            offset += numel

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm."""
        if not self._is_pytorch:
            # NumPy mode: use stored gradient
            if self._current_gradient is not None:
                return float(np.linalg.norm(self._current_gradient))
            return 0.0

        total_norm = 0.0
        for p in self._params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _apply_lr_factor(self, factor: float):
        """Apply learning rate multiplier."""
        if self._is_pytorch:
            for pg in self._base_optimizer.param_groups:
                pg['lr'] = self.base_lr * factor

    def zero_grad(self):
        """Zero gradients (PyTorch mode only)."""
        if self._is_pytorch and self._base_optimizer:
            self._base_optimizer.zero_grad()

    def new_run(self, params=None):
        """Reset for new optimization run (multi-seed experiments).

        Args:
            params: Optional new initial parameters (required for NumPy mode)
        """
        # End current cloud session
        self._end_cloud_session()

        # Reset tracking
        self.energy_history.clear()
        self.lr_history.clear()
        self._step_count = 0
        self._current_gradient = None

        # Reset components
        if self._frustration_engine:
            self._frustration_engine.reset()

        # Reset optimizer state
        if self._is_pytorch and self._base_optimizer:
            self._base_optimizer.state.clear()
            for pg in self._base_optimizer.param_groups:
                pg['lr'] = self.base_lr
        else:
            # NumPy mode: reset Adam state and optionally set new params
            self._m = np.zeros_like(self._params)
            self._v = np.zeros_like(self._params)
            self._t = 0
            if params is not None:
                self._params = np.array(params)
                self._m = np.zeros_like(self._params)
                self._v = np.zeros_like(self._params)

        # Reset warmup for new run
        self.warmup = WarmupPhaseManager(self.warmup.warmup_steps)
        self.is_configured = False
        self._config = None

        # Start new cloud session if configured
        if self._config and self._config.use_cloud and self._has_license:
            self._start_cloud_session()

    def _end_cloud_session(self):
        """End the current Cloud API session."""
        if not self._cloud_session_id or not HAS_REQUESTS:
            return

        try:
            requests.post(API_ENDPOINT, json={
                'action': 'end',
                'license_key': self.license_key,
                'session_id': self._cloud_session_id
            }, timeout=5)
        except:
            pass

        self._cloud_session_id = None

    def end(self):
        """End optimization session."""
        self._end_cloud_session()

    @property
    def config(self) -> Optional[MobiuConfig]:
        """Get current configuration (available after warmup)."""
        return self._config

    @property
    def param_groups(self):
        """Expose param_groups for framework compatibility."""
        if self._is_pytorch and self._base_optimizer:
            return self._base_optimizer.param_groups
        return []

    @property
    def state(self):
        """Expose optimizer state for framework compatibility."""
        if self._is_pytorch and self._base_optimizer:
            return self._base_optimizer.state
        return {}

    @property
    def params(self) -> np.ndarray:
        """Get current parameters (NumPy mode)."""
        if not self._is_pytorch:
            return self._params
        # PyTorch mode: return flattened params
        return self._get_params_flat()

    def get_params(self) -> np.ndarray:
        """Get current parameters as numpy array."""
        return self._get_params_flat()

    def set_params(self, params: np.ndarray):
        """Set parameters from numpy array."""
        self._set_params_from_flat(params)

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.end()
        except:
            pass


__all__ = ["Mobiu"]
