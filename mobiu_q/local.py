"""
Mobiu-Q Local Soft Algebra Engine (v5.0)
=========================================

In-process Soft Algebra. No network. No license key. No cloud round-trips.

This module mirrors the behavior of the server-side `optimization_step()`
function in `server/main.py`. Given the same inputs (energy, gradient, method,
base_lr, soft state) it produces bitwise-identical outputs — verified by
the parity test suite in `tests/test_local_parity.py`.

Design intent
-------------
The cloud server has always been licensing and telemetry infrastructure
wrapped around a small pure-NumPy algorithm. `soft_algebra_core.py` is ~200
lines of math; it can run anywhere Python and NumPy run. `LocalSAEngine`
exposes that math directly to the client, eliminating per-step network calls
for customers who can't or won't route gradient signals through an external
endpoint (air-gapped training, enterprise VPCs, latency-sensitive pipelines).

For the v5.0 release, local mode is trust-based: no license enforcement in
the pure-library path. This is appropriate given the current customer count
(zero). When paying customers exist, a signed-license layer can be added
without touching the algorithmic core (see P2_SPEC_pure_library_mode.md §
"Licensing").

Public contract
---------------
`LocalSAEngine.step(energy, gradient=None)` returns a dict with the same
keys the cloud `/step` endpoint returns:

    {
        'adaptive_lr': float,
        'warp_factor': float,
        'converged': bool,
        'metric_value': float,
    }

Client-side code paths that previously dispatched on a cloud response can
dispatch on this dict with no other changes.
"""

from typing import Optional, Dict, Any
import numpy as np

from .soft_algebra_core import (
    SoftNumber,
    signal_energy_curvature,
    signal_realized_improvement,
    soft_momentum_update,
    compute_standard_scaling,
    compute_deep_scaling,
    compute_adaptive_scaling,
)


# ── Hybrid warp factor (client-side gradient warping) ───────────────────────
# Duplicated here from server/main.py::_compute_hybrid_warp_factor so that
# LocalSAEngine has zero server-side dependencies. The formula is intentionally
# weaker than compute_soft_factor in soft_algebra_core (clip [0.5, 2.0] vs
# [0.9, 3.0]) because the client's base optimizer (Adam) already does adaptive
# moment adjustment — paper-strength warping would double-count.
#
# If you change this formula, also change the server copy, and re-run
# tests/test_local_parity.py to confirm bitwise agreement.
def _compute_hybrid_warp_factor(sn_state: SoftNumber) -> float:
    trust = abs(sn_state.real) / (abs(sn_state.real) + abs(sn_state.soft) + 1e-8)
    warp = 1.0 + 0.1 * sn_state.soft * (1 - trust)
    return float(np.clip(warp, 0.5, 2.0))


# ── Method alias resolution (legacy: vqe/qaoa/rl) ───────────────────────────
_METHOD_ALIASES = {
    "standard": "standard", "deep": "deep", "adaptive": "adaptive",
    "vqe": "standard",      "qaoa": "deep", "rl": "adaptive",
}


class LocalSAEngine:
    """
    In-process Soft Algebra session.

    Holds its own state (soft number, energy history, step counter). Pure
    NumPy, no threading, no I/O. One engine per optimization run — not
    thread-safe; create a fresh engine per worker in multi-seed experiments.

    Parameters
    ----------
    method : {"standard", "deep", "adaptive"} or legacy {"vqe", "qaoa", "rl"}
        Scaling regime. Matches the cloud server's `method` parameter.
    base_lr : float
        Base learning rate. The engine returns `adaptive_lr` in [clip_low ·
        base_lr, clip_high · base_lr] depending on the regime.
    use_soft_algebra : bool, default True
        If False, the engine returns `base_lr` and `warp_factor=1.0` every
        step — equivalent to no wrapping at all. For A/B testing.
    maximize : bool, default False
        If True, `signal_realized_improvement` treats higher energy as better
        (RL mode). If False, lower is better (VQE / loss minimization mode).
    """

    def __init__(
        self,
        method: str = "adaptive",
        base_lr: float = 3e-4,
        use_soft_algebra: bool = True,
        maximize: bool = False,
    ):
        resolved = _METHOD_ALIASES.get(method)
        if resolved is None:
            raise ValueError(
                f"method must be one of 'standard', 'deep', 'adaptive' "
                f"(or legacy 'vqe', 'qaoa', 'rl'); got {method!r}"
            )
        self.method = resolved
        self.base_lr = float(base_lr)
        self.use_soft_algebra = bool(use_soft_algebra)
        self.maximize = bool(maximize)

        self.soft_state = SoftNumber(soft=0.0, real=0.0)
        self.energy_history: list = []
        self.t: int = 0

    # ── Core step ───────────────────────────────────────────────────────────
    def step(
        self,
        energy: float,
        gradient: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform one SA step. Returns LR and warp for the client to apply.

        Returns
        -------
        dict with keys:
            'adaptive_lr'  : float, the LR to apply to the base optimizer
            'warp_factor'  : float, gradient multiplier in [0.5, 2.0]
            'converged'    : bool, True means true minimum detected (halt recommended)
            'metric_value' : float, diagnostic (trust ratio or emergence score)
        """
        # 1. Update step counter and energy history (keep last 10, matching server)
        self.t += 1
        self.energy_history.append(float(energy))
        if len(self.energy_history) > 10:
            self.energy_history = self.energy_history[-10:]

        # 2. If SA disabled or not enough history, short-circuit to base behavior.
        if not self.use_soft_algebra or len(self.energy_history) < 2:
            return {
                "adaptive_lr": self.base_lr,
                "warp_factor": 1.0,
                "converged": False,
                "metric_value": 0.0,
            }

        # 3. Extract raw signals (identical to server).
        a_t = signal_energy_curvature(self.energy_history)
        b_t = signal_realized_improvement(self.energy_history, maximize=self.maximize)

        # 4. Accumulate into soft state: S_{t+1} = (gamma * S_t) * delta + delta
        gamma = 0.9
        delta_sn = SoftNumber(soft=a_t, real=b_t)
        self.soft_state = soft_momentum_update(self.soft_state, delta_sn, gamma)

        # 5. Method-specific scaling.
        grad_placeholder = gradient if gradient is not None else np.zeros(1)

        if self.method == "deep":
            alpha_t, _, metric = compute_deep_scaling(
                self.soft_state, self.base_lr, grad_placeholder
            )
        elif self.method == "adaptive":
            alpha_t, _, metric = compute_adaptive_scaling(
                self.soft_state, self.base_lr, grad_placeholder
            )
        else:  # standard
            alpha_t, _, metric = compute_standard_scaling(
                self.soft_state, self.base_lr, grad_placeholder
            )

        # 6. True-minimum check
        if alpha_t == 0.0:
            return {
                "adaptive_lr": 0.0,
                "warp_factor": 1.0,
                "converged": True,
                "metric_value": float(metric),
            }

        # 7. Hybrid warp factor for client-side gradient modification
        warp = _compute_hybrid_warp_factor(self.soft_state)

        return {
            "adaptive_lr": float(alpha_t),
            "warp_factor": float(warp),
            "converged": False,
            "metric_value": float(metric),
        }

    # ── Lifecycle methods (API parity with cloud backend) ───────────────────
    def reset(self) -> None:
        """Clear accumulated state for a new run within the same engine."""
        self.soft_state = SoftNumber(soft=0.0, real=0.0)
        self.energy_history = []
        self.t = 0

    def end(self) -> None:
        """
        No-op in local mode; present for API parity with cloud mode.
        """
        pass


__all__ = ["LocalSAEngine"]
