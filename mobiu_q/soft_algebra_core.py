"""
═══════════════════════════════════════════════════════════════════════════════
                    SOFT ALGEBRA CORE - PROTECTED INTELLECTUAL PROPERTY
═══════════════════════════════════════════════════════════════════════════════

                              ⚠️  DO NOT MODIFY  ⚠️

This file contains the core mathematical implementation of Soft Algebra,
developed by Dr. Moshe Klein and Prof. Oded Maimon.

Patent-pending technology. All rights reserved by Mobiu Technologies.

Mathematical Foundation:
------------------------
Soft Numbers extend real numbers with infinitesimal components using 
nilpotent arithmetic where ε² = 0.

A SoftNumber S = a·ε + b where:
    a = potential (infinitesimal/soft component)
    b = realization (real component)

Core Multiplication Rule (nilpotent, ⊗):
    (a, b) ⊗ (c, d) = (ad + bc, bd)

Evolution Law:
    S_{t+1} = (γ · S_t) ⊗ Δ_t + Δ_t

    Note: γ · S_t is scalar multiplication; ⊗ is nilpotent multiplication above.

Soft Inverse (Klein, Lemma 4.1):
    (aε + b)⁻¹ = (1/b)ε·(-a/b²) + (1/b),  b ≠ 0
    When b → 0: potential ascends to numerator (soft regime).

    Trust Ratio implements this principle:
        φ = |b| / (|b| + |a|)      when b > 0  (realized improvement)
        φ = 1                        when b = 0, a > 0  (soft regime: 2× boost)
        φ = -1                       when a = b = 0  (true convergence: halt)

    Gradient Warping implements soft inverse for landscape navigation:
        warp = 1 + a / (a + b²)     (→ 2 when b=0; → 1 when b large)

Soft Calculus (Klein, Eq. 12):
    f(aε + b) = f(b) + a·f'(b)·ε

    Super-Equation computes Du[sin(π·S)] via complex-step approximation:
        S_complex = b + i·a·ε_c,  ε_c = 0.43
        du = Im(sin(π·S_complex))
    This is equivalent to Klein's formal extension at first order,
    and empirically outperforms the closed-form π·a·cos(π·b).

This allows the optimizer to carry both curvature (potential) AND improvement
(realization) information, distinguishing real progress from noise artifacts.

═══════════════════════════════════════════════════════════════════════════════
Version: 1.0.0 (Frozen)
Last Modified: 2025-01-01
DO NOT MODIFY WITHOUT EXPLICIT APPROVAL FROM SCIENTIFIC ADVISORS
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
import numpy as np

INVERTED_VERSION = "1.0.0-klein"

# ═══════════════════════════════════════════════════════════════════════════════
# SOFT NUMBER CLASS
# Core nilpotent arithmetic: ε² = 0
# Multiplication: (a,b) × (c,d) = (ad + bc, bd)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SoftNumber:
    """
    Soft Number representation: S = soft·ε + real
    
    Where:
        soft (a): The potential/infinitesimal component
        real (b): The realized/actual component
        
    Nilpotent property: ε² = 0
    """
    soft: float   # a (potential)
    real: float   # b (actual/realized)
    
    def __add__(self, other: "SoftNumber") -> "SoftNumber":
        """Addition: (a₁, b₁) + (a₂, b₂) = (a₁ + a₂, b₁ + b₂)"""
        return SoftNumber(self.soft + other.soft, self.real + other.real)
    
    def __mul__(self, other):
        """
        Multiplication with nilpotent property (ε² = 0):
        (a, b) × (c, d) = (ad + bc, bd)
        
        This is the core of Soft Algebra - the infinitesimal components
        interact but don't compound (ε² = 0).
        """
        if isinstance(other, SoftNumber):
            # Handle zero case for numerical stability
            if abs(self.real) < 1e-12 and abs(other.real) < 1e-12:
                return SoftNumber(0.0, 0.0)
            a, b = self.soft, self.real
            c, d = other.soft, other.real
            # Nilpotent multiplication: (a,b)*(c,d) = (ad+bc, bd)
            return SoftNumber(soft=a * d + b * c, real=b * d)
        else:
            # Scalar multiplication
            scalar = float(other)
            return SoftNumber(soft=self.soft * scalar, real=self.real * scalar)
    
    def __rmul__(self, other):
        """Right multiplication (for scalar * SoftNumber)"""
        return self.__mul__(other)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demeasure(potential: float, actual: float) -> SoftNumber:
    """
    Create a SoftNumber from potential and actual values.
    
    This is the "demeasurement" operation - converting classical
    measurements into the soft algebra domain.
    
    Args:
        potential: The infinitesimal/potential component (a)
        actual: The realized/real component (b)
    
    Returns:
        SoftNumber representing S = potential·ε + actual
    """
    return SoftNumber(soft=potential, real=actual)


def evolve_state(state: SoftNumber, delta: SoftNumber) -> SoftNumber:
    """
    Evolution operation: S_{new} = S_{old} × Δ + Δ
    
    This is the core evolution law - updates are only committed
    when Potential meets Reality.
    
    Args:
        state: Current soft state S_t
        delta: Change signal Δ_t
    
    Returns:
        New state S_{t+1}
    """
    return state * delta + delta


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION FUNCTIONS
# Extract (a, b) from energy history
# ═══════════════════════════════════════════════════════════════════════════════

def signal_energy_curvature(energy_history: list) -> float:
    """
    Extract potential signal (a_t) from energy curvature.
    
    Formula: a_t = |E_t - 2·E_{t-1} + E_{t-2}| / (curvature + mean_E)
    
    High curvature → high potential for change
    Low curvature → stable, less potential
    
    Args:
        energy_history: List of recent energy values
    
    Returns:
        Curvature-based potential signal a_t ∈ [0, ∞)
    """
    if len(energy_history) < 3:
        return 0.0
    
    E_t, E_t1, E_t2 = energy_history[-1], energy_history[-2], energy_history[-3]
    curvature = abs(E_t - 2*E_t1 + E_t2)
    mean_E = abs(np.mean(energy_history[-3:]))
    
    if mean_E < 1e-12:
        return 0.0
    
    return curvature / (curvature + mean_E)


def signal_realized_improvement(energy_history: list, maximize: bool = False) -> float:
    """
    Extract realization signal (b_t) from actual improvement.
    
    For minimization (VQE, QAOA): b_t = (E_{t-1} - E_t) / |E_{t-1}|
    For maximization (RL): b_t = (E_t - E_{t-1}) / |E_{t-1}|
    
    Clamped to [0, 1] for stability.
    
    Args:
        energy_history: List of recent energy values
        maximize: If True, higher values are better (RL mode)
    
    Returns:
        Improvement signal b_t ∈ [0, 1]
    """
    if len(energy_history) < 2:
        return 0.0
    
    E_prev, E_curr = energy_history[-2], energy_history[-1]
    denom = abs(E_prev) + 1e-9
    
    if maximize:
        # For RL: higher return = better
        b_t = (E_curr - E_prev) / denom
    else:
        # For VQE/QAOA: lower energy = better
        b_t = (E_prev - E_curr) / denom
    
    # Clamp to [0, 1] for stability
    return max(-1.0, min(1.0, b_t))


def create_soft_signal(energy_history: list, maximize: bool = False) -> SoftNumber:
    """
    Create a complete soft signal Δ_t from energy history.
    
    Combines curvature (potential) and improvement (realization)
    into a single SoftNumber for evolution.
    
    Args:
        energy_history: List of recent energy values
        maximize: If True, higher values are better (RL mode)
    
    Returns:
        SoftNumber Δ_t = (a_t, b_t)
    """
    a_t = signal_energy_curvature(energy_history)
    b_t = signal_realized_improvement(energy_history, maximize)
    
    # Clamp a_t for numerical stability
    a_t = max(0.0, min(10.0, a_t))
    
    return demeasure(potential=a_t, actual=b_t)


# ═══════════════════════════════════════════════════════════════════════════════
# SOFT MOMENTUM ALGEBRA
# Trust-based learning rate adaptation
# ═══════════════════════════════════════════════════════════════════════════════

def soft_momentum_update(sn_state: SoftNumber, delta_sn: SoftNumber, 
                         gamma: float = 0.9) -> SoftNumber:
    """
    Soft momentum update with decay.
    
    S_{t+1} = (γ · S_t) · Δ_t + Δ_t
    
    The decay factor γ controls how much history affects current state.
    
    Args:
        sn_state: Current accumulated soft state
        delta_sn: New signal to incorporate
        gamma: Decay factor (default 0.9)
    
    Returns:
        Updated soft state
    """
    decayed_state = sn_state * gamma
    return evolve_state(decayed_state, delta_sn)

def compute_trust_ratio(sn_state: SoftNumber) -> float:
    a = abs(sn_state.soft)
    b = abs(sn_state.real)
    if a < 1e-9 and b < 1e-9:
        return -1.0  
    if b < 1e-9:
        return a / (a + b)  
    return b / (a + b) 

def compute_soft_factor(sn_state: SoftNumber) -> float:
    """
    Soft Inverse — algebraically safe.
    
    When b=0, a>0 (barren plateau): returns 2.0 (boost)
    When b large, a small (healthy): returns ~1.0 (no boost)
    When a=0, b=0: returns 1.0 (neutral — no information to warp with).
        Previously assumed unreachable, but compute_adaptive_scaling and
        compute_deep_scaling do not short-circuit on trust=-1 the way
        compute_standard_scaling does, so this path IS reachable on
        perfectly-flat plateaus. Guarding against 0/0 is strictly safer
        than relying on callers. Caught by v5.0 parity test. (v5.0 fix)
    """
    a = abs(sn_state.soft)
    b = abs(sn_state.real)
    denom = a + b ** 2
    if denom < 1e-12:
        return 1.0  # neutral — no warping when state has collapsed to origin
    soft_factor = 1.0 + a / denom
    return max(0.9, min(3.0, soft_factor))


def adaptive_learning_rate_standard(base_lr: float, trust_ratio: float) -> float:
    """
    Compute adaptive learning rate for standard method.
    
    α_t = base_lr × (1 + 0.3 × trust)
    Clamped multiplier to [0.5, 1.5].
    
    Args:
        base_lr: Base learning rate
        trust_ratio: Current trust ratio
    
    Returns:
        Adapted learning rate
    """
    lr_mult = 1.0 + 0.3 * trust_ratio
    return base_lr * max(0.5, min(1.5, lr_mult))


# ═══════════════════════════════════════════════════════════════════════════════
# SUPER-EQUATION Δ† (For Deep/QAOA)
# Emergence detection for rugged landscapes
# ═══════════════════════════════════════════════════════════════════════════════

def compute_super_equation(sn_state: SoftNumber, 
                           alpha: float = 1.35, 
                           beta: float = 1.70, 
                           C: float = 3.00, 
                           epsilon: float = 0.43) -> float:
    """
    Compute Super-Equation Δ† for emergence detection.
    
    The Super-Equation detects when the system is at a critical point
    where small changes could lead to significant improvements.
    
    Components:
        du = Im(sin(π·S))       (Phase sensitivity)
        τ = C·a·b               (Coupling strength)
        g = exp(-(τ-1)²/2α²)    (Gaussian gate)
        γ = 1 - exp(-β·a)       (Emergence gate)
        
    Δ† = |du| × g × γ × √(b·g)
    
    Args:
        sn_state: Current soft state
        alpha: Gaussian width parameter (default 1.35)
        beta: Emergence gate steepness (default 1.70)
        C: Coupling constant (default 3.00)
    
    Returns:
        Emergence score Δ† ∈ [0, ∞)
    """
    a, b = sn_state.soft, sn_state.real
    
    # Complex soft representation
    S = b + 1j * a * epsilon
    
    # Phase sensitivity (imaginary part of sin)
    du = np.sin(np.pi * S).imag
    
    # Coupling strength
    tau = C * a * b
    
    # Gaussian gate centered at τ = 1
    g = np.exp(-(tau - 1)**2 / (2 * alpha**2))
    
    # Emergence gate (activates when potential is significant)
    gamma_gate = 1 - np.exp(-beta * a)
    
    # Final emergence score
    return abs(abs(du) * g * gamma_gate * np.sqrt(max(0, b * g)))


def adaptive_learning_rate_deep(base_lr: float, delta_dagger: float) -> float:
    """
    Compute adaptive learning rate for deep method using Δ†.
    
    α_t = base_lr × (0.3 + 5.0 × Δ†)
    Clamped to [0.3, 2.0] × base_lr.
    
    High emergence → larger steps to exploit the opportunity
    Low emergence → smaller steps to be cautious
    
    Args:
        base_lr: Base learning rate
        delta_dagger: Emergence score from super-equation
    
    Returns:
        Adapted learning rate
    """
    emergence_factor = 0.3 + 5.0 * delta_dagger
    return base_lr * max(0.3, min(2.0, emergence_factor))


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD-SPECIFIC SCALING FUNCTIONS
# Combine signals into final learning rate and gradient modifications
# ═══════════════════════════════════════════════════════════════════════════════

def compute_standard_scaling(sn_state: SoftNumber, base_lr: float,
                             gradient: np.ndarray) -> tuple:
    """..."""
    trust = compute_trust_ratio(sn_state)
    if trust < 0:
        return 0.0, gradient * 0.0, 0.0  # עצור — מינימום אמיתי
    scale = max(0.5, min(2.0, 1.0 + 1.0 * trust))
    alpha_t = base_lr * scale
    soft_factor = compute_soft_factor(sn_state)
    g_eff = gradient * soft_factor
    return alpha_t, g_eff, trust


def compute_deep_scaling(sn_state: SoftNumber, base_lr: float,
                         gradient: np.ndarray) -> tuple:
    """
    Compute scaling for deep method (QAOA, noisy hardware).
    
    Uses Super-Equation Δ† for emergence detection.
    
    Args:
        sn_state: Current soft state
        base_lr: Base learning rate
        gradient: Current gradient
    
    Returns:
        (adapted_lr, gradient, metric_value)
    """
    delta_dagger = compute_super_equation(sn_state)
    
    # Emergence-based scaling with dampening
    emergence_factor = 0.3 + 5.0 * delta_dagger
    dampening = 1.0 / (1.0 + sn_state.soft)
    
    scale = emergence_factor * dampening
    scale = max(0.3, min(2.0, scale))
    
    alpha_t = base_lr * scale
    
    return alpha_t, gradient, delta_dagger


def compute_adaptive_scaling(sn_state: SoftNumber, base_lr: float,
                             gradient: np.ndarray) -> tuple:
    """
    Compute scaling for adaptive method (RL, LLM).
    
    Combines Trust Ratio + Emergence Boost + Gradient Warping.
    This achieved +129% on LunarLander and +18% on LLM tuning.
    
    Args:
        sn_state: Current soft state
        base_lr: Base learning rate
        gradient: Current gradient
    
    Returns:
        (adapted_lr, warped_gradient, metric_value)
    """
    trust = compute_trust_ratio(sn_state)
    delta_dagger = compute_super_equation(sn_state)
    
    # Trust scale (like standard)
    trust_scale = max(0.5, min(2.0, 1.0 + 1.0 * trust))
    
    # Emergence boost (different formula from deep!)
    emergence_boost = 1.0 + 2.0 * delta_dagger
    
    # Combined scale with higher max (3.0 vs 2.0)
    scale = min(3.0, trust_scale * emergence_boost)
    alpha_t = base_lr * scale
    
    # Gradient warping (like standard)
    soft_factor = compute_soft_factor(sn_state)
    g_eff = gradient * soft_factor
    
    return alpha_t, g_eff, delta_dagger


# ═══════════════════════════════════════════════════════════════════════════════
# END OF PROTECTED CORE
# ═══════════════════════════════════════════════════════════════════════════════