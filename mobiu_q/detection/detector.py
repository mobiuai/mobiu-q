"""
================================================================================
MOBIU-AD - Anomaly Detection Client SDK
================================================================================
Python client for Mobiu-AD Cloud API.

Uses the SAME Soft Algebra core as Mobiu-Q optimizer.

Quick Start:
    >>> from mobiu_ad import MobiuAD
    >>> 
    >>> detector = MobiuAD(license_key="your-key")
    >>> for value in data_stream:
    ...     result = detector.detect(value)
    ...     if result.is_anomaly:
    ...         print(f"Anomaly! Δ†={result.delta_dagger:.4f}")

Version: 3.9.0
Copyright (c) 2025 Ido Angel / Mobiu Technologies
================================================================================
"""

__version__ = "3.9.0"
__author__ = "Ido Angel"

import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import uuid


# Default API endpoint
DEFAULT_API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_ad"


@dataclass
class DetectionResult:
    """Result from anomaly detection."""
    value: float
    at: float              # Demeasurement (potential)
    bt: float              # Measurement (deviation)
    delta_dagger: float    # Super-Equation score
    trust_ratio: float     # Confidence
    is_anomaly: bool
    threshold: float = 0.0
    is_transition: bool = False
    warmup: bool = False
    
    def __repr__(self):
        status = "WARMUP" if self.warmup else ("ANOMALY" if self.is_anomaly else "NORMAL")
        return f"DetectionResult({status}, value={self.value:.4f}, Δ†={self.delta_dagger:.4f})"


@dataclass
class BatchResult:
    """Result from batch detection."""
    total_points: int
    total_anomalies: int
    anomaly_indices: List[int]
    results: List[DetectionResult]
    session_id: str


class MobiuAD:
    """
    Mobiu-AD Anomaly Detection Client.
    
    Uses Soft Algebra (same as Mobiu-Q optimizer) for anomaly detection.
    
    Methods:
    - standard: Trust Ratio based detection
    - deep: Super-Equation Δ† based detection (recommended)
    - transition: Regime change detection
    
    Example:
        >>> detector = MobiuAD(license_key="your-key")
        >>> 
        >>> # Single detection
        >>> result = detector.detect(value=42.5)
        >>> 
        >>> # Batch detection
        >>> results = detector.detect_batch(values=[1.0, 2.0, 100.0, 3.0])
        >>> print(f"Anomalies at: {results.anomaly_indices}")
    """
    
    def __init__(
        self,
        license_key: str,
        method: str = "auto",  # Changed default to auto!
        threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        api_url: str = DEFAULT_API_URL
    ):
        """
        Initialize Mobiu-AD client.
        
        Args:
            license_key: Your Mobiu license key
            method: Detection method:
                - "auto" (default): Automatically adapts to data
                - "deep": Super-Equation Δ† for complex patterns
                - "transition": For regime changes
                - "standard": Simple Trust Ratio based
            threshold: Custom detection threshold (uses default if None)
            session_id: Optional session ID (auto-generated if None)
            api_url: API endpoint URL
        """
        self.license_key = license_key
        self.method = method
        self.threshold = threshold
        self.session_id = session_id or str(uuid.uuid4())
        self.api_url = api_url
        
        # Local counters
        self._total_points = 0
        self._total_anomalies = 0
    
    def _call_api(self, action: str, **kwargs) -> dict:
        """Make API call."""
        payload = {
            "action": action,
            "license_key": self.license_key,
            "session_id": self.session_id,
            "method": self.method,
            **kwargs
        }
        
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API call failed: {e}")
    
    def detect(self, value: float) -> DetectionResult:
        """
        Detect anomaly for single value.
        
        Args:
            value: The data value to analyze
            
        Returns:
            DetectionResult with detection details
        """
        result = self._call_api("detect", value=value)
        
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Detection failed"))
        
        self._total_points += 1
        if result.get("is_anomaly"):
            self._total_anomalies += 1
        
        return DetectionResult(
            value=value,
            at=result.get("at", 0),
            bt=result.get("bt", 0),
            delta_dagger=result.get("delta_dagger", 0),
            trust_ratio=result.get("trust_ratio", 0),
            is_anomaly=result.get("is_anomaly", False),
            threshold=result.get("threshold", 0),
            is_transition=result.get("is_transition", False),
            warmup=result.get("warmup", False)
        )
    
    def detect_batch(self, values: List[float]) -> BatchResult:
        """
        Detect anomalies in batch.
        
        Args:
            values: List of data values
            
        Returns:
            BatchResult with all results
        """
        result = self._call_api("detect_batch", values=values)
        
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Batch detection failed"))
        
        self._total_points += len(values)
        self._total_anomalies += result.get("total_anomalies", 0)
        
        results = [
            DetectionResult(
                value=r.get("value", 0),
                at=r.get("at", 0),
                bt=r.get("bt", 0),
                delta_dagger=r.get("delta_dagger", 0),
                trust_ratio=r.get("trust_ratio", 0),
                is_anomaly=r.get("is_anomaly", False),
                is_transition=r.get("is_transition", False),
            )
            for r in result.get("results", [])
        ]
        
        return BatchResult(
            total_points=result.get("total_points", 0),
            total_anomalies=result.get("total_anomalies", 0),
            anomaly_indices=result.get("anomaly_indices", []),
            results=results,
            session_id=self.session_id
        )
    
    def get_session(self) -> dict:
        """Get current session state."""
        result = self._call_api("get_session")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Failed to get session"))
        return result.get("session", {})
    
    def reset(self):
        """Reset session state."""
        result = self._call_api("reset_session")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Failed to reset session"))
        self._total_points = 0
        self._total_anomalies = 0
    
    def end_session(self):
        """End and delete session."""
        result = self._call_api("end_session")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Failed to end session"))
    
    def get_usage(self) -> dict:
        """Get license usage info."""
        result = self._call_api("get_usage")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Failed to get usage"))
        return {
            "tier": result.get("tier"),
            "usage": result.get("usage"),
            "limit": result.get("limit"),
            "remaining": result.get("remaining")
        }
    
    @property
    def stats(self) -> dict:
        """Get local detection statistics."""
        return {
            "total_points": self._total_points,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate": self._total_anomalies / max(self._total_points, 1)
        }


# =============================================================================
# LOCAL DETECTOR (No API, uses soft_algebra_core directly)
# =============================================================================

class MobiuADLocal:
    """
    Local Mobiu-AD detector (no API calls).
    
    Uses the Soft Algebra core directly for offline detection.
    Useful for testing or when API access is not needed.
    
    NEW: Auto mode - automatically selects best method!
    """
    
    def __init__(
        self,
        method: str = "auto",  # Changed default to auto!
        threshold: float = 0.05,
        window_size: int = 50,
        warmup: int = 30,
        gamma: float = 0.9
    ):
        """
        Initialize local detector.
        
        Args:
            method: Detection method:
                - "auto" (default): Automatically adapts to data
                - "deep": Super-Equation Δ† for complex patterns
                - "transition": For regime changes
                - "standard": Simple Trust Ratio based
            threshold: Base detection threshold
            window_size: Data history window
            warmup: Warmup period before detection starts
            gamma: Soft momentum decay factor
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.warmup_period = warmup
        self.gamma = gamma
        
        # Auto mode state
        self._volatility_history: List[float] = []
        
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self._soft_state = (0.0, 0.0)  # (soft, real)
        self._history: List[float] = []
        self._delta_history: List[float] = []
        self._volatility_history: List[float] = []  # For auto mode
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._total_points = 0
        self._total_anomalies = 0
    
    def _update_stats(self, value: float):
        """Welford's online algorithm."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2
    
    @property
    def _std(self) -> float:
        if self._n < 2:
            return 1.0
        return np.sqrt(self._M2 / (self._n - 1)) + 1e-9
    
    def _compute_at(self) -> float:
        """Curvature-based potential (same as Mobiu-Q)."""
        if len(self._history) < 3:
            return 0.0
        X_t, X_t1, X_t2 = self._history[-1], self._history[-2], self._history[-3]
        curv = abs(X_t - 2*X_t1 + X_t2)
        mean_X = abs(np.mean(self._history[-3:]))
        if mean_X < 1e-12:
            return 0.0
        return curv / (curv + mean_X)
    
    def _compute_bt(self, value: float) -> float:
        """Deviation-based realization."""
        z_score = abs(value - self._mean) / self._std
        return max(0.0, min(1.0, 1 - np.exp(-z_score / 3.0)))
    
    def _soft_multiply(self, s1: tuple, s2: tuple) -> tuple:
        """Nilpotent multiplication: (a,b) × (c,d) = (ad+bc, bd)."""
        a, b = s1
        c, d = s2
        return (a*d + b*c, b*d)
    
    def _soft_add(self, s1: tuple, s2: tuple) -> tuple:
        return (s1[0] + s2[0], s1[1] + s2[1])
    
    def _compute_super_equation(self, state: tuple) -> float:
        """Super-Equation Δ† (same as Mobiu-Q)."""
        a, b = state
        if abs(a) < 1e-9 or abs(b) < 1e-9:
            return 0.0
        
        alpha, beta, C, epsilon = 1.35, 1.70, 3.00, 0.43
        
        S = b + 1j * a * epsilon
        du = np.sin(np.pi * S).imag
        tau = C * a * b
        g = np.exp(-(tau - 1)**2 / (2 * alpha**2))
        gamma_gate = 1 - np.exp(-beta * a)
        
        return abs(abs(du) * g * gamma_gate * np.sqrt(max(0, b * g)))
    
    def _compute_trust(self, state: tuple) -> float:
        """Trust ratio (same as Mobiu-Q)."""
        a, b = state
        denom = abs(a) + abs(b) + 1e-9
        return (abs(b) + 1e-9) / denom
    
    def detect(self, value: float) -> DetectionResult:
        """Detect anomaly for single value."""
        self._update_stats(value)
        self._history.append(value)
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]
        
        self._total_points += 1
        
        # Warmup
        if self._n <= self.warmup_period:
            return DetectionResult(
                value=value, at=0, bt=0, delta_dagger=0,
                trust_ratio=0, is_anomaly=False, warmup=True
            )
        
        # AUTO MODE: Select method based on data characteristics
        effective_method = self.method
        if self.method == "auto":
            effective_method = self._auto_select_method()
        
        # Compute signals
        at = min(10.0, max(0.0, self._compute_at()))
        bt = self._compute_bt(value)
        delta_sn = (at, bt)
        
        # Evolve soft state: S_new = (γ·S) × Δ + Δ
        decayed = (self._soft_state[0] * self.gamma, self._soft_state[1] * self.gamma)
        evolved = self._soft_multiply(decayed, delta_sn)
        self._soft_state = self._soft_add(evolved, delta_sn)
        
        # Compute metrics
        delta_dagger = self._compute_super_equation(self._soft_state)
        trust = self._compute_trust(self._soft_state)
        
        # Track for adaptive threshold
        self._delta_history.append(delta_dagger)
        if len(self._delta_history) > 100:
            self._delta_history = self._delta_history[-100:]
        
        # Adaptive threshold
        if len(self._delta_history) >= 20:
            adaptive_threshold = max(self.threshold, np.percentile(self._delta_history, 95))
        else:
            adaptive_threshold = self.threshold
        
        # Detection
        is_anomaly = delta_dagger > adaptive_threshold
        
        # Transition detection
        is_transition = False
        if effective_method == "transition":
            soft_mag = np.sqrt(self._soft_state[0]**2 + self._soft_state[1]**2)
            is_transition = soft_mag > 0.5 and trust < 0.3
            is_anomaly = is_anomaly or is_transition
        
        if is_anomaly:
            self._total_anomalies += 1
        
        return DetectionResult(
            value=value,
            at=at,
            bt=bt,
            delta_dagger=delta_dagger,
            trust_ratio=trust,
            is_anomaly=is_anomaly,
            threshold=adaptive_threshold,
            is_transition=is_transition,
            warmup=False
        )
    
    def _auto_select_method(self) -> str:
        """
        Automatically select best method based on data characteristics.
        
        - High volatility / changing patterns → transition
        - Stable patterns → deep
        """
        if len(self._history) < 10:
            return "deep"
        
        # Calculate recent volatility
        recent = self._history[-10:]
        volatility = np.std(recent) / (np.mean(np.abs(recent)) + 1e-9)
        
        # Track volatility history
        self._volatility_history.append(volatility)
        if len(self._volatility_history) > 20:
            self._volatility_history = self._volatility_history[-20:]
        
        # If volatility is changing (regime shift) → use transition
        if len(self._volatility_history) >= 5:
            vol_change = abs(self._volatility_history[-1] - np.mean(self._volatility_history[-5:-1]))
            if vol_change > 0.05:  # Volatility regime is changing
                return "transition"
        
        return "deep"
    
    def detect_batch(self, values: List[float]) -> BatchResult:
        """Detect anomalies in batch."""
        results = [self.detect(v) for v in values]
        anomaly_indices = [i for i, r in enumerate(results) if r.is_anomaly]
        
        return BatchResult(
            total_points=len(values),
            total_anomalies=len(anomaly_indices),
            anomaly_indices=anomaly_indices,
            results=results,
            session_id="local"
        )
    
    @property
    def stats(self) -> dict:
        return {
            "total_points": self._total_points,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate": self._total_anomalies / max(self._total_points, 1)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_anomalies(
    values: List[float],
    license_key: Optional[str] = None,
    method: str = "deep",
    threshold: float = 0.05
) -> BatchResult:
    """
    One-liner anomaly detection.
    
    If license_key is provided, uses cloud API.
    Otherwise, uses local detection.
    """
    if license_key:
        detector = MobiuAD(license_key=license_key, method=method, threshold=threshold)
        return detector.detect_batch(values)
    else:
        detector = MobiuADLocal(method=method, threshold=threshold)
        return detector.detect_batch(values)


def find_transitions(
    values: List[float],
    license_key: Optional[str] = None,
    min_gap: int = 10
) -> List[int]:
    """
    Find regime transition points.
    
    Returns indices where regime changes occur.
    """
    result = detect_anomalies(values, license_key=license_key, method="transition")
    
    # Filter for true transitions (using 0→peak→0 profile)
    transitions = []
    deltas = [r.delta_dagger for r in result.results]
    
    if len(deltas) < 35:
        return result.anomaly_indices
    
    threshold = np.percentile(deltas[30:], 95)
    
    for i in range(35, len(deltas) - 5):
        if deltas[i] > threshold:
            if deltas[i] >= max(deltas[i-5:i+5]):
                if not transitions or i - transitions[-1] >= min_gap:
                    transitions.append(i)
    
    return transitions


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MobiuAD",
    "MobiuADLocal",
    "DetectionResult",
    "BatchResult",
    "detect_anomalies",
    "find_transitions",
    "__version__",
]
