"""
Warmup Phase Manager (v3.6.18)
====================
Collects initial data during warmup to auto-detect optimization characteristics.

The warmup phase runs for the first N steps (default: 30) and analyzes:
- Direction: maximize (reward) vs minimize (loss)
- Variance: high variance suggests RL/trading
- Curvature: high curvature suggests rugged landscape (QAOA)
- Noise level: helps decide cloud vs local mode
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class WarmupAnalysis:
    """Results from warmup phase analysis."""
    direction: str          # 'maximize' or 'minimize'
    variance: float         # normalized variance [0, 1]
    curvature: float        # landscape curvature [0, 1]
    noise_level: float      # measurement noise [0, 1]
    mean_value: float       # average metric value
    trend: float            # slope of metric trajectory


class WarmupPhaseManager:
    """
    Manages the warmup phase for auto-detection.

    Collects metrics and gradient norms during warmup,
    then analyzes them to configure the optimizer.

    Usage:
        warmup = WarmupPhaseManager(warmup_steps=30)

        for step in range(1000):
            loss = compute_loss()
            grad_norm = compute_grad_norm()

            if not warmup.is_complete:
                if warmup.record(loss, grad_norm):
                    # Warmup complete - configure optimizer
                    analysis = warmup.analyze()
                    print(f"Direction: {analysis.direction}")
    """

    def __init__(self, warmup_steps: int = 30):
        """
        Initialize warmup manager.

        Args:
            warmup_steps: Number of steps to collect before analysis
        """
        self.warmup_steps = warmup_steps
        self.metrics: List[float] = []
        self.grad_norms: List[float] = []
        self.is_complete = False
        self._analysis: Optional[WarmupAnalysis] = None

    def record(self, metric: float, grad_norm: float = 0.0) -> bool:
        """
        Record a metric and gradient norm.

        Args:
            metric: Loss or reward value
            grad_norm: Gradient norm (optional)

        Returns:
            True when warmup is complete
        """
        self.metrics.append(metric)
        self.grad_norms.append(grad_norm)

        if len(self.metrics) >= self.warmup_steps:
            self.is_complete = True
            return True
        return False

    def analyze(self) -> WarmupAnalysis:
        """
        Analyze collected warmup data.

        Returns:
            WarmupAnalysis with detected characteristics
        """
        if self._analysis is not None:
            return self._analysis

        if len(self.metrics) < 5:
            # Not enough data - return defaults
            return WarmupAnalysis(
                direction='minimize',
                variance=0.0,
                curvature=0.0,
                noise_level=0.0,
                mean_value=0.0,
                trend=0.0
            )

        # Compute all analysis components
        direction = self._detect_direction()
        variance = self._compute_variance()
        curvature = self._compute_curvature()
        noise_level = self._estimate_noise()
        mean_value = float(np.mean(self.metrics))
        trend = self._compute_trend()

        self._analysis = WarmupAnalysis(
            direction=direction,
            variance=variance,
            curvature=curvature,
            noise_level=noise_level,
            mean_value=mean_value,
            trend=trend
        )

        return self._analysis

    def _detect_direction(self) -> str:
        """
        Detect if user is maximizing or minimizing.

        Key insight: Look at the TRAJECTORY, not just the values.
        - If values are IMPROVING (going in a "better" direction) = user wants this
        - VQE: values go MORE NEGATIVE (slope < 0, mean < 0) = minimize
        - RL: values INCREASE (slope > 0) regardless of sign = maximize
        - Loss: values DECREASE (slope < 0, mean > 0) = minimize

        The key signal is:
        - slope > 0 + large absolute change = maximize (RL rewards improving)
        - slope < 0 + small values (0-10) = minimize (loss decreasing)
        - slope < 0 + negative values = minimize (VQE energy decreasing)
        """
        if len(self.metrics) < 5:
            return 'minimize'

        # Compute slope (trend)
        x = np.arange(len(self.metrics))
        slope = np.polyfit(x, self.metrics, 1)[0]

        # Statistics
        mean_val = np.mean(self.metrics)
        std_val = np.std(self.metrics)
        value_range = max(self.metrics) - min(self.metrics)

        # Coefficient of variation (normalized variance)
        cv = std_val / (abs(mean_val) + 1e-9)

        # Key signals
        is_increasing = slope > 0.1  # Significant positive trend
        has_high_variance = cv > 0.3 or std_val > 10  # Lower threshold!
        is_large_scale = abs(mean_val) > 20 or value_range > 50

        # ========================
        # MAXIMIZE detection (RL/Trading)
        # ========================

        # Pattern 1: Values increasing significantly = user wants higher values
        if is_increasing and (has_high_variance or is_large_scale):
            return 'maximize'

        # Pattern 2: Negative values becoming less negative (improving towards 0)
        # This is classic RL: starts at -200, improves towards 0 or positive
        if mean_val < 0 and slope > 0.5 and value_range > 20:
            return 'maximize'

        # Pattern 3: High variance + any improvement = probably RL
        if has_high_variance and slope > 0:
            return 'maximize'

        # ========================
        # MINIMIZE detection (VQE/Loss)
        # ========================

        # Pattern 1: VQE - negative values going MORE negative WITH LOW VARIANCE
        # High variance negative values = RL penalties, not VQE
        if mean_val < 0 and slope < -0.01 and cv < 0.5:
            return 'minimize'

        # Pattern 2: Loss function - small positive values decreasing
        if 0 < mean_val < 20 and slope < 0:
            return 'minimize'

        # Pattern 3: Very low variance = likely VQE/quantum (not RL)
        if cv < 0.1 and std_val < 1:
            return 'minimize'

        # ========================
        # Tie-breaker: Use variance as signal
        # ========================

        # Large scale values with any positive trend = rewards
        if is_large_scale and slope > 0:
            return 'maximize'

        # HIGH VARIANCE is a strong signal for RL/Trading
        # RL rewards have high variance, VQE/loss have low variance
        # Trading returns: small values with high variance near zero
        if cv > 0.8 and abs(mean_val) < 1:
            # Very high variance near zero = likely trading returns
            return 'maximize'

        # Default to minimize (supervised learning is more common)
        return 'minimize'

    def _compute_variance(self) -> float:
        """
        Compute normalized variance.

        High variance (>0.5) indicates:
        - RL (episode returns vary wildly)
        - Trading (market volatility)
        - Noisy gradients

        Low variance (<0.2) indicates:
        - Supervised learning
        - VQE/QAOA
        """
        if len(self.metrics) < 5:
            return 0.0

        std = np.std(self.metrics)
        mean = abs(np.mean(self.metrics)) + 1e-9

        # Normalized variance, clamped to [0, 1]
        return min(1.0, std / mean)

    def _compute_curvature(self) -> float:
        """
        Estimate loss landscape curvature.

        High curvature indicates:
        - Rugged landscape (QAOA, deep circuits)
        - Local minima present
        - Need for careful exploration
        """
        if len(self.metrics) < 5:
            return 0.0

        # Compute second differences
        second_diff = np.diff(self.metrics, n=2)
        curvature = np.mean(np.abs(second_diff))

        # Normalize by scale
        scale = abs(np.mean(self.metrics)) + 1e-9

        return min(1.0, curvature / scale)

    def _estimate_noise(self) -> float:
        """
        Estimate measurement noise level.

        High noise indicates need for:
        - More sophisticated Soft Algebra
        - Lower sync_interval
        - Trust region caution
        """
        if len(self.metrics) < 10:
            return 0.0

        # Fit linear trend
        x = np.arange(len(self.metrics))
        coeffs = np.polyfit(x, self.metrics, 1)
        trend = np.polyval(coeffs, x)

        # Residuals indicate noise
        residuals = np.array(self.metrics) - trend
        noise = np.std(residuals) / (abs(np.mean(self.metrics)) + 1e-9)

        return min(1.0, noise)

    def _compute_trend(self) -> float:
        """Compute linear trend (slope) of metrics."""
        if len(self.metrics) < 5:
            return 0.0

        x = np.arange(len(self.metrics))
        slope = np.polyfit(x, self.metrics, 1)[0]
        return float(slope)

    def reset(self):
        """Reset warmup for new run."""
        self.metrics.clear()
        self.grad_norms.clear()
        self.is_complete = False
        self._analysis = None


__all__ = ["WarmupPhaseManager", "WarmupAnalysis"]
