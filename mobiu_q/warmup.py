"""
Warmup Phase Manager (v3.6.0)
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

        Logic:
        - Compute trend (slope)
        - Check value ranges and patterns
        - Typical losses: small positive, decreasing
        - Typical rewards: can be large/negative, increasing
        """
        if len(self.metrics) < 5:
            return 'minimize'

        # Compute slope
        x = np.arange(len(self.metrics))
        slope = np.polyfit(x, self.metrics, 1)[0]

        # Mean value
        mean_val = np.mean(self.metrics)

        # Heuristics:
        # 1. If values are large (>100) and increasing -> likely reward
        # 2. If values are small positive (0-10) and decreasing -> likely loss
        # 3. If values are negative -> likely reward (many RL envs have negative rewards)

        # Check for typical RL patterns
        has_negative = any(m < 0 for m in self.metrics)
        is_large_scale = abs(mean_val) > 50

        if has_negative:
            # Negative values suggest RL (many envs have negative rewards initially)
            return 'maximize'
        elif is_large_scale and slope > 0:
            # Large positive values increasing -> reward
            return 'maximize'
        elif 0 < mean_val < 20 and slope < 0:
            # Small positive values decreasing -> loss
            return 'minimize'
        elif slope > 0:
            # Generally increasing -> maximize
            return 'maximize'
        else:
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
