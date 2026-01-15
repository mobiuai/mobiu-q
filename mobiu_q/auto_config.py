"""
Auto Configuration Engine (v3.6.18)
=========================
Automatically selects optimal configuration based on warmup analysis.

Decision Rules:
- method: Based on variance and curvature
- cloud vs local: Based on noise and complexity
- sync_interval: Based on network conditions and noise
"""

from dataclasses import dataclass
from typing import Optional

from .warmup import WarmupAnalysis


@dataclass
class MobiuConfig:
    """Configuration determined by auto-detection."""
    maximize: bool          # True for reward, False for loss
    method: str             # 'standard', 'deep', or 'adaptive'
    mode: str               # 'hardware' or 'simulation'
    use_cloud: bool         # Whether to use cloud Soft Algebra
    sync_interval: int      # How often to sync with cloud
    use_trust_region: bool  # Whether to use Trust Region
    use_super_equation: bool  # Whether to use Super-Equation
    base_lr: float          # Auto-selected learning rate


class AutoConfigEngine:
    """
    Automatically configures Mobiu based on warmup analysis.

    Decision Tree:
    1. Direction: Based on trend and value ranges
    2. Method: Based on variance and curvature
    3. Cloud: Based on noise and complexity
    4. Features: Based on landscape characteristics

    Usage:
        engine = AutoConfigEngine()
        config = engine.configure(warmup_analysis)
    """

    def __init__(self, has_license: bool = True, has_connection: bool = True):
        """
        Initialize config engine.

        Args:
            has_license: Whether user has valid license key
            has_connection: Whether cloud is reachable
        """
        self.has_license = has_license
        self.has_connection = has_connection

    def configure(
        self,
        analysis: WarmupAnalysis,
        forced_mode: Optional[str] = None,
        forced_method: Optional[str] = None,
        forced_maximize: Optional[bool] = None
    ) -> MobiuConfig:
        """
        Generate configuration from warmup analysis.

        Args:
            analysis: Results from WarmupPhaseManager.analyze()
            forced_mode: If set, override auto-detected mode ('hardware'/'simulation')
            forced_method: If set, override auto-detected method ('standard'/'deep'/'adaptive')
            forced_maximize: If set, override auto-detected direction (True/False)

        Returns:
            MobiuConfig with all settings determined
        """
        # 1. Direction (maximize/minimize) - use forced if provided
        if forced_maximize is not None:
            maximize = forced_maximize
        else:
            maximize = analysis.direction == 'maximize'

        # 2. Method selection (use forced if provided)
        method = forced_method if forced_method else self._select_method(analysis)

        # 3. Mode selection (use forced if provided)
        mode = forced_mode if forced_mode else self._select_mode(analysis)

        # 4. Cloud decision
        use_cloud = self._should_use_cloud(analysis)

        # 5. Sync interval
        sync_interval = self._compute_sync_interval(analysis)

        # 6. Feature activation
        use_trust_region = analysis.variance < 0.4
        use_super_equation = analysis.curvature > 0.2

        # 7. Auto-select learning rate based on method AND mode
        base_lr = self._select_lr(method, analysis, mode)

        return MobiuConfig(
            maximize=maximize,
            method=method,
            mode=mode,
            use_cloud=use_cloud,
            sync_interval=sync_interval,
            use_trust_region=use_trust_region,
            use_super_equation=use_super_equation,
            base_lr=base_lr
        )

    def _select_method(self, analysis: WarmupAnalysis) -> str:
        """
        Select optimal method based on landscape characteristics.

        Score-based Selection:
        - STANDARD (VQE): low variance, low curvature, smooth landscape
        - DEEP (QAOA): high curvature (rugged landscape), discrete optimization
        - ADAPTIVE (RL/Crypto): high variance OR high noise, policy gradients

        Real benchmark configurations:
        - VQE: method="standard", LR=0.01
        - QAOA: method="deep", LR=0.1
        - RL/Crypto: method="adaptive", LR=0.0003
        """
        # Score for STANDARD (VQE): low variance, low curvature, smooth
        standard_score = 0.0
        if analysis.variance < 0.25:
            standard_score += 0.4
        if analysis.curvature < 0.25:
            standard_score += 0.4
        if analysis.noise_level < 0.3:
            standard_score += 0.2

        # Score for DEEP (QAOA): high curvature (rugged), any variance
        deep_score = 0.0
        if analysis.curvature > 0.25:
            deep_score += 0.6  # curvature is key signal for QAOA
        if analysis.noise_level > 0.2:
            deep_score += 0.2
        if analysis.variance < 0.4:
            deep_score += 0.2  # QAOA typically has moderate variance

        # Score for ADAPTIVE (RL/Crypto): high variance OR high noise
        adaptive_score = 0.0
        if analysis.variance > 0.35:  # lower threshold than before
            adaptive_score += 0.5
        if analysis.noise_level > 0.35:
            adaptive_score += 0.3
        if abs(analysis.trend) > 0.1:  # unstable trajectory
            adaptive_score += 0.2

        # Select method with highest score
        scores = {
            'standard': standard_score,
            'deep': deep_score,
            'adaptive': adaptive_score
        }
        return max(scores, key=scores.get)

    def _select_mode(self, analysis: WarmupAnalysis) -> str:
        """
        Select mode (hardware vs simulation) based on noise characteristics.

        Hardware mode is better for:
        - High noise environments (quantum hardware, FakeFez)
        - Shot noise from real measurements
        - Systematic bias in gradients

        Simulation mode is better for:
        - Clean gradients (autodiff, backprop)
        - RL/Trading (high variance in REWARDS, not gradients)
        - Standard deep learning

        Key insight:
        - Quantum: moderate variance (0.1-0.5) + noise in measurements
        - RL/Trading: VERY high variance (>0.5) from reward stochasticity
        - Deep Learning: low variance, clean gradients
        """
        # RL/Trading pattern: very high variance suggests reward stochasticity
        # not gradient noise - use simulation mode
        if analysis.variance > 0.5:
            return 'simulation'

        # High noise with moderate variance suggests quantum hardware
        if analysis.noise_level > 0.15 and analysis.variance < 0.5:
            return 'hardware'

        # Moderate noise (typical of quantum shot noise)
        if analysis.noise_level > 0.05:
            # Check for quantum patterns:
            # - moderate variance (not RL-like)
            # - or high curvature (rugged quantum landscape)
            if analysis.variance > 0.1 and analysis.variance < 0.5:
                return 'hardware'
            if analysis.curvature > 0.15 and analysis.variance < 0.4:
                return 'hardware'

        # Very clean gradients (autodiff, simulation)
        if analysis.noise_level < 0.02:
            return 'simulation'

        # Low-moderate noise with low variance - likely quantum
        if analysis.noise_level > 0.03 and analysis.variance < 0.4:
            return 'hardware'

        return 'simulation'

    def _should_use_cloud(self, analysis: WarmupAnalysis) -> bool:
        """
        Determine if cloud Soft Algebra should be used.

        Cloud is ALWAYS used when license and connection are available.
        Soft Algebra is our core IP and runs only in the cloud.
        """
        # Cloud is required when available - no local Soft Algebra
        return self.has_license and self.has_connection

    def _compute_sync_interval(self, analysis: WarmupAnalysis) -> int:
        """
        Compute optimal sync interval.

        Lower interval = more frequent cloud calls = better adaptation
        Higher interval = less latency overhead

        MobiuOptimizer default is 10 - we match that for consistency.
        """
        # Match MobiuOptimizer default of 10
        # This ensures Soft Algebra runs frequently enough to adapt
        return 10

    def _select_lr(self, method: str, analysis: WarmupAnalysis, mode: str) -> float:
        """
        Auto-select learning rate based on detected method and mode.

        Base LRs from real benchmarks (matching MobiuQCore defaults):
        | Method    | Mode       | Default LR |
        |-----------|------------|------------|
        | standard  | simulation | 0.01       |
        | standard  | hardware   | 0.02       |
        | deep      | simulation | 0.1        |
        | deep      | hardware   | 0.1        |
        | adaptive  | any        | 0.0003     |

        Adjustments:
        - High noise reduces LR (avoid chasing noise)
        - High variance may require lower LR for stability
        """
        # Base LR per method + mode (matching MobiuQCore.get_default_lr)
        if method == 'adaptive':
            base_lr = 0.0003
        elif method == 'deep':
            base_lr = 0.1
        elif mode == 'hardware':
            base_lr = 0.02  # Hardware needs higher LR
        else:  # standard + simulation
            base_lr = 0.01

        # Adjust based on noise level (high noise → lower LR)
        noise_scale = 1.0 - 0.3 * min(1.0, analysis.noise_level)

        # Adjust based on variance for stability (method-specific)
        if method == 'adaptive':
            # For RL: very high variance might need even lower LR
            variance_scale = 1.0 - 0.2 * max(0, analysis.variance - 0.3)
        else:
            variance_scale = 1.0

        return base_lr * noise_scale * variance_scale


__all__ = ["AutoConfigEngine", "MobiuConfig"]
