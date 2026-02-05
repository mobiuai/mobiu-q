"""
MobiuSignal - Trading Signal Generator using Soft Algebra
==========================================================

🚧 IN DEVELOPMENT - API may change

Applies the same potential/realized framework from Mobiu-Q optimization
to market data for generating trading signals.

Quick Start:
    from mobiu_q.signal import MobiuSignal, compute_signal
    
    # Simple usage
    signal = MobiuSignal(lookback=20)
    result = signal.compute(prices)
    
    if result.is_strong:
        print(f"Strong {result.direction} signal: {result.magnitude:.2f}")
    
    # Backtest
    backtest = signal.backtest(historical_prices, future_window=5)
    print(f"Correlation: {backtest.correlation:.3f}")
    print(f"Q4/Q1 Ratio: {backtest.q4_q1_ratio:.2f}x")

Mathematical Framework:
    - Potential (aₜ): Normalized volatility σ/μ × scale
    - Realized (bₜ): Price change (Pₜ - Pₜ₋₁)/Pₜ₋₁
    - Magnitude: √(aₜ² + bₜ²) - Euclidean interaction strength

Validated Results (3,080 days BTC/USDT):
    - Spearman correlation: +0.222 (p<0.0001)
    - Q4/Q1 ratio: 1.83x larger moves
    - Precision lift: 1.18x vs random
"""

from .signal import (
    # Main class
    MobiuSignal,
    MobiuSignalOptimized,
    
    # Data classes
    SignalResult,
    BacktestResult,
    
    # Convenience functions
    compute_signal,
    backtest_signal,
)

__all__ = [
    # Main classes
    "MobiuSignal",
    "MobiuSignalOptimized",
    
    # Data classes
    "SignalResult", 
    "BacktestResult",
    
    # Functions
    "compute_signal",
    "backtest_signal",
]

__version__ = "4.3.1"
