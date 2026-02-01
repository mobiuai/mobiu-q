"""
MobiuSignal Example - Trading Signal Generator
==============================================

This example demonstrates how to use MobiuSignal for:
1. Computing signals from price data
2. Streaming mode for live data
3. Backtesting signal quality
4. Integration with MobiuOptimizer for parameter optimization

Requirements:
    pip install mobiu-q scipy pandas  # pandas optional, for data loading
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Basic Signal Computation
# ═══════════════════════════════════════════════════════════════════════════════

def example_basic():
    """Basic signal computation from price array."""
    from mobiu_q.signal import MobiuSignal, compute_signal
    
    # Sample price data (21 days minimum for lookback=20)
    prices = np.array([
        100, 101, 99, 102, 103, 101, 104, 105, 103, 106,
        107, 105, 108, 109, 107, 110, 112, 110, 113, 115, 118
    ])
    
    # Method 1: Using class
    signal = MobiuSignal(lookback=20)
    result = signal.compute(prices)
    
    print("=== Basic Signal Computation ===")
    print(f"Potential (volatility): {result.potential:.3f}")
    print(f"Realized (price change): {result.realized:.3f}%")
    print(f"Signal Magnitude: {result.magnitude:.3f}")
    print(f"Direction: {'+1 (Bullish)' if result.is_bullish else '-1 (Bearish)' if result.is_bearish else '0 (Neutral)'}")
    print(f"Quartile: Q{result.quartile}")
    print(f"Strong Signal: {result.is_strong}")
    
    # Method 2: Using convenience function
    result2 = compute_signal(prices, lookback=20)
    print(f"\nConvenience function result: {result2.magnitude:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Streaming Mode
# ═══════════════════════════════════════════════════════════════════════════════

def example_streaming():
    """Streaming mode for live price updates."""
    from mobiu_q.signal import MobiuSignal
    
    signal = MobiuSignal(lookback=20)
    
    # Simulate streaming prices
    initial_prices = np.random.randn(25).cumsum() + 100
    
    print("\n=== Streaming Mode ===")
    
    for i, price in enumerate(initial_prices):
        result = signal.update(price)
        
        if result is not None:
            status = "🔥 STRONG" if result.is_strong else ""
            direction = "📈" if result.is_bullish else "📉" if result.is_bearish else "➡️"
            print(f"Step {i+1}: Price={price:.2f}, Mag={result.magnitude:.2f}, {direction} {status}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Backtesting
# ═══════════════════════════════════════════════════════════════════════════════

def example_backtest():
    """Backtest signal quality on historical data."""
    from mobiu_q.signal import MobiuSignal, backtest_signal
    
    # Generate synthetic price data (or load real data)
    np.random.seed(42)
    n_days = 500
    returns = np.random.randn(n_days) * 0.02  # 2% daily vol
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Backtest
    signal = MobiuSignal(lookback=20)
    result = signal.backtest(prices, future_window=5)
    
    print("\n=== Backtest Results ===")
    print(f"Total Signals: {result.total_signals}")
    print(f"Strong Signals (Q4): {result.strong_signals}")
    print(f"Avg Magnitude: {result.avg_magnitude:.3f}")
    print(f"Correlation with future moves: {result.correlation:.3f} (p={result.correlation_pvalue:.4f})")
    print(f"Q4/Q1 Ratio: {result.q4_q1_ratio:.2f}x")
    print(f"Precision Lift: {result.precision_lift:.2f}x vs random")
    
    # Convenience function
    result2 = backtest_signal(prices, lookback=20, future_window=5)
    print(f"\nConvenience function correlation: {result2.correlation:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Series Computation
# ═══════════════════════════════════════════════════════════════════════════════

def example_series():
    """Compute signals for entire price series."""
    from mobiu_q.signal import MobiuSignal
    
    # Generate data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    
    signal = MobiuSignal(lookback=20)
    results = signal.compute_series(prices)
    
    print("\n=== Series Analysis ===")
    print(f"Total signals: {len(results)}")
    
    # Find strong signals
    strong_signals = [r for r in results if r.is_strong]
    bullish_strong = [r for r in strong_signals if r.is_bullish]
    bearish_strong = [r for r in strong_signals if r.is_bearish]
    
    print(f"Strong signals: {len(strong_signals)} ({len(strong_signals)/len(results)*100:.1f}%)")
    print(f"  - Bullish: {len(bullish_strong)}")
    print(f"  - Bearish: {len(bearish_strong)}")
    
    # Magnitude distribution
    magnitudes = [r.magnitude for r in results]
    print(f"\nMagnitude stats:")
    print(f"  Min: {min(magnitudes):.3f}")
    print(f"  Max: {max(magnitudes):.3f}")
    print(f"  Mean: {np.mean(magnitudes):.3f}")
    print(f"  Std: {np.std(magnitudes):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Real BTC Data (if available)
# ═══════════════════════════════════════════════════════════════════════════════

def example_btc_data():
    """Example with real BTC data (requires pandas and data file)."""
    try:
        import pandas as pd
        from mobiu_q.signal import MobiuSignal
        
        # Try to load BTC data (user needs to provide this)
        # Example: df = pd.read_csv('btc_daily.csv')
        # prices = df['close'].values
        
        print("\n=== BTC Data Example ===")
        print("To use with real data:")
        print("  import pandas as pd")
        print("  df = pd.read_csv('btc_daily.csv')")
        print("  prices = df['close'].values")
        print("  signal = MobiuSignal(lookback=20)")
        print("  result = signal.backtest(prices, future_window=5)")
        print("\nExpected results (from 3,080 days validation):")
        print("  - Correlation: ~0.222")
        print("  - Q4/Q1 Ratio: ~1.83x")
        print("  - Precision Lift: ~1.18x")
        
    except ImportError:
        print("\nInstall pandas for real data examples: pip install pandas")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Integration with MobiuOptimizer
# ═══════════════════════════════════════════════════════════════════════════════

def example_with_optimizer():
    """Use MobiuOptimizer to optimize signal parameters."""
    from mobiu_q.signal import MobiuSignalOptimized
    
    print("\n=== Parameter Optimization ===")
    print("MobiuSignalOptimized.fit() can find optimal parameters:")
    print("  - lookback: volatility window (10-50)")
    print("  - vol_scale: scaling factor (50-200)")
    print("\nUsage:")
    print("  signal = MobiuSignalOptimized.fit(train_prices, license_key='your-key')")
    print("  result = signal.compute(test_prices)")
    
    # Note: Actual optimization requires license key and training data


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MobiuSignal Examples")
    print("=" * 60)
    
    example_basic()
    example_streaming()
    example_backtest()
    example_series()
    example_btc_data()
    example_with_optimizer()
    
    print("\n" + "=" * 60)
    print("Done! For more info: from mobiu_q.signal import MobiuSignal; help(MobiuSignal)")
    print("=" * 60)
