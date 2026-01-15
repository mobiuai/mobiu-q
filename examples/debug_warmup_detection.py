#!/usr/bin/env python3
"""
Debug warmup detection for different scenarios.
"""

import numpy as np
from mobiu_q.warmup import WarmupPhaseManager, WarmupAnalysis
from mobiu_q.auto_config import AutoConfigEngine

def test_detection(name, metrics):
    """Test detection with given metrics."""
    print(f"\n--- {name} ---")
    print(f"Metrics: min={min(metrics):.2f}, max={max(metrics):.2f}, mean={np.mean(metrics):.2f}")
    print(f"Trend: {'increasing' if metrics[-1] > metrics[0] else 'decreasing'}")

    warmup = WarmupPhaseManager(warmup_steps=len(metrics))
    for m in metrics:
        warmup.record(m, 0.0)

    analysis = warmup.analyze()
    print(f"\nAnalysis:")
    print(f"  direction: {analysis.direction}")
    print(f"  variance: {analysis.variance:.3f}")
    print(f"  curvature: {analysis.curvature:.3f}")
    print(f"  noise_level: {analysis.noise_level:.3f}")
    print(f"  trend: {analysis.trend:.3f}")

    engine = AutoConfigEngine(has_license=True, has_connection=True)
    config = engine.configure(analysis)
    print(f"\nConfig:")
    print(f"  maximize: {config.maximize}")
    print(f"  method: {config.method}")
    print(f"  mode: {config.mode}")
    print(f"  base_lr: {config.base_lr}")
    print(f"  sync_interval: {config.sync_interval}")

    return analysis, config


def main():
    print("=" * 70)
    print("DEBUG: Warmup Detection for Different Scenarios")
    print("=" * 70)

    # Scenario 1: VQE - negative energies, decreasing (improving)
    np.random.seed(42)
    vqe_metrics = [-0.5 - i * 0.03 + np.random.randn() * 0.02 for i in range(30)]
    test_detection("VQE (negative, decreasing)", vqe_metrics)

    # Scenario 2: RL - large positive rewards, increasing
    np.random.seed(42)
    rl_metrics = [50 + i * 5 + np.random.randn() * 30 for i in range(30)]
    test_detection("RL (large positive, increasing)", rl_metrics)

    # Scenario 3: RL with negative start - starts negative, becomes positive
    np.random.seed(42)
    rl_neg_metrics = [-100 + i * 10 + np.random.randn() * 20 for i in range(30)]
    test_detection("RL (negative start, increasing)", rl_neg_metrics)

    # Scenario 4: Crypto trading - high variance, regime switching
    np.random.seed(42)
    crypto_metrics = []
    for i in range(30):
        if i < 10:
            m = np.random.randn() * 0.05  # Small returns
        elif i < 20:
            m = 0.1 + np.random.randn() * 0.1  # Bull market
        else:
            m = -0.05 + np.random.randn() * 0.08  # Bear market
        crypto_metrics.append(m)
    test_detection("Crypto (regime switching)", crypto_metrics)

    # Scenario 5: Standard loss - small positive, decreasing
    np.random.seed(42)
    loss_metrics = [2.0 - i * 0.05 + np.random.randn() * 0.1 for i in range(30)]
    test_detection("Standard Loss (small positive, decreasing)", loss_metrics)

    # Scenario 6: Actual crypto from benchmark (simulated episode returns)
    np.random.seed(42)
    actual_crypto = []
    for i in range(30):
        # Simulating REINFORCE episode returns
        episode_return = np.random.randn() * 0.5  # High variance, near zero mean
        actual_crypto.append(episode_return)
    test_detection("Actual Crypto Episode Returns", actual_crypto)

    # Scenario 7: LunarLander-like (starts very negative, improves)
    np.random.seed(42)
    lunar_metrics = [-200 + i * 5 + np.random.randn() * 50 for i in range(30)]
    test_detection("LunarLander (very negative, improving)", lunar_metrics)

if __name__ == "__main__":
    main()
