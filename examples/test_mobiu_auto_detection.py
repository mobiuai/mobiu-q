#!/usr/bin/env python3
"""
================================================================================
TEST: Mobiu Auto-Detection for Different Problem Types
================================================================================

This test verifies that Mobiu correctly auto-detects the optimal configuration
for different problem types:

1. VQE-like (Quantum): low variance, low curvature → standard, LR ~0.01
2. QAOA-like (Quantum): high curvature, moderate variance → deep, LR ~0.1
3. RL-like (LunarLander): high variance, increasing rewards → adaptive, LR ~0.0003
4. Crypto-like (Trading): high variance, volatile → adaptive, LR ~0.0003
================================================================================
"""

import numpy as np
from mobiu_q import Mobiu
from mobiu_q.warmup import WarmupPhaseManager
from mobiu_q.auto_config import AutoConfigEngine, MobiuConfig

np.random.seed(42)


def generate_vqe_scenario(n_steps=35):
    """
    VQE-like: smooth energy landscape, decreasing loss.

    Characteristics:
    - Low variance (<0.25)
    - Low curvature (<0.25)
    - Low noise (<0.3)
    - Decreasing trend (minimize)
    """
    np.random.seed(123)

    # Start from high energy, smoothly decrease
    start_energy = 2.0
    target_energy = 0.5

    energies = []
    for i in range(n_steps):
        # Smooth exponential decay with very small noise
        progress = i / n_steps
        energy = start_energy * (1 - progress) + target_energy * progress
        # Very small Gaussian noise (~2%) to keep variance low
        energy += np.random.randn() * 0.02 * energy
        energies.append(max(0.1, energy))

    return energies


def generate_qaoa_scenario(n_steps=35):
    """
    QAOA-like: rugged landscape, discrete optimization.

    Characteristics:
    - Moderate variance (<0.4)
    - HIGH curvature (>0.25)
    - Moderate noise (>0.2)
    - Decreasing trend (minimize)
    """
    np.random.seed(456)

    # Rugged landscape with local minima
    energies = []
    base = 2.0

    for i in range(n_steps):
        progress = i / n_steps
        # Base decreasing trend
        trend = base * (1 - 0.3 * progress)
        # STRONG rugged oscillations (creates high curvature)
        oscillation = 0.8 * np.sin(i * 1.5) + 0.4 * np.cos(i * 2.3)
        # Moderate shot noise from quantum measurement
        noise = np.random.randn() * 0.15

        energy = trend + oscillation + noise
        energies.append(max(0.1, energy))

    return energies


def generate_rl_scenario(n_steps=35):
    """
    RL-like: policy gradient with high variance episode returns.

    Characteristics:
    - HIGH variance (>0.35)
    - Any curvature
    - High noise (>0.35)
    - Increasing trend (maximize reward)
    """
    np.random.seed(789)

    # RL rewards: start negative, increase with high variance
    rewards = []

    for i in range(n_steps):
        # Base increasing trend
        base_reward = -150 + i * 8
        # High episode-to-episode variance
        variance = np.random.randn() * 60

        reward = base_reward + variance
        rewards.append(reward)

    return rewards


def generate_crypto_scenario(n_steps=35):
    """
    Crypto-like: trading with regime switching and volatility.

    Characteristics:
    - HIGH variance (>0.35)
    - Any curvature
    - High noise (>0.35)
    - Volatile/mixed trend (maximize profit)
    """
    np.random.seed(101112)

    # Trading returns with regime switching
    returns = []
    regime = 1  # 1 = bull, -1 = bear

    for i in range(n_steps):
        # Occasional regime switch
        if np.random.random() < 0.08:
            regime *= -1

        # Base return with regime
        base_return = regime * 5 + np.random.randn() * 15

        # Occasional large moves
        if np.random.random() < 0.1:
            base_return += regime * np.random.random() * 30

        returns.append(base_return)

    return returns


def analyze_scenario(name: str, metrics: list):
    """Analyze a scenario using WarmupPhaseManager and AutoConfigEngine."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    # Use WarmupPhaseManager to analyze
    warmup = WarmupPhaseManager(warmup_steps=30)

    for i, m in enumerate(metrics[:30]):
        warmup.record(m, grad_norm=0.1)

    analysis = warmup.analyze()

    print(f"\nWarmup Analysis:")
    print(f"  Direction: {analysis.direction}")
    print(f"  Variance: {analysis.variance:.4f}")
    print(f"  Curvature: {analysis.curvature:.4f}")
    print(f"  Noise Level: {analysis.noise_level:.4f}")
    print(f"  Mean Value: {analysis.mean_value:.4f}")
    print(f"  Trend: {analysis.trend:.6f}")

    # Use AutoConfigEngine to configure
    config_engine = AutoConfigEngine(has_license=True, has_connection=True)
    config = config_engine.configure(analysis)

    print(f"\nAuto-Configuration:")
    print(f"  Maximize: {config.maximize}")
    print(f"  Method: {config.method}")
    print(f"  Base LR: {config.base_lr:.6f}")
    print(f"  Use Cloud: {config.use_cloud}")
    print(f"  Use Trust Region: {config.use_trust_region}")
    print(f"  Use Super-Equation: {config.use_super_equation}")

    return analysis, config


def test_vqe_detection():
    """Test that VQE-like scenarios get standard method and ~0.01 LR."""
    metrics = generate_vqe_scenario()
    analysis, config = analyze_scenario("VQE (Quantum Eigenvalue)", metrics)

    # Verify
    assert config.maximize == False, f"VQE should minimize, got maximize={config.maximize}"
    assert config.method == 'standard', f"VQE should use 'standard', got '{config.method}'"
    assert 0.005 < config.base_lr < 0.02, f"VQE LR should be ~0.01, got {config.base_lr}"

    print(f"\n✅ VQE detection PASSED!")
    print(f"   Expected: method='standard', LR~0.01, maximize=False")
    print(f"   Got: method='{config.method}', LR={config.base_lr:.4f}, maximize={config.maximize}")


def test_qaoa_detection():
    """Test that QAOA-like scenarios get deep method and ~0.1 LR."""
    metrics = generate_qaoa_scenario()
    analysis, config = analyze_scenario("QAOA (Discrete Optimization)", metrics)

    # Verify
    assert config.maximize == False, f"QAOA should minimize, got maximize={config.maximize}"
    assert config.method == 'deep', f"QAOA should use 'deep', got '{config.method}'"
    assert 0.05 < config.base_lr < 0.15, f"QAOA LR should be ~0.1, got {config.base_lr}"

    print(f"\n✅ QAOA detection PASSED!")
    print(f"   Expected: method='deep', LR~0.1, maximize=False")
    print(f"   Got: method='{config.method}', LR={config.base_lr:.4f}, maximize={config.maximize}")


def test_rl_detection():
    """Test that RL-like scenarios get adaptive method and ~0.0003 LR."""
    metrics = generate_rl_scenario()
    analysis, config = analyze_scenario("RL (Policy Gradient)", metrics)

    # Verify
    assert config.maximize == True, f"RL should maximize, got maximize={config.maximize}"
    assert config.method == 'adaptive', f"RL should use 'adaptive', got '{config.method}'"
    assert 0.0001 < config.base_lr < 0.001, f"RL LR should be ~0.0003, got {config.base_lr}"

    print(f"\n✅ RL detection PASSED!")
    print(f"   Expected: method='adaptive', LR~0.0003, maximize=True")
    print(f"   Got: method='{config.method}', LR={config.base_lr:.6f}, maximize={config.maximize}")


def test_crypto_detection():
    """Test that Crypto-like scenarios get adaptive method and ~0.0003 LR."""
    metrics = generate_crypto_scenario()
    analysis, config = analyze_scenario("Crypto (Trading)", metrics)

    # Verify
    assert config.maximize == True, f"Crypto should maximize, got maximize={config.maximize}"
    assert config.method == 'adaptive', f"Crypto should use 'adaptive', got '{config.method}'"
    assert 0.0001 < config.base_lr < 0.001, f"Crypto LR should be ~0.0003, got {config.base_lr}"

    print(f"\n✅ Crypto detection PASSED!")
    print(f"   Expected: method='adaptive', LR~0.0003, maximize=True")
    print(f"   Got: method='{config.method}', LR={config.base_lr:.6f}, maximize={config.maximize}")


def main():
    print("="*60)
    print("🔍 MOBIU AUTO-DETECTION TEST SUITE")
    print("="*60)
    print("\nTesting auto-detection for 4 problem types:")
    print("  1. VQE (Quantum) → standard, LR ~0.01")
    print("  2. QAOA (Quantum) → deep, LR ~0.1")
    print("  3. RL (LunarLander) → adaptive, LR ~0.0003")
    print("  4. Crypto (Trading) → adaptive, LR ~0.0003")

    # Run tests
    try:
        test_vqe_detection()
    except AssertionError as e:
        print(f"\n❌ VQE detection FAILED: {e}")

    try:
        test_qaoa_detection()
    except AssertionError as e:
        print(f"\n❌ QAOA detection FAILED: {e}")

    try:
        test_rl_detection()
    except AssertionError as e:
        print(f"\n❌ RL detection FAILED: {e}")

    try:
        test_crypto_detection()
    except AssertionError as e:
        print(f"\n❌ Crypto detection FAILED: {e}")

    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)


if __name__ == "__main__":
    main()
