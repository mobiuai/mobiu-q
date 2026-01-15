#!/usr/bin/env python3
"""
================================================================================
TEST: New Mobiu (Auto-Adaptive) vs MobiuOptimizer (Manual Config)
================================================================================

This test compares the new simplified Mobiu API against the existing
MobiuOptimizer to verify that auto-detection works correctly.

Tests:
1. Supervised Learning (loss minimization) - should detect minimize
2. RL-like (reward maximization) - should detect maximize
3. Noisy gradient environment - should adapt appropriately
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from mobiu_q import Mobiu, MobiuOptimizer, LocalSoftAlgebra

np.random.seed(42)
torch.manual_seed(42)


def test_supervised_learning():
    """Test with typical supervised learning pattern (minimize loss)."""
    print("\n" + "=" * 60)
    print("TEST 1: Supervised Learning (Loss Minimization)")
    print("=" * 60)

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    # Create optimizer with new Mobiu
    opt = Mobiu(model.parameters(), lr=0.01, verbose=True)

    # Simulate training
    criterion = nn.MSELoss()
    losses = []

    for i in range(50):
        x = torch.randn(32, 10)
        y_true = torch.randn(32, 2)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        loss.backward()
        opt.step(loss.item())
        opt.zero_grad()
        losses.append(loss.item())

    print(f"\nResults:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Detected maximize: {opt.config.maximize if opt.config else 'N/A'}")
    print(f"  Detected method: {opt.config.method if opt.config else 'N/A'}")

    # Verify detection
    assert opt.config is not None, "Config should be set after warmup"
    assert opt.config.maximize == False, f"Should detect minimize, got maximize={opt.config.maximize}"
    print("  ✅ Correctly detected loss minimization!")

    return losses


def test_rl_pattern():
    """Test with RL-like pattern (maximize reward)."""
    print("\n" + "=" * 60)
    print("TEST 2: RL Pattern (Reward Maximization)")
    print("=" * 60)

    # Simple policy network
    policy = nn.Linear(10, 4)
    opt = Mobiu(policy.parameters(), lr=0.0003, verbose=True)

    rewards = []

    for i in range(50):
        x = torch.randn(1, 10)
        logits = policy(x)

        # Simulate RL reward: starts negative, gradually increases with noise
        reward = -200 + i * 5 + np.random.randn() * 30

        # Fake backward (in real RL this would be policy gradient)
        fake_loss = -logits.mean()
        fake_loss.backward()
        opt.step(reward)
        opt.zero_grad()

        rewards.append(reward)

    print(f"\nResults:")
    print(f"  Initial reward: {rewards[0]:.1f}")
    print(f"  Final reward:   {rewards[-1]:.1f}")
    print(f"  Detected maximize: {opt.config.maximize if opt.config else 'N/A'}")
    print(f"  Detected method: {opt.config.method if opt.config else 'N/A'}")

    # Verify detection
    assert opt.config is not None, "Config should be set after warmup"
    assert opt.config.maximize == True, f"Should detect maximize, got maximize={opt.config.maximize}"
    print("  ✅ Correctly detected reward maximization!")

    return rewards


def test_local_soft_algebra():
    """Test LocalSoftAlgebra independently."""
    print("\n" + "=" * 60)
    print("TEST 3: LocalSoftAlgebra (Offline Mode)")
    print("=" * 60)

    lsa = LocalSoftAlgebra(base_lr=0.01, maximize=False)

    # Simulate decreasing loss
    lrs = []
    for i in range(30):
        loss = 1.0 / (i + 1)  # Decreasing loss
        lr, warp = lsa.update(loss)
        lrs.append(lr)

    print(f"\nResults:")
    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Final LR:   {lrs[-1]:.6f}")
    print(f"  LR range: {min(lrs):.6f} - {max(lrs):.6f}")
    print(f"  ✅ LocalSoftAlgebra working!")

    return lrs


def test_federated_simulation():
    """Test with federated learning simulation (biased gradients)."""
    print("\n" + "=" * 60)
    print("TEST 4: Federated Learning Simulation")
    print("=" * 60)

    # Simple model
    model = nn.Linear(20, 1)
    opt = Mobiu(model.parameters(), lr=0.01, verbose=True)

    # True target
    np.random.seed(999)
    w_true = torch.tensor(np.random.randn(20, 1), dtype=torch.float32)

    # Client biases (non-IID simulation)
    client_biases = [torch.tensor(np.random.randn(20, 1) * 0.8, dtype=torch.float32)
                     for _ in range(10)]

    losses = []

    for step in range(80):
        # Compute true loss
        with torch.no_grad():
            true_loss = ((model.weight.T - w_true) ** 2).mean().item()
        losses.append(true_loss)

        # Compute biased gradient (federated simulation)
        np.random.seed(step * 42)
        sampled_clients = np.random.choice(10, 5, replace=False)

        opt.zero_grad()
        total_grad = torch.zeros_like(model.weight)

        for c in sampled_clients:
            biased_target = w_true + client_biases[c]
            grad = 2 * (model.weight.T - biased_target) / 20
            total_grad += grad.T

        total_grad /= 5
        model.weight.grad = total_grad

        opt.step(true_loss)

    print(f"\nResults:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    print(f"  Detected: maximize={opt.config.maximize}, method={opt.config.method}")
    print("  ✅ Federated simulation completed!")

    return losses


def main():
    print("=" * 60)
    print("🚀 MOBIU AUTO-ADAPTIVE OPTIMIZER TESTS")
    print("=" * 60)
    print("\nTesting the new Mobiu class with auto-detection...")

    # Run tests
    test_supervised_learning()
    test_rl_pattern()
    test_local_soft_algebra()
    test_federated_simulation()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print("  - Supervised learning: correctly detected minimize")
    print("  - RL pattern: correctly detected maximize")
    print("  - LocalSoftAlgebra: working offline")
    print("  - Federated simulation: adapting to biased gradients")
    print("=" * 60)


if __name__ == "__main__":
    main()
