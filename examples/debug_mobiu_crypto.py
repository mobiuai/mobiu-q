#!/usr/bin/env python3
"""
Debug Mobiu behavior in Crypto trading scenario.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy

from mobiu_q import Mobiu, MobiuOptimizer

# Simple policy
class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

def simulate_rl_training(opt, use_mobiu_api=False, num_episodes=20):
    """Simulate RL training and track LR changes."""
    policy = SimplePolicy()

    lr_changes = []
    rewards_seen = []

    for ep in range(num_episodes):
        # Simulate episode
        state = torch.randn(1, 10)
        logits = policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Simulate reward (RL-like: large positive, increasing with variance)
        reward = 50 + ep * 5 + np.random.randn() * 30
        rewards_seen.append(reward)

        # Compute loss
        loss = -log_prob * reward

        opt.zero_grad()
        loss.backward()

        # Track LR before step
        if hasattr(opt, 'param_groups'):
            lr_before = opt.param_groups[0]['lr']
        elif hasattr(opt, '_base_optimizer'):
            lr_before = opt._base_optimizer.param_groups[0]['lr']
        else:
            lr_before = 0

        # Step
        if use_mobiu_api:
            opt.step(reward)
        else:
            opt.step()

        # Track LR after step
        if hasattr(opt, 'param_groups'):
            lr_after = opt.param_groups[0]['lr']
        elif hasattr(opt, '_base_optimizer'):
            lr_after = opt._base_optimizer.param_groups[0]['lr']
        else:
            lr_after = 0

        if lr_before != lr_after:
            lr_changes.append((ep, lr_before, lr_after))

    return rewards_seen, lr_changes

def main():
    print("=" * 70)
    print("DEBUG: Mobiu Crypto/RL Behavior")
    print("=" * 70)

    # Test 1: Plain Adam
    print("\n--- Test 1: Plain Adam ---")
    torch.manual_seed(42)
    np.random.seed(42)
    opt_adam = optim.Adam(SimplePolicy().parameters(), lr=0.0003)
    rewards, lr_changes = simulate_rl_training(opt_adam, use_mobiu_api=False)
    print(f"Rewards range: {min(rewards):.1f} to {max(rewards):.1f}")
    print(f"LR changes: {len(lr_changes)}")

    # Test 2: MobiuOptimizer (old API)
    print("\n--- Test 2: MobiuOptimizer (old API) ---")
    torch.manual_seed(42)
    np.random.seed(42)
    policy = SimplePolicy()
    opt_old = MobiuOptimizer(
        optim.Adam(policy.parameters(), lr=0.0003),
        license_key="YOUR_KEY",
        method="adaptive",
        maximize=True,
        use_soft_algebra=True,
        verbose=True
    )
    rewards, lr_changes = simulate_rl_training(opt_old, use_mobiu_api=True)
    print(f"Rewards range: {min(rewards):.1f} to {max(rewards):.1f}")
    print(f"LR changes: {len(lr_changes)}")
    for ep, before, after in lr_changes[:5]:
        print(f"  Episode {ep}: {before:.6f} -> {after:.6f}")
    opt_old.end()

    # Test 3: Mobiu (new API) - auto detection
    print("\n--- Test 3: Mobiu (new API) - auto detection ---")
    torch.manual_seed(42)
    np.random.seed(42)
    policy = SimplePolicy()
    opt_new = Mobiu(
        policy.parameters(),
        lr=0.0003,
        license_key="YOUR_KEY",
        verbose=True
    )
    rewards, lr_changes = simulate_rl_training(opt_new, use_mobiu_api=True)
    print(f"Rewards range: {min(rewards):.1f} to {max(rewards):.1f}")
    print(f"LR changes: {len(lr_changes)}")
    for ep, before, after in lr_changes[:5]:
        print(f"  Episode {ep}: {before:.6f} -> {after:.6f}")

    # Check config
    print(f"\nMobiu config after training:")
    if opt_new.config:
        print(f"  maximize: {opt_new.config.maximize}")
        print(f"  method: {opt_new.config.method}")
        print(f"  mode: {opt_new.config.mode}")
        print(f"  sync_interval: {opt_new.config.sync_interval}")
        print(f"  base_lr: {opt_new.base_lr}")
    else:
        print("  Config not set (still in warmup?)")

    print(f"\nInternal state:")
    print(f"  is_configured: {opt_new.is_configured}")
    print(f"  _step_count: {opt_new._step_count}")
    print(f"  _cloud_session_id: {opt_new._cloud_session_id}")
    print(f"  energy_history length: {len(opt_new.energy_history)}")
    print(f"  lr_history length: {len(opt_new.lr_history)}")

    opt_new.end()

    # Test 4: Mobiu with forced maximize
    print("\n--- Test 4: Mobiu with forced maximize=True ---")
    torch.manual_seed(42)
    np.random.seed(42)
    policy = SimplePolicy()

    # Collect warmup data first
    warmup_rewards = [50 + i * 5 + np.random.randn() * 30 for i in range(30)]

    opt_forced = Mobiu(
        policy.parameters(),
        lr=0.0003,
        license_key="YOUR_KEY",
        method="adaptive",  # Force method
        verbose=True
    )

    # Force warmup with RL-like data
    opt_forced.warmup_only(warmup_rewards)

    print(f"\nConfig after warmup_only:")
    if opt_forced.config:
        print(f"  maximize: {opt_forced.config.maximize}")
        print(f"  method: {opt_forced.config.method}")
        print(f"  base_lr: {opt_forced.base_lr}")

    opt_forced.end()

if __name__ == "__main__":
    main()
