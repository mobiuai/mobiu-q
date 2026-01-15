#!/usr/bin/env python3
"""
TELEMETRY TEST - Compare Mobiu vs MobiuOptimizer step by step
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from mobiu_q import Mobiu, MobiuOptimizer

LICENSE_KEY = "e756ce65-186e-4747-aaaf-5a1fb1473b7e"

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def get_param_norm(model):
    """Get L2 norm of all parameters."""
    total = 0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5

def test_optimizer(name, model, optimizer, use_metric=False, num_steps=50):
    """Run optimizer and collect telemetry."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Telemetry
    losses = []
    lrs = []
    param_norms = []

    torch.manual_seed(42)
    np.random.seed(42)

    for step in range(num_steps):
        # Generate batch
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        # Forward
        pred = model(x)
        loss = ((pred - y) ** 2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Get LR before step
        if hasattr(optimizer, 'param_groups'):
            lr_before = optimizer.param_groups[0]['lr']
        elif hasattr(optimizer, '_base_optimizer'):
            lr_before = optimizer._base_optimizer.param_groups[0]['lr']
        else:
            lr_before = 0

        # Step
        if use_metric:
            optimizer.step(loss.item())
        else:
            optimizer.step()

        # Get LR after step
        if hasattr(optimizer, 'param_groups'):
            lr_after = optimizer.param_groups[0]['lr']
        elif hasattr(optimizer, '_base_optimizer'):
            lr_after = optimizer._base_optimizer.param_groups[0]['lr']
        else:
            lr_after = 0

        # Record
        losses.append(loss.item())
        lrs.append(lr_after)
        param_norms.append(get_param_norm(model))

        # Print every 10 steps
        if step % 10 == 0 or lr_before != lr_after:
            lr_change = "  LR CHANGED!" if lr_before != lr_after else ""
            print(f"Step {step:3d}: loss={loss.item():.4f}, lr={lr_after:.6f}, params={param_norms[-1]:.4f}{lr_change}")

    print(f"\nSummary:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Improvement:  {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    print(f"  LR changes:   {sum(1 for i in range(1, len(lrs)) if lrs[i] != lrs[i-1])}")
    print(f"  Final LR:     {lrs[-1]:.6f}")

    return losses, lrs, param_norms

def main():
    print("="*60)
    print("TELEMETRY TEST: Mobiu vs MobiuOptimizer vs Adam")
    print("="*60)

    # Create identical models
    torch.manual_seed(0)
    model_adam = SimpleModel()
    model_mobiu_opt = copy.deepcopy(model_adam)
    model_mobiu_new = copy.deepcopy(model_adam)

    print(f"\nInitial param norm: {get_param_norm(model_adam):.4f}")

    # Test 1: Plain Adam
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.01)
    losses_adam, lrs_adam, params_adam = test_optimizer(
        "Plain Adam", model_adam, opt_adam, use_metric=False
    )

    # Test 2: MobiuOptimizer (old API)
    opt_old = MobiuOptimizer(
        optim.Adam(model_mobiu_opt.parameters(), lr=0.01),
        license_key=LICENSE_KEY,
        method="standard",
        maximize=False,
        use_soft_algebra=True,
        sync_interval=10,
        verbose=True
    )
    losses_old, lrs_old, params_old = test_optimizer(
        "MobiuOptimizer (old)", model_mobiu_opt, opt_old, use_metric=True
    )
    opt_old.end()

    # Test 3: Mobiu (new API)
    opt_new = Mobiu(
        model_mobiu_new.parameters(),
        lr=0.01,
        license_key=LICENSE_KEY,
        verbose=True
    )
    losses_new, lrs_new, params_new = test_optimizer(
        "Mobiu (new)", model_mobiu_new, opt_new, use_metric=True
    )

    # Print config after warmup
    print(f"\nMobiu config:")
    if opt_new.config:
        print(f"  maximize: {opt_new.config.maximize}")
        print(f"  method: {opt_new.config.method}")
        print(f"  mode: {opt_new.config.mode}")
        print(f"  sync_interval: {opt_new.config.sync_interval}")
        print(f"  use_cloud: {opt_new.config.use_cloud}")
    else:
        print("  Config not set!")

    print(f"\nMobiu internal state:")
    print(f"  is_configured: {opt_new.is_configured}")
    print(f"  _cloud_session_id: {opt_new._cloud_session_id}")
    print(f"  _step_count: {opt_new._step_count}")
    print(f"  lr_history length: {len(opt_new.lr_history)}")

    opt_new.end()

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Optimizer':<20} {'Initial':<10} {'Final':<10} {'Improve':<10} {'LR changes'}")
    print(f"{'Adam':<20} {losses_adam[0]:<10.4f} {losses_adam[-1]:<10.4f} {(losses_adam[0]-losses_adam[-1])/losses_adam[0]*100:<10.1f}% {0}")
    print(f"{'MobiuOptimizer':<20} {losses_old[0]:<10.4f} {losses_old[-1]:<10.4f} {(losses_old[0]-losses_old[-1])/losses_old[0]*100:<10.1f}% {sum(1 for i in range(1, len(lrs_old)) if lrs_old[i] != lrs_old[i-1])}")
    print(f"{'Mobiu (new)':<20} {losses_new[0]:<10.4f} {losses_new[-1]:<10.4f} {(losses_new[0]-losses_new[-1])/losses_new[0]*100:<10.1f}% {sum(1 for i in range(1, len(lrs_new)) if lrs_new[i] != lrs_new[i-1])}")

if __name__ == "__main__":
    main()
