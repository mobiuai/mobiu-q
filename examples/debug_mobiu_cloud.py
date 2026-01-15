#!/usr/bin/env python3
"""Debug Mobiu cloud communication."""

import numpy as np
from mobiu_q import Mobiu

LICENSE_KEY = "YOUR_KEY"

# Create simple test
print("=" * 60)
print("DEBUG: Mobiu Cloud Communication")
print("=" * 60)

# Initialize with small params
init_params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
print(f"\nInit params: {init_params}")

# Create Mobiu
opt = Mobiu(
    params=init_params.copy(),
    lr=0.02,
    license_key=LICENSE_KEY,
    mode='hardware',
    verbose=True
)

# Warmup with VQE-like data (negative, decreasing)
warmup_data = [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4]
print(f"\nWarmup data (VQE-like): {warmup_data}")
opt.warmup_only(warmup_data)

print(f"\nConfig after warmup:")
print(f"  maximize: {opt.config.maximize}")
print(f"  method: {opt.config.method}")
print(f"  mode: {opt.config.mode}")
print(f"  base_lr: {opt.base_lr}")
print(f"  cloud_session_id: {opt._cloud_session_id}")

# Test new_run
print("\n" + "=" * 60)
print("Testing new_run + step")
print("=" * 60)

test_params = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
opt.new_run(test_params.copy())

print(f"\nAfter new_run:")
print(f"  _params: {opt._params}")
print(f"  cloud_session_id: {opt._cloud_session_id}")
print(f"  is_configured: {opt.is_configured}")

# Run a few steps
for i in range(5):
    energy = -1.0 - i * 0.1  # Decreasing (improving for VQE)
    gradient = np.random.randn(5) * 0.1

    print(f"\nStep {i+1}:")
    print(f"  Energy: {energy}")
    print(f"  Gradient: {gradient}")
    print(f"  Params before: {opt._params}")

    new_params = opt.step(energy, gradient=gradient)

    print(f"  Params after (returned): {new_params}")
    print(f"  Params after (internal): {opt._params}")

    if new_params is not None and not np.allclose(new_params, opt._params):
        print("  ⚠️ MISMATCH between returned and internal params!")

opt.end()
print("\n✅ Test complete")
