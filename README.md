# Mobiu-Q

**Soft Algebra Optimizer for Quantum Computing**

Mobiu-Q achieves **+23% on VQE** and **+46% on QAOA** compared to Adam optimizer, using a novel Soft Algebra approach that's resilient to quantum noise.

## Installation

```bash
pip install mobiu-q
```

## Quick Start

### VQE (Molecular Simulation)

```python
from mobiu_q import MobiuQCore, Demeasurement
import numpy as np

# Initialize optimizer for VQE
opt = MobiuQCore(
    license_key="your-license-key",
    problem="vqe"         # Uses Trust Ratio, lr=0.01
)

# Optimization loop
params = np.random.uniform(-np.pi, np.pi, n_params)
for step in range(60):
    energy = energy_fn(params)
    gradient = Demeasurement.finite_difference(energy_fn, params)
    params = opt.step(params, gradient, energy)

opt.end()
```

### QAOA (Combinatorial Optimization)

```python
from mobiu_q import MobiuQCore
import numpy as np

# Initialize optimizer for QAOA
opt = MobiuQCore(
    license_key="your-license-key",
    mode="noisy",         # For SPSA gradients
    problem="qaoa",       # Uses Super-Equation Δ†
    base_lr=0.1           # Important! Default is 0.02
)

# SPSA gradient function (built-in to your workflow)
def spsa_gradient(fn, params, c=0.1):
    delta = np.random.choice([-1, 1], size=len(params))
    E_plus = fn(params + c * delta)
    E_minus = fn(params - c * delta)
    grad = (E_plus - E_minus) / (2 * c) * delta
    energy = (E_plus + E_minus) / 2
    return grad, energy

# Optimization loop
params = np.random.uniform(-np.pi, np.pi, n_params)
for step in range(150):
    gradient, energy = spsa_gradient(qaoa_energy_fn, params)
    params = opt.step(params, gradient, energy)

opt.end()
```

## Multi-Seed Experiments (Counts as 1 Run)

```python
opt = MobiuQCore(license_key="your-license-key", problem="vqe")

for seed in range(10):
    opt.new_run()  # Reset optimizer state, keep session
    np.random.seed(seed)
    params = np.random.uniform(-np.pi, np.pi, n_params)
    
    for step in range(60):
        gradient = Demeasurement.finite_difference(energy_fn, params)
        params = opt.step(params, gradient, energy_fn(params))

opt.end()  # All 10 seeds count as 1 run!
```

## Validate Your Results

Run our validation script to see the improvement on your machine:

```bash
# Download validation script
curl -O https://raw.githubusercontent.com/mobiuai/mobiu-q/main/examples/customer_validation.py

# Edit LICENSE_KEY in the file, then run:
python customer_validation.py
```

Expected output:
```
TEST 1: VQE - H2 Molecule (60 steps)
📊 IMPROVEMENT: ~23% better accuracy with Mobiu-Q

TEST 2: QAOA - MaxCut Ising (150 steps)  
📊 IMPROVEMENT: ~46% (Mobiu-Q wins 10/10 seeds)
```

## Benchmarks

### VQE (Quantum Chemistry)

| Molecule | Improvement vs Adam | p-value |
|----------|---------------------|---------|
| H₂       | +23%                | < 0.05  |
| LiH      | +50.6%              | < 10⁻¹² |
| HeH⁺     | +68%                | < 10⁻⁴⁰ |

### QAOA (Combinatorial, noise=10%)

| Problem             | Depth | Improvement | Win Rate |
|---------------------|-------|-------------|----------|
| MaxCut              | p=5   | +46%        | 10/10    |
| Vertex Cover        | p=5   | +35.77%     | 55/60    |
| Max Independent Set | p=5   | +30.62%     | 52/60    |

### Hardware Validation

Tested on **IBM Quantum Fez (156-qubit)**:
- **Adam**: Crashed to non-physical energy
- **Mobiu-Q**: Converged to 99.6% accuracy

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem` | `"vqe"` | `"vqe"` or `"qaoa"` |
| `mode` | `"standard"` | `"standard"` or `"noisy"` |
| `base_lr` | auto | VQE: 0.01, QAOA: **use 0.1** |

### Recommended Settings

| Use Case | Settings |
|----------|----------|
| VQE (simulator) | `problem="vqe"` |
| VQE (hardware) | `problem="vqe", mode="noisy"` |
| QAOA | `problem="qaoa", mode="noisy", base_lr=0.1` |

## How It Works

### VQE: Trust Ratio

For smooth energy landscapes (molecular chemistry), Mobiu-Q uses the **Trust Ratio**:

```
φ = |S.real| / (|S.real| + |S.soft| + ε)
```

High trust = stable gradient = larger learning rate.

### QAOA: Super-Equation Δ†

For rugged combinatorial landscapes, Mobiu-Q uses the **Super-Equation**:

```
Δ† = |Du[sin(πS)]| · g(τ,α) · Γ(a,β) · √(b·g(τ,α))
```

This identifies optimal "emergence points" where the optimization should act aggressively.

## Pricing

- **Free**: 20 runs/month
- **Pro**: $19/month unlimited

Get your license at [app.mobiu.ai](https://app.mobiu.ai)

## Support

- Email: ai@mobiu.ai

## License

Proprietary - All rights reserved.

---

Made with ❤️ by [Mobiu Technologies](https://mobiu.ai)