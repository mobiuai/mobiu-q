# Mobiu-Q v2.8.4

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://badge.fury.io/py/mobiu-q)
[![License](https://img.shields.io/badge/License-Proprietary-blue)](https://mobiu.ai)

**Mobiu-Q** wraps your existing optimizer with **Soft Algebra** to filter noise and improve convergence. Same API, better results.

---

## ⚡ Quick Start

```python
from mobiu_q import MobiuOptimizer

# PyTorch (RL, LLM, Deep Learning)
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, license_key="YOUR_KEY", method="adaptive")

# Quantum (VQE, QAOA) - pass params instead of optimizer
params = np.random.randn(10)
opt = MobiuOptimizer(params, license_key="YOUR_KEY", method="standard", mode="simulation")
```

> **Note:** `MobiuOptimizer` auto-detects PyTorch optimizers vs numpy arrays and uses the appropriate backend.

---

## 🔧 Configuration

### Methods

| Method | Best For | Default LR |
|--------|----------|------------|
| `standard` | VQE, Chemistry, smooth landscapes | 0.01 (sim) / 0.02 (hw) |
| `deep` | QAOA, combinatorial, rugged landscapes | 0.1 |
| `adaptive` | RL, LLM, high-variance problems | 0.0003 |

### Modes (Quantum only)

| Mode | When to Use | Gradient Method |
|------|-------------|-----------------|
| `simulation` | Clean simulator (Qiskit Aer, PennyLane default) | Finite Difference (2N evals) |
| `hardware` | Real quantum hardware, FakeFez, noisy backends | SPSA (2 evals, noise-resilient) |

**Rule of thumb:** If your backend has noise → use `hardware`. If it's a perfect simulator → use `simulation`.

### Base Optimizers (MobiuQCore only)

Available: `Adam` (default), `AdamW`, `NAdam`, `AMSGrad`, `SGD`, `Momentum`, `LAMB`

> **Note:** Optimizer names are case-sensitive!

### A/B Testing Parameter

```python
# For fair comparisons, toggle Soft Algebra:
opt = MobiuOptimizer(base_opt, use_soft_algebra=True)   # Default - SA enabled
opt = MobiuOptimizer(base_opt, use_soft_algebra=False)  # Baseline - SA disabled
```

---

## 📦 Installation

```bash
pip install mobiu-q
```

---

## 🎯 Usage Examples

### PyTorch (RL / LLM / Deep Learning)

Use `MobiuOptimizer` - wraps your PyTorch optimizer:

```python
import torch
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key"

model = MyModel()
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)

opt = MobiuOptimizer(
    base_opt,                    # Your PyTorch optimizer
    license_key=LICENSE_KEY,
    method="adaptive",           # Best for RL/LLM
    use_soft_algebra=True,       # Enable Soft Algebra (default)
    sync_interval=50,            # Sync with cloud every N steps (default: 50)
    verbose=True
)

for epoch in range(100):
    loss = criterion(model(x), y)
    loss.backward()
    opt.step(loss.item())  # Pass loss value for Soft Algebra
    opt.zero_grad()

opt.end()
```

### Quantum VQE (Simulation)

```python
from mobiu_q import MobiuOptimizer
import numpy as np

LICENSE_KEY = "your-license-key"

params = np.random.randn(10)

opt = MobiuOptimizer(
    params,                      # Numpy array - auto-detects quantum mode
    license_key=LICENSE_KEY,
    method="standard",
    mode="simulation",           # Clean simulator
    use_soft_algebra=True
)

for step in range(100):
    params = opt.step(params, energy_fn)  # Auto-computes gradient

opt.end()
```

### Quantum VQE (Real Hardware / FakeFez)

```python
from mobiu_q import MobiuOptimizer

opt = MobiuOptimizer(
    params,
    license_key=LICENSE_KEY,
    method="standard",
    mode="hardware",             # Noisy hardware - uses SPSA
)

for step in range(100):
    params = opt.step(params, energy_fn)

opt.end()
```

### QAOA (Combinatorial Optimization)

```python
from mobiu_q import MobiuOptimizer

opt = MobiuOptimizer(
    params,
    license_key=LICENSE_KEY,
    method="deep",               # Best for rugged landscapes
    mode="hardware",
)

for step in range(150):
    params = opt.step(params, maxcut_cost_fn)

opt.end()
```

### Manual Gradient (Quantum)

```python
from mobiu_q import MobiuOptimizer, Demeasurement

opt = MobiuOptimizer(params, license_key=LICENSE_KEY, method="standard")

for step in range(100):
    energy = energy_fn(params)
    gradient = Demeasurement.finite_difference(energy_fn, params)
    params = opt.step(params, gradient, energy)

opt.end()
```

---

## 🔑 License Key

Get your key at [app.mobiu.ai](https://app.mobiu.ai)

```python
# Option 1: Pass directly
opt = MobiuOptimizer(base_opt, license_key="your-key")

# Option 2: Environment variable
export MOBIU_Q_LICENSE_KEY="your-key"

# Option 3: Save to file (one time)
from mobiu_q import save_license_key
save_license_key("your-key")
```

---

## 🏆 Verified Benchmark Results

All benchmarks use fair A/B testing: **Soft Algebra ON vs OFF**, same seeds, same conditions.

### 🧪 Fair Testing Methodology

```python
# PyTorch A/B test:
opt_baseline = MobiuOptimizer(base_opt, license_key=KEY, use_soft_algebra=False)
opt_mobiu = MobiuOptimizer(base_opt, license_key=KEY, use_soft_algebra=True)

# Quantum A/B test:
opt_baseline = MobiuOptimizer(params, license_key=KEY, use_soft_algebra=False)
opt_mobiu = MobiuOptimizer(params, license_key=KEY, use_soft_algebra=True)

# Same seed, same problem, same everything - only SA differs
```

### ⚛️ Quantum VQE on IBM FakeFez

| Molecule | Qubits | Improvement | Win Rate |
|----------|--------|-------------|----------|
| **H₂** | 2 | **+52.5%** | 100% |
| **BeH₂** | 6 | **+55.1%** | 100% |
| **LiH** | 4 | **+34.5%** | 100% |

### 🎯 QAOA on IBM FakeFez

| Problem | Improvement | p-value |
|---------|-------------|---------|
| **MaxCut** | **+45.1%** | 0.0003 |

### 🎮 Reinforcement Learning

| Environment | Improvement | Win Rate |
|-------------|-------------|----------|
| **LunarLander-v3** | **+127.8%** | 96.7% |
| **MuJoCo InvertedPendulum** | **+111%** | 100% |
| **MuJoCo Hopper** | **+41%** | 80% |

### 💰 Finance

| Problem | Improvement |
|---------|-------------|
| **Credit Risk** | **+52.3%** |
| **Portfolio Optimization** | **+51.7%** |

---

## 🛠️ Troubleshooting

### Not Improving?

1. **Switch optimizer**: Try `NAdam` or `Momentum` (Quantum mode)
2. **Switch method**: `standard` ↔ `adaptive` ↔ `deep`
3. **Adjust LR**: Diverging → lower by 2-5x, stuck → raise by 2x
4. **Reduce sync_interval**: Try `sync_interval=1` for more frequent updates (PyTorch)

### Quantum Specific

- **Noisy results?** Use `mode="hardware"` (enables SPSA)
- **Clean simulator?** Use `mode="simulation"` (uses finite difference)

### PyTorch Specific

- **High latency?** Increase `sync_interval` (default: 50)
- **Not learning?** Decrease `sync_interval` to 1

---

## 📖 API Reference

### MobiuOptimizer (Universal)

```python
MobiuOptimizer(
    optimizer_or_params,          # torch.optim.Optimizer OR np.ndarray
    license_key: str,
    method: str = "adaptive",     # "standard", "deep", "adaptive"
    mode: str = "simulation",     # "simulation", "hardware"
    use_soft_algebra: bool = True,
    sync_interval: int = 50,      # Cloud sync frequency (PyTorch only)
    verbose: bool = True
)
```

**Auto-detection:**
- If `optimizer_or_params` has `.step()`, `.param_groups`, `.zero_grad()` → PyTorch mode
- Otherwise → Quantum mode (delegates to MobiuQCore)

### MobiuQCore (Quantum - Low-level)

For advanced quantum use cases:

```python
from mobiu_q import MobiuQCore

MobiuQCore(
    license_key: str,
    method: str = "standard",
    mode: str = "simulation",
    base_optimizer: str = "Adam", # Optimizer name (string!)
    base_lr: float = None,        # Auto-computed if None
    use_soft_algebra: bool = True,
    verbose: bool = True
)
```

---

## 🔬 How It Works

Mobiu-Q is based on **Soft Algebra** from Klein/Maimon theory:

```
SoftNumber multiplication (ε²=0):
(a, b) × (c, d) = (ad + bc, bd)
```

The **Super-Equation Δ†** detects emergence moments for adaptive scaling.

---

## 💰 Pricing

| Tier | Price | Runs |
|------|-------|------|
| **Free** | $0 | 20 runs/month |
| **Pro** | $19/month | Unlimited |

Get your key at [app.mobiu.ai](https://app.mobiu.ai)

---

## 🧑‍🔬 Scientific Foundation

Based on **Soft Numbers** theory developed by **Dr. Moshe Klein** and **Prof. Oded Maimon** (Tel Aviv University), as presented in their book on Soft Logic and Soft Numbers.

---

## 📚 Links

- **Website**: [mobiu.ai](https://mobiu.ai)
- **App**: [app.mobiu.ai](https://app.mobiu.ai)
- **PyPI**: [pypi.org/project/mobiu-q](https://pypi.org/project/mobiu-q)

---

© 2025 Mobiu Technologies. All rights reserved.