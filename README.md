# Mobiu-Q v2.7.5

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://badge.fury.io/py/mobiu-q)
[![License](https://img.shields.io/badge/License-Proprietary-blue)](https://mobiu.ai)

**Mobiu-Q** wraps your existing optimizer with **Soft Algebra** to filter noise and improve convergence. Same API, better results.

---

## ⚡ Quick Start

```python
from mobiu_q import MobiuOptimizer

# PyTorch (RL, LLM, Deep Learning)
opt = MobiuOptimizer(torch_optimizer, method="adaptive")

# Quantum VQE (clean simulator)
opt = MobiuOptimizer(params, method="standard", mode="simulation")

# Quantum VQE (real/noisy hardware)
opt = MobiuOptimizer(params, method="standard", mode="hardware")

# QAOA (combinatorial optimization)
opt = MobiuOptimizer(params, method="deep", mode="hardware")
```

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

### Base Optimizers

Available: `Adam` (default), `AdamW`, `NAdam`, `AMSGrad`, `SGD`, `Momentum`, `LAMB`

> **Note:** Optimizer names are case-sensitive!

---

## 📦 Installation

```bash
pip install mobiu-q
```

---

## 🎯 Usage Examples

### PyTorch (RL / LLM / Deep Learning)

```python
import torch
from mobiu_q import MobiuOptimizer

model = MyModel()
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, method="adaptive")

for epoch in range(100):
    loss = criterion(model(x), y)
    loss.backward()
    opt.step(loss.item())  # Pass loss for Soft Algebra
    opt.zero_grad()

opt.end()
```

### Quantum VQE (Simulation)

```python
from mobiu_q import MobiuOptimizer
import numpy as np

params = np.random.randn(10)
opt = MobiuOptimizer(params, method="standard", mode="simulation")

for step in range(100):
    params = opt.step(params, energy_fn)  # Auto-computes gradient

opt.end()
```

### Quantum VQE (Real Hardware / FakeFez)

```python
from mobiu_q import MobiuOptimizer

opt = MobiuOptimizer(params, method="standard", mode="hardware")

for step in range(100):
    params = opt.step(params, energy_fn)  # Uses SPSA gradient

opt.end()
```

### QAOA (Combinatorial Optimization)

```python
from mobiu_q import MobiuOptimizer

opt = MobiuOptimizer(params, method="deep", mode="hardware")

for step in range(150):
    params = opt.step(params, maxcut_cost_fn)

opt.end()
```

---

## 🏆 Verified Benchmark Results

All benchmarks use fair A/B testing: **Soft Algebra ON vs OFF**, same seeds, same conditions.

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

1. **Switch optimizer**: Try `NAdam` or `Momentum`
2. **Switch method**: `standard` ↔ `adaptive` ↔ `deep`
3. **Adjust LR**: Diverging → lower by 2-5x, stuck → raise by 2x

### Quantum Specific

- **Noisy results?** Use `mode="hardware"` (enables SPSA)
- **Clean simulator?** Use `mode="simulation"` (uses finite difference)

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