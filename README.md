# Mobiu-Q v2.7.3

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://badge.fury.io/py/mobiu-q)
[![License](https://img.shields.io/badge/License-Proprietary-blue)](https://mobiu.ai)

**Mobiu-Q** wraps your existing optimizer with **Soft Algebra** to filter noise and improve convergence. Same API, better results.

---

## 🚀 What's New in v2.7

- **MobiuOptimizer**: Universal wrapper - auto-detects PyTorch optimizers!
- **Hybrid Mode**: Cloud intelligence + local PyTorch performance
- **Zero Friction**: One-line integration for PyTorch users
- **Full Backward Compatibility**: All existing code continues to work

---

## ⚡ Quick Start

### PyTorch Users (NEW in v2.7!)

```python
import torch
from mobiu_q import MobiuOptimizer

# Your existing code
model = MyModel()
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)

# Wrap with Mobiu-Q (one line!)
opt = MobiuOptimizer(base_opt, method="adaptive")

# Training loop stays the same
for epoch in range(100):
    loss = criterion(model(x), y)
    loss.backward()
    opt.step(loss.item())  # Pass loss for Soft Algebra
    opt.zero_grad()

opt.end()
```

### Quantum Users (unchanged)

```python
from mobiu_q import MobiuQCore

opt = MobiuQCore(method="standard")
for step in range(100):
    params = opt.step(params, energy_fn)
opt.end()
```

---

## 🏆 Verified Benchmark Results

All benchmarks compare **Optimizer + Soft Algebra** vs **Optimizer alone**. Same learning rate, same seeds, fair A/B test.

### 🎮 Reinforcement Learning
| Environment | Improvement | p-value | Win Rate |
|-------------|-------------|---------|----------|
| **LunarLander-v3** | **+129.7%** | <0.001 | 96.7% |
| **MuJoCo InvertedPendulum** | **+118.6%** | 0.001 | 100% |
| **MuJoCo Hopper** | **+41.2%** | 0.007 | 80% |

### 📐 Classical Optimization
| Function | Improvement | Description |
|----------|-------------|-------------|
| **Rosenbrock** | **+75.8%** | Valley navigation |
| **Beale** | **+62.0%** | Plateau escape |
| **Sphere** | **+31.1%** | Convex baseline |

### ⚛️ Quantum VQE - Condensed Matter
| Model | Improvement |
|-------|-------------|
| **SSH Model (Topological)** | **+61.0%** |
| **XY Model** | **+60.8%** |
| **Ferro Ising** | **+45.1%** |
| **Transverse Ising** | **+42.0%** |
| **Heisenberg XXZ** | **+20.8%** |
| **Kitaev Chain** | **+20.4%** |
| **Hubbard Dimer** | **+14.1%** |

### ⚛️ Quantum VQE - Chemistry
| Molecule | Improvement |
|----------|-------------|
| **FakeFez H₂** | **+52.4%** (p=0.043) |
| **He Atom** | **+51.2%** |
| **H₂ Molecule** | **+46.6%** |
| **H₃⁺ Chain** | **+42.0%** |
| **LiH Molecule** | **+41.4%** |
| **BeH₂ Molecule** | **+37.8%** |

### 🎯 QAOA (Combinatorial Optimization)
| Problem | Improvement | Wins |
|---------|-------------|------|
| **FakeFez MaxCut** | **+45.1%** | p=0.0003 |
| **Vertex Cover** | **+31.9%** | 51/60 |
| **Max Independent Set** | **+31.9%** | 51/60 |
| **MaxCut** | **+21.5%** | 45/60 |

### 💰 Finance (QUBO)
| Problem | Improvement |
|---------|-------------|
| **Credit Risk** | **+52.3%** |
| **Portfolio Optimization** | **+51.7%** |

### 💊 Drug Discovery
| Task | Improvement | Config |
|------|-------------|--------|
| **Binding Affinity** | **+12.2%** | AMSGrad + standard |

---

## 📦 Installation

```bash
pip install mobiu-q
```

---

## 🔧 Configuration

📖 **See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for complete details**

### Which Class to Use?

| Use Case | Class | Mode |
|----------|-------|------|
| **PyTorch (RL, LLM, Deep Learning)** | `MobiuOptimizer` | Hybrid |
| **Quantum (VQE, QAOA)** | `MobiuQCore` | Cloud |
| **NumPy optimization** | `MobiuQCore` | Cloud |

### Methods

| Method | Best For | Default LR |
|--------|----------|------------|
| `standard` | VQE, Chemistry, Finance | 0.01 |
| `deep` | QAOA, Noisy Hardware | 0.1 |
| `adaptive` | RL, LLM, High-variance | 0.0003 |

### Base Optimizers

| Optimizer | Best For |
|-----------|----------|
| `Adam` | Default, most cases |
| `AdamW` | LLM, weight decay |
| `SGD` | QAOA |
| `AMSGrad` | Drug Discovery |
| `NAdam` | Alternative to Adam |
| `Momentum` | RL alternative |
| `LAMB` | Large batch |

**Important:** Optimizer names are case-sensitive!

---

## 🆕 MobiuOptimizer (v2.7)

The new `MobiuOptimizer` auto-detects your optimizer type:

```python
from mobiu_q import MobiuOptimizer

# PyTorch optimizer → Hybrid mode (recommended for RL/LLM)
base_opt = torch.optim.Adam(model.parameters())
opt = MobiuOptimizer(base_opt, method="adaptive")

# NumPy params → Cloud mode (same as MobiuQCore)
params = np.random.randn(10)
opt = MobiuOptimizer(params, method="standard")
```

### How Hybrid Mode Works

```
┌─────────────────┐     loss/return    ┌─────────────────┐
│  Local PyTorch  │ ──────────────────▶│   Mobiu Cloud   │
│                 │                    │                 │
│  • Gradients    │   adaptive_lr      │  • Soft Algebra │
│  • Weight Update│ ◀──────────────────│  • Super-Eq Δ†  │
│  • Momentum     │                    │  • Trust Ratio  │
└─────────────────┘                    └─────────────────┘
```

**Benefits:**
- ✅ PyTorch handles precision & GPU acceleration
- ✅ Cloud provides Soft Algebra intelligence
- ✅ Minimal network overhead (only sends loss value)
- ✅ No momentum state corruption

---

## 🛠️ Troubleshooting

If optimization is not improving or diverging:

### 1. Switch Base Optimizer

```python
opt = MobiuQCore(license_key="KEY", base_optimizer="NAdam")
opt = MobiuQCore(license_key="KEY", base_optimizer="Momentum")
```

### 2. Switch Method

| If This Fails | Try This |
|---------------|----------|
| `standard` | `adaptive` |
| `adaptive` | `deep` |
| `deep` | `standard` |

### 3. Switch Mode (Quantum)

```python
opt = MobiuQCore(license_key="KEY", mode="hardware")
```

### 4. Adjust Learning Rate

| Scenario | Recommendation |
|----------|---------------|
| Diverging | Lower LR by 2-5x |
| No improvement | Increase LR by 2x |

---

## 🔬 How It Works

Mobiu-Q is based on **Soft Algebra** (ε²=0):

```
(a, b) × (c, d) = (ad + bc, bd)
```

Evolution Law:
```
S_{t+1} = (γ · S_t) · Δ_t + Δ_t
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

## 📊 Summary by Domain

| Domain | Best Result | Avg Improvement |
|--------|-------------|-----------------|
| **RL** | +129.7% | ~96% |
| **Classical Opt** | +75.8% | ~56% |
| **Condensed Matter** | +61.0% | ~38% |
| **Quantum Chemistry** | +52.4% | ~45% |
| **Finance** | +52.3% | ~52% |
| **QAOA** | +45.1% | ~32% |
| **Drug Discovery** | +12.2% | +12% |

---

## 🧑‍🔬 Scientific Foundation

- **Dr. Moshe Klein** – Soft Logic and Soft Numbers
- **Prof. Oded Maimon** – Tel Aviv University

---

## 📚 Links

- **Website**: [mobiu.ai](https://mobiu.ai)
- **App**: [app.mobiu.ai](https://app.mobiu.ai)
- **PyPI**: [pypi.org/project/mobiu-q](https://pypi.org/project/mobiu-q)

---

© 2025 Mobiu Technologies. All rights reserved.