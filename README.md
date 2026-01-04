# Mobiu-Q v3.0.4

**Soft Algebra for Optimization & Attention**

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://pypi.org/project/mobiu-q/)
[![License](https://img.shields.io/badge/license-Proprietary-blue.svg)](LICENSE)

---

## Overview

Mobiu-Q is a framework built on **Soft Algebra** (nilpotent ε²=0) that provides:

1. **MobiuOptimizer** - Stable optimization in noisy environments
2. **MobiuAttention** 🧪 - O(N) linear attention for long sequences

Both share the same mathematical foundation but serve different purposes.

---

## Installation

```bash
pip install mobiu-q
```

---

## Quick Start

### MobiuOptimizer (Stable API)

```python
from mobiu_q import MobiuOptimizer
import torch

# Wrap any PyTorch optimizer
model = MyModel()
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, method="adaptive", use_soft_algebra=True)

for batch in dataloader:
    loss = criterion(model(batch))
    loss.backward()
    opt.step(loss.item())  # Pass loss for Soft Algebra

opt.end()  # Important: release resources
```

### MobiuAttention (🧪 Experimental)

```python
from mobiu_q.experimental import MobiuAttention, MobiuBlock

# Drop-in replacement for nn.MultiheadAttention
attn = MobiuAttention(d_model=512, num_heads=8)
out = attn(x)  # x: [batch, seq, dim]

# Or use complete block
block = MobiuBlock(d_model=512, num_heads=8)
out = block(x)
```

---

## MobiuOptimizer

### Methods

| Method     | Use Case                                    | Default LR |
|------------|---------------------------------------------|------------|
| `standard` | Smooth landscapes, chemistry, physics       | 0.01       |
| `deep`     | Deep circuits, noisy hardware, complex opt  | 0.1        |
| `adaptive` | RL, LLM fine-tuning, high-variance problems | 0.0003     |

### Benchmarks

| Domain                  | Improvement | Win Rate | p-value |
|-------------------------|-------------|----------|---------|
| Crypto Trading 🆕       | **+56% profit** | **100%** | **<0.001** |
| LunarLander-v3          | +128%       | 97%      | <0.001  |
| MuJoCo InvertedPendulum | +111%       | 100%     | <0.001  |
| VQE H₂ (FakeFez)        | +52%        | 100%     | <0.001  |
| QAOA MaxCut             | +45%        | 95%      | <0.001  |

#### Crypto Trading Details

Tested on synthetic crypto market with regime switching (bull/bear), flash crashes, and high volatility:

| Metric | Adam Baseline | Mobiu Optimizer |
|--------|---------------|-----------------|
| Profit | -0.9% | **+55.9%** |
| Episode Return | -0.17 | **+0.46** |

*500 episodes × 10 seeds, p < 0.001*

### Maximize vs Minimize

By default, Mobiu-Q assumes you're **minimizing** (loss, energy). For RL/Trading where you **maximize** (reward, profit), use the explicit keyword arguments:
```python
# Loss minimization - for supervised learning, VQE
opt.step(loss=loss.item())  # Lower is better

# Reward maximization - for RL, trading
opt.step(reward=episode_return)  # Higher is better
```

| Use Case | What to pass | Example |
|----------|--------------|---------|
| Supervised Learning | `loss=` | `opt.step(loss=loss.item())` |
| VQE / QAOA | `loss=` | `opt.step(loss=energy)` |
| RL (policy gradient) | `reward=` | `opt.step(reward=episode_return)` |
| Trading | `reward=` | `opt.step(reward=profit)` |

**Legacy API** (still supported):
```python
# Old way - use maximize flag at init
opt = MobiuOptimizer(base_opt, method="adaptive", maximize=True)
opt.step(episode_return)
```

**Why does this matter?** Soft Algebra tracks the "direction of improvement". Passing reward as loss (or vice versa) confuses the optimizer.

### A/B Testing

```python
# Test with Soft Algebra
opt_on = MobiuOptimizer(base_opt, use_soft_algebra=True)

# Test without (baseline)
opt_off = MobiuOptimizer(base_opt, use_soft_algebra=False)
```

---

## Base Optimizers

Mobiu-Q enhances these base optimizers with Soft Algebra:

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `Adam` | Adaptive moments, most popular | Default, most cases |
| `AdamW` | Adam with decoupled weight decay | LLM, Transformers |
| `NAdam` | Adam with Nesterov momentum | Alternative to Adam |
| `AMSGrad` | Adam with max(v) for stability | Drug discovery, unstable loss |
| `SGD` | Simple gradient descent | QAOA, convex problems |
| `Momentum` | SGD with momentum | RL, LLM fine-tuning |
| `LAMB` | Layer-wise adaptive scaling | Large batch training |

### Choosing an Optimizer

**PyTorch mode** - Choose your optimizer when creating the base optimizer:

```python
import torch
from mobiu_q import MobiuOptimizer

# Using Adam (default, recommended for most cases)
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, method="adaptive")

# Using AdamW (recommended for LLM/Transformers)
base_opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
opt = MobiuOptimizer(base_opt, method="adaptive")

# Using SGD with Momentum (recommended for RL)
base_opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
opt = MobiuOptimizer(base_opt, method="adaptive", maximize=True)

# Using NAdam
base_opt = torch.optim.NAdam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, method="deep")
```

**Quantum mode** - Choose your optimizer via the `base_optimizer` parameter:

```python
from mobiu_q import MobiuOptimizer
import numpy as np

params = np.random.randn(10)

# Using Adam (default)
opt = MobiuOptimizer(params, method="standard")

# Using NAdam
opt = MobiuOptimizer(params, method="standard", base_optimizer="NAdam")

# Using AMSGrad
opt = MobiuOptimizer(params, method="deep", base_optimizer="AMSGrad")
```

**⚠️ Important:** In Quantum mode, optimizer names are **case-sensitive!**

```python
# ✅ Correct
opt = MobiuOptimizer(params, base_optimizer="NAdam")

# ❌ Wrong - will fall back to Adam
opt = MobiuOptimizer(params, base_optimizer="nadam")
```

---

## 🛠️ Troubleshooting

If optimization is not improving or diverging, try these adjustments:

### 1. Switch Base Optimizer

Different optimizers work better for different problems:

| Problem Type | Recommended Optimizer |
|--------------|----------------------|
| LoRA / LLM | `Momentum` or `AdamW` |
| VQE / Chemistry | `Adam` |
| QAOA | `NAdam` |
| RL / Trading | `Momentum` |
| Drug Discovery | `AMSGrad` |
| Large Batch | `LAMB` |

```python
# PyTorch: If Adam isn't working, try Momentum:
base_opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
opt = MobiuOptimizer(base_opt, method="adaptive")

# Quantum: If Adam isn't working, try NAdam:
opt = MobiuOptimizer(params, base_optimizer="NAdam", method="adaptive")
```

### 2. Switch Method

| If This Fails | Try This |
|---------------|----------|
| `standard` | `adaptive` |
| `adaptive` | `deep` |
| `deep` | `standard` |

```python
# If standard isn't working for your problem:
opt = MobiuOptimizer(base_opt, method="adaptive")
```

### 3. Switch Mode (Quantum only)

| If This Fails | Try This |
|---------------|----------|
| `simulation` | `hardware` |

```python
opt = MobiuOptimizer(params, method="standard", mode="hardware")
```

### 4. Adjust Learning Rate

```python
# Try lower LR if diverging
base_opt = torch.optim.Adam(model.parameters(), lr=0.0001)

# Try higher LR if stuck
base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 5. Common Fixes by Domain

| Domain | Common Issue | Fix |
|--------|--------------|-----|
| **LoRA** | SGD + high LR diverges | Use `Momentum` + LR=0.02 |
| **Drug Discovery** | BCE loss unstable | Use `AMSGrad` + `standard` method |
| **Crypto/RL** | High variance | Use `Momentum` + `adaptive` method |
| **QAOA** | Local minima | Use `NAdam` + `deep` method |

---

## MobiuAttention 🧪

### Why?

Standard Transformer attention is O(N²) in sequence length. MobiuAttention is **O(N)**.

| Seq Length | Transformer | MobiuAttention | Speedup |
|------------|-------------|----------------|---------|
| 2,048      | 21s         | 9s             | 2.3x    |
| 4,096      | 39s         | 10s            | 3.9x    |
| 8,192      | 42s         | 7s             | 6.0x    |
| 16,384     | **OOM** 💥  | 5s ✅          | ∞       |

### Quality (Same as Transformer)

| Benchmark            | Transformer | MobiuAttention |
|----------------------|-------------|----------------|
| Shakespeare PPL      | 12.8        | 13.5           |
| ListOps Accuracy     | 81%         | 82%            |
| Needle-in-Haystack   | 100%        | 100%           |

### Usage

```python
from mobiu_q.experimental import MobiuBlock

class LongContextLM(nn.Module):
    def __init__(self, vocab, d=512, h=8, layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.Sequential(*[MobiuBlock(d, h) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)
    
    def forward(self, x):
        return self.head(self.blocks(self.embed(x)))

# Works with 16K+ tokens!
model = LongContextLM(50000)
x = torch.randint(0, 50000, (1, 16384))
out = model(x)  # No OOM!
```

### ⚠️ Experimental Status

- Functional and tested
- API may change in future versions
- Feedback welcome!

---

## How It Works

### Soft Algebra

Both optimizer and attention use the nilpotent property ε²=0:

```
SoftNumber multiplication: (a,b) × (c,d) = (ad + bc, bd)
```

This enables tracking both "potential" and "realized" components.

### In Optimization

```python
lr_t = base_lr × (1 + soft_component)
```

Soft Algebra adapts learning rate based on loss landscape curvature.

### In Attention

```python
S(t) = γ·S(t-1) + k_t ⊗ v_t  # O(N) state update
```

Instead of O(N²) pairwise attention, we track state with O(N) complexity.

---

## License

Free tier: 20 API calls/month (optimizer only)
Pro tier: Unlimited - https://app.mobiu.ai

**Note:** MobiuAttention runs locally, no API calls required.

---

## Links

- [PyPI](https://pypi.org/project/mobiu-q/)
- [GitHub Issues](https://github.com/mobiu-ai/mobiu-q/issues)

---

## Citation

```bibtex
@software{mobiu_q,
  title={Mobiu-Q: Soft Algebra for Optimization and Attention},
  author={Mobiu Technologies},
  year={2026},
  url={https://github.com/mobiu-ai/mobiu-q}
}
```