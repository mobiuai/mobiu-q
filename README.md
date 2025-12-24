# Mobiu-Q (v2.5.1)

**Universal Physics-Aware Optimizer for Stochastic Systems**

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://badge.fury.io/py/mobiu-q)
[![License](https://img.shields.io/badge/License-Proprietary-blue)](https://mobiu.ai)

**Mobiu-Q** is the first optimizer based on **Soft Algebra**, developed by Dr. Moshe Klein and Prof. Oded Maimon. By mathematically decomposing gradients into *Potential* ($a_t$) and *Realization* ($b_t$), it filters out noise in real-time.

Works across **Quantum Computing**, **Reinforcement Learning**, **LLM Fine-Tuning**, and **FinTech**.

---

## 🚀 What's New in v2.5.0

- **New Method Names**: `standard`, `deep`, `adaptive` (legacy `vqe`/`qaoa`/`rl` still work!)
- **Noise Robustness**: +32.5% more robust to quantum hardware noise
- **80% Win Rate**: Outperforms standard optimizers across all noise levels
- **LLM Support**: +18% improvement on soft prompt tuning

---

## 🏆 Benchmark Results

### Noise Robustness (IBM FakeBackend)

| Condition | Momentum | Mobiu-Q | Winner |
|-----------|----------|---------|--------|
| IDEAL     | -2.19    | -1.29   | Momentum |
| **NOISY** | +0.20    | **-0.30** | **Mobiu-Q ✅** |

**Key Finding:**
- Momentum degradation under noise: **+109%** (breaks down)
- Mobiu-Q degradation under noise: **+77%** (stays stable)
- **Mobiu-Q is 32.5% MORE ROBUST to noise!**

### Comprehensive Noise Test

| Qubits | Noise Level | SA Gain | Win Rate |
|--------|-------------|---------|----------|
| 2      | all levels  | +27-65% | 5/5 ✅   |
| 4      | all levels  | +5-19%  | 4/5 ✅   |
| 6      | all levels  | +2-14%  | 3/5 ✅   |

**Overall: 80% win rate (12/15 tests) with +5% to +65% improvement**

### LLM Soft Prompt Tuning

| Config | Improvement | Win Rate |
|--------|-------------|----------|
| Momentum+SA | **+18.1%** | 3/3 ✅ |

### Quantum Chemistry (VQE)

| Molecule | Improvement |
|----------|-------------|
| H2       | +46.6%      |
| LiH      | +41.4%      |
| BeH2     | +37.8%      |
| He Atom  | +51.2%      |

### Reinforcement Learning

| Environment | Improvement | Win Rate |
|-------------|-------------|----------|
| LunarLander | +129.7%     | 96.7%    |
| MuJoCo      | +118.6%     | 100%     |

---

## 📦 Installation

```bash
pip install mobiu-q
```

---

## ⚡ Quick Start

### 1. Standard (Quantum VQE, Chemistry)

```python
from mobiu_q import MobiuQCore, Demeasurement

opt = MobiuQCore(license_key="YOUR-KEY", method="standard")

for step in range(100):
    grad = Demeasurement.finite_difference(energy_fn, params)
    params = opt.step(params, grad, energy_fn(params))

opt.end()
```

### 2. Deep (Complex Circuits, Noisy Hardware)

```python
opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="deep",
    mode="hardware"  # For quantum hardware / noisy simulation
)

for step in range(150):
    grad, energy = Demeasurement.spsa(energy_fn, params)
    params = opt.step(params, grad, energy)

opt.end()
```

### 3. Adaptive (RL, LLM Fine-Tuning)

```python
opt = MobiuQCore(license_key="YOUR-KEY", method="adaptive")

for episode in range(1000):
    episode_return = run_episode(policy)
    gradient = compute_policy_gradient()
    policy_params = opt.step(policy_params, gradient, episode_return)

opt.end()
```

### 4. LLM Soft Prompt Tuning

```python
opt = MobiuQCore(license_key="YOUR-KEY", method="adaptive")

for step in range(50):
    loss = compute_loss(soft_tokens, model, batch)
    grad = compute_gradient(loss, soft_tokens)
    soft_tokens = opt.step(soft_tokens, grad, loss)

opt.end()
```

### 5. Multi-Seed Experiments (1 billing session)

```python
opt = MobiuQCore(license_key="YOUR-KEY")

for seed in range(10):
    opt.new_run()  # Resets state, keeps session open
    params = init_params(seed)
    # ... optimization loop ...

opt.end()  # All 10 seeds count as 1 run
```

---

## 🎛️ Configuration

### Methods

| Method   | Legacy | Use Case                              | Default LR |
|----------|--------|---------------------------------------|------------|
| standard | vqe    | Smooth landscapes, chemistry, physics | 0.01-0.02  |
| deep     | qaoa   | Deep circuits, noisy hardware         | 0.1        |
| adaptive | rl     | RL, LLM fine-tuning, high-variance    | 0.0003     |

### Modes

| Mode       | Use Case                    |
|------------|------------------------------|
| simulation | Clean simulations            |
| hardware   | Quantum hardware, noisy sims |

### Optimizers

⚠️ **Optimizer names are case-sensitive!**

```python
# Use default (Adam)
opt = MobiuQCore(method="standard")

# Alternative optimizer
opt = MobiuQCore(method="deep", base_optimizer="NAdam")
```

Available optimizers:
- `Adam` (default) - Best overall
- `NAdam` - Strong on deep circuits
- `Momentum` - Best for noisy hardware (+18.1% on LLM)
- `AMSGrad` - Alternative for standard
- `SGD` - Simple baseline
- `LAMB` - Large batch training

### Disable Soft Algebra

For A/B testing:

```python
opt = MobiuQCore(method="standard", use_soft_algebra=False)
```

---

## 🧠 How It Works

### The Core Innovation: "Noise Hallucination" Prevention

Standard optimizers assume lower objective values always indicate better solutions. In noisy environments, this fails. Mobiu-Q uses **Soft Algebra** to distinguish real progress from noise.

### SoftNumber Multiplication (Nilpotent ε²=0)

```
(a, b) * (c, d) = (ad + bc, bd)
```

### State Evolution

```
S_{t+1} = (γ · S_t) · Δ_t + Δ_t
```

Where:
- `a_t` (Potential): Curvature signal
- `b_t` (Realized): Actual improvement
- `Δ†` (Super-Equation): Emergence detection for deep/adaptive

### Method-Specific Logic

| Method   | Primary Mechanism              | Best For                    |
|----------|--------------------------------|-----------------------------|
| standard | Trust Ratio + Gradient Warping | Smooth energy landscapes    |
| deep     | Super-Equation Δ†              | Rugged, multimodal, noisy   |
| adaptive | Trust + Emergence + Warping    | High-variance, sparse reward|

---

## 📊 When to Use Mobiu-Q

✅ **Use Mobiu-Q when:**
- High noise/variance (quantum hardware, RL, stochastic finance)
- Deep circuits with many parameters
- Noisy quantum hardware (IBM, IonQ, etc.)
- LLM fine-tuning with limited data
- Standard optimizers diverge or get stuck

❌ **Skip Mobiu-Q when:**
- Clean, convex problems
- Deterministic, low-noise environments

---

## 🔑 Pricing

| Tier     | Runs/Month | Features              |
|----------|------------|-----------------------|
| **Free** | 20         | Testing & students    |
| **Pro**  | Unlimited  | Priority, all features|

**[Get your License Key](https://app.mobiu.ai)**

---

## 📚 API Reference

### MobiuQCore

```python
MobiuQCore(
    license_key: str,
    method: str = "standard",      # "standard", "deep", "adaptive"
    mode: str = "simulation",      # "simulation" or "hardware"
    base_lr: float = None,         # Auto if None
    base_optimizer: str = "Adam",  # Case-sensitive!
    use_soft_algebra: bool = True,
    offline_fallback: bool = True
)
```

**Methods:**
- `step(params, gradient, energy)` → Updated params
- `new_run()` → Reset for new seed (same session)
- `end()` → End session (counts usage)
- `check_usage()` → Get remaining runs

### Demeasurement

```python
# For standard (smooth)
grad = Demeasurement.finite_difference(energy_fn, params)
grad = Demeasurement.parameter_shift(circuit_fn, params)

# For deep/hardware (noisy)
grad, energy = Demeasurement.spsa(energy_fn, params)
```

---

## 🔬 Scientific Foundation

Mobiu-Q is based on **Soft Algebra**, developed by:

- **Dr. Moshe Klein** - Mathematician, Soft Logic and Soft Numbers
- **Prof. Oded Maimon** - Tel Aviv University, Industrial Engineering

---

## 📖 Citation

```bibtex
@software{mobiu_q,
  title = {Mobiu-Q: Soft Algebra Optimizer for Stochastic Systems},
  author = {Angel, Ido and Klein, Moshe and Maimon, Oded},
  year = {2024},
  url = {https://mobiu.ai}
}
```

---

*Proprietary technology. All rights reserved by Mobiu Technologies.*