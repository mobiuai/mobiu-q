# Mobiu-Q v3.1.2

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

# Your license key (get one at https://app.mobiu.ai)
LICENSE_KEY = "your-license-key-here"

# Wrap any PyTorch optimizer
model = MyModel()
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="adaptive",
    use_soft_algebra=True
)

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
# Note: MobiuAttention runs locally, no license key needed!
attn = MobiuAttention(d_model=512, num_heads=8)
out = attn(x)  # x: [batch, seq, dim]

# Or use complete block
block = MobiuBlock(d_model=512, num_heads=8)
out = block(x)
```

---

## License Key

MobiuOptimizer requires a license key to access the cloud API:

```python
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# PyTorch mode (pass optimizer)
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")

# Quantum/NumPy mode (pass params array)
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, method="standard")
```

**Get your key:** https://app.mobiu.ai

| Tier | API Calls | Price |
|------|-----------|-------|
| Free | 20/month | $0 |
| Pro | Unlimited | $19/month |

**Note:** MobiuAttention runs locally and does NOT require a license key.

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

By default, Mobiu-Q assumes you're **minimizing** (loss, energy). For RL/Trading where you **maximize** (reward, profit), set `maximize=True`:

```python
LICENSE_KEY = "your-license-key-here"

# Loss minimization (default) - for supervised learning, VQE
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")
opt.step(loss.item())

# Reward maximization - for RL, trading
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive", maximize=True)
opt.step(episode_return)
```

| Use Case | maximize= | Example |
|----------|-----------|---------|
| Supervised Learning | `False` (default) | `opt.step(loss.item())` |
| VQE / QAOA | `False` (default) | `opt.step(energy)` |
| RL (policy gradient) | `True` | `opt.step(episode_return)` |
| Trading | `True` | `opt.step(profit)` |

**Why does this matter?** Soft Algebra tracks the "direction of improvement". Using the wrong setting confuses the optimizer.

### A/B Testing

```python
LICENSE_KEY = "your-license-key-here"

# Test with Soft Algebra
opt_on = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, use_soft_algebra=True)

# Test without (baseline)
opt_off = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, use_soft_algebra=False)
```

---

## Examples by Domain

### Reinforcement Learning (REINFORCE)

```python
import torch
import torch.nn.functional as F
import gymnasium as gym
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# Simple policy network
policy = torch.nn.Sequential(
    torch.nn.Linear(8, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, 4)
)

# Wrap optimizer with maximize=True for RL
base_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="adaptive",
    maximize=True,       # Important: RL maximizes reward!
    sync_interval=50,    # Sync with cloud every 50 steps
    verbose=True
)

env = gym.make("LunarLander-v3")

for episode in range(1000):
    state, _ = env.reset()
    log_probs, rewards = [], []
    
    # Collect episode
    done = False
    while not done:
        logits = policy(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    
    # REINFORCE update
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = sum(-lp * G for lp, G in zip(log_probs, returns))
    
    opt.zero_grad()
    loss.backward()
    opt.step(sum(rewards))  # Pass episode return for Soft Algebra

opt.end()
```

### Quantum Chemistry (VQE with Qiskit)

```python
import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# H₂ Hamiltonian
hamiltonian = SparsePauliOp.from_list([
    ("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347),
    ("IZ", 0.5716), ("XX", 0.0910), ("YY", 0.0910)
])

# Setup
backend = AerSimulator()
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = 4096

ansatz = EfficientSU2(2, reps=2, entanglement="linear")
params = np.random.uniform(-0.3, 0.3, ansatz.num_parameters)

# Optimizer (NumPy mode - auto-delegates to MobiuQCore)
opt = MobiuOptimizer(
    params,
    license_key=LICENSE_KEY,
    method="standard",
    mode="hardware",        # Use hardware mode for noisy backends
    use_soft_algebra=True
)

# VQE loop with SPSA gradient
for step in range(100):
    # SPSA gradient estimation (2 circuit evaluations)
    delta = np.random.choice([-1, 1], size=len(params))
    shift = 0.1
    
    job = estimator.run([
        (ansatz, hamiltonian, params),
        (ansatz, hamiltonian, params + shift * delta),
        (ansatz, hamiltonian, params - shift * delta)
    ])
    results = job.result()
    
    energy = float(results[0].data.evs)
    grad = (float(results[1].data.evs) - float(results[2].data.evs)) / (2 * shift) * delta
    
    # Update params via Mobiu-Q
    params = opt.step(params, grad, energy)
    
    if step % 20 == 0:
        print(f"Step {step}: energy = {energy:.4f}")

opt.end()
print(f"Final energy: {energy:.4f}")  # Should approach -1.85
```

### Combinatorial Optimization (QAOA)

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# MaxCut graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
n_qubits = 4
p = 2  # QAOA layers

def qaoa_circuit(params):
    gammas, betas = params[:p], params[p:]
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for layer in range(p):
        for i, j in edges:
            qc.rzz(2 * gammas[layer], i, j)
        for i in range(n_qubits):
            qc.rx(2 * betas[layer], i)
    qc.measure_all()
    return qc

def evaluate(params, shots=1024):
    qc = qaoa_circuit(params)
    counts = AerSimulator().run(qc, shots=shots).result().get_counts()
    cost = 0
    for bitstring, count in counts.items():
        for i, j in edges:
            if bitstring[-(i+1)] != bitstring[-(j+1)]:
                cost += count
    return -cost / shots  # Negative for minimization

# Optimizer
params = np.random.uniform(-np.pi, np.pi, 2 * p)
opt = MobiuOptimizer(
    params,
    license_key=LICENSE_KEY,
    method="deep",
    mode="simulation"
)

for step in range(100):
    # SPSA gradient
    delta = np.random.choice([-1, 1], size=len(params))
    shift = 0.1
    e_plus = evaluate(params + shift * delta)
    e_minus = evaluate(params - shift * delta)
    energy = evaluate(params)
    grad = (e_plus - e_minus) / (2 * shift) * delta
    
    params = opt.step(params, grad, energy)
    
    if step % 20 == 0:
        print(f"Step {step}: MaxCut = {-energy:.2f}")

opt.end()
print(f"Final MaxCut value: {-energy:.2f}")
```

### Trading / Finance

```python
import torch
import numpy as np
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# Simple trading policy: state → action probabilities
policy = torch.nn.Sequential(
    torch.nn.Linear(20, 64), torch.nn.ReLU(),
    torch.nn.Linear(64, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 3)  # Hold, Buy, Sell
)

base_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="adaptive",
    maximize=True,       # Maximize profit!
    sync_interval=50,
    verbose=True
)

# Training loop
for episode in range(500):
    state = get_market_state()  # Your market data
    log_probs, rewards = [], []
    
    for step in range(episode_length):
        logits = policy(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        
        state, reward = execute_trade(action.item())  # Your trading logic
        rewards.append(reward)
    
    # Policy gradient update
    returns = compute_returns(rewards, gamma=0.99)
    loss = sum(-lp * G for lp, G in zip(log_probs, returns))
    
    opt.zero_grad()
    loss.backward()
    opt.step(sum(rewards))  # Pass episode profit

opt.end()
```

### Stable-Baselines3 (PPO, SAC, etc.)

SB3 calls `optimizer.step()` internally without arguments. Use `set_metric()` to provide the reward:
```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

class MobiuSB3Callback(BaseCallback):
    """Callback that integrates Mobiu-Q with SB3."""
    
    def __init__(self, method="adaptive", use_soft_algebra=True, verbose=0):
        super().__init__(verbose=verbose)
        self.method = method
        self.use_soft_algebra = use_soft_algebra
        self._mobiu = None
        self._ep_returns = []
    
    def _on_training_start(self):
        base_opt = self.model.policy.optimizer
        self._mobiu = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method=self.method,
            use_soft_algebra=self.use_soft_algebra,
            maximize=True,
            sync_interval=50,
            verbose=True
        )
        # Replace SB3's optimizer
        self.model.policy.optimizer = self._mobiu
    
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                self._ep_returns.append(ep_return)
                # Update metric with rolling average
                recent = self._ep_returns[-4:]
                self._mobiu.set_metric(np.mean(recent))
        return True
    
    def _on_training_end(self):
        if self._mobiu:
            self._mobiu.end()


# Usage
env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0)
model.learn(total_timesteps=200_000, callback=MobiuSB3Callback())
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

LICENSE_KEY = "your-license-key-here"

# Using Adam (default, recommended for most cases)
base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")

# Using AdamW (recommended for LLM/Transformers)
base_opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")

# Using SGD with Momentum (recommended for RL)
base_opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive", maximize=True)

# Using NAdam
base_opt = torch.optim.NAdam(model.parameters(), lr=0.0003)
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="deep")
```

**Quantum mode** - Choose your optimizer via the `base_optimizer` parameter:

```python
from mobiu_q import MobiuOptimizer
import numpy as np

LICENSE_KEY = "your-license-key-here"
params = np.random.randn(10)

# Using Adam (default)
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, method="standard")

# Using NAdam
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, method="standard", base_optimizer="NAdam")

# Using AMSGrad
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, method="deep", base_optimizer="AMSGrad")
```

**⚠️ Important:** Optimizer names are **case-sensitive!**

```python
# ✅ Correct
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, base_optimizer="NAdam")

# ❌ Wrong - will fall back to Adam
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, base_optimizer="nadam")
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
LICENSE_KEY = "your-license-key-here"

# PyTorch: If Adam isn't working, try Momentum:
base_opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")

# Quantum: If Adam isn't working, try NAdam:
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, base_optimizer="NAdam", method="adaptive")
```

### 2. Switch Method

| If This Fails | Try This |
|---------------|----------|
| `standard` | `adaptive` |
| `adaptive` | `deep` |
| `deep` | `standard` |

```python
# If standard isn't working for your problem:
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")
```

### 3. Switch Mode (Quantum only)

| If This Fails | Try This |
|---------------|----------|
| `simulation` | `hardware` |

```python
opt = MobiuOptimizer(params, license_key=LICENSE_KEY, method="standard", mode="hardware")
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

# No license key needed - runs locally!
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

## Full Examples

For complete working examples with benchmarking, see the `examples/` folder:

| File | Domain | Description |
|------|--------|-------------|
| `test_lunarlander_hybrid.py` | RL | LunarLander with REINFORCE |
| `test_mujoco_maximize.py` | RL | MuJoCo continuous control |
| `ppo_mobiu_test.py` | RL | PPO from scratch |
| `crypto_trading_benchmark.py` | Trading | Crypto with regime switching |
| `test_fakefez_h2.py` | VQE | H₂ molecule on FakeFez |
| `test_fakefez_lih.py` | VQE | LiH molecule |
| `test_fakefez_qaoa.py` | QAOA | MaxCut optimization |

---

## License

| Tier | API Calls | Price | Get Started |
|------|-----------|-------|-------------|
| Free | 20/month | $0 | [Sign up](https://app.mobiu.ai) |
| Pro | Unlimited | $19/month | [Get one](https://app.mobiu.ai) |

**Note:** MobiuAttention runs locally, no API calls required.

---

## Links

- [PyPI](https://pypi.org/project/mobiu-q/)
- [GitHub](https://github.com/mobiuai/mobiu-q/)

---

## Citation

```bibtex
@software{mobiu_q,
  title={Mobiu-Q: Soft Algebra for Optimization and Attention},
  author={Mobiu Technologies},
  year={2026},
  url={https://mobiu.ai}
}
```