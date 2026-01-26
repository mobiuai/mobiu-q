# Mobiu-Q v4.0.1

**Soft Algebra for Optimization & Attention**

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://pypi.org/project/mobiu-q/)
[![License](https://img.shields.io/badge/license-Proprietary-blue.svg)](LICENSE)

---

## Overview

Mobiu-Q is a framework built on **Soft Algebra** (nilpotent ε²=0) that provides:

1. **MobiuOptimizer** - Stable optimization in noisy environments
2. **MobiuAttention** 🧪 - O(N) linear attention for long sequences
3. **MobiuSignal** 🆕 - Trading signals using Soft Algebra
4. **MobiuAD** - Streaming anomaly detection        
5. **TrainGuard** - Safe ML training monitor      

---

## Installation

```bash
pip install mobiu-q
```

---

## Quick Start

### MobiuOptimizer (Stable API)

```python
from mobiu_q import MobiuOptimizer, MobiuAD, TrainGuard
import torch

# Your license key (get one at https://app.mobiu.ai)
LICENSE_KEY = "your-license-key-here"

# Wrap any PyTorch optimizer
model = MyModel()
base_opt = torch.optim.Adam(model.parameters())
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="adaptive",  # LR auto-set to 0.0003
    use_soft_algebra=True
)

# Or with explicit LR:
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="standard",
    base_lr=0.001  # Override default LR
)

for batch in dataloader:
    loss = criterion(model(batch))
    loss.backward()
    opt.step(loss.item())  # Pass loss for Soft Algebra

opt.end()  # Important: release resources
```

### Monitoring Training
```python
opt = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method="adaptive")
# ... training ...

# Track metrics
print(opt.lr_history)    # Learning rates over time
print(opt.warp_history)  # Gradient warp factors (new in v3.1.3)
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

### MobiuAD (🆕 NEW in v3.9.0)
```python
from mobiu_q import MobiuAD, TrainGuard

LICENSE_KEY = "your-license-key-here"

# Streaming anomaly detection
detector = MobiuAD(license_key=LICENSE_KEY)
result = detector.detect(value)

# Training monitor (Q + AD combined)
guard = TrainGuard(license_key=LICENSE_KEY)
result = guard.step(loss, gradient, val_loss)
```

### MobiuSignal (🆕 NEW in v3.10.0)
```python
from mobiu_q.signal import MobiuSignal

# No license required - runs locally!
signal = MobiuSignal(lookback=20)

# Compute signal from prices
result = signal.compute(prices)
if result.is_strong:
    print(f"Strong {'📈' if result.is_bullish else '📉'} signal: {result.magnitude:.2f}")

# Backtest on historical data
backtest = signal.backtest(historical_prices, future_window=5)
print(f"Correlation: {backtest.correlation:.3f}")
print(f"Q4/Q1 Ratio: {backtest.q4_q1_ratio:.2f}x")
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

**Note:** Default LR is auto-applied in PyTorch mode. Override with `base_lr=...` if needed.

### Benchmarks

#### Reinforcement Learning & Trading

| Domain                  | Improvement | Win Rate | p-value |
|-------------------------|-------------|----------|---------|
| Crypto Trading          | **+56%** profit | 100% | <0.001  |
| LunarLander-v3          | +128%       | 97%      | <0.001  |
| MuJoCo InvertedPendulum | +111%       | 100%     | <0.001  |
| RL Trading (MobiuSignal) | **+168%** | 83% | <0.001 |

#### Quantum Computing

| Domain                  | Improvement | Win Rate | p-value |
|-------------------------|-------------|----------|---------|
| VQE H₂ (FakeFez)        | +52%        | 100%     | <0.001  |
| QAOA MaxCut             | +45%        | 95%      | <0.001  |

#### Noisy & Distributed Learning 🆕

These domains have **systematic gradient bias** - exactly where Soft Algebra excels:

| Domain              | Improvement | Win Rate | p-value | Bias Source |
|---------------------|-------------|----------|---------|-------------|
| Federated Learning  | **+67%**    | 100%     | <0.001  | Non-IID client data |
| Imbalanced Data     | **+52%**    | 100%     | <0.001  | Majority class dominates |
| Sim-to-Real         | **+47%**    | 100%     | <0.001  | Simulator ≠ reality |
| Noisy Labels        | **+40%**    | 100%     | <0.001  | Systematic mislabeling |

*All tests: 10 seeds, same energy & gradient for both, only `use_soft_algebra` differs*

### Why Soft Algebra Works Here

In these domains, the **gradient is systematically biased**:
- Federated: Each client sees different data distribution
- Imbalanced: Gradient dominated by majority class
- Sim-to-Real: Simulator has wrong physics parameters
- Noisy Labels: Labels consistently confused (e.g., 3↔8)

Soft Algebra detects the gap between gradient direction and actual loss improvement, then corrects for it.

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

### Federated Learning 🆕

```python
import numpy as np
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# Simulate federated aggregation with non-IID clients
class FederatedTrainer:
    def __init__(self, n_clients=10, non_iid_strength=0.5):
        self.n_clients = n_clients
        self.non_iid = non_iid_strength
        # Each client has biased local data
        self.client_biases = [np.random.randn(dim) * non_iid_strength 
                             for _ in range(n_clients)]
    
    def aggregate_gradients(self, params, sampled_clients):
        """Aggregate gradients from subset of clients (FedAvg style)"""
        grads = []
        for c in sampled_clients:
            # Each client's gradient is biased by their local data
            local_grad = compute_gradient(params) + self.client_biases[c]
            grads.append(local_grad)
        return np.mean(grads, axis=0)

# Mobiu-Q handles the systematic bias from non-IID aggregation
params = np.random.randn(100)
opt = MobiuOptimizer(
    params,
    license_key=LICENSE_KEY,
    method="standard",
    base_lr=0.01
)

for round in range(100):
    # Sample random clients (realistic FL scenario)
    clients = np.random.choice(n_clients, size=5, replace=False)
    gradient = trainer.aggregate_gradients(params, clients)
    loss = compute_global_loss(params)
    
    params = opt.step(params, gradient, loss)

opt.end()
```

### Imbalanced Data Classification 🆕

```python
import torch
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# Dataset with 90% class 0, 10% class 1 (fraud detection, medical diagnosis)
train_loader = create_imbalanced_loader(imbalance_ratio=0.9)

model = FraudDetector()
base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="standard"
)

for batch in train_loader:
    # Gradient dominated by majority class
    loss = criterion(model(batch))
    loss.backward()
    
    # Soft Algebra corrects for class imbalance bias
    opt.step(loss.item())

opt.end()
```

### Sim-to-Real Robotics 🆕

```python
import torch
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# Policy trained in simulator, deployed in real world
policy = RobotPolicy()
base_opt = torch.optim.Adam(policy.parameters(), lr=0.0003)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="adaptive",
    maximize=True
)

for episode in range(1000):
    # Gradient from SIMULATOR (biased - wrong friction, mass, etc.)
    sim_loss = run_simulator_episode(policy)
    sim_loss.backward()
    
    # Periodically evaluate in REAL environment
    if episode % 10 == 0:
        real_reward = run_real_episode(policy)
    
    # Soft Algebra uses real reward to correct simulator bias
    opt.step(real_reward)

opt.end()
```

### Noisy Labels 🆕

```python
import torch
from mobiu_q import MobiuOptimizer

LICENSE_KEY = "your-license-key-here"

# Dataset with systematic label noise (crowdsourced, OCR errors)
# e.g., "3" often mislabeled as "8", "cat" confused with "dog"
train_loader = create_noisy_label_loader(noise_rate=0.3)

model = Classifier()
base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="standard"
)

for batch_x, noisy_labels in train_loader:
    # Gradient points toward WRONG targets due to label noise
    loss = criterion(model(batch_x), noisy_labels)
    loss.backward()
    
    # Validate on clean held-out set
    clean_loss = evaluate_clean(model)
    
    # Soft Algebra detects mismatch and corrects
    opt.step(clean_loss)

opt.end()
```

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

**Tip:** For even better results, use MobiuSignal features as your state representation. 
See the MobiuSignal section for integration example.

---

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

## MobiuSignal 🆕

Trading signal generator using the same Soft Algebra potential/realized framework.

### Validated Results (3,080 days BTC/USDT)

| Metric | Result |
|--------|--------|
| Spearman correlation | **+0.222** (p<0.0001) |
| Q4/Q1 ratio | **1.83x** larger moves |
| Precision lift | **1.18x** vs random |

### Mathematical Framework
```
Potential (aₜ) = σₜ/μₜ × scale    # Normalized volatility
Realized (bₜ)  = (Pₜ - Pₜ₋₁)/Pₜ₋₁  # Price change
Magnitude      = √(aₜ² + bₜ²)      # Signal strength
```

### Usage
```python
from mobiu_q.signal import MobiuSignal, SignalResult

# Basic usage
signal = MobiuSignal(lookback=20, vol_scale=100)
result = signal.compute(prices)

print(f"Potential: {result.potential:.3f}")
print(f"Realized: {result.realized:.3f}%")
print(f"Magnitude: {result.magnitude:.3f}")
print(f"Direction: {result.direction}")  # +1, -1, or 0
print(f"Quartile: Q{result.quartile}")   # 1-4 (4=strongest)

# Check signal strength
if result.is_strong:  # Top quartile
    if result.is_bullish:
        print("🚀 Strong bullish signal!")
    else:
        print("🔻 Strong bearish signal!")
```

### Streaming Mode
```python
signal = MobiuSignal(lookback=20)

for price in live_price_stream:
    result = signal.update(price)
    if result and result.is_strong:
        execute_trade(result.direction)
```

### Backtesting
```python
from mobiu_q.signal import MobiuSignal, backtest_signal

# Full backtest
signal = MobiuSignal(lookback=20)
result = signal.backtest(prices, future_window=5)

print(f"Total signals: {result.total_signals}")
print(f"Strong signals: {result.strong_signals}")
print(f"Correlation: {result.correlation:.3f} (p={result.correlation_pvalue:.4f})")
print(f"Q4/Q1 Ratio: {result.q4_q1_ratio:.2f}x")
print(f"Precision Lift: {result.precision_lift:.2f}x")

# Quick backtest
result = backtest_signal(prices, lookback=20, future_window=5)
```

### Series Analysis
```python
signal = MobiuSignal(lookback=20)
results = signal.compute_series(prices)

# Find all strong signals
strong = [r for r in results if r.is_strong]
print(f"Found {len(strong)} strong signals out of {len(results)}")
```

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Day trading | `lookback=10-20` |
| Swing trading | `lookback=20-50` |
| High volatility | Lower `vol_scale` |
| Low volatility | Higher `vol_scale` |

**Note:** MobiuSignal runs **100% locally** - no API calls, no license key needed.

---

### Integration with MobiuOptimizer (RL Trading)

Combine MobiuSignal features with MobiuOptimizer for RL-based trading:
```python
from mobiu_q import MobiuOptimizer
from mobiu_q.signal import MobiuSignal
import torch
import torch.nn as nn

LICENSE_KEY = "your-license-key-here"

# Policy uses MobiuSignal features as state
class TradingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)  # Hold, Buy, Sell
        )
    
    def forward(self, signal_features):
        # signal_features: [potential, realized, magnitude, position, pnl]
        return self.net(signal_features)

# Setup
signal = MobiuSignal(lookback=20)
policy = TradingPolicy()
base_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
opt = MobiuOptimizer(
    base_opt,
    license_key=LICENSE_KEY,
    method="adaptive",
    maximize=True,  # Maximize profit!
    use_soft_algebra=True
)

# Training loop
for episode in range(500):
    signal.reset()
    # ... collect episode using signal.update(price) for state ...
    # ... REINFORCE update ...
    opt.step(episode_profit)

opt.end()
```

**Validated Results (30 seeds, 500 episodes, regime switching):**

| Metric | Adam | Mobiu | Δ |
|--------|------|-------|---|
| Final PnL | -4.7 | +3.2 | **+168.7%** |
| Win Rate | - | 83.3% | 25/30 seeds |
| p-value | - | - | 0.000012 |

*Adam lost money on average; Mobiu turned it profitable.*

See `examples/rl_trading_mobiu_benchmark.py` for full implementation.

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
| Federated Learning | `Adam` |
| Imbalanced Data | `Adam` |
| Sim-to-Real | `Adam` + `adaptive` |
| Noisy Labels | `Adam` |

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
| **Federated** | Non-IID variance | Use `Adam` + `standard` + LR=0.01 |
| **Imbalanced** | Majority bias | Use `Adam` + `standard` + LR=0.01 |

---

## MobiuAttention 🧪

### Why?

Standard Transformer attention is O(N²) in sequence length. MobiuAttention is **O(N)**.

| Seq Length | Transformer | MobiuAttention | Speedup | Memory Saving |
|------------|-------------|----------------|---------|---------------|
| 4,096      | 16.9ms      | 16.4ms         | ~1x     | ~Equal        |
| 8,192      | 75.3ms      | **33.8ms**     | **2.2x** ✅ | **50%** ✅   |
| 16,384     | **OOM** 💥  | Works          | ∞       | ∞             |

*Tested on T4 GPU, batch=2, d_model=128*

### Quality (Same as Transformer)

| Benchmark            | Transformer | MobiuAttention |
|----------------------|-------------|----------------|
| Shakespeare PPL      | 12.3        | **12.2** ✅            |
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

### When to Use MobiuAttention

| Scenario | Recommendation |
|----------|----------------|
| **GPU + seq > 4K** | ✅ **Use MobiuAttention** - 2x faster, 50% less memory |
| **GPU + seq < 4K** | Standard Transformer is fine |
| **Apple Silicon (MPS)** | Equal quality, Transformer slightly faster |
| **Very long context (>8K)** | ✅ **MobiuAttention only option** - Transformer OOMs |

**Key insight:** MobiuAttention maintains the same quality (PPL) while enabling 
longer contexts that would otherwise cause out-of-memory errors.

**Note:** MobiuAttention runs **100% locally** - no API calls, no license key needed.
It's included in the `mobiu-q` package but operates independently of the cloud service.

### ⚠️ Combining MobiuAttention with MobiuOptimizer

Based on our testing, **combining both products may cause interference**:

| Configuration | PPL | Result |
|---------------|-----|--------|
| Standard + Adam | 11.86 | Baseline |
| Standard + MobiuOptimizer | **3.85** | ✅ Best! (+67.5%) |
| MobiuAttention + Adam | 11.05 | Good (+6.8%) |
| MobiuAttention + MobiuOptimizer | 9.84 | ⚠️ Interference (+17%) |

**Recommendation:**
- For **best quality**: Use MobiuOptimizer alone (Standard Attention)
- For **long context (>4K tokens)**: Use MobiuAttention alone
- **Don't combine** unless you've tested on your specific use case

### Verified Performance (Fair A/B Test)

**Shakespeare Language Modeling:**
| Configuration | PPL | Improvement |
|---------------|-----|-------------|
| Adam (baseline) | 11.86 | - |
| **MobiuOptimizer (adaptive)** | **3.85** | **+67.5%** ✅ |

*Same API, same hyperparameters, only `use_soft_algebra` differs*

**Key Insight:** Soft Algebra provides real value - not just adaptive LR!

### ⚠️ Experimental Status

- Functional and tested
- API may change in future versions
- Feedback welcome!

---

## 🛡️ Anomaly Detection (NEW in v3.9.0)

Mobiu-Q now includes **anomaly detection** using the same Soft Algebra mathematics as the optimizer.

### MobiuAD - Streaming Detector

Detect anomalies in real-time data streams:

```python
from mobiu_q import MobiuAD

LICENSE_KEY = "your-license-key-here"

detector = MobiuAD(license_key=LICENSE_KEY, method="deep")

for value in data_stream:
    result = detector.detect(value)
    if result.is_anomaly:
        print(f"⚠️ Anomaly! Δ†={result.delta_dagger:.4f}")
```

### Detection Methods

| Method | Use Case | Best For |
|--------|----------|----------|
| `standard` | Trust Ratio based | Simple anomalies |
| `deep` | Super-Equation Δ† | Behavioral changes (recommended) |
| `transition` | Regime detection | Pattern shifts |

### TrainGuard - Safe ML Training

Combines MobiuOptimizer + MobiuAD to monitor training health:

```python
from mobiu_q import TrainGuard

LICENSE_KEY = "your-license-key-here"

guard = TrainGuard(license_key=LICENSE_KEY)

for epoch in range(100):
    loss = train_epoch(model)
    val_loss = evaluate(model)
    gradient = get_gradient_norm(model)
    
    result = guard.step(
        loss=loss,
        gradient=gradient,
        val_loss=val_loss
    )
    
    # Apply optimized gradient
    apply_gradient(result.adjusted_gradient)
    
    # Check for training issues
    if result.alert:
        if result.alert_type == 'GRADIENT_EXPLOSION':
            print("💥 Gradient explosion detected!")
            reduce_lr()
        elif result.alert_type == 'OVERFITTING':
            print("📈 Overfitting detected!")
            apply_regularization()
        elif result.alert_type == 'LOSS_SPIKE':
            print("⚡ Loss spike detected!")
            restore_checkpoint()

guard.end()
```

### TrainGuard Alerts

| Alert Type | Trigger | Suggested Action |
|------------|---------|------------------|
| `GRADIENT_EXPLOSION` | Gradient norm spike | Reduce LR, clip gradients |
| `OVERFITTING` | Val loss ↑ while train loss ↓ | Early stop, regularize |
| `LOSS_SPIKE` | Sudden loss increase | Restore checkpoint |
| `PLATEAU` | No improvement | Increase LR, change optimizer |

### Batch Detection

```python
from mobiu_q import MobiuAD

LICENSE_KEY = "your-license-key-here"

detector = MobiuAD(license_key=LICENSE_KEY)

# Detect anomalies in batch
results = detector.detect_batch(data_array)
print(f"Found {results.total_anomalies} anomalies at indices: {results.anomaly_indices}")
```

### MobiuAD vs PyOD

| Feature | MobiuAD | PyOD |
|---------|---------|------|
| **Type** | Streaming | Batch |
| **Detects** | Behavioral changes | Statistical outliers |
| **Real-time** | ✅ Yes | ❌ No |
| **Early warning** | ✅ Yes | ❌ No |
| **Pattern changes** | ✅ Excellent | ⚠️ Limited |
| **Value outliers** | ⚠️ Good | ✅ Excellent |

**Use MobiuAD when:**
- Real-time streaming detection needed
- Detecting behavioral/pattern changes
- Early warning before anomalies manifest
- Monitoring ML training (TrainGuard)

**Use PyOD when:**
- Batch analysis of static data
- Detecting statistical outliers
- Value-based anomaly detection

### Enhanced Detection (PyOD + Soft Algebra)

Combine both approaches for maximum coverage:

```python
from mobiu_q.detection import MobiuADEnhanced

# Requires: pip install pyod
detector = MobiuADEnhanced(base_detector="IForest")
results = detector.detect_batch(data)
```

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

### In Anomaly Detection

```python
Δ† = |a_t + i·b_t|  # Super-Equation score
```

Soft Algebra tracks behavioral patterns and detects deviations.

### In Signal Generation
```python
a_t = volatility / mean_price    # Potential: what COULD happen
b_t = price_change               # Realized: what DID happen
signal = √(a² + b²)              # Interaction magnitude
```

Soft Algebra detects when high potential meets significant realization.

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
| `test_federated_fair.py` | FL | Federated learning benchmark |
| `test_noisy_labels_fair.py` | Noisy | Noisy labels benchmark |
| `test_sim_to_real_fair.py` | Robotics | Sim-to-real benchmark |
| `test_imbalanced_fair.py` | Classification | Imbalanced data benchmark |
| `test_mobiu_attention_real.py` | Attention | Shakespeare + Code + Scaling benchmarks |
| `test_double_mobiu.py` | Fair A/B Test | Soft Algebra ON vs OFF comparison |
| `test_trainguard.py` | Training | TrainGuard monitoring demo |
| `benchmark_behavioral.py` | Detection | Behavioral anomaly benchmark |
| `example_signal.py` | Trading | MobiuSignal demo |
| `rl_trading_mobiu_benchmark.py` | Trading | RL Trading with MobiuSignal + regime switching |

---

## License

| Tier | API Calls | Price | Get Started |
|------|-----------|-------|-------------|
| Free | 20/month | $0 | [Sign up](https://app.mobiu.ai) |
| Pro | Unlimited | $19/month | [Get one](https://app.mobiu.ai) |

**Note:** MobiuAttention & MobiuSignal run locally, no API calls required.

---

## Links

- [PyPI](https://pypi.org/project/mobiu-q/)
- [GitHub](https://github.com/mobiuai/mobiu-q/)

---

## Citation

```bibtex
@software{mobiu_q,
  title={Mobiu-Q: Soft Algebra for Optimization, Attention and Anomaly Detection},
  author={Mobiu Technologies},
  year={2026},
  url={https://mobiu.ai}
}
```