# Mobiu-Q (v2.1)

**Universal Physics-Aware Optimizer for Stochastic Systems**

[![PyPI version](https://badge.fury.io/py/mobiu-q.svg)](https://badge.fury.io/py/mobiu-q)
[![Win Rate](https://img.shields.io/badge/Win%20Rate-99.3%25-brightgreen)](https://mobiu.ai)
[![License](https://img.shields.io/badge/License-Proprietary-blue)](https://mobiu.ai)

**Mobiu-Q** is the first optimizer based on **Soft Algebra**. By mathematically decomposing gradients into *Potential* ($a_t$) and *Realization* ($b_t$), it filters out noise in real-time.

Originally designed for Quantum Computing, Mobiu-Q v2.0 is now validated for **FinTech Risk Modeling**, **Reinforcement Learning**, and **Complex Engineering** problems.

---

## 🚀 The Core Innovation: "Noise Hallucination" Prevention

Standard optimizers (Adam, SGD) assume that lower objective values always indicate better solutions. In noisy environments—like NISQ processors or stochastic financial markets—this fails. Optimizers "tunnel" into noise, creating **Noise Hallucinations** (non-physical solutions).

**The Solution:**
Mobiu-Q utilizes a cross-coupled state evolution law:
```math
S_{t+1} = (\gamma \cdot S_t) \cdot \Delta_t + \Delta_t

```

This ensures that a parameter update is only committed if the *Potential Field* () is validated by a *Realized Improvement* ().

---

## 🏆 Universal Benchmarks (v2.0)

Validated across **24 distinct problem domains** with 1,000+ random seeds.

| Domain | Problem | Improvement (vs Adam) | Significance |
| --- | --- | --- | --- |
| **💰 Finance** | **Credit Risk (VaR)** | **+52.3%** | Superior stability in high-volatility noise |
| **💰 Finance** | **Portfolio Opt** | **+51.7%** | Better Sharpe ratio convergence |
| **🤖 AI / RL** | **LunarLander** | **+129.7%** | 96% Win rate vs Adam's crashing |
| **📐 Classical** | **Rosenbrock Valley** | **+75.8%** | Navigates narrow, curved valleys |
| **⚛️ Quantum** | **H2 Molecule** | **+49.1%** | Chemical accuracy in noisy simulations |
| **🌀 Topology** | **SSH Model** | **+61.0%** | Identifies topological phases |
| **🕸️ Graph** | **MaxCut (QAOA)** | **+21.5%** | Escapes local minima via  |

### Hardware Verification (IBM Fez)

Tested on IBM's 127-qubit *Fez* processor:

* **Adam:** Diverged to -1.68 Ha (Noise Hallucination).
* **Mobiu-Q:** Stabilized exactly at the physical ground state (-1.176 Ha).

---

## 📦 Installation

```bash
pip install mobiu-q

```

---

## ⚡ Quick Start

### 1. Universal Stochastic Optimization (Finance / AI)

For noisy classical problems (Credit Risk, RL, Engineering).

```python
from mobiu_q import MobiuQCore, Demeasurement

# Initialize Cloud Optimizer
opt = MobiuQCore(
    license_key="YOUR-KEY", 
    problem="vqe"  # 'vqe' logic (Chemistry/Finance)
)

# Optimization Loop
for step in range(100):
    # 1. Calculate Gradient (Local)
    grad = Demeasurement.finite_difference(energy_fn, params)
    
    # 2. Step with Cloud Brain
    # Note: Pass energy for curvature analysis
    params = opt.step(params, grad, energy_fn(params))

# End Session
opt.end()

```

### 2. Rugged Landscapes (Combinatorial / QAOA)

For problems with many local minima (MaxCut, Rastrigin, Ackley).

```python
opt = MobiuQCore(
    license_key="YOUR-KEY",
    problem="qaoa",  # Activates Super-Equation logic
    mode="noisy"     # Handles hardware noise
)

for step in range(150):
    # SPSA is efficient for QAOA (2 evals only)
    grad, energy = Demeasurement.spsa(energy_fn, params)
    params = opt.step(params, grad, energy)

opt.end()

```

### 3. Multi-Seed Experiments

To run multiple seeds as a single billing session:

```python
opt = MobiuQCore(license_key="YOUR-KEY")

for seed in range(10):
    opt.new_run()  # Resets state, keeps session open
    params = init_params(seed)
    # ... optimization loop ...

opt.end() # Counts as 1 run

```

---

## 🔑 Pricing

* **Free Tier:** For students & testing (Limited runs).
* **Pro Tier:** Unlimited runs, Priority Processing, FinTech/AI models.

**[Get your License Key](https://app.mobiu.ai)**

---

*Proprietary technology. All rights reserved by Mobiu Technologies.*