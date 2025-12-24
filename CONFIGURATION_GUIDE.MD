# Mobiu-Q Configuration Guide

## Complete Benchmark Results & Recommended Configurations

This document summarizes all validated benchmark results and provides the recommended configuration for each use case.

---

## 🏆 Quick Reference Table

| Domain | Method | Optimizer | Best Result | Win Rate | Recommended LR |
|--------|--------|-----------|-------------|----------|----------------|
| **Materials - Bulk Modulus** | standard | Adam | +98.3% | 100% | 0.001 |
| **Materials - Band Gap** | standard | Adam | +66.8% | 100% | 0.001 |
| **LoRA r16** | adaptive | Momentum | +97.6% | 100% | 0.01-0.03 |
| **LoRA r32** | adaptive | SGD | +90.2% | 100% | 0.01-0.03 |
| **RL - LunarLander** | adaptive | Momentum | +129.7% | 96.7% | server default |
| **RL - MuJoCo** | adaptive | Momentum | +118.6% | 100% | server default |
| **RL - Crypto Trading** | adaptive | Momentum | +10.9% | 90% | server default |
| **VQE - XY Model** | standard | Adam | +60.8% | - | 0.01 |
| **VQE - He Atom** | standard | Adam | +51.2% | - | 0.01 |
| **VQE - H2 Molecule** | standard | Adam | +46.6% | - | 0.01 |
| **VQE - H3+ Chain** | standard | Adam | +42.0% | - | 0.01 |
| **VQE - LiH Molecule** | standard | Adam | +41.4% | - | 0.01 |
| **VQE - BeH2 Molecule** | standard | Adam | +37.8% | - | 0.01 |
| **LLM Full Fine-tuning** | adaptive | Momentum | +23.3% | 100% | server default |
| **LLM GPT-2 (Perplexity)** | adaptive | Momentum | +21.2% | 100% | server default |
| **LLM Soft Prompt** | adaptive | Momentum | +18.1% | 100% | server default |
| **QAOA MaxCut 4q** | deep | NAdam | +27.2% | - | 0.1 |
| **QAOA MaxCut 5q** | deep | NAdam | +23.7% | - | 0.1 |
| **Noise Robustness** | standard | Momentum | +32.5% | 80% | 0.02 |
| **IBM FakeFez VQE** | standard | Adam | +50.9% | - | 0.02 |

---

## 📊 Detailed Results by Domain

### 🔬 Materials Science

Best overall results. Use `method="standard"` with `Adam`.

| Task | Improvement | p-value | Win Rate | Configuration |
|------|-------------|---------|----------|---------------|
| **Bulk Modulus (GPa)** | **+98.3%** | <0.001 | 100% | `method="standard", base_optimizer="Adam", lr=0.001` |
| **Band Gap (eV)** | **+66.8%** | <0.001 | 100% | `method="standard", base_optimizer="Adam", lr=0.001` |
| **Formation Energy** | **+26.9%** | <0.001 | 100% | `method="standard", base_optimizer="Adam", lr=0.0008` |

```python
# Recommended for Materials Science
opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="standard",
    base_optimizer="Adam"
)
```

---

### 🤖 LLM & LoRA Fine-tuning

Use `method="adaptive"` with `Momentum` (not Adam or SGD!).

| Task | Improvement | p-value | Win Rate | Configuration |
|------|-------------|---------|----------|---------------|
| **LoRA r16** | **+97.6%** | <0.001 | 100% | `method="adaptive", base_optimizer="Momentum"` |
| **LoRA r32** | **+90.2%** | <0.001 | 100% | `method="adaptive", base_optimizer="SGD"` |
| **Full Fine-tuning** | **+23.3%** | 0.002 | 100% | `method="adaptive", base_optimizer="Momentum"` |
| **GPT-2 Medium PPL** | **+21.2%** | <0.001 | 100% | `method="adaptive", base_optimizer="Momentum"` |
| **Soft Prompt Tuning** | **+18.1%** | <0.05 | 100% | `method="adaptive", base_optimizer="Momentum"` |

⚠️ **Known Issues:**
- DistilBERT: -31.6% (use different architecture or Adam)
- GPT-2 Low Batch (batch=8): -28.6% (increase batch size to 16+)

```python
# Recommended for LoRA
opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="adaptive",
    base_optimizer="Momentum"
)
```

---

### 🎮 Reinforcement Learning

Excellent results. Use `method="adaptive"` with `Momentum`.

| Environment | Improvement | p-value | Win Rate | Configuration |
|-------------|-------------|---------|----------|---------------|
| **LunarLander-v3** | **+129.7%** | <0.001 | 96.7% | `method="adaptive", base_optimizer="Momentum"` |
| **MuJoCo InvertedPendulum** | **+118.6%** | 0.001 | 100% | `method="adaptive", base_optimizer="Momentum"` |
| **Crypto Trading** | **+10.9%** | 0.005 | 90% | `method="adaptive", base_optimizer="Momentum"` |

❌ **Not Recommended:**
- Simple Trading: +2.3%, p=0.615 (not significant)
- CartPole: too easy, no advantage

```python
# Recommended for RL
opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="adaptive",
    base_optimizer="Momentum"
)
```

---

### 🧪 Quantum Chemistry (VQE)

Consistent improvements. Use `method="standard"` with `Adam`.

| Problem | Improvement | Configuration |
|---------|-------------|---------------|
| **XY Model** | **+60.8%** | `method="standard", base_optimizer="Adam", lr=0.01` |
| **He Atom** | **+51.2%** | `method="standard", base_optimizer="Adam", lr=0.01` |
| **H2 Molecule** | **+46.6%** | `method="standard", base_optimizer="Adam", lr=0.01` |
| **H3+ Chain** | **+42.0%** | `method="standard", base_optimizer="Adam", lr=0.01` |
| **LiH Molecule** | **+41.4%** | `method="standard", base_optimizer="Adam", lr=0.01` |
| **BeH2 Molecule** | **+37.8%** | `method="standard", base_optimizer="Adam", lr=0.01` |

**Historical 100-seed validation:**
- Average improvement vs Adam: +43.88%
- H2: +53.7%, Heisenberg: +28%

```python
# Recommended for VQE
from mobiu_q import MobiuQCore, Demeasurement

opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="standard"
    # base_optimizer defaults to Adam
)

for step in range(100):
    grad = Demeasurement.finite_difference(energy_fn, params)
    params = opt.step(params, grad, energy_fn(params))
```

---

### 🎯 QAOA (Combinatorial Optimization)

Use `method="deep"` with `NAdam` or `Adam`.

| Problem | Improvement | p-value | Configuration |
|---------|-------------|---------|---------------|
| **MaxCut 4-qubit** | **+27.2%** | 0.04 | `method="deep", base_optimizer="NAdam", lr=0.1` |
| **MaxCut 5-qubit** | **+23.7%** | 0.004 | `method="deep", base_optimizer="NAdam", lr=0.1` |
| **MaxCut p=3** | **+15.6%** | 0.008 | `method="deep", base_optimizer="NAdam", lr=0.1` |

**Optimizer comparison on QAOA:**
| Optimizer | SA Effect |
|-----------|-----------|
| SGD | +30.5% ✅ |
| LAMB | +14.4% ✅ |
| Adam | +6.0% ✅ |
| AMSGrad | +4.1% |
| Momentum | -0.8% ❌ |
| NAdam | -2.1% (but best baseline) |

```python
# Recommended for QAOA
opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="deep",
    base_optimizer="NAdam",
    mode="simulation"
)
```

---

### 🛡️ Noise Robustness (Quantum Hardware)

Mobiu-Q **shines under noise**. Use `mode="hardware"`.

| Metric | Result | Configuration |
|--------|--------|---------------|
| **Robustness Advantage** | **+32.5%** | `mode="hardware"` |
| **Win Rate (all noise)** | **80%** (12/15) | `mode="hardware"` |
| **IBM FakeFez VQE** | **+50.9%** (p=0.03) | `method="standard", mode="hardware"` |

**Key Insight:** Adam diverges under noise, Mobiu-Q stays stable.
- Adam on IBM Fez: diverged to -1.68 Ha (noise hallucination)
- Mobiu-Q on IBM Fez: stable at -1.176 Ha (ground state)
- Result: **93.3% improvement** at convergence

```python
# Recommended for noisy hardware
opt = MobiuQCore(
    license_key="YOUR-KEY",
    method="standard",
    mode="hardware",
    base_optimizer="Adam"
)
```

---

## 🔧 Default Learning Rates (Server)

The server automatically selects optimal learning rates:

| Method | Mode | Default LR |
|--------|------|------------|
| standard | simulation | 0.01 |
| standard | hardware | 0.02 |
| deep | any | 0.1 |
| adaptive | any | 0.0003 |

---

## ⚠️ Known Limitations

### ❌ Domains Where SA Struggled

| Domain | Result | Reason | Workaround |
|--------|--------|--------|------------|
| **Drug Discovery** | -28% to -515% | BCE loss instability | Try `method="standard"`, lower LR |
| **DistilBERT** | -31.6% | Classification vs regression | Use Adam, not Momentum |
| **GPT-2 Low Batch** | -28.6% | High variance | Increase batch to 16+ |
| **Simple Trading** | +2.3% (ns) | Too simple | Use on complex strategies |
| **Multimodal extremes** | ~0% | Rastrigin/Ackley | Not suitable |

### ⚠️ Important Notes

1. **Optimizer names are case-sensitive!**
   - ✅ `"Adam"`, `"Momentum"`, `"NAdam"`
   - ❌ `"adam"`, `"momentum"`, `"nadam"`

2. **Method matters:**
   - `standard` = VQE, Chemistry, Materials (smooth landscapes)
   - `deep` = QAOA, noisy hardware (rugged landscapes)
   - `adaptive` = RL, LLM, LoRA (high variance)

3. **Noise helps Mobiu-Q:**
   - SA advantage is stronger under noise
   - Use `mode="hardware"` for noisy quantum systems

---

## 📋 Configuration Decision Tree

```
What problem are you solving?
│
├─► Quantum (VQE/Chemistry)
│   └─► method="standard", base_optimizer="Adam"
│       ├─► Clean simulation: mode="simulation", lr=0.01
│       └─► Noisy hardware: mode="hardware", lr=0.02
│
├─► Quantum (QAOA/MaxCut)
│   └─► method="deep", base_optimizer="NAdam", lr=0.1
│
├─► Reinforcement Learning
│   └─► method="adaptive", base_optimizer="Momentum"
│
├─► LLM/LoRA Fine-tuning
│   └─► method="adaptive", base_optimizer="Momentum"
│       ⚠️ Batch size >= 16 recommended
│
├─► Materials Science
│   └─► method="standard", base_optimizer="Adam", lr=0.001
│
└─► Other / Unknown
    └─► Start with: method="adaptive", base_optimizer="Adam"
        If fails: try Momentum, then try different method
```

---

## 📊 Summary Statistics

| Category | Tests | Significant Wins | Win Rate | Avg Improvement |
|----------|-------|------------------|----------|-----------------|
| Materials Science | 3 | 3 | 100% | +64% |
| LoRA/LLM | 5 | 5 | 100% | +50% |
| RL | 4 | 3 | 75% | +68% |
| VQE | 6 | 6 | 100% | +47% |
| QAOA | 3 | 3 | 100% | +22% |
| Noise | 3 | 2 | 67% | +38% |
| **Overall** | **24** | **22** | **92%** | **~50%** |

---

## 🔗 Links

- **PyPI:** [pypi.org/project/mobiu-q](https://pypi.org/project/mobiu-q)
- **App:** [app.mobiu.ai](https://app.mobiu.ai)
- **Website:** [mobiu.ai](https://mobiu.ai)

---

*Last updated: December 2025*
*Version: 2.5*