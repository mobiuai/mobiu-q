# Mobiu-Q Configuration Guide
## Verified Benchmark Results & Recommended Settings

**Last Updated:** December 2024  
**Version:** 2.5.5

---

## 📊 Executive Summary

| Domain | Best Improvement | Recommended Config |
|--------|------------------|-------------------|
| **RL Games** | +129.7% | Adam, method=adaptive |
| **RL Robotics** | +118.6% | Adam, method=adaptive |
| **Materials Science** | +98.3% | Default |
| **LoRA Fine-tuning** | +97.6% | Momentum |
| **Drug Discovery** | +80.5% | Task-specific |
| **Quantum VQE** | +53.1% | Adam, mode=hardware |
| **QAOA MaxCut** | +30.5% | SGD, method=deep |
| **LLM Training** | +21.2% PPL | Default |
| **LLM Soft Prompt** | +18.1% | Momentum |

---

## 🎮 Reinforcement Learning

### Configuration
```python
from mobiu_q import MobiuQCore

opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="adaptive",
    mode="simulation",
    base_optimizer="Adam",
    base_lr=0.0003
)
```

### Verified Results

| Environment | Improvement | p-value | Win Rate |
|-------------|-------------|---------|----------|
| **LunarLander-v3** | **+129.7%** | <0.001 | 96.7% |
| **MuJoCo InvertedPendulum** | **+118.6%** | <0.001 | 100% |
| **Crypto Trading** | **+10.9%** profit | <0.005 | 90% |

---

## ⚛️ Quantum Computing

### VQE Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    mode="hardware",
    base_optimizer="Adam"
)
```

### QAOA Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="deep",
    mode="simulation",
    base_optimizer="SGD",
    base_lr=0.1
)
```

### Verified Results

| Test | Improvement | Win Rate | Config |
|------|-------------|----------|--------|
| **FakeFez VQE H2** | **+53.1%** | 100% | Adam, mode=hardware |
| **Noise Robustness** | **+32.5%** | 80% | Momentum, mode=hardware |
| **QAOA MaxCut** | **+30.5%** | - | SGD, method=deep |

### Deep Circuits (depth ≥ 8)
Use **method=deep** for best results (+7-12% improvement)

---

## 🔬 Materials Science

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    mode="simulation"
)
```

### Verified Results

| Task | Improvement | Win Rate |
|------|-------------|----------|
| **Bulk Modulus** | **+98.3%** | 100% |
| **Band Gap** | **+66.8%** | 100% |
| **Formation Energy** | **+26.9%** | 100% |

---

## 💊 Drug Discovery

### ⚠️ Task-Specific Configuration Required

| Task | Improvement | method | base_lr |
|------|-------------|--------|---------|
| **Solubility** | **+80.5%** | deep | 0.0008 |
| **Binding Affinity** | **+54.0%** | adaptive | 0.001 |
| **Toxicity** | **+23.5%** | standard | 0.0005 |

---

## 🤖 LLM Fine-tuning

### Soft Prompt Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    base_optimizer="Momentum",
    base_lr=0.01
)
```

### Verified Results

| Task | Improvement | Config |
|------|-------------|--------|
| **GPT-2 Medium** | **+21.2% PPL** | Adam |
| **Soft Prompt** | **+18.1%** | Momentum |
| **GPT-2 Small** | **+3.9% PPL** | Adam |

---

## 🔧 LoRA Fine-tuning

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    base_optimizer="Momentum",
    base_lr=0.02
)
```

### Verified Results

| Config | Improvement | Win Rate |
|--------|-------------|----------|
| **r16-Momentum** | **+97.6%** | 100% |
| **r8-HighLR** | **+92.6%** | 100% |
| **r32-SGD** | **+90.2%** | 100% |

---

## 📋 Quick Reference

| Domain | method | mode | base_optimizer | base_lr |
|--------|--------|------|----------------|---------|
| **RL** | adaptive | simulation | Adam | 0.0003 |
| **VQE Clean** | standard | simulation | Adam | 0.01 |
| **VQE Noisy** | standard | hardware | Adam | (auto) |
| **QAOA** | deep | simulation | SGD | 0.1 |
| **Materials** | standard | simulation | Adam | 0.01 |
| **Drug - Solubility** | deep | simulation | Adam | 0.0008 |
| **Drug - Binding** | adaptive | simulation | Adam | 0.001 |
| **Drug - Toxicity** | standard | simulation | Adam | 0.0005 |
| **LLM Soft Prompt** | standard | simulation | Momentum | 0.01 |
| **LLM Training** | standard | simulation | Adam | 5e-5 |
| **LoRA** | standard | simulation | Momentum | 0.02 |

---

## 🔗 API Reference

### Optimizers
- `Adam` - Default, most cases
- `Momentum` - LLM/LoRA
- `SGD` - QAOA/deep circuits
- `NAdam`, `AMSGrad` - Alternatives

### Methods
- `standard` - VQE, LLM, Materials, most cases
- `deep` - QAOA, combinatorial optimization, deep circuits
- `adaptive` - RL, high-variance environments

### Modes
- `simulation` - Clean environments
- `hardware` - Noisy/real hardware

---

*All results verified with p < 0.05*