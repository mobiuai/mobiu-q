# Mobiu-Q Configuration Guide
## Verified Benchmark Results & Recommended Settings

**Last Updated:** December 2024  
**Version:** 2.7.4

---

## 📊 Executive Summary

| Domain | Best Improvement | Recommended Config |
|--------|------------------|-------------------|
| **RL Games** | +129.7% | Adam, adaptive, lr=0.0003 |
| **RL Robotics** | +118.6% | Adam, adaptive, lr=0.0003 |
| **Classical Opt** | +75.8% | Adam, standard, lr=0.01 |
| **Condensed Matter** | +61.0% | Adam, standard, lr=0.01 |
| **Finance** | +52.3% | Adam, standard, lr=0.01 |
| **Quantum VQE** | +52.4% | Adam, standard, mode=hardware |
| **QAOA** | +45.1% | SGD, deep, lr=0.1 |
| **Drug Discovery** | +12.2% | AMSGrad, standard, lr=0.01 |

---

## 🎮 Reinforcement Learning

### Configuration
```python
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
| **MuJoCo InvertedPendulum** | **+118.6%** | 0.001 | 100% |
| **MuJoCo Hopper** | **+41.2%** | 0.007 | 80% |

---

## 📐 Classical Optimization

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    base_optimizer="Adam",
    base_lr=0.01
)
```

### Verified Results

| Function | Improvement | Landscape |
|----------|-------------|-----------|
| **Rosenbrock** | **+75.8%** | Valley |
| **Beale** | **+62.0%** | Plateau |
| **Sphere** | **+31.1%** | Convex |

---

## ⚛️ Quantum VQE - Condensed Matter

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    mode="simulation",  # or "hardware" for noisy
    base_optimizer="Adam"
)
```

### Verified Results

| Model | Improvement |
|-------|-------------|
| **SSH Model (Topological)** | **+61.0%** |
| **XY Model** | **+60.8%** |
| **Ferro Ising** | **+45.1%** |
| **Transverse Ising** | **+42.0%** |
| **Heisenberg XXZ** | **+20.8%** |
| **Kitaev Chain** | **+20.4%** |
| **Hubbard Dimer** | **+14.1%** |

---

## ⚛️ Quantum VQE - Chemistry

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    mode="hardware",  # Recommended for real hardware
    base_optimizer="Adam"
)
```

### Verified Results

| Molecule | Improvement | Notes |
|----------|-------------|-------|
| **FakeFez H₂** | **+52.4%** | p=0.043, IBM noise |
| **He Atom** | **+51.2%** | |
| **H₂ Molecule** | **+46.6%** | |
| **H₃⁺ Chain** | **+42.0%** | |
| **LiH Molecule** | **+41.4%** | |
| **BeH₂ Molecule** | **+37.8%** | |

---

## 🎯 QAOA (Combinatorial Optimization)

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="deep",
    mode="hardware",
    base_optimizer="SGD",
    base_lr=0.1
)
```

### Verified Results

| Problem | Improvement | Wins | Notes |
|---------|-------------|------|-------|
| **FakeFez MaxCut** | **+45.1%** | 10/10 | p=0.0003 |
| **Vertex Cover** | **+31.9%** | 51/60 | |
| **Max Independent Set** | **+31.9%** | 51/60 | |
| **MaxCut** | **+21.5%** | 45/60 | |

---

## 💰 Finance (QUBO)

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    mode="simulation",
    base_optimizer="Adam",
    base_lr=0.01
)
```

### Verified Results

| Problem | Improvement |
|---------|-------------|
| **Credit Risk** | **+52.3%** |
| **Portfolio Optimization** | **+51.7%** |

---

## 💊 Drug Discovery

### Configuration
```python
opt = MobiuQCore(
    license_key="YOUR_KEY",
    method="standard",
    mode="simulation",
    base_optimizer="AMSGrad",
    base_lr=0.01
)
```

### Verified Results

| Task | Improvement |
|------|-------------|
| **Binding Affinity** | **+12.2%** |

---

## 📋 Quick Reference

| Domain | method | mode | base_optimizer | base_lr |
|--------|--------|------|----------------|---------|
| **RL** | adaptive | simulation | Adam | 0.0003 |
| **Classical** | standard | simulation | Adam | 0.01 |
| **VQE Clean** | standard | simulation | Adam | 0.01 |
| **VQE Noisy** | standard | hardware | Adam | 0.02 |
| **QAOA** | deep | hardware | SGD | 0.1 |
| **Finance** | standard | simulation | Adam | 0.01 |
| **Drug** | standard | simulation | AMSGrad | 0.01 |

---

## 🔧 Default Learning Rates

| Method | Mode | Default LR |
|--------|------|------------|
| standard | simulation | **0.01** |
| standard | hardware | **0.02** |
| deep | any | **0.1** |
| adaptive | any | **0.0003** |

---

## 🔗 Available Optimizers

| Optimizer | Best For |
|-----------|----------|
| **Adam** | Default, most cases |
| **AdamW** | LLM, weight decay |
| **SGD** | QAOA |
| **AMSGrad** | Drug Discovery |
| **NAdam** | Alternative to Adam |
| **Momentum** | RL alternative |
| **LAMB** | Large batch |

**Note:** Optimizer names are case-sensitive!

---

## 🔧 Troubleshooting

### 1. Switch Method
| If This Fails | Try This |
|---------------|----------|
| standard | adaptive |
| adaptive | deep |
| deep | standard |

### 2. Adjust Learning Rate
| Issue | Action |
|-------|--------|
| Diverging | Lower LR by 2-5x |
| No progress | Increase LR by 2x |

---

*All results verified with p < 0.05*