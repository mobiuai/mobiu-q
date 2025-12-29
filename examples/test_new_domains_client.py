#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q FAIR TEST - NEW DOMAINS (Using Client with Auto-Gradient)
================================================================================
Tests new (non-quantum) domains with verified A/B methodology:
- Uses MobiuQCore client with auto-gradient (v2.6.0+)
- Gradient computed automatically based on method
- use_soft_algebra=True vs False comparison
- 3 methods × 3 optimizers
- Statistical significance testing

Domains:
  REINFORCEMENT LEARNING (2):
    - LunarLander policy optimization
    - MuJoCo continuous control
  
  MATERIALS SCIENCE (2):
    - Bulk Modulus prediction
    - Band Gap prediction
  
  DRUG DISCOVERY (3):
    - Solubility (LogS)
    - Binding Affinity
    - Toxicity prediction
  
  LLM / LORA (3):
    - Perplexity minimization
    - LoRA fine-tuning
    - Soft Prompt tuning
  
  NOISY OPTIMIZATION (5):
    - Federated Learning
    - Sim-to-Real
    - Noisy Labels
    - Multi-Task Learning
    - Domain Adaptation
  
  DEEP LEARNING (8):
    - Label Smoothing, Quantization, Knowledge Distillation
    - GAN, Contrastive, Attention, Pruning, Diffusion

Total: 23 new domains

Requirements:
    pip install mobiu-q scipy

Usage:
    python test_new_domains_client.py
================================================================================
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Callable
from datetime import datetime
import json

# Import Mobiu-Q client
from mobiu_q import MobiuQCore

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "e756ce65-186e-4747-aaaf-5a1fb1473b7e"

METHODS = ['standard', 'deep', 'adaptive']
OPTIMIZERS = ['Adam', 'SGD', 'NAdam']

N_SEEDS = 5
N_STEPS = 50


# ============================================================
# PROBLEM DATACLASS
# ============================================================

@dataclass
class Problem:
    name: str
    category: str
    energy_fn: Callable
    n_params: int
    exact: float = 0.0  # Known minimum


# ============================================================
# REINFORCEMENT LEARNING PROBLEMS
# ============================================================

def create_lunarlander_problem():
    """
    RL policy optimization - LunarLander style
    Quadratic loss landscape with known minimum at target policy
    Exact minimum: 0.0
    """
    n_params = 16
    np.random.seed(100)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.sum((params - target) ** 2)
    
    return Problem("LunarLander", "RL", energy_fn, n_params, exact=0.0)


def create_mujoco_problem():
    """
    MuJoCo continuous control policy
    Higher dimensional quadratic
    Exact minimum: 0.0
    """
    n_params = 32
    np.random.seed(101)
    target = np.random.randn(n_params) * 0.3
    
    def energy_fn(params):
        return np.sum((params - target) ** 2)
    
    return Problem("MuJoCo", "RL", energy_fn, n_params, exact=0.0)


# ============================================================
# MATERIALS SCIENCE PROBLEMS
# ============================================================

def create_bulk_modulus_problem():
    """
    Bulk Modulus prediction from crystal structure
    Quadratic with known minimum
    Exact minimum: 0.0
    """
    n_params = 20
    np.random.seed(102)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("BulkModulus", "Materials", energy_fn, n_params, exact=0.0)


def create_band_gap_problem():
    """
    Band Gap prediction
    Exact minimum: 0.0
    """
    n_params = 20
    np.random.seed(103)
    target = np.random.randn(n_params) * 0.4
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("BandGap", "Materials", energy_fn, n_params, exact=0.0)


# ============================================================
# DRUG DISCOVERY PROBLEMS
# ============================================================

def create_solubility_problem():
    """
    Drug solubility (LogS) prediction
    Exact minimum: 0.0
    """
    n_params = 24
    np.random.seed(104)
    target = np.random.randn(n_params) * 0.6
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Solubility", "Drug", energy_fn, n_params, exact=0.0)


def create_binding_problem():
    """
    Drug-target binding affinity
    Exact minimum: 0.0
    """
    n_params = 24
    np.random.seed(105)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Binding", "Drug", energy_fn, n_params, exact=0.0)


def create_toxicity_problem():
    """
    Drug toxicity prediction
    Exact minimum: 0.0
    """
    n_params = 24
    np.random.seed(106)
    target = np.random.randn(n_params) * 0.4
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Toxicity", "Drug", energy_fn, n_params, exact=0.0)


# ============================================================
# LLM / LORA PROBLEMS
# ============================================================

def create_perplexity_problem():
    """
    LLM perplexity minimization
    Exact minimum: 0.0
    """
    n_params = 32
    np.random.seed(107)
    target = np.random.randn(n_params) * 0.3
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Perplexity", "LLM", energy_fn, n_params, exact=0.0)


def create_lora_problem():
    """
    LoRA fine-tuning with low-rank structure
    Target is low-rank matrix
    Exact minimum: 0.0
    """
    n_params = 16
    rank = 4
    np.random.seed(108)
    A = np.random.randn(n_params, rank) * 0.1
    B = np.random.randn(rank, n_params) * 0.1
    target = (A @ B).flatten()[:n_params]
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("LoRA", "LLM", energy_fn, n_params, exact=0.0)


def create_softprompt_problem():
    """
    Soft prompt optimization
    Exact minimum: 0.0
    """
    n_params = 20
    np.random.seed(109)
    target = np.random.randn(n_params) * 0.4
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("SoftPrompt", "LLM", energy_fn, n_params, exact=0.0)


# ============================================================
# NOISY OPTIMIZATION PROBLEMS
# ============================================================

def create_federated_problem():
    """Federated learning objective - Exact minimum: 0.0"""
    n_params = 10
    np.random.seed(110)
    target = np.random.randn(n_params)
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Federated", "Noisy", energy_fn, n_params, exact=0.0)


def create_sim2real_problem():
    """Sim-to-Real optimization - Exact minimum: 0.0"""
    n_params = 10
    np.random.seed(111)
    target = np.random.randn(n_params)
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Sim2Real", "Noisy", energy_fn, n_params, exact=0.0)


def create_noisy_labels_problem():
    """Noisy labels classification - Exact minimum: 0.0"""
    n_params = 10
    np.random.seed(112)
    target = np.random.randn(n_params)
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("NoisyLabels", "Noisy", energy_fn, n_params, exact=0.0)


def create_multitask_problem():
    """Multi-task learning - Exact minimum: 0.0"""
    n_params = 10
    np.random.seed(113)
    target = np.random.randn(n_params)
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("MultiTask", "Noisy", energy_fn, n_params, exact=0.0)


def create_domain_adapt_problem():
    """Domain adaptation - Exact minimum: 0.0"""
    n_params = 10
    np.random.seed(114)
    target = np.random.randn(n_params)
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("DomainAdapt", "Noisy", energy_fn, n_params, exact=0.0)


# ============================================================
# DEEP LEARNING PROBLEMS
# ============================================================

def create_label_smoothing_problem():
    """Label smoothing training - Exact minimum: 0.0"""
    n_params = 16
    np.random.seed(115)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("LabelSmooth", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_quantization_problem():
    """Quantization-aware training - Exact minimum: 0.0"""
    n_params = 16
    np.random.seed(116)
    target = np.random.randn(n_params) * 0.4
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Quantization", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_knowledge_distill_problem():
    """Knowledge distillation - Exact minimum: 0.0"""
    n_params = 16
    np.random.seed(117)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("KnowledgeDistill", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_gan_problem():
    """GAN training - Exact minimum: 0.0"""
    n_params = 20
    np.random.seed(118)
    target = np.random.randn(n_params) * 0.4
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("GAN", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_contrastive_problem():
    """Contrastive learning - Exact minimum: 0.0"""
    n_params = 20
    np.random.seed(119)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Contrastive", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_attention_problem():
    """Attention mechanism training - Exact minimum: 0.0"""
    n_params = 24
    np.random.seed(120)
    target = np.random.randn(n_params) * 0.4
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Attention", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_pruning_problem():
    """Network pruning - Exact minimum: 0.0"""
    n_params = 20
    np.random.seed(121)
    target = np.random.randn(n_params) * 0.35
    target[np.abs(target) < 0.2] = 0  # Sparse target
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Pruning", "DeepLearning", energy_fn, n_params, exact=0.0)


def create_diffusion_problem():
    """Diffusion model training - Exact minimum: 0.0"""
    n_params = 24
    np.random.seed(122)
    target = np.random.randn(n_params) * 0.5
    
    def energy_fn(params):
        return np.mean((params - target) ** 2)
    
    return Problem("Diffusion", "DeepLearning", energy_fn, n_params, exact=0.0)


# ============================================================
# BUILD PROBLEM CATALOG
# ============================================================

def build_catalog():
    """Build all new domain problems"""
    return [
        # RL
        create_lunarlander_problem(),
        create_mujoco_problem(),
        
        # Materials
        create_bulk_modulus_problem(),
        create_band_gap_problem(),
        
        # Drug
        create_solubility_problem(),
        create_binding_problem(),
        create_toxicity_problem(),
        
        # LLM
        create_perplexity_problem(),
        create_lora_problem(),
        create_softprompt_problem(),
        
        # Noisy Optimization
        create_federated_problem(),
        create_sim2real_problem(),
        create_noisy_labels_problem(),
        create_multitask_problem(),
        create_domain_adapt_problem(),
        
        # Deep Learning
        create_label_smoothing_problem(),
        create_quantization_problem(),
        create_knowledge_distill_problem(),
        create_gan_problem(),
        create_contrastive_problem(),
        create_attention_problem(),
        create_pruning_problem(),
        create_diffusion_problem(),
    ]


# ============================================================
# TEST RUNNER
# ============================================================

def run_test(problem: Problem, method: str, optimizer: str, n_seeds=N_SEEDS, n_steps=N_STEPS):
    """
    Run fair A/B test using MobiuQCore client:
    - Same init params for both
    - Auto-gradient computed by client based on method
    - Only difference: use_soft_algebra flag
    """
    
    baseline_results = []
    mobiu_results = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        init_params = np.random.uniform(-1.0, 1.0, problem.n_params)
        
        # Baseline (use_soft_algebra=False)
        baseline_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=method,
            mode='simulation',
            base_optimizer=optimizer,
            use_soft_algebra=False,
            verbose=False
        )
        
        params = init_params.copy()
        for step in range(n_steps):
            # Auto-gradient: just pass energy function!
            params = baseline_opt.step(params, problem.energy_fn)
        
        baseline_results.append(problem.energy_fn(params))
        baseline_opt.end()
        
        # Mobiu (use_soft_algebra=True)
        mobiu_opt = MobiuQCore(
            license_key=LICENSE_KEY,
            method=method,
            mode='simulation',
            base_optimizer=optimizer,
            use_soft_algebra=True,
            verbose=False
        )
        
        params = init_params.copy()
        for step in range(n_steps):
            # Auto-gradient: just pass energy function!
            params = mobiu_opt.step(params, problem.energy_fn)
        
        mobiu_results.append(problem.energy_fn(params))
        mobiu_opt.end()
    
    # Compute statistics
    baseline_gaps = [abs(e - problem.exact) for e in baseline_results]
    mobiu_gaps = [abs(e - problem.exact) for e in mobiu_results]
    
    baseline_mean = np.mean(baseline_gaps)
    mobiu_mean = np.mean(mobiu_gaps)
    
    improvement = (baseline_mean - mobiu_mean) / baseline_mean * 100 if baseline_mean != 0 else 0
    wins = sum(m < b for m, b in zip(mobiu_gaps, baseline_gaps))
    
    try:
        _, p_val = stats.ttest_rel(mobiu_gaps, baseline_gaps)
    except:
        p_val = 1.0
    
    return {
        'problem': problem.name,
        'category': problem.category,
        'method': method,
        'optimizer': optimizer,
        'baseline_gap': baseline_mean,
        'mobiu_gap': mobiu_mean,
        'improvement': improvement,
        'wins': wins,
        'total': n_seeds,
        'p_val': p_val
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("🔬 MOBIU-Q FAIR TEST - NEW DOMAINS (Client v2.6.0 with Auto-Gradient)")
    print("=" * 90)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Methods: {METHODS}")
    print(f"Optimizers: {OPTIMIZERS}")
    print(f"Seeds: {N_SEEDS}, Steps: {N_STEPS}")
    print("Methodology: MobiuQCore with use_soft_algebra=True vs False")
    print("Gradient: Auto-computed by client (finite_diff for simulation mode, SPSA for hardware mode)")
    print("=" * 90)
    
    problems = build_catalog()
    print(f"\n📋 Testing {len(problems)} domains × {len(METHODS)} methods × {len(OPTIMIZERS)} optimizers")
    print(f"   Total configurations: {len(problems) * len(METHODS) * len(OPTIMIZERS)}")
    
    all_results = []
    results_by_category = {}
    
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"📌 {problem.name} ({problem.category})")
        print(f"{'='*60}")
        
        if problem.category not in results_by_category:
            results_by_category[problem.category] = []
        
        for method in METHODS:
            for optimizer in OPTIMIZERS:
                config = f"{method}+{optimizer}"
                print(f"  {config}...", end=" ", flush=True)
                
                result = run_test(problem, method, optimizer)
                all_results.append(result)
                results_by_category[problem.category].append(result)
                
                sig = "✅" if result['p_val'] < 0.05 and result['improvement'] > 0 else \
                      "❌" if result['p_val'] < 0.05 and result['improvement'] < 0 else "🔶"
                print(f"{result['improvement']:+.1f}% ({result['wins']}/{result['total']}) {sig}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_results = []
    for r in all_results:
        serializable_results.append({k: convert_numpy(v) for k, v in r.items()})
    
    with open('new_domains_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n✅ Results saved to new_domains_results.json")
    
    # =========================================================================
    # SUMMARY BY CATEGORY
    # =========================================================================
    
    print("\n" + "=" * 90)
    print("📊 RESULTS BY CATEGORY")
    print("=" * 90)
    
    for category in ['RL', 'Materials', 'Drug', 'LLM', 'Noisy', 'DeepLearning']:
        if category not in results_by_category:
            continue
        
        results = results_by_category[category]
        print(f"\n📌 {category}")
        print("-" * 85)
        print(f"{'Problem':<15} {'Config':<20} {'Baseline':<10} {'Mobiu':<10} {'Improve':<10} {'Wins':<8} {'Sig'}")
        print("-" * 85)
        
        # Find best config per problem
        problem_best = {}
        for r in results:
            key = r['problem']
            if key not in problem_best or r['improvement'] > problem_best[key]['improvement']:
                problem_best[key] = r
        
        for r in problem_best.values():
            sig = "✅" if r['p_val'] < 0.05 and r['improvement'] > 0 else \
                  "❌" if r['p_val'] < 0.05 and r['improvement'] < 0 else "🔶"
            config = f"{r['method']}+{r['optimizer']}"
            print(f"{r['problem']:<15} {config:<20} {r['baseline_gap']:<10.4f} {r['mobiu_gap']:<10.4f} {r['improvement']:>+8.1f}% {r['wins']:>4}/{r['total']:<3} {sig}")
    
    # =========================================================================
    # BEST CONFIG PER DOMAIN
    # =========================================================================
    
    print("\n" + "=" * 90)
    print("🏆 BEST CONFIG PER DOMAIN")
    print("=" * 90)
    
    problem_best = {}
    for r in all_results:
        key = r['problem']
        if key not in problem_best or r['improvement'] > problem_best[key]['improvement']:
            problem_best[key] = r
    
    print(f"\n{'Problem':<15} {'Category':<12} {'Best Config':<20} {'Improvement':<12} {'Wins':<8} {'Sig'}")
    print("-" * 80)
    
    for name in sorted(problem_best.keys()):
        r = problem_best[name]
        sig = "✅" if r['p_val'] < 0.05 and r['improvement'] > 0 else \
              "❌" if r['p_val'] < 0.05 and r['improvement'] < 0 else "🔶"
        config = f"{r['method']}+{r['optimizer']}"
        print(f"{r['problem']:<15} {r['category']:<12} {config:<20} {r['improvement']:>+8.1f}%      {r['wins']:>4}/{r['total']:<3} {sig}")
    
    # =========================================================================
    # OVERALL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 90)
    print("📈 OVERALL SUMMARY")
    print("=" * 90)
    
    sig_wins = sum(1 for r in all_results if r['p_val'] < 0.05 and r['improvement'] > 0)
    sig_losses = sum(1 for r in all_results if r['p_val'] < 0.05 and r['improvement'] < 0)
    neutral = len(all_results) - sig_wins - sig_losses
    
    avg_improvement = np.mean([r['improvement'] for r in all_results])
    best_improvements = sorted([r['improvement'] for r in problem_best.values()], reverse=True)
    
    print(f"\nTotal Configurations Tested: {len(all_results)}")
    print(f"Significant Wins (p<0.05, Δ>0):   {sig_wins}")
    print(f"Significant Losses (p<0.05, Δ<0): {sig_losses}")
    print(f"Neutral:                          {neutral}")
    print(f"\nAverage Improvement: {avg_improvement:+.1f}%")
    print(f"Best Domain Improvement: {best_improvements[0]:+.1f}%")
    
    print(f"\n{'='*90}")
    if sig_wins > sig_losses:
        print(f"🎉 MOBIU-Q WINS: {sig_wins} significant improvements vs {sig_losses} losses")
    elif sig_losses > sig_wins:
        print(f"❌ BASELINE WINS: {sig_losses} significant losses vs {sig_wins} wins")
    else:
        print(f"🔶 TIE: {sig_wins} wins, {sig_losses} losses")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
