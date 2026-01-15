#!/usr/bin/env python3
"""
================================================================================
BENCHMARK: Mobiu (Auto-Adaptive) vs Plain Adam
================================================================================

Fair comparison on federated learning simulation.
Both optimizers see the same gradients and losses.
Only difference: Mobiu uses LocalSoftAlgebra for adaptive LR.

This tests if the LOCAL (offline) Soft Algebra provides benefit.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

np.random.seed(42)
torch.manual_seed(42)


class FederatedProblem:
    """Federated Learning simulation with biased gradients."""

    def __init__(self, dim=20, n_clients=10, non_iid=0.8):
        self.dim = dim
        self.n_clients = n_clients
        self.non_iid = non_iid

        # Fixed true target
        np.random.seed(999)
        self.w_true = np.random.randn(dim)

        # Client biases
        self.client_biases = [
            np.random.randn(dim) * non_iid
            for _ in range(n_clients)
        ]

    def init_params(self, seed):
        np.random.seed(seed)
        return np.random.randn(self.dim) * 0.5

    def global_loss(self, w):
        return float(np.mean((w - self.w_true) ** 2))

    def federated_gradient(self, w, rng_state):
        np.random.seed(rng_state)
        sampled = np.random.choice(self.n_clients, 5, replace=False)

        grads = []
        for c in sampled:
            target = self.w_true + self.client_biases[c]
            grad = 2 * (w - target) / self.dim
            grads.append(grad)

        return np.mean(grads, axis=0)


def run_adam(problem, seed, n_steps=80, lr=0.01):
    """Run plain Adam optimizer."""
    params = problem.init_params(seed)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')

    for step in range(n_steps):
        loss = problem.global_loss(params)
        gradient = problem.federated_gradient(params, seed * 10000 + step)

        best_loss = min(best_loss, loss)

        # Adam update
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))

        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

    return best_loss


def run_mobiu_local(problem, seed, n_steps=80, lr=0.01):
    """Run Adam + LocalSoftAlgebra (what Mobiu does offline)."""
    from mobiu_q import LocalSoftAlgebra

    params = problem.init_params(seed)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # LocalSoftAlgebra for adaptive LR
    lsa = LocalSoftAlgebra(base_lr=lr, maximize=False)

    best_loss = float('inf')

    for step in range(n_steps):
        loss = problem.global_loss(params)
        gradient = problem.federated_gradient(params, seed * 10000 + step)

        best_loss = min(best_loss, loss)

        # Get adaptive LR from LocalSoftAlgebra
        adaptive_lr, warp_factor = lsa.update(loss)

        # Apply gradient warping
        warped_gradient = gradient * warp_factor

        # Adam update with adaptive LR
        m = beta1 * m + (1 - beta1) * warped_gradient
        v = beta2 * v + (1 - beta2) * (warped_gradient ** 2)

        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))

        params = params - adaptive_lr * m_hat / (np.sqrt(v_hat) + eps)

    return best_loss


def main():
    print("=" * 70)
    print("📊 BENCHMARK: Mobiu (LocalSoftAlgebra) vs Plain Adam")
    print("=" * 70)
    print("Problem: Federated Learning with Non-IID Clients")
    print("Both see same gradients. Difference: Mobiu uses adaptive LR.")
    print("=" * 70)

    problem = FederatedProblem(dim=20, n_clients=10, non_iid=0.8)

    n_seeds = 10
    seeds = [42 + s * 17 for s in range(n_seeds)]

    adam_results = []
    mobiu_results = []

    print(f"\n{'Seed':<8} {'Adam':<12} {'Mobiu':<12} {'Δ%':<10} {'Winner'}")
    print("-" * 55)

    for seed in seeds:
        adam_loss = run_adam(problem, seed)
        mobiu_loss = run_mobiu_local(problem, seed)

        adam_results.append(adam_loss)
        mobiu_results.append(mobiu_loss)

        if adam_loss > 0:
            improvement = (adam_loss - mobiu_loss) / adam_loss * 100
        else:
            improvement = 0

        winner = "🏆 Mobiu" if mobiu_loss < adam_loss else "Adam"
        print(f"{seed:<8} {adam_loss:<12.6f} {mobiu_loss:<12.6f} {improvement:>+8.1f}%  {winner}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    adam_avg = np.mean(adam_results)
    mobiu_avg = np.mean(mobiu_results)
    improvement = (adam_avg - mobiu_avg) / adam_avg * 100

    wins_mobiu = sum(m < a for m, a in zip(mobiu_results, adam_results))

    print(f"\nAdam avg:  {adam_avg:.6f}")
    print(f"Mobiu avg: {mobiu_avg:.6f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"\nWin rate: Mobiu {wins_mobiu}/{n_seeds}")

    # Statistical test
    t_stat, p_value = stats.ttest_rel(adam_results, mobiu_results)
    print(f"\np-value: {p_value:.6f}")

    if p_value < 0.05 and mobiu_avg < adam_avg:
        print("✅ SIGNIFICANT: Mobiu is better!")
    elif p_value < 0.05:
        print("⚠️ SIGNIFICANT: Adam is better")
    else:
        print("⚪ NOT SIGNIFICANT")

    print("=" * 70)


if __name__ == "__main__":
    main()
