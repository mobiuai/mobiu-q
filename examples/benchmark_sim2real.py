#!/usr/bin/env python3
"""
================================================================================
BENCHMARK: Mobiu vs Adam on Sim-to-Real
================================================================================

Simulation of sim-to-real transfer learning.
Gradient comes from simulator (biased physics).
Energy comes from real environment.

This creates systematic gradient-energy mismatch.
================================================================================
"""

import numpy as np
from scipy import stats

np.random.seed(42)


class Sim2RealProblem:
    """Sim-to-Real with biased simulator."""

    def __init__(self, dim=20, sim_bias=0.4):
        self.dim = dim
        self.sim_bias = sim_bias

        # True target
        np.random.seed(777)
        self.w_true = np.random.randn(dim)

        # Simulator has systematic bias
        self.w_sim_bias = np.random.randn(dim) * sim_bias

    def init_params(self, seed):
        np.random.seed(seed)
        return np.random.randn(self.dim) * 0.5

    def real_loss(self, w):
        """True real-world loss."""
        return float(np.mean((w - self.w_true) ** 2))

    def sim_gradient(self, w, rng_state):
        """Gradient from simulator (biased)."""
        np.random.seed(rng_state)
        # Simulator thinks target is slightly different
        sim_target = self.w_true + self.w_sim_bias
        # Add some noise
        noise = np.random.randn(self.dim) * 0.1
        return 2 * (w - sim_target) / self.dim + noise


def run_adam(problem, seed, n_steps=80, lr=0.01):
    """Run plain Adam optimizer."""
    params = problem.init_params(seed)

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')

    for step in range(n_steps):
        loss = problem.real_loss(params)
        gradient = problem.sim_gradient(params, seed * 10000 + step)

        best_loss = min(best_loss, loss)

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))

        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

    return best_loss


def run_mobiu_local(problem, seed, n_steps=80, lr=0.01):
    """Run Adam + LocalSoftAlgebra."""
    from mobiu_q import LocalSoftAlgebra

    params = problem.init_params(seed)

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    lsa = LocalSoftAlgebra(base_lr=lr, maximize=False)

    best_loss = float('inf')

    for step in range(n_steps):
        loss = problem.real_loss(params)
        gradient = problem.sim_gradient(params, seed * 10000 + step)

        best_loss = min(best_loss, loss)

        # Adaptive LR
        adaptive_lr, warp_factor = lsa.update(loss)
        warped_gradient = gradient * warp_factor

        m = beta1 * m + (1 - beta1) * warped_gradient
        v = beta2 * v + (1 - beta2) * (warped_gradient ** 2)

        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))

        params = params - adaptive_lr * m_hat / (np.sqrt(v_hat) + eps)

    return best_loss


def main():
    print("=" * 70)
    print("🤖 BENCHMARK: Mobiu vs Adam on Sim-to-Real")
    print("=" * 70)
    print("Simulator bias: 0.4 (wrong physics parameters)")
    print("Energy = real loss, Gradient = from biased simulator")
    print("=" * 70)

    problem = Sim2RealProblem(dim=20, sim_bias=0.4)

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

        improvement = (adam_loss - mobiu_loss) / adam_loss * 100 if adam_loss > 0 else 0
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
