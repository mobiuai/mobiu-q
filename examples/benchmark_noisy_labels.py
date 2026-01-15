#!/usr/bin/env python3
"""
================================================================================
BENCHMARK: Mobiu vs Adam on Noisy Labels
================================================================================

Simulation of training with systematic label noise (e.g., crowdsourced data).
Labels are systematically confused: class i → class (i+1) with 30% probability.

The gradient is computed from noisy labels, but energy is true loss.
This creates gradient-energy mismatch that Soft Algebra can detect.
================================================================================
"""

import numpy as np
from scipy import stats

np.random.seed(42)


class NoisyLabelsProblem:
    """Classification with systematic label noise."""

    def __init__(self, dim=20, n_classes=5, noise_rate=0.3):
        self.dim = dim
        self.n_classes = n_classes
        self.noise_rate = noise_rate

        # True class centers
        np.random.seed(888)
        self.class_centers = np.random.randn(n_classes, dim)

    def init_params(self, seed):
        np.random.seed(seed)
        return np.random.randn(self.n_classes, self.dim) * 0.5

    def generate_batch(self, rng_state, batch_size=32):
        """Generate a batch with systematic label noise."""
        np.random.seed(rng_state)

        # True labels
        true_labels = np.random.randint(0, self.n_classes, batch_size)

        # Noisy labels: class i → class (i+1) % n_classes
        noisy_labels = true_labels.copy()
        noise_mask = np.random.rand(batch_size) < self.noise_rate
        noisy_labels[noise_mask] = (noisy_labels[noise_mask] + 1) % self.n_classes

        # Generate data
        X = self.class_centers[true_labels] + np.random.randn(batch_size, self.dim) * 0.5

        return X, true_labels, noisy_labels

    def compute_loss(self, W, X, labels):
        """Compute cross-entropy-like loss."""
        logits = X @ W.T  # (batch, n_classes)
        # Simple softmax loss approximation
        correct_scores = logits[np.arange(len(labels)), labels]
        loss = -np.mean(correct_scores) + np.mean(np.log(np.sum(np.exp(logits), axis=1) + 1e-9))
        return float(loss)

    def compute_gradient(self, W, X, labels):
        """Compute gradient w.r.t. W."""
        batch_size = len(labels)
        logits = X @ W.T
        probs = np.exp(logits) / (np.sum(np.exp(logits), axis=1, keepdims=True) + 1e-9)

        grad = np.zeros_like(W)
        for c in range(self.n_classes):
            mask = (labels == c).astype(float).reshape(-1, 1)
            grad[c] = np.mean((probs[:, c:c+1] - mask) * X, axis=0)

        return grad


def run_adam(problem, seed, n_steps=80, lr=0.01):
    """Run plain Adam optimizer."""
    W = problem.init_params(seed)

    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')

    for step in range(n_steps):
        X, true_labels, noisy_labels = problem.generate_batch(seed * 10000 + step)

        # True loss (what we want to minimize)
        true_loss = problem.compute_loss(W, X, true_labels)
        best_loss = min(best_loss, true_loss)

        # Gradient from noisy labels (systematic bias!)
        gradient = problem.compute_gradient(W, X, noisy_labels)

        # Adam update
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))

        W = W - lr * m_hat / (np.sqrt(v_hat) + eps)

    return best_loss


def run_mobiu_local(problem, seed, n_steps=80, lr=0.01):
    """Run Adam + LocalSoftAlgebra."""
    from mobiu_q import LocalSoftAlgebra

    W = problem.init_params(seed)

    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    lsa = LocalSoftAlgebra(base_lr=lr, maximize=False)

    best_loss = float('inf')

    for step in range(n_steps):
        X, true_labels, noisy_labels = problem.generate_batch(seed * 10000 + step)

        # True loss
        true_loss = problem.compute_loss(W, X, true_labels)
        best_loss = min(best_loss, true_loss)

        # Gradient from noisy labels
        gradient = problem.compute_gradient(W, X, noisy_labels)

        # Adaptive LR from LocalSoftAlgebra
        adaptive_lr, warp_factor = lsa.update(true_loss)

        # Warp gradient
        warped_gradient = gradient * warp_factor

        # Adam update
        m = beta1 * m + (1 - beta1) * warped_gradient
        v = beta2 * v + (1 - beta2) * (warped_gradient ** 2)

        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))

        W = W - adaptive_lr * m_hat / (np.sqrt(v_hat) + eps)

    return best_loss


def main():
    print("=" * 70)
    print("🏷️  BENCHMARK: Mobiu vs Adam on Noisy Labels")
    print("=" * 70)
    print("Noise: 30% systematic confusion (class i → class i+1)")
    print("Energy = true loss, Gradient = from noisy labels")
    print("=" * 70)

    problem = NoisyLabelsProblem(dim=20, n_classes=5, noise_rate=0.3)

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

        improvement = (adam_loss - mobiu_loss) / abs(adam_loss) * 100 if adam_loss != 0 else 0
        winner = "🏆 Mobiu" if mobiu_loss < adam_loss else "Adam"
        print(f"{seed:<8} {adam_loss:<12.6f} {mobiu_loss:<12.6f} {improvement:>+8.1f}%  {winner}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    adam_avg = np.mean(adam_results)
    mobiu_avg = np.mean(mobiu_results)
    improvement = (adam_avg - mobiu_avg) / abs(adam_avg) * 100

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
