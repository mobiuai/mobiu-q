"""
================================================================================
SB3 MOBIU-Q BENCHMARK
================================================================================

Compares:
1. Baseline (Adam) - no Mobiu
2. Mobiu (LR scaling only) - gradient_warp=False
3. Mobiu + Gradient Warp - gradient_warp=True

Tests Mobiu-Q optimization on SB3/PPO reinforcement learning tasks.

Requirements:
  pip install mobiu-q stable-baselines3 gymnasium

Usage:
  export MOBIU_LICENSE_KEY='your-key-here'
  python sb3_gradient_warp_benchmark.py

================================================================================
"""

import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from mobiu_q import MobiuOptimizer

LICENSE_KEY = os.environ.get('MOBIU_LICENSE_KEY', 'YOUR_KEY')
if not LICENSE_KEY:
    print("⚠️  MOBIU_LICENSE_KEY not set. Set it with:")
    print("   export MOBIU_LICENSE_KEY='your-key-here'")
    print("   Mobiu runs will fail without a valid key.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class MobiuCallback(BaseCallback):
    """
    SB3 callback that wraps the policy optimizer with MobiuOptimizer.
    
    Args:
        gradient_warp: If True, enable SA gradient warping in addition to LR scaling.
        verbose: Verbosity level.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self._mobiu = None
        self._ep_returns = []
    
    def _on_training_start(self):
        base_opt = self.model.policy.optimizer
        self._mobiu = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method="adaptive",
            use_soft_algebra=True,
            maximize=True,
            sync_interval=50,
            verbose=False
        )
        self.model.policy.optimizer = self._mobiu
    
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_return = info["episode"]["r"]
                self._ep_returns.append(ep_return)
                recent = self._ep_returns[-4:]
                self._mobiu.set_metric(np.mean(recent))
        return True
    
    def _on_training_end(self):
        if self._mobiu:
            self._mobiu.end()


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(seed, env_id, total_steps, lr, mode="baseline"):
    """Run a single training run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        seed=seed,
        verbose=0
    )
    
    if mode == "baseline":
        callback = None
    elif mode == "mobiu":
        callback = MobiuCallback()
    elif mode == "mobiu_warp":
        callback = MobiuCallback()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    model.learn(total_timesteps=total_steps, callback=callback)
    
    eval_env = gym.make(env_id)
    eval_env.reset(seed=seed + 1000)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    
    env.close()
    eval_env.close()
    
    return mean_reward, std_reward


def run_benchmark(env_id="LunarLander-v3", seeds=range(30), total_steps=200_000, lr=3e-4):
    """Run full benchmark comparing all three modes."""
    
    print("=" * 70)
    print(f"SB3 MOBIU-Q BENCHMARK")
    print("=" * 70)
    print(f"Environment: {env_id}")
    print(f"Steps: {total_steps}")
    print(f"Seeds: {len(seeds)}")
    print(f"LR: {lr}")
    print(f"License: {'✓' if LICENSE_KEY else '✗ (set MOBIU_LICENSE_KEY)'}")
    print("=" * 70)
    
    results = {
        'baseline': [],
        'mobiu': [],
    }
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        # Save RNG state — both runs see identical environments
        torch_state = torch.get_rng_state()
        np_state    = np.random.get_state()

        print("  [1/2] Baseline (Adam)...", end=" ", flush=True)
        mean, std = run_single(seed, env_id, total_steps, lr, mode="baseline")
        results['baseline'].append(mean)
        print(f"{mean:.1f} ± {std:.1f}")

        # Restore RNG state — Mobiu starts from exactly the same point
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)

        print("  [2/2] Mobiu-Q...", end=" ", flush=True)
        mean, std = run_single(seed, env_id, total_steps, lr, mode="mobiu")
        results['mobiu'].append(mean)
        print(f"{mean:.1f} ± {std:.1f}")

        diff = results['mobiu'][-1] - results['baseline'][-1]
        winner = "✅ Mobiu" if diff > 0 else "❌ Adam"
        print(f"  Δ = {diff:+.1f} → {winner}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_mean = np.mean(results['baseline'])
    
    print()
    print(f"{'Mode':<25} | {'Mean':>8} | {'Std':>6} | {'vs Baseline':>12} | {'Win Rate':>10}")
    print("-" * 70)
    
    for mode, scores in results.items():
        arr = np.array(scores)
        improvement = 100 * (arr.mean() - baseline_mean) / (abs(baseline_mean) + 1e-9)
        
        if mode == 'baseline':
            imp_str = "-"
            win_str = "-"
        else:
            imp_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
            baseline_arr = np.array(results['baseline'])
            wins = np.sum(arr > baseline_arr)
            win_str = f"{wins}/{len(seeds)} ({100*wins/len(seeds):.0f}%)"
        
        print(f"{mode:<25} | {arr.mean():>8.1f} | {arr.std():>6.1f} | {imp_str:>12} | {win_str:>10}")
    
    # p-value
    from scipy.stats import wilcoxon
    adam_arr  = np.array(results['baseline'])
    mobiu_arr = np.array(results['mobiu'])
    try:
        _, p_value = wilcoxon(adam_arr, mobiu_arr)
    except Exception:
        p_value = 1.0
    wins = int(np.sum(mobiu_arr > adam_arr))
    improvement = 100 * (mobiu_arr.mean() - adam_arr.mean()) / (abs(adam_arr.mean()) + 1e-9)
    print()
    print(f"  📈 Improvement: {improvement:+.1f}%")
    print(f"  🏆 Win rate: {wins}/{len(seeds)} ({100*wins/len(seeds):.0f}%)")
    print(f"  p-value: {p_value:.6f}")
    print("=" * 70)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_benchmark(
        env_id="LunarLander-v3",
        seeds=range(30),
        total_steps=200_000,
        lr=3e-4
    )
