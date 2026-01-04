#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q MUJOCO TEST
================================================================================
Tests Soft Algebra + Frustration Engine on MuJoCo continuous control.

Fair A/B Test:
- Baseline: use_soft_algebra=False
- Mobiu: use_soft_algebra=True

Requirements:
    pip install mobiu-q gymnasium[mujoco] torch numpy scipy

Usage:
    python test_mujoco_maximize.py
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.stats import wilcoxon
from datetime import datetime
import json

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("⚠️ gymnasium not installed")

try:
    from mobiu_q import MobiuOptimizer
    HAS_MOBIU = True
except ImportError:
    HAS_MOBIU = False
    print("⚠️ mobiu-q not installed")

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY_HERE"

# Environments to test
ENVIRONMENTS = [
    "InvertedPendulum-v5",
    "Hopper-v5",
]

NUM_SEEDS = 10
TOTAL_TIMESTEPS = 50_000
STEPS_PER_UPDATE = 2048
BASE_LR = 3e-4

# ============================================================
# ACTOR-CRITIC FOR CONTINUOUS CONTROL
# ============================================================

class ActorCritic(nn.Module):
    """Actor-Critic for continuous action spaces."""
    
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        
        # Actor (policy) - outputs mean and log_std
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic (value function)
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor_mean(features), self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        features = self.shared(x)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features).squeeze(-1)
        
        return action, log_prob, entropy, value


# ============================================================
# PPO TRAINING
# ============================================================

def train_ppo(env_name, seed, use_mobiu=True):
    """Train PPO on MuJoCo environment."""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Model
    model = ActorCritic(obs_dim, act_dim)
    
    # Optimizer
    base_opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=1e-5)
    
    if use_mobiu and HAS_MOBIU:
        optimizer = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method='adaptive',
            use_soft_algebra=True,            
            sync_interval=50,
            verbose=False
        )
    else:
        optimizer = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method='adaptive',
            use_soft_algebra=False,
            sync_interval=50,
            verbose=False
        ) if HAS_MOBIU else base_opt
    
    # PPO hyperparameters
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    n_epochs = 10
    batch_size = 64
    
    # Buffers
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    done_buf = []
    val_buf = []
    
    # Training
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    episode_rewards = []
    current_reward = 0
    
    for step in range(TOTAL_TIMESTEPS):
        # Collect step
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs)
        
        action_np = action.numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        current_reward += reward
        
        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(log_prob)
        rew_buf.append(reward)
        done_buf.append(done)
        val_buf.append(value)
        
        obs = torch.tensor(next_obs, dtype=torch.float32)
        
        if done:
            episode_rewards.append(current_reward)
            current_reward = 0
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
        
        # Update when buffer is full
        if len(obs_buf) >= STEPS_PER_UPDATE:
            # Compute advantages (GAE)
            with torch.no_grad():
                _, _, _, last_value = model.get_action_and_value(obs)
            
            advantages = []
            returns = []
            gae = 0
            values = val_buf + [last_value]
            
            for t in reversed(range(len(rew_buf))):
                if done_buf[t]:
                    delta = rew_buf[t] - values[t]
                    gae = delta
                else:
                    delta = rew_buf[t] + gamma * values[t + 1] - values[t]
                    gae = delta + gamma * gae_lambda * gae
                
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            # Convert to tensors
            b_obs = torch.stack(obs_buf)
            b_act = torch.stack(act_buf)
            b_logp = torch.stack(logp_buf)
            b_adv = torch.tensor(advantages, dtype=torch.float32)
            b_ret = torch.tensor(returns, dtype=torch.float32)
            
            # Normalize advantages
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            
            # PPO update
            indices = np.arange(len(obs_buf))
            
            for _ in range(n_epochs):
                np.random.shuffle(indices)
                
                for start in range(0, len(indices), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    
                    _, new_logp, entropy, new_val = model.get_action_and_value(
                        b_obs[batch_idx], b_act[batch_idx]
                    )
                    
                    # Policy loss
                    ratio = torch.exp(new_logp - b_logp[batch_idx])
                    pg_loss1 = -b_adv[batch_idx] * ratio
                    pg_loss2 = -b_adv[batch_idx] * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    v_loss = F.mse_loss(new_val, b_ret[batch_idx])
                    
                    # Entropy bonus
                    ent_loss = -entropy.mean()
                    
                    # Total loss
                    loss = pg_loss + 0.5 * v_loss + 0.01 * ent_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    if HAS_MOBIU:
                        metric = episode_rewards[-1] if episode_rewards else 0
                        optimizer.step(reward=metric)
                    else:
                        optimizer.step()
            
            # Clear buffers
            obs_buf.clear()
            act_buf.clear()
            logp_buf.clear()
            rew_buf.clear()
            done_buf.clear()
            val_buf.clear()
    
    env.close()
    
    if HAS_MOBIU:
        optimizer.end()
    
    # Return average of last 20 episodes
    if len(episode_rewards) >= 20:
        return np.mean(episode_rewards[-20:])
    return np.mean(episode_rewards) if episode_rewards else 0


# ============================================================
# MAIN TEST
# ============================================================

def main():
    if not HAS_GYM:
        print("❌ Please install: pip install gymnasium[mujoco]")
        return
    
    print("=" * 70)
    print("🤖 MUJOCO CONTINUOUS CONTROL TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Environments: {ENVIRONMENTS}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,} | Seeds: {NUM_SEEDS}")
    print()
    print("Fair Test:")
    print("  • Baseline: use_soft_algebra=False")
    print("  • Mobiu:    use_soft_algebra=True")
    print("=" * 70)
    
    all_results = {}
    
    for env_name in ENVIRONMENTS:
        print(f"\n{'='*70}")
        print(f"🎮 {env_name}")
        print("=" * 70)
        
        baseline_results = []
        mobiu_results = []
        
        for seed in range(NUM_SEEDS):
            print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
            
            # Baseline
            print("    Training Baseline...", end=" ", flush=True)
            baseline_score = train_ppo(env_name, seed, use_mobiu=False)
            print(f"Score: {baseline_score:.1f}")
            
            # Mobiu
            print("    Training Mobiu...", end=" ", flush=True)
            mobiu_score = train_ppo(env_name, seed, use_mobiu=True)
            print(f"Score: {mobiu_score:.1f}")
            
            baseline_results.append(baseline_score)
            mobiu_results.append(mobiu_score)
            
            diff = mobiu_score - baseline_score
            winner = "✅ Mobiu" if diff > 0 else "❌ Base"
            print(f"    Δ = {diff:+.1f} → {winner}")
        
        # Statistics
        baseline_arr = np.array(baseline_results)
        mobiu_arr = np.array(mobiu_results)
        diff = mobiu_arr - baseline_arr
        
        win_rate = np.mean(diff > 0)
        improvement = 100 * (mobiu_arr.mean() - baseline_arr.mean()) / (abs(baseline_arr.mean()) + 1e-9)
        
        if len(baseline_results) >= 5:
            _, p_value = wilcoxon(baseline_arr, mobiu_arr, alternative='less')
        else:
            p_value = 1.0
        
        cohen_d = diff.mean() / (diff.std() + 1e-9)
        
        print(f"\n  Results ({env_name}):")
        print(f"    Baseline: {baseline_arr.mean():.1f} ± {baseline_arr.std():.1f}")
        print(f"    Mobiu:    {mobiu_arr.mean():.1f} ± {mobiu_arr.std():.1f}")
        print(f"    Improvement: {improvement:+.1f}%")
        print(f"    p-value: {p_value:.6f}")
        print(f"    Win rate: {win_rate*100:.1f}%")
        
        all_results[env_name] = {
            'baseline_mean': float(baseline_arr.mean()),
            'baseline_std': float(baseline_arr.std()),
            'mobiu_mean': float(mobiu_arr.mean()),
            'mobiu_std': float(mobiu_arr.std()),
            'improvement': float(improvement),
            'p_value': float(p_value),
            'cohen_d': float(cohen_d),
            'win_rate': float(win_rate)
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY - MuJoCo Continuous Control")
    print("=" * 70)
    print(f"\n{'Environment':<25} | {'Baseline':>10} | {'Mobiu':>10} | {'Δ%':>8} | {'Win':>6} | {'p-value':<10}")
    print("-" * 85)
    
    for env_name, results in all_results.items():
        sig = "🏆" if results['p_value'] < 0.05 else ""
        print(f"{env_name:<25} | {results['baseline_mean']:>10.1f} | {results['mobiu_mean']:>10.1f} | {results['improvement']:>+7.1f}% | {results['win_rate']*100:>5.0f}% | {results['p_value']:.4f} {sig}")
    
    # Save
    filename = f"mujoco_maximize_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Saved: {filename}")


if __name__ == "__main__":
    main()
