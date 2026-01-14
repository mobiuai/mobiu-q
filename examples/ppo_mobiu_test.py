#!/usr/bin/env python3
"""
================================================================================
PPO from Scratch with Mobiu-Q
================================================================================
Clean PPO implementation for LunarLander-v3 with fair A/B testing.

Usage:
    python ppo_mobiu_test.py
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
from scipy.stats import wilcoxon
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

try:
    from mobiu_q import MobiuOptimizer
    HAS_MOBIU = True
except ImportError:
    HAS_MOBIU = False

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"
METHOD = "adaptive"  
ENV_NAME = "LunarLander-v3"

TOTAL_TIMESTEPS = 100_000
STEPS_PER_UPDATE = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NUM_SEEDS = 30


# ============================================================
# ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)


# ============================================================
# ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
    
    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
    
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        advantages, returns = [], []
        gae = 0
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def get_tensors(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(self.values, dtype=torch.float32)
        )


# ============================================================
# PPO TRAINING
# ============================================================

def train_ppo(seed, use_mobiu=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    model = ActorCritic(obs_dim, act_dim)
    base_opt = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    
    if use_mobiu and HAS_MOBIU:
        optimizer = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method=METHOD,
            use_soft_algebra=True,
            maximize=True,
            sync_interval=50,
            verbose=False
        )
    else:
        optimizer = base_opt
    
    buffer = RolloutBuffer()
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(TOTAL_TIMESTEPS):
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        current_episode_reward += reward
        
        buffer.add(obs, action, log_prob, reward, done, value.item())
        
        obs = torch.tensor(next_obs, dtype=torch.float32)
        
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
        
        if len(buffer.obs) >= STEPS_PER_UPDATE:
            with torch.no_grad():
                _, _, _, last_value = model.get_action_and_value(obs)
            
            advantages, returns = buffer.compute_returns_and_advantages(last_value.item(), GAMMA, GAE_LAMBDA)
            
            b_obs, b_actions, b_log_probs, b_values = buffer.get_tensors()
            b_advantages = torch.tensor(advantages, dtype=torch.float32)
            b_returns = torch.tensor(returns, dtype=torch.float32)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            indices = np.arange(len(buffer.obs))
            
            for epoch in range(N_EPOCHS):
                np.random.shuffle(indices)
                
                for start in range(0, len(indices), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    batch_indices = indices[start:end]
                    
                    mb_obs = b_obs[batch_indices]
                    mb_actions = b_actions[batch_indices]
                    mb_log_probs = b_log_probs[batch_indices]
                    mb_advantages = b_advantages[batch_indices]
                    mb_returns = b_returns[batch_indices]
                    
                    _, new_log_probs, entropy, new_values = model.get_action_and_value(mb_obs, mb_actions)
                    
                    log_ratio = new_log_probs - mb_log_probs
                    ratio = log_ratio.exp()
                    
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = F.mse_loss(new_values, mb_returns)
                    entropy_loss = -entropy.mean()
                    loss = pg_loss + VF_COEF * v_loss + ENT_COEF * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    
                    if use_mobiu and HAS_MOBIU:
                        metric = episode_rewards[-1] if episode_rewards else 0
                        optimizer.step(metric)
                    else:
                        optimizer.step()
            
            buffer.clear()
    
    env.close()
    
    if use_mobiu and HAS_MOBIU:
        optimizer.end()
    
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else -200


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 PPO + MOBIU TEST")
    print("=" * 70)
    print(f"Environment: {ENV_NAME}")
    print(f"Method: {METHOD}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Seeds: {NUM_SEEDS}")
    print("=" * 70)
    
    baseline_results, mobiu_results = [], []
    
    for seed in range(NUM_SEEDS):
        print(f"\nSeed {seed + 1}/{NUM_SEEDS}")
        
        print("  Training Baseline...", end=" ", flush=True)
        baseline_score = train_ppo(seed, use_mobiu=False)
        print(f"Score: {baseline_score:.1f}")
        
        print("  Training Mobiu...", end=" ", flush=True)
        mobiu_score = train_ppo(seed, use_mobiu=True)
        print(f"Score: {mobiu_score:.1f}")
        
        baseline_results.append(baseline_score)
        mobiu_results.append(mobiu_score)
        
        diff = mobiu_score - baseline_score
        winner = "✅ Mobiu" if diff > 0 else "❌ Baseline"
        print(f"  Δ = {diff:+.1f} → {winner}")
    
    baseline_arr = np.array(baseline_results)
    mobiu_arr = np.array(mobiu_results)
    diff = mobiu_arr - baseline_arr
    
    win_rate = np.mean(diff > 0)
    
    if len(baseline_results) >= 5:
        _, p_value = wilcoxon(baseline_arr, mobiu_arr, alternative='less')
    else:
        p_value = 1.0
    
    improvement = 100 * (mobiu_arr.mean() - baseline_arr.mean()) / (abs(baseline_arr.mean()) + 1e-9)
    cohen_d = diff.mean() / (diff.std() + 1e-9)
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - PPO")
    print("=" * 70)
    print(f"Baseline (Adam):   {baseline_arr.mean():.1f} ± {baseline_arr.std():.1f}")
    print(f"Mobiu:      {mobiu_arr.mean():.1f} ± {mobiu_arr.std():.1f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"p-value: {p_value:.6f}")
    print(f"Cohen's d: {cohen_d:+.2f}")
    print(f"Win rate: {win_rate*100:.1f}%")
    print("=" * 70)
    
    if improvement > 5 and p_value < 0.05:
        print("🏆 SIGNIFICANT IMPROVEMENT!")
    elif p_value < 0.05:
        print("✅ Statistically significant")
    else:
        print("🔶 Not statistically significant")


if __name__ == "__main__":
    main()
