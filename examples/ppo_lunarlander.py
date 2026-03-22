#!/usr/bin/env python3
"""
================================================================================
🚀 PPO: Pure Adam vs Adam + Mobiu-Q (THE ULTIMATE FAIR TEST)
================================================================================
Fair comparison: Exactly the same PPO algorithm, same LR, same seeds.
The ONLY difference is wrapping Adam with MobiuOptimizer.

This demonstrates how Mobiu-Q acts as a "Drop-in Replacement" that structurally 
filters noise out of the PPO surrogate loss landscape.
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
import json
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"
ENV_NAME = "LunarLander-v3"

TOTAL_TIMESTEPS = 100_000
STEPS_PER_UPDATE = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LR = 3e-4  # Industry standard for PPO
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

def train_ppo(seed, optimizer_name="adam"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    model = ActorCritic(obs_dim, act_dim)
    
    # ========================================
    # OPTIMIZER SETUP - THE ONLY DIFFERENCE
    # ========================================
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
        use_mobiu = False
        
    elif optimizer_name == "adam+mobiu":
        base_adam = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
        optimizer = MobiuOptimizer(
            base_adam,
            license_key=LICENSE_KEY,
            boost="aggressive",
            method="adaptive", # Best for RL variance
            verbose=False
        )
        use_mobiu = True
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    
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
                    
                    # The surrogate loss we want to minimize
                    loss = pg_loss + VF_COEF * v_loss + ENT_COEF * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    
                    # THE ONLY DIFFERENCE IN THE PPO UPDATE LOOP:
                    if use_mobiu:
                        optimizer.step(loss.item()) # <--- CRITICAL FIX: Passing dynamic loss!
                    else:
                        optimizer.step()
            
            buffer.clear()
    
    env.close()
    
    if use_mobiu:
        optimizer.end()
    
    # Return average of the last 20 episodes to gauge final stable performance
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else -200


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 PPO: Pure Adam vs Adam + Mobiu-Q (FAIR BENCHMARK)")
    print("=" * 70)
    print(f"Environment: {ENV_NAME}")
    print(f"Timesteps:   {TOTAL_TIMESTEPS:,}")
    print(f"Learning R:  {LR} (Industry Standard)")
    print(f"Seeds:       {NUM_SEEDS}")
    print()
    print("Testing the exact same PPO implementation. The only difference")
    print("is passing the surrogate loss to MobiuOptimizer.step(loss.item())")
    print("=" * 70)
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\n[Seed {seed + 1}/{NUM_SEEDS}]")
        
        # Save RNG state — both runs see identical environments
        torch_state = torch.get_rng_state()
        np_state    = np.random.get_state()

        # Test 1: Plain Adam
        print("  Running Pure Adam...    ", end="", flush=True)
        adam_score = train_ppo(seed, optimizer_name="adam")
        print(f"Final Score: {adam_score:6.1f}")

        # Restore RNG state — Mobiu starts from exactly the same point
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)

        # Test 2: Adam + Mobiu-Q wrapper
        print("  Running Adam+Mobiu-Q... ", end="", flush=True)
        mobiu_score = train_ppo(seed, optimizer_name="adam+mobiu")
        print(f"Final Score: {mobiu_score:6.1f}")
        
        adam_results.append(adam_score)
        mobiu_results.append(mobiu_score)
        
        diff = mobiu_score - adam_score
        winner = "✅ Mobiu wins" if diff > 0 else "❌ Adam wins"
        print(f"  -> Δ = {diff:+.1f} ({winner})")
    
    # Statistics
    adam_arr = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    diff = mobiu_arr - adam_arr
    
    win_rate = np.mean(diff > 0)
    
    try:
        _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative='less')
    except:
        p_value = 1.0
    
    improvement = 100 * (mobiu_arr.mean() - adam_arr.mean()) / (abs(adam_arr.mean()) + 1e-9)
    cohen_d = diff.mean() / (diff.std() + 1e-9)
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Pure Adam:       {adam_arr.mean():>8.1f} ± {adam_arr.std():.1f}")
    print(f"Adam + Mobiu-Q:  {mobiu_arr.mean():>8.1f} ± {mobiu_arr.std():.1f}")
    print("-" * 70)
    print(f"📈 Improvement:  {improvement:+.1f}%")
    print(f"🏆 Win rate:     {win_rate*100:.1f}% ({int(win_rate*NUM_SEEDS)}/{NUM_SEEDS})")
    print(f"⚖️ p-value:      {p_value:.6f}")
    print(f"📏 Cohen's d:    {cohen_d:+.2f}")
    print("=" * 70)
    
    # Save Results
    filename = f'ppo_lunarlander_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump({
            'environment': ENV_NAME,
            'timesteps': TOTAL_TIMESTEPS,
            'adam_mean': float(adam_arr.mean()),
            'mobiu_mean': float(mobiu_arr.mean()),
            'improvement_pct': float(improvement),
            'win_rate': float(win_rate),
            'p_value': float(p_value),
            'adam_scores': [float(x) for x in adam_arr],
            'mobiu_scores': [float(x) for x in mobiu_arr],
        }, f, indent=2)

if __name__ == "__main__":
    main()