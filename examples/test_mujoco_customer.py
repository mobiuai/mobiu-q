#!/usr/bin/env python3
"""
================================================================================
🤖 MUJOCO CONTINUOUS CONTROL - CUSTOMER VIEW TEST (FAIR BENCHMARK)
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer
- Test: Adam + Mobiu-Q 

Requirements:
    pip install mobiu-q gymnasium[mujoco] torch numpy scipy
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.stats import wilcoxon
import json

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from mobiu_q import MobiuOptimizer

LICENSE_KEY = "YOUR_KEY"
METHOD = "adaptive"
ENVIRONMENTS = ["InvertedPendulum-v5", "Hopper-v5"]
NUM_SEEDS = 10
TOTAL_TIMESTEPS = 50_000
STEPS_PER_UPDATE = 2048
BASE_LR = 3e-4

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
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


def train_ppo(env_name, seed, init_weights, use_mobiu=False):
    """Unified training function for perfect fairness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(init_weights) # IDENTICAL INIT
    
    base_opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=1e-5)
    
    if use_mobiu:
        optimizer = MobiuOptimizer(
            base_opt, license_key=LICENSE_KEY, method=METHOD,
            # Maximize=True is intentional: aggressive LR boost aids PPO exploration 
            base_lr=BASE_LR, maximize=True, sync_interval=50, verbose=False
        )
    else:
        optimizer = base_opt
        
    gamma, gae_lambda, clip_epsilon = 0.99, 0.95, 0.2
    n_epochs, batch_size = 10, 64
    
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    episode_rewards, current_reward = [], 0
    
    for step in range(TOTAL_TIMESTEPS):
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs)
        
        action_np = np.clip(action.numpy(), env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        current_reward += reward
        
        obs_buf.append(obs); act_buf.append(action); logp_buf.append(log_prob)
        rew_buf.append(reward); done_buf.append(done); val_buf.append(value)
        
        obs = torch.tensor(next_obs, dtype=torch.float32)
        
        if done:
            episode_rewards.append(current_reward)
            current_reward = 0
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
        
        if len(obs_buf) >= STEPS_PER_UPDATE:
            with torch.no_grad():
                _, _, _, last_value = model.get_action_and_value(obs)
            
            advantages, returns, gae = [], [], 0
            values = val_buf + [last_value]
            
            for t in reversed(range(len(rew_buf))):
                delta = rew_buf[t] + gamma * values[t + 1] * (not done_buf[t]) - values[t]
                gae = delta + gamma * gae_lambda * gae * (not done_buf[t])
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            b_obs, b_act, b_logp = torch.stack(obs_buf), torch.stack(act_buf), torch.stack(logp_buf)
            b_adv, b_ret = torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            
            indices = np.arange(len(obs_buf))
            for _ in range(n_epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    _, new_logp, entropy, new_val = model.get_action_and_value(b_obs[batch_idx], b_act[batch_idx])
                    
                    ratio = torch.exp(new_logp - b_logp[batch_idx])
                    pg_loss = torch.max(-b_adv[batch_idx] * ratio, -b_adv[batch_idx] * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)).mean()
                    v_loss = F.mse_loss(new_val, b_ret[batch_idx])
                    loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    if use_mobiu:
                        optimizer.step(loss.item()) # FIX: dynamic loss
                    else:
                        optimizer.step()
            
            obs_buf.clear(); act_buf.clear(); logp_buf.clear()
            rew_buf.clear(); done_buf.clear(); val_buf.clear()
    
    env.close()
    if use_mobiu: optimizer.end()
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else 0


def main():
    if not HAS_GYM:
        print("❌ Please install: pip install gymnasium[mujoco]")
        return
    
    print("=" * 70)
    print("🤖 MUJOCO CONTINUOUS CONTROL (FAIR BENCHMARK)")
    print("=" * 70)
    
    all_results = {}
    
    for env_name in ENVIRONMENTS:
        print(f"\n🎮 {env_name}")
        print("-" * 50)
        
        baseline_results, mobiu_results = [], []
        
        # Get dummy env just to initialize shapes for identical weights
        dummy_env = gym.make(env_name)
        obs_dim, act_dim = dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0]
        dummy_env.close()
        
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            dummy_model = ActorCritic(obs_dim, act_dim)
            init_weights = {k: v.clone() for k, v in dummy_model.state_dict().items()}
            
            print(f"  Seed {seed + 1:<2}/{NUM_SEEDS} | ", end="", flush=True)
            
            # Sync RNG
            torch_state = torch.get_rng_state()
            np_state = np.random.get_state()
            
            baseline_score = train_ppo(env_name, seed, init_weights, use_mobiu=False)
            print(f"Adam: {baseline_score:6.1f} | ", end="", flush=True)
            
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)
            
            mobiu_score = train_ppo(env_name, seed, init_weights, use_mobiu=True)
            print(f"Mobiu: {mobiu_score:6.1f}", end="", flush=True)
            
            baseline_results.append(baseline_score)
            mobiu_results.append(mobiu_score)
            
            diff = mobiu_score - baseline_score
            winner = "✅ Mobiu" if diff > 0 else "❌ Adam"
            print(f"  -> Δ={diff:+.1f} {winner}")
        
        b_arr, m_arr = np.array(baseline_results), np.array(mobiu_results)
        win_rate = np.mean((m_arr - b_arr) > 0)
        improvement = 100 * (m_arr.mean() - b_arr.mean()) / (abs(b_arr.mean()) + 1e-9)
        
        all_results[env_name] = {'improvement': float(improvement), 'win_rate': float(win_rate)}
    
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    for env_name, results in all_results.items():
        print(f"{env_name:<20}: {results['improvement']:>+6.1f}% improvement | {results['win_rate']*100:.0f}% wins")

if __name__ == "__main__":
    main()