#!/usr/bin/env python3
"""
================================================================================
🤖 MUJOCO CONTINUOUS CONTROL - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

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
from datetime import datetime
import json

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"
METHOD = "adaptive"

ENVIRONMENTS = [
    "InvertedPendulum-v5",
    "Hopper-v5",
]

NUM_SEEDS = 10
TOTAL_TIMESTEPS = 50_000
STEPS_PER_UPDATE = 2048
BASE_LR = 3e-4

# ============================================================
# ACTOR-CRITIC
# ============================================================

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


# ============================================================
# PPO TRAINING - PURE ADAM (Baseline)
# ============================================================

def train_ppo_pure_adam(env_name, seed):
    """Train with Pure Adam - what customer has BEFORE adding Mobiu-Q"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=1e-5)
    
    gamma, gae_lambda, clip_epsilon = 0.99, 0.95, 0.2
    n_epochs, batch_size = 10, 64
    
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, done_buf, val_buf = [], [], []
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    episode_rewards = []
    current_reward = 0
    
    for step in range(TOTAL_TIMESTEPS):
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
        
        if len(obs_buf) >= STEPS_PER_UPDATE:
            with torch.no_grad():
                _, _, _, last_value = model.get_action_and_value(obs)
            
            advantages, returns = [], []
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
            
            b_obs = torch.stack(obs_buf)
            b_act = torch.stack(act_buf)
            b_logp = torch.stack(logp_buf)
            b_adv = torch.tensor(advantages, dtype=torch.float32)
            b_ret = torch.tensor(returns, dtype=torch.float32)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            
            indices = np.arange(len(obs_buf))
            
            for _ in range(n_epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    
                    _, new_logp, entropy, new_val = model.get_action_and_value(
                        b_obs[batch_idx], b_act[batch_idx]
                    )
                    
                    ratio = torch.exp(new_logp - b_logp[batch_idx])
                    pg_loss1 = -b_adv[batch_idx] * ratio
                    pg_loss2 = -b_adv[batch_idx] * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = F.mse_loss(new_val, b_ret[batch_idx])
                    ent_loss = -entropy.mean()
                    loss = pg_loss + 0.5 * v_loss + 0.01 * ent_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            
            obs_buf.clear()
            act_buf.clear()
            logp_buf.clear()
            rew_buf.clear()
            done_buf.clear()
            val_buf.clear()
    
    env.close()
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else 0


# ============================================================
# PPO TRAINING - WITH MOBIU-Q
# ============================================================

def train_ppo_with_mobiu(env_name, seed):
    """Train with Mobiu-Q - what customer has AFTER adding Mobiu-Q"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model = ActorCritic(obs_dim, act_dim)
    base_opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=1e-5)
    
    # Customer adds Mobiu-Q like this:
    optimizer = MobiuOptimizer(
        base_opt,
        license_key=LICENSE_KEY,
        method=METHOD,
        maximize=True,
        sync_interval=50,
        verbose=False
    )
    
    gamma, gae_lambda, clip_epsilon = 0.99, 0.95, 0.2
    n_epochs, batch_size = 10, 64
    
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, done_buf, val_buf = [], [], []
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    episode_rewards = []
    current_reward = 0
    
    for step in range(TOTAL_TIMESTEPS):
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
        
        if len(obs_buf) >= STEPS_PER_UPDATE:
            with torch.no_grad():
                _, _, _, last_value = model.get_action_and_value(obs)
            
            advantages, returns = [], []
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
            
            b_obs = torch.stack(obs_buf)
            b_act = torch.stack(act_buf)
            b_logp = torch.stack(logp_buf)
            b_adv = torch.tensor(advantages, dtype=torch.float32)
            b_ret = torch.tensor(returns, dtype=torch.float32)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            
            indices = np.arange(len(obs_buf))
            
            for _ in range(n_epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    
                    _, new_logp, entropy, new_val = model.get_action_and_value(
                        b_obs[batch_idx], b_act[batch_idx]
                    )
                    
                    ratio = torch.exp(new_logp - b_logp[batch_idx])
                    pg_loss1 = -b_adv[batch_idx] * ratio
                    pg_loss2 = -b_adv[batch_idx] * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = F.mse_loss(new_val, b_ret[batch_idx])
                    ent_loss = -entropy.mean()
                    loss = pg_loss + 0.5 * v_loss + 0.01 * ent_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    # Mobiu-Q step with reward metric
                    metric = episode_rewards[-1] if episode_rewards else 0
                    optimizer.step(metric)
            
            obs_buf.clear()
            act_buf.clear()
            logp_buf.clear()
            rew_buf.clear()
            done_buf.clear()
            val_buf.clear()
    
    env.close()
    optimizer.end()
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else 0


# ============================================================
# MAIN
# ============================================================

def main():
    if not HAS_GYM:
        print("❌ Please install: pip install gymnasium[mujoco]")
        return
    
    print("=" * 70)
    print("🤖 MUJOCO CONTINUOUS CONTROL - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Method: {METHOD}")
    print(f"Environments: {ENVIRONMENTS}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,} | Seeds: {NUM_SEEDS}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)
    
    all_results = {}
    
    for env_name in ENVIRONMENTS:
        print(f"\n{'='*70}")
        print(f"🎮 {env_name}")
        print("=" * 70)
        
        baseline_results, mobiu_results = [], []
        
        for seed in range(NUM_SEEDS):
            print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
            
            print("    Pure Adam...", end=" ", flush=True)
            baseline_score = train_ppo_pure_adam(env_name, seed)
            print(f"Score: {baseline_score:.1f}")
            
            print("    Adam + Mobiu...", end=" ", flush=True)
            mobiu_score = train_ppo_with_mobiu(env_name, seed)
            print(f"Score: {mobiu_score:.1f}")
            
            baseline_results.append(baseline_score)
            mobiu_results.append(mobiu_score)
            
            diff = mobiu_score - baseline_score
            winner = "✅ Mobiu" if diff > 0 else "❌ Adam"
            print(f"    Δ = {diff:+.1f} → {winner}")
        
        baseline_arr = np.array(baseline_results)
        mobiu_arr = np.array(mobiu_results)
        
        win_rate = np.mean((mobiu_arr - baseline_arr) > 0)
        improvement = 100 * (mobiu_arr.mean() - baseline_arr.mean()) / (abs(baseline_arr.mean()) + 1e-9)
        
        print(f"\n  Results ({env_name}):")
        print(f"    Pure Adam:     {baseline_arr.mean():.1f} ± {baseline_arr.std():.1f}")
        print(f"    Adam + Mobiu:  {mobiu_arr.mean():.1f} ± {mobiu_arr.std():.1f}")
        print(f"    Improvement: {improvement:+.1f}%")
        print(f"    Win rate: {win_rate*100:.1f}%")
        
        all_results[env_name] = {
            'baseline_mean': float(baseline_arr.mean()),
            'mobiu_mean': float(mobiu_arr.mean()),
            'improvement': float(improvement),
            'win_rate': float(win_rate)
        }
    
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    for env_name, results in all_results.items():
        print(f"{env_name}: {results['improvement']:+.1f}% improvement, {results['win_rate']*100:.0f}% wins")


if __name__ == "__main__":
    main()
