#!/usr/bin/env python3
"""
================================================================================
🚀 LUNARLANDER - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

LunarLander-v3:
- Classic RL benchmark
- Discrete action space (4 actions)
- Target: Land safely between flags
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from scipy.stats import wilcoxon
from datetime import datetime
import json
import copy

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

NUM_EPISODES = 1000
NUM_SEEDS = 30
BASE_LR = 0.0003
METHOD = "adaptive"
ENV_NAME = "LunarLander-v3"


# ============================================================
# POLICY NETWORK
# ============================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, state):
        with torch.no_grad():
            logits = self.forward(torch.FloatTensor(state).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def get_log_prob(self, state, action):
        logits = self.forward(torch.FloatTensor(state).unsqueeze(0))
        return F.log_softmax(logits, dim=-1)[0, action]


# ============================================================
# TRAINING - PURE ADAM (Baseline)
# ============================================================

def train_pure_adam(env_name, policy, num_episodes, seed, label=""):
    """Train with Pure Adam - what customer has BEFORE adding Mobiu-Q"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=BASE_LR)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    returns = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        
        while not done:
            action = policy.get_action(state)
            log_probs.append(policy.get_log_prob(state, action))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        
        G = 0
        disc_returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            disc_returns.insert(0, G)
        disc_returns = torch.tensor(disc_returns)
        
        baseline = disc_returns.mean()
        advantages = (disc_returns - baseline) / (disc_returns.std() + 1e-8)
        
        loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        returns.append(sum(rewards))
        
        if (ep + 1) % 200 == 0:
            print(f"  {label} ep {ep+1}: avg last 100 = {np.mean(returns[-100:]):.1f}")
    
    env.close()
    return returns


# ============================================================
# TRAINING - WITH MOBIU-Q
# ============================================================

def train_with_mobiu(env_name, policy, num_episodes, seed, label=""):
    """Train with Mobiu-Q - what customer has AFTER adding Mobiu-Q"""
    base_opt = torch.optim.Adam(policy.parameters(), lr=BASE_LR)
    optimizer = MobiuOptimizer(
        base_opt,
        license_key=LICENSE_KEY,
        method=METHOD,
        maximize=True,
        verbose=False
    )
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    returns = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        
        while not done:
            action = policy.get_action(state)
            log_probs.append(policy.get_log_prob(state, action))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        
        G = 0
        disc_returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            disc_returns.insert(0, G)
        disc_returns = torch.tensor(disc_returns)
        
        baseline = disc_returns.mean()
        advantages = (disc_returns - baseline) / (disc_returns.std() + 1e-8)
        
        loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(sum(rewards))  # Pass reward metric to Mobiu
        
        returns.append(sum(rewards))
        
        if (ep + 1) % 200 == 0:
            print(f"  {label} ep {ep+1}: avg last 100 = {np.mean(returns[-100:]):.1f}")
    
    optimizer.end()
    env.close()
    return returns


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 LUNARLANDER - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Environment: {ENV_NAME}")
    print(f"Episodes: {NUM_EPISODES} | Seeds: {NUM_SEEDS} | LR: {BASE_LR}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)
    
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\nSeed {seed + 1}/{NUM_SEEDS}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        template = PolicyNetwork(state_dim, action_dim)
        adam_policy = copy.deepcopy(template)
        mobiu_policy = copy.deepcopy(template)
        
        # Train with Pure Adam
        adam_returns = train_pure_adam(ENV_NAME, adam_policy, NUM_EPISODES, seed, "Adam ")
        
        # Train with Mobiu-Q
        mobiu_returns = train_with_mobiu(ENV_NAME, mobiu_policy, NUM_EPISODES, seed, "Mobiu")
        
        adam_final = np.mean(adam_returns[-100:])
        mobiu_final = np.mean(mobiu_returns[-100:])
        
        adam_results.append(adam_final)
        mobiu_results.append(mobiu_final)
        
        winner = "✅ Mobiu" if mobiu_final > adam_final else "❌ Adam"
        diff = mobiu_final - adam_final
        print(f"  Pure Adam: {adam_final:.1f} | Adam+Mobiu: {mobiu_final:.1f} | Δ={diff:+.1f} → {winner}")
    
    # Statistics
    adam_arr = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    diff = mobiu_arr - adam_arr
    
    _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative="less")
    pct_improvement = 100 * diff.mean() / (abs(adam_arr.mean()) + 1e-9)
    cohen_d = diff.mean() / (diff.std() + 1e-9)
    win_rate = sum(d > 0 for d in diff) / len(diff)
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Pure Adam:      {adam_arr.mean():.1f} ± {adam_arr.std():.1f}")
    print(f"Adam + Mobiu:   {mobiu_arr.mean():.1f} ± {mobiu_arr.std():.1f}")
    print(f"\nImprovement: {pct_improvement:+.1f}%")
    print(f"p-value: {p_value:.6f}")
    print(f"Cohen's d: {cohen_d:+.2f}")
    print(f"Win rate: {win_rate*100:.1f}%")
    
    if pct_improvement > 5 and p_value < 0.05:
        print("\n🏆 SIGNIFICANT IMPROVEMENT!")
    print("=" * 70)
    
    with open('lunarlander_customer_results.json', 'w') as f:
        json.dump({
            'adam_mean': float(adam_arr.mean()),
            'mobiu_mean': float(mobiu_arr.mean()),
            'improvement_pct': float(pct_improvement),
            'p_value': float(p_value),
            'win_rate': float(win_rate)
        }, f, indent=2)


if __name__ == "__main__":
    main()
