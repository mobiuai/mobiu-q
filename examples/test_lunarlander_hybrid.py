#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q REAL TEST - LunarLander-v3
================================================================================
Tests Mobiu-Q on actual LunarLander-v3 using fair A/B comparison.

Fair Test Methodology:
- Both use MobiuOptimizer wrapping PyTorch Adam
- Adam baseline: use_soft_algebra=False (constant LR)
- Mobiu-Q: use_soft_algebra=True (adaptive LR from Soft Algebra)
- Same seeds, same initial weights, same everything except Soft Algebra

Requirements:
    pip install mobiu-q gymnasium torch numpy scipy

Usage:
    python test_lunarlander_hybrid.py
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
# TRAINING FUNCTION
# ============================================================

def train_policy(env_name, policy, optimizer, num_episodes, seed, label=""):
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
        optimizer.step(sum(rewards))
        
        returns.append(sum(rewards))
        
        if (ep + 1) % 200 == 0:
            print(f"  {label} ep {ep+1}: avg last 100 = {np.mean(returns[-100:]):.1f}")
    
    env.close()
    return returns


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 70)
    print("🚀 MOBIU-Q FAIR A/B TEST - LunarLander-v3")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Environment: {ENV_NAME}")
    print(f"Episodes: {NUM_EPISODES} | Seeds: {NUM_SEEDS} | LR: {BASE_LR}")
    print(f"Method: {METHOD}")
    print()
    print("Fair Test: Both use MobiuOptimizer")
    print("  • Adam:  use_soft_algebra=False (constant LR)")
    print("  • Mobiu: use_soft_algebra=True  (AUTO adaptive LR)")
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
        
        # Train with Adam (use_soft_algebra=False)
        base_opt_adam = torch.optim.Adam(adam_policy.parameters(), lr=BASE_LR)
        adam_opt = MobiuOptimizer(
            base_opt_adam,
            license_key=LICENSE_KEY,
            method=METHOD,
            use_soft_algebra=False,
            verbose=False
        )
        
        adam_returns = train_policy(ENV_NAME, adam_policy, adam_opt, NUM_EPISODES, seed, "Adam")
        adam_opt.end()
        
        # Train with Mobiu-Q (use_soft_algebra=True)
        base_opt_mobiu = torch.optim.Adam(mobiu_policy.parameters(), lr=BASE_LR)
        mobiu_opt = MobiuOptimizer(
            base_opt_mobiu,
            license_key=LICENSE_KEY,
            method=METHOD,
            maximize=True,
            use_soft_algebra=True,
            verbose=False
        )
        
        mobiu_returns = train_policy(ENV_NAME, mobiu_policy, mobiu_opt, NUM_EPISODES, seed, "Mobiu")
        mobiu_opt.end()
        
        adam_final = np.mean(adam_returns[-100:])
        mobiu_final = np.mean(mobiu_returns[-100:])
        
        adam_results.append(adam_final)
        mobiu_results.append(mobiu_final)
        
        winner = "✅ Mobiu" if mobiu_final > adam_final else "❌ Adam"
        diff = mobiu_final - adam_final
        print(f"  Adam: {adam_final:.1f} | Mobiu: {mobiu_final:.1f} | Δ={diff:+.1f} → {winner}")
    
    # Statistics
    adam_arr = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    diff = mobiu_arr - adam_arr
    
    _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative="less")
    pct_improvement = 100 * diff.mean() / (abs(adam_arr.mean()) + 1e-9)
    cohen_d = diff.mean() / (diff.std() + 1e-9)
    win_rate = sum(d > 0 for d in diff) / len(diff)
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - LunarLander-v3")
    print("=" * 70)
    print(f"Adam (SA=off):     {adam_arr.mean():.1f} ± {adam_arr.std():.1f}")
    print(f"Mobiu (SA=on): {mobiu_arr.mean():.1f} ± {mobiu_arr.std():.1f}")
    print(f"\nImprovement: {pct_improvement:+.1f}%")
    print(f"p-value: {p_value:.6f}")
    print(f"Cohen's d: {cohen_d:+.2f}")
    print(f"Win rate: {win_rate*100:.1f}%")
    
    if pct_improvement > 5 and p_value < 0.05:
        print("\n🏆 SIGNIFICANT IMPROVEMENT!")
    elif p_value < 0.05:
        print("\n✅ Statistically significant (p < 0.05)")
    else:
        print("\n🔶 Not statistically significant")
    
    print("=" * 70)
    
    results = {
        'method': METHOD,
        'adam_mean': float(adam_arr.mean()),
        'mobiu_mean': float(mobiu_arr.mean()),
        'improvement_pct': float(pct_improvement),
        'p_value': float(p_value),
        'win_rate': float(win_rate)
    }
    
    with open('lunarlander_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
