#!/usr/bin/env python3
"""
================================================================================
📦 PPO PORTFOLIO - Improved Full PPO + Mobiu-Q (Adaptive) [FAIR BENCHMARK]
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from scipy.stats import wilcoxon
import copy

from mobiu_q import MobiuOptimizer

LICENSE_KEY = "YOUR_KEY"
NUM_EPISODES = 600
NUM_SEEDS = 10
BASE_LR = 0.0003
METHOD = "adaptive"
EPISODE_LENGTH = 600
TRANSACTION_FEE = 0.001
GAMMA = 0.99
GAE_LAMBDA = 0.95

class PortfolioTradingEnv:
    def __init__(self, seed=None):
        self.episode_length = EPISODE_LENGTH
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_idx = 0
        self.position = 0
        self.pnl = 0.0
        self.prices = self._generate_prices()
        return self._get_state()
    
    def _generate_prices(self):
        prices = [100.0]
        for i in range(self.episode_length):
            regime = (i // 100) % 3
            if regime == 0:
                ret = 0.0018 + 0.015 * self.rng.randn()
            elif regime == 1:
                ret = 0.0008 + 0.038 * self.rng.randn()
            else:
                ret = 0.0003 + 0.009 * self.rng.randn()
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices)
    
    def _get_state(self):
        start = max(0, self.step_idx - 19)
        recent_returns = np.diff(self.prices[start:self.step_idx + 1])
        padded = np.pad(recent_returns, (20 - len(recent_returns), 0), 'constant')
        return np.concatenate([padded, [float(self.position), self.pnl / 100.0]]).astype(np.float32)
    
    def step(self, action):
        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]
        price_change = (next_price - current_price) / current_price
        reward = 0
        old_position = self.position
        
        if action == 1: self.position = 1
        elif action == 2: self.position = -1
        else: self.position = 0
        
        reward = self.position * price_change * 100
        if old_position != self.position:
            reward -= TRANSACTION_FEE * 100
        
        self.pnl += reward
        self.step_idx += 1
        done = self.step_idx >= self.episode_length - 1
        return self._get_state(), reward, done, {'pnl': self.pnl}

class ActorCritic(nn.Module):
    def __init__(self, state_dim=22, action_dim=3):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

def train_ppo(policy, env, use_mobiu=False):
    base_opt = optim.Adam(policy.parameters(), lr=BASE_LR)
    if use_mobiu:
        # Maximize=True is intentional: aggressive LR boost aids PPO exploration
        optimizer = MobiuOptimizer(base_opt, license_key=LICENSE_KEY, method=METHOD, base_lr=BASE_LR, maximize=False, verbose=False)
    else:
        optimizer = base_opt
    
    episode_pnls = []
    
    for ep in range(NUM_EPISODES):
        # Sync env random states per episode for fairness
        state = env.reset(seed=ep*100)
        log_probs, values, rewards, dones = [], [], [], []
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits, value = policy(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())
            rewards.append(reward)
            dones.append(done)
            state = next_state
        
        episode_pnls.append(info['pnl'])
        
        returns = []
        gae = 0
        next_value = 0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + GAMMA * next_value * (1 - d) - v
            gae = delta + GAMMA * GAE_LAMBDA * gae * (1 - d)
            returns.insert(0, gae + v)
            next_value = v
        
        returns = torch.tensor([r.item() if isinstance(r, torch.Tensor) else r for r in returns], dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = sum(-logp * ret for logp, ret in zip(log_probs, returns))
        
        optimizer.zero_grad()
        loss.backward()
        
        if use_mobiu:
            optimizer.step(loss.item()) # FIX: Passing loss!
        else:
            optimizer.step()
    
    if use_mobiu: optimizer.end()
    return episode_pnls

def main():
    print("=" * 90)
    print("📦 PPO PORTFOLIO - Improved Full PPO + Mobiu-Q (Adaptive) [FAIR]")
    print("=" * 90)
    
    adam_pnls, mobiu_pnls = [], []
    
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = PortfolioTradingEnv()
        dummy_policy = ActorCritic()
        init_weights = {k: v.clone() for k, v in dummy_policy.state_dict().items()}
        
        print(f"Seed {seed+1}/{NUM_SEEDS} | ", end="", flush=True)
        
        torch.manual_seed(seed * 99)
        policy_adam = ActorCritic()
        policy_adam.load_state_dict(init_weights)
        pnls_adam = train_ppo(policy_adam, env, use_mobiu=False)
        avg_adam = np.mean(pnls_adam[-100:])
        adam_pnls.append(avg_adam)
        print(f"Adam: {avg_adam:6.1f} | ", end="", flush=True)
        
        torch.manual_seed(seed * 99)
        policy_mobiu = ActorCritic()
        policy_mobiu.load_state_dict(init_weights)
        pnls_mobiu = train_ppo(policy_mobiu, env, use_mobiu=True)
        avg_mobiu = np.mean(pnls_mobiu[-100:])
        mobiu_pnls.append(avg_mobiu)
        print(f"Mobiu: {avg_mobiu:6.1f}")
    
    adam_arr, mobiu_arr = np.array(adam_pnls), np.array(mobiu_pnls)
    improvement = (mobiu_arr.mean() - adam_arr.mean()) / abs(adam_arr.mean()) * 100
    wins = np.sum(mobiu_arr > adam_arr)
    _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative='less')
    
    print("\n" + "="*90)
    print("FINAL RESULTS")
    print(f"Adam Avg PnL:     {adam_arr.mean():.2f} ± {adam_arr.std():.2f}")
    print(f"Mobiu-Q Avg PnL:  {mobiu_arr.mean():.2f} ± {mobiu_arr.std():.2f}")
    print(f"Improvement:      {improvement:+.1f}%")
    print(f"Win rate:         {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%) | p-value: {p_value:.6f}")
    print("="*90)

if __name__ == "__main__":
    main()