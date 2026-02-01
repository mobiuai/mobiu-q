#!/usr/bin/env python3
"""
================================================================================
🪙 CRYPTO TRADING - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

Crypto Trading Environment:
- Simulated crypto market with regime switching
- High volatility and transaction costs
- Tests Mobiu-Q on non-stationary rewards
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
from scipy.stats import wilcoxon

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"
METHOD = "adaptive"

NUM_EPISODES = 500
NUM_SEEDS = 10
BASE_LR = 0.0003
WINDOW_SIZE = 20
EPISODE_LENGTH = 500
TRANSACTION_FEE = 0.001

print("=" * 70)
print("🪙 CRYPTO TRADING - CUSTOMER VIEW TEST")
print("=" * 70)
print(f"Method: {METHOD}")
print(f"Episodes: {NUM_EPISODES} | Seeds: {NUM_SEEDS}")
print()
print("This test shows what a CUSTOMER would experience:")
print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
print("  • Test: Adam + Mobiu-Q enhancement")
print("=" * 70)


# ============================================================
# CRYPTO TRADING ENVIRONMENT
# ============================================================

class CryptoTradingEnv:
    def __init__(self, window_size=20, episode_length=500, seed=None):
        self.window_size = window_size
        self.episode_length = episode_length
        self.rng = np.random.RandomState(seed)
        self.state_dim = window_size + 4
        self.action_space_n = 4
        
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._generate_crypto_prices()
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trade_count = 0
        return self._get_state(), {}
    
    def _generate_crypto_prices(self):
        n = self.episode_length + self.window_size + 10
        base_vol = self.rng.uniform(0.02, 0.05)
        drift = self.rng.uniform(-0.001, 0.002)
        returns = []
        vol = base_vol
        regime = 1
        
        for i in range(n):
            if self.rng.random() < 0.02:
                regime *= -1
            vol = 0.85 * vol + 0.15 * base_vol * (1 + abs(self.rng.randn()))
            if self.rng.random() < 0.005:
                shock = self.rng.choice([-1, 1]) * self.rng.uniform(0.05, 0.15)
            else:
                shock = 0
            ret = regime * drift + vol * self.rng.randn() + shock
            returns.append(ret)
        
        self.prices = 1000 * np.exp(np.cumsum(returns))
        self.returns = np.array(returns)
        self.volatilities = self._compute_rolling_vol()
    
    def _compute_rolling_vol(self, window=10):
        vols = []
        for i in range(len(self.returns)):
            start = max(0, i - window)
            vols.append(np.std(self.returns[start:i+1]) if i > 0 else 0.02)
        return np.array(vols)
    
    def _get_state(self):
        window_returns = self.returns[self.current_step - self.window_size:self.current_step]
        window_returns = window_returns / (np.std(window_returns) + 1e-8)
        current_vol = self.volatilities[self.current_step] / 0.05
        position_ind = float(self.position)
        
        if self.position != 0:
            current_price = self.prices[self.current_step]
            unrealized = self.position * (current_price - self.entry_price) / self.entry_price
        else:
            unrealized = 0
        unrealized = np.tanh(unrealized * 10)
        trade_freq = min(self.trade_count / 50, 1.0)
        
        return np.concatenate([
            window_returns,
            [current_vol, position_ind, unrealized, trade_freq]
        ]).astype(np.float32)
    
    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        
        if action == 1 and self.position != 1:  # Buy/Long
            if self.position == -1:
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - TRANSACTION_FEE
                self.total_profit += pnl - TRANSACTION_FEE
            self.position = 1
            self.entry_price = current_price
            self.trade_count += 1
            reward -= TRANSACTION_FEE
            
        elif action == 2 and self.position != -1:  # Sell/Short
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
                reward += pnl - TRANSACTION_FEE
                self.total_profit += pnl - TRANSACTION_FEE
            self.position = -1
            self.entry_price = current_price
            self.trade_count += 1
            reward -= TRANSACTION_FEE
            
        elif action == 3 and self.position != 0:  # Close position
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            reward += pnl - TRANSACTION_FEE
            self.total_profit += pnl - TRANSACTION_FEE
            self.position = 0
            self.entry_price = 0
            self.trade_count += 1
        
        self.current_step += 1
        done = self.current_step >= self.episode_length + self.window_size - 1
        
        if done and self.position != 0:
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            reward += pnl - TRANSACTION_FEE
            self.total_profit += pnl - TRANSACTION_FEE
        
        return self._get_state(), reward, done, False, {'total_profit': self.total_profit}


# ============================================================
# POLICY NETWORK
# ============================================================

class TradingPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state):
        logits = self.forward(torch.FloatTensor(state).unsqueeze(0))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# ============================================================
# TRAINING - PURE ADAM (Baseline)
# ============================================================

def train_pure_adam(policy, env, num_episodes):
    """Train with Pure Adam - what customer has BEFORE adding Mobiu-Q"""
    optimizer = optim.Adam(policy.parameters(), lr=BASE_LR)
    
    episode_returns, episode_profits = [], []
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep * 12345)
        log_probs, rewards = [], []
        
        done = False
        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_returns.append(sum(rewards))
        episode_profits.append(info['total_profit'])
    
    return episode_returns, episode_profits


# ============================================================
# TRAINING - WITH MOBIU-Q
# ============================================================

def train_with_mobiu(policy, env, num_episodes):
    """Train with Mobiu-Q - what customer has AFTER adding Mobiu-Q"""
    base_opt = optim.Adam(policy.parameters(), lr=BASE_LR)
    optimizer = MobiuOptimizer(
        base_opt,
        license_key=LICENSE_KEY,
        method=METHOD,
        maximize=True,
        verbose=False
    )
    
    episode_returns, episode_profits = [], []
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep * 12345)
        log_probs, rewards = [], []
        
        done = False
        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(sum(rewards))  # Pass reward metric to Mobiu
        
        episode_returns.append(sum(rewards))
        episode_profits.append(info['total_profit'])
    
    optimizer.end()
    return episode_returns, episode_profits


# ============================================================
# MAIN
# ============================================================

def main():
    adam_results, adam_profits = [], []
    mobiu_results, mobiu_profits = [], []
    
    for seed in range(NUM_SEEDS):
        print(f"\nSeed {seed+1}/{NUM_SEEDS} ", end="", flush=True)
        
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        env = CryptoTradingEnv(WINDOW_SIZE, EPISODE_LENGTH)
        
        policy_adam = TradingPolicy(env.state_dim)
        policy_mobiu = copy.deepcopy(policy_adam)
        
        # Train with Pure Adam
        print("[Adam", end="", flush=True)
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        ret_adam, prof_adam = train_pure_adam(policy_adam, env, NUM_EPISODES)
        print(".....] ", end="", flush=True)
        
        # Train with Mobiu-Q
        print("[Mobiu", end="", flush=True)
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        ret_mobiu, prof_mobiu = train_with_mobiu(policy_mobiu, env, NUM_EPISODES)
        print(".....] ", end="", flush=True)
        
        avg_adam = np.mean(prof_adam[-50:])
        avg_mobiu = np.mean(prof_mobiu[-50:])
        
        adam_results.append(np.mean(ret_adam[-50:]))
        adam_profits.append(avg_adam)
        mobiu_results.append(np.mean(ret_mobiu[-50:]))
        mobiu_profits.append(avg_mobiu)
        
        winner = "✅" if avg_mobiu > avg_adam else "❌"
        print(f"| Profit: {avg_adam*100:+.1f}% vs {avg_mobiu*100:+.1f}% {winner}")
    
    adam_prof_arr = np.array(adam_profits)
    mobiu_prof_arr = np.array(mobiu_profits)
    
    _, p_profits = wilcoxon(adam_prof_arr, mobiu_prof_arr, alternative='less')
    profit_wins = np.sum(mobiu_prof_arr > adam_prof_arr) / NUM_SEEDS * 100
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"\n💰 Trading Profit:")
    print(f"  Pure Adam:     {adam_prof_arr.mean()*100:+.2f}% ± {adam_prof_arr.std()*100:.2f}%")
    print(f"  Adam + Mobiu:  {mobiu_prof_arr.mean()*100:+.2f}% ± {mobiu_prof_arr.std()*100:.2f}%")
    print(f"  p-value: {p_profits:.6f}")
    print(f"  Win rate: {profit_wins:.1f}%")
    
    print("\n" + "=" * 70)
    if p_profits < 0.05 and profit_wins >= 70:
        print("🏆 SIGNIFICANT IMPROVEMENT!")
    elif profit_wins >= 60:
        print("✅ Mobiu shows advantage")
    else:
        print("📊 Results inconclusive")
    print("=" * 70)


if __name__ == "__main__":
    main()
