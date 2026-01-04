#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q CRYPTO TRADING BENCHMARK (ORIGINAL VERSION)
================================================================================
Simulates crypto trading with high volatility and regime switching.

DETERMINISM:
- Both optimizers start from IDENTICAL policy initialization (deep copy)
- Both use the SAME torch/numpy seed for action sampling  
- Both see the SAME market data per episode (via env.rng seed)
- Only difference: how the optimizer updates the policy
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
from scipy.stats import wilcoxon
import os

# Mobiu Import
try:
    from mobiu_q import MobiuOptimizer
    HAS_MOBIU = True
except ImportError:
    HAS_MOBIU = False
    print("⚠️ mobiu-q not installed")

# ============================================================================
# CONFIGURATION
# ============================================================================

LICENSE_KEY = os.environ.get("MOBIU_LICENSE_KEY", "YOUR_KEY_HERE")

NUM_EPISODES = 500
NUM_SEEDS = 10
BASE_LR = 0.0003
WINDOW_SIZE = 20
EPISODE_LENGTH = 500
TRANSACTION_FEE = 0.001  # 0.1%

print("=" * 70)
print("🪙 MOBIU-Q CRYPTO TRADING BENCHMARK (ORIGINAL)")
print("=" * 70)
print(f"Episodes: {NUM_EPISODES} | Seeds: {NUM_SEEDS}")
print(f"Episode length: {EPISODE_LENGTH} steps")
print(f"Transaction fee: {TRANSACTION_FEE*100:.2f}%")
print("=" * 70)

# ============================================================================
# CRYPTO TRADING ENVIRONMENT (ORIGINAL WITH REGIME SWITCHING)
# ============================================================================

class CryptoTradingEnv:
    """
    Crypto trading environment with:
    - High volatility (2-5%)
    - Regime switching (bull/bear markets)
    - Flash crashes/pumps
    - Transaction costs
    
    State: [price_returns[-window:], volatility, position, unrealized_pnl, trade_count]
    Actions: 0=Hold, 1=Long, 2=Short, 3=Close
    """
    
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
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0
        self.total_profit = 0
        self.trade_count = 0
        
        return self._get_state(), {}
    
    def _generate_crypto_prices(self):
        """Generate crypto-like price series with high volatility and patterns"""
        n = self.episode_length + self.window_size + 10
        
        # Crypto characteristics
        base_vol = self.rng.uniform(0.02, 0.05)  # 2-5% base volatility
        drift = self.rng.uniform(-0.001, 0.002)  # Slight upward bias
        
        # Generate with regime switching (bull/bear)
        returns = []
        vol = base_vol
        regime = 1  # 1=bull, -1=bear
        
        for i in range(n):
            # Regime switching (momentum)
            if self.rng.random() < 0.02:  # 2% chance to switch
                regime *= -1
            
            # Volatility clustering
            vol = 0.85 * vol + 0.15 * base_vol * (1 + abs(self.rng.randn()))
            
            # Flash crash / pump (rare events)
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
        """Compute rolling volatility"""
        vols = []
        for i in range(len(self.returns)):
            start = max(0, i - window)
            vols.append(np.std(self.returns[start:i+1]) if i > 0 else 0.02)
        return np.array(vols)
    
    def _get_state(self):
        # Normalized returns
        window_returns = self.returns[self.current_step - self.window_size:self.current_step]
        window_returns = window_returns / (np.std(window_returns) + 1e-8)
        
        # Current volatility (normalized)
        current_vol = self.volatilities[self.current_step] / 0.05
        
        # Position info
        position_ind = float(self.position)
        
        # Unrealized PnL
        if self.position != 0:
            current_price = self.prices[self.current_step]
            unrealized = self.position * (current_price - self.entry_price) / self.entry_price
        else:
            unrealized = 0
        unrealized = np.tanh(unrealized * 10)
        
        # Trade frequency
        trade_freq = min(self.trade_count / 50, 1.0)
        
        state = np.concatenate([
            window_returns,
            [current_vol, position_ind, unrealized, trade_freq]
        ]).astype(np.float32)
        
        return state
    
    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        
        # Execute action
        if action == 1 and self.position != 1:  # Go Long
            if self.position == -1:  # Close short first
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - TRANSACTION_FEE
                self.total_profit += pnl - TRANSACTION_FEE
            self.position = 1
            self.entry_price = current_price
            self.trade_count += 1
            reward -= TRANSACTION_FEE
            
        elif action == 2 and self.position != -1:  # Go Short
            if self.position == 1:  # Close long first
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
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.episode_length + self.window_size - 1
        
        # Force close at end
        if done and self.position != 0:
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            reward += pnl - TRANSACTION_FEE
            self.total_profit += pnl - TRANSACTION_FEE
        
        return self._get_state(), reward, done, False, {'total_profit': self.total_profit}

# ============================================================================
# POLICY NETWORK
# ============================================================================

class TradingPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state):
        logits = self.forward(torch.FloatTensor(state).unsqueeze(0))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# ============================================================================
# TRAINING
# ============================================================================

def train_reinforce(policy, optimizer, env, num_episodes, use_mobiu=False):
    """Train using REINFORCE"""
    episode_returns = []
    episode_profits = []
    
    for ep in range(num_episodes):
        # Reset with deterministic seed per episode
        state, _ = env.reset(seed=ep * 12345)
        
        log_probs = []
        rewards = []
        
        done = False
        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        
        if use_mobiu:
            optimizer.step(sum(rewards))
        else:
            optimizer.step()
        
        episode_returns.append(sum(rewards))
        episode_profits.append(info['total_profit'])
    
    return episode_returns, episode_profits

# ============================================================================
# MAIN
# ============================================================================

def main():
    if not HAS_MOBIU:
        print("❌ mobiu-q not installed!")
        return
    
    print(f"\n📊 Environment:")
    print(f"  State dim: {WINDOW_SIZE + 4}")
    print(f"  Actions: Hold, Long, Short, Close")
    print("=" * 70)
    
    adam_results = []
    adam_profits = []
    mobiu_results = []
    mobiu_profits = []
    
    for seed in range(NUM_SEEDS):
        print(f"\nSeed {seed+1}/{NUM_SEEDS} ", end="", flush=True)
        
        # Set master seed
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        # Create environment
        env = CryptoTradingEnv(WINDOW_SIZE, EPISODE_LENGTH)
        
        # Create policy and DEEP COPY for fair comparison
        policy_adam = TradingPolicy(env.state_dim)
        policy_mobiu = copy.deepcopy(policy_adam)  # IDENTICAL starting point!
        
        # Optimizers
        opt_adam = optim.Adam(policy_adam.parameters(), lr=BASE_LR)
        opt_mobiu = MobiuOptimizer(
            optim.Adam(policy_mobiu.parameters(), lr=BASE_LR),
            license_key=LICENSE_KEY,
            method='adaptive',
            maximize=True,
            use_soft_algebra=True,
            verbose=False
        )
        
        # Train Adam
        print("[Adam", end="", flush=True)
        torch.manual_seed(seed * 42)  # Reset seed for identical sampling
        np.random.seed(seed * 42)
        ret_adam, prof_adam = train_reinforce(policy_adam, opt_adam, env, NUM_EPISODES, use_mobiu=False)
        print(".....] ", end="", flush=True)
        
        # Train Mobiu
        print("[Mobiu", end="", flush=True)
        torch.manual_seed(seed * 42)  # Reset seed for identical sampling
        np.random.seed(seed * 42)
        ret_mobiu, prof_mobiu = train_reinforce(policy_mobiu, opt_mobiu, env, NUM_EPISODES, use_mobiu=True)
        print(".....] ", end="", flush=True)
        
        opt_mobiu.end()
        
        # Average last 50 episodes
        avg_adam = np.mean(prof_adam[-50:])
        avg_mobiu = np.mean(prof_mobiu[-50:])
        
        adam_results.append(np.mean(ret_adam[-50:]))
        adam_profits.append(avg_adam)
        mobiu_results.append(np.mean(ret_mobiu[-50:]))
        mobiu_profits.append(avg_mobiu)
        
        print(f"| Profit: {avg_adam*100:+.1f}% vs {avg_mobiu*100:+.1f}%")
    
    # Statistics
    adam_ret_arr = np.array(adam_results)
    mobiu_ret_arr = np.array(mobiu_results)
    adam_prof_arr = np.array(adam_profits)
    mobiu_prof_arr = np.array(mobiu_profits)
    
    _, p_returns = wilcoxon(adam_ret_arr, mobiu_ret_arr, alternative='less')
    _, p_profits = wilcoxon(adam_prof_arr, mobiu_prof_arr, alternative='less')
    
    return_wins = np.sum(mobiu_ret_arr > adam_ret_arr) / NUM_SEEDS * 100
    profit_wins = np.sum(mobiu_prof_arr > adam_prof_arr) / NUM_SEEDS * 100
    
    ret_improvement = (mobiu_ret_arr.mean() - adam_ret_arr.mean()) / (abs(adam_ret_arr.mean()) + 1e-9) * 100
    prof_improvement = (mobiu_prof_arr.mean() - adam_prof_arr.mean()) / (abs(adam_prof_arr.mean()) + 1e-9) * 100
    
    print("\n" + "=" * 70)
    print("🪙 CRYPTO TRADING RESULTS")
    print("=" * 70)
    
    print(f"\n📈 Episode Returns:")
    print(f"  Adam:  {adam_ret_arr.mean():.4f} ± {adam_ret_arr.std():.4f}")
    print(f"  Mobiu: {mobiu_ret_arr.mean():.4f} ± {mobiu_ret_arr.std():.4f}")
    print(f"  Improvement: {ret_improvement:+.1f}%")
    print(f"  p-value: {p_returns:.6f}")
    print(f"  Win rate: {return_wins:.1f}%")
    
    print(f"\n💰 Trading Profit:")
    print(f"  Adam:  {adam_prof_arr.mean()*100:+.2f}% ± {adam_prof_arr.std()*100:.2f}%")
    print(f"  Mobiu: {mobiu_prof_arr.mean()*100:+.2f}% ± {mobiu_prof_arr.std()*100:.2f}%")
    print(f"  p-value: {p_profits:.6f}")
    print(f"  Profit win rate: {profit_wins:.1f}%")
    
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
