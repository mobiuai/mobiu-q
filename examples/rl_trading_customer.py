#!/usr/bin/env python3
"""
================================================================================
🚀 RL TRADING BENCHMARK - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

Trading Environment with MobiuSignal features:
- Regime switching (trending, mean-reverting, volatile)
- Creates systematic bias that Mobiu-Q can exploit
- 30 seeds for statistical significance
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy.stats import wilcoxon
import copy
import warnings

warnings.filterwarnings('ignore')

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

NUM_EPISODES = 500
NUM_SEEDS = 30
BASE_LR = 0.0003
METHOD = "adaptive"
LOOKBACK = 20


# ============================================================
# MOBIU SIGNAL (LOCAL)
# ============================================================

@dataclass
class SignalResult:
    potential: float
    realized: float
    magnitude: float
    direction: int


class MobiuSignal:
    """Local MobiuSignal for RL environment."""
    
    def __init__(self, lookback: int = 20, vol_scale: float = 100.0):
        self.lookback = lookback
        self.vol_scale = vol_scale
        self.price_history = []
    
    def update(self, price: float) -> Optional[SignalResult]:
        self.price_history.append(price)
        
        if len(self.price_history) < self.lookback + 1:
            return None
        
        if len(self.price_history) > self.lookback + 10:
            self.price_history = self.price_history[-(self.lookback + 1):]
        
        prices = np.array(self.price_history[-(self.lookback + 1):])
        returns = np.diff(np.log(prices))
        
        volatility = np.std(returns[-self.lookback:])
        mean_price = np.mean(np.abs(prices[-self.lookback:]))
        a_t = (volatility / (mean_price + 1e-9)) * self.vol_scale * 100
        
        b_t = (prices[-1] - prices[-2]) / (np.abs(prices[-2]) + 1e-9) * 100
        
        magnitude = np.sqrt(a_t**2 + b_t**2)
        direction = 1 if b_t > 0.01 else (-1 if b_t < -0.01 else 0)
        
        return SignalResult(potential=a_t, realized=b_t, magnitude=magnitude, direction=direction)
    
    def reset(self):
        self.price_history = []


# ============================================================
# TRADING ENVIRONMENT
# ============================================================

class TradingEnv:
    """Trading environment with regime switching."""
    
    def __init__(self, n_steps: int = 500, regime_length: int = 100):
        self.n_steps = n_steps
        self.regime_length = regime_length
        self.signal = MobiuSignal(lookback=LOOKBACK)
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.signal.reset()
        self.step_idx = 0
        self.position = 0
        self.pnl = 0.0
        self.prices = self._generate_prices()
        
        for i in range(LOOKBACK + 1):
            self.signal.update(self.prices[i])
            self.step_idx = i
        
        return self._get_state()
    
    def _generate_prices(self) -> np.ndarray:
        prices = [100.0]
        
        for i in range(self.n_steps):
            regime = (i // self.regime_length) % 3
            
            if regime == 0:  # Trending
                drift = 0.001 * (1 if (i // self.regime_length) % 2 == 0 else -1)
                vol = 0.01
            elif regime == 1:  # Mean-reverting
                drift = -0.0005 * (prices[-1] - 100) / 100
                vol = 0.02
            else:  # Volatile
                drift = 0.0005
                vol = 0.03
            
            ret = drift + vol * np.random.randn()
            prices.append(prices[-1] * np.exp(ret))
        
        return np.array(prices)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_idx += 1
        
        if self.step_idx >= len(self.prices) - 1:
            return self._get_state(), 0.0, True, {'pnl': self.pnl}
        
        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]
        
        self.signal.update(current_price)
        
        price_change = (next_price - current_price) / current_price
        old_position = self.position
        
        if action == 1 and self.position <= 0:
            self.position = 1
        elif action == 2 and self.position >= 0:
            self.position = -1
        
        reward = self.position * price_change * 100
        if old_position != self.position:
            reward -= 0.1
        
        self.pnl += reward
        done = self.step_idx >= len(self.prices) - 2
        
        return self._get_state(), reward, done, {'pnl': self.pnl}
    
    def _get_state(self) -> np.ndarray:
        sig = self.signal.update(self.prices[self.step_idx])
        if sig is None:
            return np.zeros(5)
        
        return np.array([
            sig.potential / 10.0,
            sig.realized / 5.0,
            sig.magnitude / 10.0,
            float(self.position),
            self.pnl / 100.0
        ], dtype=np.float32)


# ============================================================
# POLICY NETWORK
# ============================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            logits = self.forward(torch.FloatTensor(state).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def get_log_prob(self, state: np.ndarray, action: int) -> torch.Tensor:
        logits = self.forward(torch.FloatTensor(state).unsqueeze(0))
        return F.log_softmax(logits, dim=-1)[0, action]


# ============================================================
# TRAINING - PURE ADAM (Baseline)
# ============================================================

def train_pure_adam(policy: PolicyNetwork, num_episodes: int, seed: int, label: str = "") -> List[float]:
    """Train with Pure Adam - what customer has BEFORE adding Mobiu-Q"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=BASE_LR)
    env = TradingEnv()
    returns = []
    
    for ep in range(num_episodes):
        np.random.seed(seed * 10000 + ep)
        
        state = env.reset()
        log_probs, rewards = [], []
        done = False
        
        while not done:
            action = policy.get_action(state)
            log_probs.append(policy.get_log_prob(state, action))
            state, reward, done, info = env.step(action)
            rewards.append(reward)
        
        episode_return = info['pnl']
        returns.append(episode_return)
        
        G = 0
        disc_returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            disc_returns.insert(0, G)
        disc_returns = torch.tensor(disc_returns)
        
        if disc_returns.std() > 1e-6:
            advantages = (disc_returns - disc_returns.mean()) / (disc_returns.std() + 1e-8)
        else:
            advantages = disc_returns - disc_returns.mean()
        
        loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % 100 == 0:
            avg = np.mean(returns[-100:])
            print(f"    {label} ep {ep+1}: avg100 = {avg:.1f}")
    
    return returns


# ============================================================
# TRAINING - WITH MOBIU-Q
# ============================================================

def train_with_mobiu(policy: PolicyNetwork, num_episodes: int, seed: int, label: str = "") -> List[float]:
    """Train with Mobiu-Q - what customer has AFTER adding Mobiu-Q"""
    base_opt = torch.optim.Adam(policy.parameters(), lr=BASE_LR)
    optimizer = MobiuOptimizer(
        base_opt,
        license_key=LICENSE_KEY,
        method=METHOD,
        maximize=True,
        verbose=False
    )
    
    env = TradingEnv()
    returns = []
    
    for ep in range(num_episodes):
        np.random.seed(seed * 10000 + ep)
        
        state = env.reset()
        log_probs, rewards = [], []
        done = False
        
        while not done:
            action = policy.get_action(state)
            log_probs.append(policy.get_log_prob(state, action))
            state, reward, done, info = env.step(action)
            rewards.append(reward)
        
        episode_return = info['pnl']
        returns.append(episode_return)
        
        G = 0
        disc_returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            disc_returns.insert(0, G)
        disc_returns = torch.tensor(disc_returns)
        
        if disc_returns.std() > 1e-6:
            advantages = (disc_returns - disc_returns.mean()) / (disc_returns.std() + 1e-8)
        else:
            advantages = disc_returns - disc_returns.mean()
        
        loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(episode_return)
        
        if (ep + 1) % 100 == 0:
            avg = np.mean(returns[-100:])
            print(f"    {label} ep {ep+1}: avg100 = {avg:.1f}")
    
    optimizer.end()
    return returns


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 RL TRADING BENCHMARK - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Episodes: {NUM_EPISODES} | Seeds: {NUM_SEEDS} | LR: {BASE_LR}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\nSeed {seed + 1}/{NUM_SEEDS}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        template = PolicyNetwork()
        adam_policy = copy.deepcopy(template)
        mobiu_policy = copy.deepcopy(template)
        
        # Pure Adam
        adam_returns = train_pure_adam(adam_policy, NUM_EPISODES, seed, "Adam ")
        
        # Adam + Mobiu
        mobiu_returns = train_with_mobiu(mobiu_policy, NUM_EPISODES, seed, "Mobiu")
        
        adam_final = np.mean(adam_returns[-100:])
        mobiu_final = np.mean(mobiu_returns[-100:])
        
        adam_results.append(adam_final)
        mobiu_results.append(mobiu_final)
        
        diff = mobiu_final - adam_final
        winner = "✅ Mobiu" if diff > 0 else "❌ Adam"
        print(f"  Pure Adam: {adam_final:.1f} | Adam+Mobiu: {mobiu_final:.1f} | Δ={diff:+.1f} → {winner}")
    
    # Statistics
    adam_arr = np.array(adam_results)
    mobiu_arr = np.array(mobiu_results)
    diff = mobiu_arr - adam_arr
    
    _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative="less")
    improvement = 100 * diff.mean() / (abs(adam_arr.mean()) + 1e-9)
    cohen_d = diff.mean() / (diff.std() + 1e-9)
    win_rate = sum(d > 0 for d in diff) / len(diff)
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Pure Adam:     {adam_arr.mean():.1f} ± {adam_arr.std():.1f}")
    print(f"Adam + Mobiu:  {mobiu_arr.mean():.1f} ± {mobiu_arr.std():.1f}")
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
