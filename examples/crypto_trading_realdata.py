#!/usr/bin/env python3
"""
PPO CRYPTO TRADING – BTC-USD real daily data + Mobiu-Q [FAIR BENCHMARK]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from scipy.stats import wilcoxon
import yfinance as yf
import pandas as pd

from mobiu_q import MobiuOptimizer

LICENSE_KEY    = "YOUR_KEY"
NUM_EPISODES   = 300
NUM_SEEDS      = 20
BASE_LR        = 0.0003
METHOD         = "adaptive"
EPISODE_LENGTH = 600
TRANSACTION_FEE = 0.001
GAMMA          = 0.99
GAE_LAMBDA     = 0.95


def download_btc_data():
    df = yf.download('BTC-USD', period='max', interval='1d', progress=False)
    if df.empty:
        raise ValueError("Failed to download BTC-USD data")
    prices = df['Close'].dropna()
    print(f"Downloaded {len(prices)} daily closes ({prices.index[0]:%Y-%m-%d} → {prices.index[-1]:%Y-%m-%d})")
    return prices


class CryptoTradingEnv:
    def __init__(self, prices_series: pd.Series, episode_length=600):
        self.prices_series = prices_series.astype(np.float32)
        self.episode_length = episode_length
        self.available_starts = self._get_possible_starts()
        if not self.available_starts:
            raise ValueError(f"Insufficient data: {len(prices_series)} < {episode_length}")
        self.reset()

    def _get_possible_starts(self):
        max_start = len(self.prices_series) - self.episode_length
        return list(range(0, max_start + 1, self.episode_length))

    def reset(self, seed=None, start_idx=None):
        if seed is not None:
            np.random.seed(seed)
        if start_idx is None:
            start_idx = np.random.choice(self.available_starts)
        self.start_idx = start_idx
        self.step_idx = 0
        self.position = 0
        self.pnl = 0.0
        
        slice_ = self.prices_series.iloc[self.start_idx : self.start_idx + self.episode_length + 1]
        self.prices = np.asarray(slice_.values).flatten().astype(np.float32)
        return self._get_state()

    def _get_state(self):
        start = max(0, self.step_idx - 19)
        prices_window = self.prices[start:self.step_idx + 1]
        prices_window = np.asarray(prices_window).flatten()
        
        if len(prices_window) < 2:
            returns_padded = np.zeros(20, dtype=np.float32)
        else:
            returns = np.diff(prices_window)
            returns = np.asarray(returns).flatten()
            
            pad_width = 20 - len(returns)
            if pad_width > 0:
                returns_padded = np.pad(returns, (pad_width, 0), mode='constant', constant_values=0.0)
            else:
                returns_padded = returns[-20:]
            
            returns_padded = np.asarray(returns_padded).flatten()[:20]

        state = np.zeros(22, dtype=np.float32)
        state[:20] = returns_padded
        state[20] = float(self.position)
        state[21] = float(self.pnl / 100.0)
        return state

    def step(self, action):
        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]
        price_change = (next_price - current_price) / current_price

        old_position = self.position
        if action == 1:    self.position = 1
        elif action == 2:  self.position = -1
        else:              self.position = 0

        reward = self.position * price_change * 100
        if old_position != self.position:
            reward -= TRANSACTION_FEE * 100

        self.pnl += reward
        self.step_idx += 1
        done = self.step_idx >= self.episode_length - 1
        return self._get_state(), reward, done, {'pnl': self.pnl}


# שאר הקוד (ActorCritic, train_ppo, main) נשאר זהה לגרסה הקודמת שלך

class ActorCritic(nn.Module):
    def __init__(self, state_dim=22, action_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.actor  = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


def train_ppo(policy, env, use_mobiu=False):
    base_opt = optim.Adam(policy.parameters(), lr=BASE_LR)

    if use_mobiu:
        optimizer = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method=METHOD,
            base_lr=BASE_LR,
            update_interval=1,
            verbose=False
        )
    else:
        optimizer = base_opt

    episode_pnls = []

    for ep in range(NUM_EPISODES):
        start_idx = env.available_starts[ep % len(env.available_starts)]
        state = env.reset(start_idx=start_idx)

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

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-logp * ret for logp, ret in zip(log_probs, returns))

        optimizer.zero_grad()
        loss.backward()

        if use_mobiu:
            optimizer.step(loss.item())
        else:
            optimizer.step()

    if use_mobiu:
        optimizer.end()

    return episode_pnls


def main():
    prices_series = download_btc_data()

    print("=" * 95)
    print("PPO CRYPTO TRADING – BTC-USD real daily data + Mobiu-Q [FAIR]")
    print(f"Episode length: {EPISODE_LENGTH} days | Available windows: {len(prices_series) // EPISODE_LENGTH}")
    print("=" * 95)

    adam_pnls, mobiu_pnls = [], []

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = CryptoTradingEnv(prices_series, EPISODE_LENGTH)

        dummy = ActorCritic()
        init_weights = {k: v.clone() for k, v in dummy.state_dict().items()}

        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()

        print(f"Seed {seed+1:2d}/{NUM_SEEDS} | ", end="", flush=True)

        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        policy_adam = ActorCritic()
        policy_adam.load_state_dict(init_weights)
        pnls_adam = train_ppo(policy_adam, env, use_mobiu=False)
        avg_adam = np.mean(pnls_adam[-100:])
        adam_pnls.append(avg_adam)
        print(f"Adam: {avg_adam:6.1f} | ", end="", flush=True)

        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        policy_mobiu = ActorCritic()
        policy_mobiu.load_state_dict(init_weights)
        pnls_mobiu = train_ppo(policy_mobiu, env, use_mobiu=True)
        avg_mobiu = np.mean(pnls_mobiu[-100:])
        mobiu_pnls.append(avg_mobiu)
        print(f"Mobiu: {avg_mobiu:6.1f}")

    adam_arr  = np.array(adam_pnls)
    mobiu_arr = np.array(mobiu_pnls)

    improvement = (mobiu_arr.mean() - adam_arr.mean()) / (abs(adam_arr.mean()) + 1e-10) * 100
    wins = np.sum(mobiu_arr > adam_arr)
    _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative='less')

    print("\n" + "="*95)
    print("FINAL RESULTS")
    print(f"Adam Avg PnL (last 100 eps):     {adam_arr.mean():.2f} ± {adam_arr.std():.2f}")
    print(f"Mobiu-Q Avg PnL (last 100 eps):  {mobiu_arr.mean():.2f} ± {mobiu_arr.std():.2f}")
    print(f"Improvement:                     {improvement:+.1f}%")
    print(f"Win rate:                        {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print(f"p-value (Wilcoxon):              {p_value:.6f}")
    print("="*95)


if __name__ == "__main__":
    main()