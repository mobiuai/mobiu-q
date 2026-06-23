#!/usr/bin/env python3
"""
Portfolio PPO — Real Mobiu (Adaptive) vs Generic Scalar Baseline (Fixed)
=======================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scipy.stats import wilcoxon

from mobiu_q import MobiuOptimizer

LICENSE_KEY = "YOUR_LICENSE_HERE"
NUM_EPISODES = 400
NUM_SEEDS = 8
BASE_LR = 0.0003
METHOD = "adaptive"
EPISODE_LENGTH = 400


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
            regime = (i // 80) % 3
            if regime == 0:
                ret = 0.0015 + 0.018 * self.rng.randn()
            elif regime == 1:
                ret = 0.0007 + 0.035 * self.rng.randn()
            else:
                ret = 0.0004 + 0.008 * self.rng.randn()
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices)

    def _get_state(self):
        start = max(0, self.step_idx - 19)
        recent = np.diff(self.prices[start:self.step_idx + 1])
        padded = np.pad(recent, (20 - len(recent), 0), 'constant')
        return np.concatenate([padded, [float(self.position), self.pnl / 100]]).astype(np.float32)

    def step(self, action):
        current = self.prices[self.step_idx]
        next_p = self.prices[self.step_idx + 1]
        ret = (next_p - current) / current
        old_pos = self.position

        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        else:
            self.position = 0

        reward = self.position * ret * 100
        if old_pos != self.position:
            reward -= 0.001 * 100
        self.pnl += reward
        self.step_idx += 1
        done = self.step_idx >= self.episode_length - 1
        return self._get_state(), reward, done, {'pnl': self.pnl}


class ActorCritic(nn.Module):
    def __init__(self, state_dim=22, action_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class GenericScalarOptimizer:
    """Clean scalar implementation (no soft algebra)."""
    def __init__(self, params, base_lr=0.0003, gamma=0.9):
        self.params = list(params)
        self.base_lr = base_lr
        self.gamma = gamma
        self.a_ema = 0.0
        self.b_ema = 0.0
        self.energy_history = []
        self.m = None
        self.v = None
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self, loss_value):
        self.energy_history.append(float(loss_value))
        if len(self.energy_history) > 10:
            self.energy_history = self.energy_history[-10:]

        a_t = self._signal_curvature()
        b_t = self._signal_improvement()
        self.a_ema = self.gamma * self.a_ema + (1 - self.gamma) * a_t
        self.b_ema = self.gamma * self.b_ema + (1 - self.gamma) * b_t

        trust = abs(self.b_ema) / (abs(self.a_ema) + abs(self.b_ema) + 1e-9)
        if abs(self.a_ema) < 1e-9 and abs(self.b_ema) < 1e-9:
            trust = -1.0

        if trust < 0:
            alpha_t = 0.0
        else:
            lr_mult = max(0.5, min(1.5, 1.0 + 0.3 * trust))
            alpha_t = self.base_lr * lr_mult

        warp = 1.0 + 0.5 * (abs(self.a_ema) / (abs(self.a_ema) + abs(self.b_ema) + 1e-9))

        if self.m is None:
            self.m = [torch.zeros_like(p) for p in self.params]
            self.v = [torch.zeros_like(p) for p in self.params]

        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g_eff = p.grad.data * warp
            self.m[i] = 0.9 * self.m[i] + 0.1 * g_eff
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g_eff ** 2)
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            if alpha_t > 0:
                p.data.add_(-alpha_t * m_hat / (torch.sqrt(v_hat) + 1e-8))

    def _signal_curvature(self):
        if len(self.energy_history) < 3:
            return 0.0
        e0, e1, e2 = self.energy_history[-1], self.energy_history[-2], self.energy_history[-3]
        curv = abs(e0 - 2*e1 + e2)
        mean_e = abs(np.mean(self.energy_history[-3:])) + 1e-12
        return min(10.0, curv / (curv + mean_e))

    def _signal_improvement(self):
        if len(self.energy_history) < 2:
            return 0.0
        prev, curr = self.energy_history[-2], self.energy_history[-1]
        b = (prev - curr) / (abs(prev) + 1e-9)
        return max(-1.0, min(1.0, b))


def train_ppo(policy, env, optimizer, use_mobiu=False):
    episode_pnls = []

    for ep in range(NUM_EPISODES):
        state = env.reset(seed=ep * 100)
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
            delta = r + 0.99 * next_value * (1 - d) - v
            gae = delta + 0.99 * 0.95 * gae * (1 - d)
            returns.insert(0, gae + v)
            next_value = v

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-logp * ret for logp, ret in zip(log_probs, returns))

        optimizer.zero_grad()
        loss.backward()

        if use_mobiu:
            optimizer.step(loss.item())
        else:
            optimizer.step(loss.item())   # ← תוקן כאן

    if use_mobiu:
        optimizer.end()

    return episode_pnls


def main():
    print("=" * 95)
    print("Portfolio PPO — Real Mobiu (Adaptive) vs Generic Scalar Baseline (Fixed)")
    print("=" * 95)

    mobiu_pnls = []
    scalar_pnls = []

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = PortfolioTradingEnv()

        # Full Mobiu (Adaptive)
        policy_mobiu = ActorCritic()
        base_opt_mobiu = torch.optim.Adam(policy_mobiu.parameters(), lr=BASE_LR)
        mobiu_opt = MobiuOptimizer(
            base_opt_mobiu, license_key=LICENSE_KEY,
            method="adaptive", base_lr=BASE_LR, verbose=False
        )
        pnls_mobiu = train_ppo(policy_mobiu, env, mobiu_opt, use_mobiu=True)
        avg_mobiu = np.mean(pnls_mobiu[-100:])
        mobiu_pnls.append(avg_mobiu)
        print(f"Seed {seed+1:2d} | Mobiu (Adaptive): {avg_mobiu:7.2f}", end=" | ")

        # Generic Scalar Baseline
        torch.manual_seed(seed)
        np.random.seed(seed)
        policy_scalar = ActorCritic()
        scalar_opt = GenericScalarOptimizer(policy_scalar.parameters(), base_lr=BASE_LR)
        pnls_scalar = train_ppo(policy_scalar, env, scalar_opt, use_mobiu=False)
        avg_scalar = np.mean(pnls_scalar[-100:])
        scalar_pnls.append(avg_scalar)
        print(f"Generic Scalar: {avg_scalar:7.2f}")

    mobiu_arr = np.array(mobiu_pnls)
    scalar_arr = np.array(scalar_pnls)

    improvement = (mobiu_arr.mean() - scalar_arr.mean()) / abs(scalar_arr.mean()) * 100
    wins = np.sum(mobiu_arr > scalar_arr)
    _, p_value = wilcoxon(scalar_arr, mobiu_arr, alternative='less')

    print("\n" + "=" * 95)
    print("FINAL RESULTS")
    print(f"Generic Scalar Baseline: {scalar_arr.mean():.2f} ± {scalar_arr.std():.2f}")
    print(f"Full Mobiu (Adaptive):   {mobiu_arr.mean():.2f} ± {mobiu_arr.std():.2f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate:    {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print(f"p-value:     {p_value:.6f}")
    print("=" * 95)


if __name__ == "__main__":
    main()