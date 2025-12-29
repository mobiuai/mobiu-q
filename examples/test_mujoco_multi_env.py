#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q MUJOCO MULTI-ENVIRONMENT BENCHMARK
================================================================================
Tests Soft Algebra across multiple MuJoCo robotics environments.

Environments:
- InvertedPendulum (simple balance)
- Hopper (1-leg jumping)
- HalfCheetah (2D running)
- Ant (4-leg walking) - optional, slower

Market: Industrial Robotics ($4.07B → $9.56B by 2030, 15.1% CAGR)

DETERMINISM:
- Both optimizers start from IDENTICAL policy initialization per seed
- Both see the SAME environment dynamics per episode
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from dataclasses import dataclass
from scipy.stats import wilcoxon
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_EPISODES = 300      # MuJoCo converges faster
NUM_SEEDS = 10
BASE_LR = 0.0003
MAX_STEPS_PER_EP = 1000

# Environments to test (ordered by complexity)
ENVIRONMENTS = [
    "InvertedPendulum-v5",   # Simple: 4 obs, 1 action
    "Hopper-v5",             # Medium: 11 obs, 3 actions
]

print("=" * 70)
print("🤖 MOBIU-Q MUJOCO MULTI-ENVIRONMENT BENCHMARK")
print("=" * 70)

# ============================================================================
# CHECK MUJOCO
# ============================================================================

try:
    import gymnasium as gym
    # Test all environments
    for env_name in ENVIRONMENTS:
        env = gym.make(env_name)
        print(f"✓ {env_name}: obs={env.observation_space.shape[0]}, act={env.action_space.shape[0]}")
        env.close()
except Exception as e:
    print(f"❌ MuJoCo error: {e}")
    print("Install: pip install gymnasium[mujoco]")
    exit(1)

print(f"\nEpisodes: {NUM_EPISODES} | Seeds: {NUM_SEEDS}")
print("=" * 70)

# ============================================================================
# SOFT ALGEBRA CORE
# ============================================================================

@dataclass
class SoftNumber:
    soft: float
    real: float

    def __add__(self, other):
        return SoftNumber(self.soft + other.soft, self.real + other.real)

    def __mul__(self, other):
        if isinstance(other, SoftNumber):
            if abs(self.real) < 1e-12 and abs(other.real) < 1e-12:
                return SoftNumber(0.0, 0.0)
            a, b = self.soft, self.real
            c, d = other.soft, other.real
            return SoftNumber(soft=a * d + b * c, real=b * d)
        else:
            return SoftNumber(self.soft * float(other), self.real * float(other))

    def __rmul__(self, other):
        return self.__mul__(other)

def compute_super_equation(sn_state: SoftNumber, alpha=1.35, beta=1.70, C=3.00, epsilon=0.43) -> float:
    a, b = sn_state.soft, sn_state.real
    S = b + 1j * a * epsilon
    du = np.sin(np.pi * S).imag
    tau = C * a * b
    g = np.exp(-(tau - 1)**2 / (2 * alpha**2))
    gamma_gate = 1 - np.exp(-beta * a)
    return abs(abs(du) * g * gamma_gate * np.sqrt(max(0, b * g)))

# ============================================================================
# CONTINUOUS POLICY NETWORK
# ============================================================================

class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        
        self.action_high = 1.0
        self.action_low = -1.0

    def set_action_bounds(self, low, high):
        self.action_low = low
        self.action_high = high

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def get_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mean, std = self.forward(state_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action.squeeze(0).numpy().clip(self.action_low, self.action_high)

    def get_log_prob(self, state, action):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = torch.FloatTensor(action).unsqueeze(0)
        mean, std = self.forward(state_t)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action_t).sum(dim=-1).squeeze(0)

# ============================================================================
# MOBIU-Q RL OPTIMIZER
# ============================================================================

class MobiuRLCore:
    def __init__(self, params, base_lr=0.0003, gamma=0.9,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.base_lr = base_lr
        self.gamma = gamma
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.sn_state = SoftNumber(0.0, 0.0)
        self.return_history = []

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, episode_return: float):
        self.t += 1
        self.return_history.append(float(episode_return))
        if len(self.return_history) > 10:
            self.return_history = self.return_history[-10:]

        alpha_t = self.base_lr
        soft_factor = 1.0

        if len(self.return_history) >= 2:
            a_t = 0.0
            if len(self.return_history) >= 3:
                E_t, E_t1, E_t2 = self.return_history[-1], self.return_history[-2], self.return_history[-3]
                curv = abs(E_t - 2*E_t1 + E_t2)
                mean_E = abs(np.mean(self.return_history[-3:]))
                a_t = curv / (curv + mean_E) if mean_E > 1e-12 else 0.0
            
            b_t = (self.return_history[-1] - self.return_history[-2]) / (abs(self.return_history[-2]) + 1e-9)
            b_t = max(0.0, min(1.0, b_t))

            delta_sn = SoftNumber(soft=a_t, real=b_t)
            self.sn_state = (self.sn_state * self.gamma) * delta_sn + delta_sn

            trust = abs(self.sn_state.real) / (
                abs(self.sn_state.real) + abs(self.sn_state.soft) + 1e-9)

            delta_dagger = compute_super_equation(self.sn_state)
            
            trust_scale = max(0.5, min(2.0, 1.0 + 1.0 * trust))
            emergence_boost = 1.0 + 2.0 * delta_dagger
            scale = min(3.0, trust_scale * emergence_boost)

            alpha_t = self.base_lr * scale
            soft_factor = max(0.9, min(1.1, 1.0 + 0.1 * self.sn_state.soft))

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data * soft_factor

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= alpha_t * m_hat / (torch.sqrt(v_hat) + self.eps)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_episode(env, policy, optimizer, gamma=0.99, use_mobiu=False, env_seed=None):
    if env_seed is not None:
        state, _ = env.reset(seed=env_seed)
    else:
        state, _ = env.reset()
    
    log_probs, rewards = [], []
    done = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_EP:
        action = policy.get_action(state)
        log_probs.append(policy.get_log_prob(state, action))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        steps += 1

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1 and returns.std() > 1e-8:
        baseline = returns.mean()
        advantages = returns - baseline
        advantages = advantages / (advantages.std() + 1e-8)
    else:
        advantages = returns

    loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)

    if use_mobiu:
        optimizer.step(episode_return=sum(rewards))
    else:
        optimizer.step()

    return sum(rewards)

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

all_results = {}

for env_name in ENVIRONMENTS:
    print(f"\n{'='*70}")
    print(f"🎮 ENVIRONMENT: {env_name}")
    print(f"{'='*70}")
    
    # Get env specs
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])
    action_low = float(env.action_space.low[0])
    env.close()
    
    adam_results = []
    mobiu_results = []
    
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}", end="", flush=True)
        
        # Create identical policies
        torch.manual_seed(seed)
        np.random.seed(seed)
        template = ContinuousPolicy(state_dim, action_dim)
        template.set_action_bounds(action_low, action_high)
        
        adam_policy = copy.deepcopy(template)
        mobiu_policy = copy.deepcopy(template)
        
        # ===== ADAM =====
        print(" [Adam", end="", flush=True)
        torch.manual_seed(seed * 1000)
        np.random.seed(seed * 1000)
        adam_opt = torch.optim.Adam(adam_policy.parameters(), lr=BASE_LR)
        env = gym.make(env_name)
        adam_returns = []
        
        for ep in range(NUM_EPISODES):
            env_seed = seed * 100000 + ep
            ret = train_episode(env, adam_policy, adam_opt, use_mobiu=False, env_seed=env_seed)
            adam_returns.append(ret)
            if (ep + 1) % 60 == 0:
                print(".", end="", flush=True)
        env.close()
        print("]", end="", flush=True)
        
        # ===== MOBIU =====
        print(" [Mobiu", end="", flush=True)
        torch.manual_seed(seed * 1000)
        np.random.seed(seed * 1000)
        mobiu_opt = MobiuRLCore(mobiu_policy.parameters(), base_lr=BASE_LR)
        env = gym.make(env_name)
        mobiu_returns = []
        
        for ep in range(NUM_EPISODES):
            env_seed = seed * 100000 + ep
            ret = train_episode(env, mobiu_policy, mobiu_opt, use_mobiu=True, env_seed=env_seed)
            mobiu_returns.append(ret)
            if (ep + 1) % 60 == 0:
                print(".", end="", flush=True)
        env.close()
        print("]", end="", flush=True)
        
        adam_final = np.mean(adam_returns[-50:])
        mobiu_final = np.mean(mobiu_returns[-50:])
        
        adam_results.append(adam_final)
        mobiu_results.append(mobiu_final)
        
        diff = mobiu_final - adam_final
        symbol = "✓" if diff > 0 else "✗"
        print(f" | Adam: {adam_final:.1f} | Mobiu: {mobiu_final:.1f} | {symbol}")
    
    # Store results
    all_results[env_name] = {
        'adam': adam_results,
        'mobiu': mobiu_results
    }

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("📊 FINAL RESULTS - ALL ENVIRONMENTS")
print("=" * 70)

print(f"\n{'Environment':<25} | {'Adam':<12} | {'Mobiu':<12} | {'Δ%':<10} | {'Win':<8} | {'p-value':<10}")
print("-" * 85)

summary = []

for env_name in ENVIRONMENTS:
    adam_arr = np.array(all_results[env_name]['adam'])
    mobiu_arr = np.array(all_results[env_name]['mobiu'])
    
    adam_mean = adam_arr.mean()
    mobiu_mean = mobiu_arr.mean()
    
    if abs(adam_mean) > 1e-6:
        improvement = 100 * (mobiu_mean - adam_mean) / abs(adam_mean)
    else:
        improvement = 0
    
    win_rate = sum(m > a for m, a in zip(mobiu_arr, adam_arr)) / len(adam_arr)
    
    try:
        _, p_value = wilcoxon(adam_arr, mobiu_arr, alternative="less")
    except:
        p_value = 1.0
    
    sig = "🏆" if p_value < 0.05 and improvement > 0 else "✅" if win_rate >= 0.7 else "🔶" if improvement > 0 else "❌"
    
    print(f"{env_name:<25} | {adam_mean:>10.1f} | {mobiu_mean:>10.1f} | {improvement:>+8.1f}% | {win_rate*100:>6.0f}% | {p_value:>9.4f} {sig}")
    
    summary.append({
        'env': env_name,
        'adam': adam_mean,
        'mobiu': mobiu_mean,
        'improvement': improvement,
        'win_rate': win_rate,
        'p_value': p_value
    })

print("-" * 85)

# Overall summary
avg_improvement = np.mean([s['improvement'] for s in summary])
avg_win_rate = np.mean([s['win_rate'] for s in summary])
significant_count = sum(1 for s in summary if s['p_value'] < 0.05 and s['improvement'] > 0)

print(f"\n📋 OVERALL SUMMARY:")
print(f"   Average improvement: {avg_improvement:+.1f}%")
print(f"   Average win rate: {avg_win_rate*100:.0f}%")
print(f"   Significant results: {significant_count}/{len(ENVIRONMENTS)}")

# Save results
results = {
    'date': datetime.now().isoformat(),
    'config': {'episodes': NUM_EPISODES, 'seeds': NUM_SEEDS, 'lr': BASE_LR},
    'environments': ENVIRONMENTS,
    'results': all_results,
    'summary': summary
}

filename = f"mujoco_multienv_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Saved: {filename}")

# Investor summary
print("\n" + "=" * 70)
print("📋 INVESTOR SUMMARY")
print("=" * 70)
print(f"Market: Industrial Robotics ($4.07B → $9.56B by 2030, 15.1% CAGR)")
print(f"Tests: {len(ENVIRONMENTS)} MuJoCo environments × {NUM_SEEDS} seeds")
print(f"Results:")
for s in summary:
    status = "✅" if s['p_value'] < 0.05 else "🔶"
    print(f"  {status} {s['env']}: {s['improvement']:+.1f}%, p={s['p_value']:.4f}")
print("=" * 70)
