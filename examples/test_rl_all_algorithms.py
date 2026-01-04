#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q RL BENCHMARK - ALL ALGORITHMS (From Scratch)
================================================================================
Tests Soft Algebra across different RL algorithms, implemented from scratch
for proper Mobiu integration (passing reward, not loss).

Algorithms:
  - REINFORCE (Policy Gradient)
  - A2C (Advantage Actor-Critic)
  - PPO (Proximal Policy Optimization)
  - DQN (Deep Q-Network)

Requirements:
    pip install mobiu-q gymnasium torch numpy scipy

Usage:
    python test_rl_all_algorithms.py
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
from datetime import datetime
from scipy import stats
import json
import copy
import warnings
warnings.filterwarnings('ignore')

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY_HERE"
N_SEEDS = 5
EVAL_EPISODES = 10

ALGORITHMS = {
    'REINFORCE': {
        'envs': ['CartPole-v1', 'LunarLander-v3'],
        'episodes': 500,
        'lr': 0.001,
    },
    'A2C': {
        'envs': ['CartPole-v1', 'LunarLander-v3'],
        'episodes': 500,
        'lr': 0.001,
    },
    'PPO': {
        'envs': ['CartPole-v1', 'LunarLander-v3'],
        'episodes': 500,
        'lr': 0.0003,
    },
    'DQN': {
        'envs': ['CartPole-v1', 'LunarLander-v3'],
        'episodes': 300,
        'lr': 0.001,
    },
}

# ============================================================
# NETWORKS
# ============================================================

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE/A2C/PPO"""
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


class ActorCritic(nn.Module):
    """Actor-Critic network for A2C/PPO"""
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action_and_value(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = F.log_softmax(logits, dim=-1)[0, action]
        return action, log_prob, value.squeeze()


class DQNNetwork(nn.Module):
    """Q-Network for DQN"""
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# REPLAY BUFFER (for DQN)
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_reinforce(env_name, num_episodes, lr, seed, use_soft_algebra):
    """REINFORCE (vanilla policy gradient)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(state_dim, action_dim)
    base_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    optimizer = MobiuOptimizer(
        base_opt,
        use_soft_algebra=use_soft_algebra,
        maximize=True,
        verbose=False
    )
    
    episode_rewards = []
    
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
        
        # Compute discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        loss = sum(-lp * G for lp, G in zip(log_probs, returns))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(sum(rewards))  # Pass REWARD, not loss!
        
        episode_rewards.append(sum(rewards))
    
    optimizer.end()
    env.close()
    return np.mean(episode_rewards[-50:])


def train_a2c(env_name, num_episodes, lr, seed, use_soft_algebra):
    """Advantage Actor-Critic"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(state_dim, action_dim)
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = MobiuOptimizer(
        base_opt,
        use_soft_algebra=use_soft_algebra,
        maximize=True,
        verbose=False
    )
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        log_probs, values, rewards = [], [], []
        done = False
        
        while not done:
            action, log_prob, value = model.get_action_and_value(state)
            log_probs.append(log_prob)
            values.append(value)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        
        # Compute returns and advantages
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)
        
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss
        actor_loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))
        
        # Critic loss
        critic_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(sum(rewards))  # Pass REWARD!
        
        episode_rewards.append(sum(rewards))
    
    optimizer.end()
    env.close()
    return np.mean(episode_rewards[-50:])


def train_ppo(env_name, num_episodes, lr, seed, use_soft_algebra):
    """Proximal Policy Optimization"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(state_dim, action_dim)
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = MobiuOptimizer(
        base_opt,
        use_soft_algebra=use_soft_algebra,
        maximize=True,
        verbose=False
    )
    
    clip_epsilon = 0.2
    n_epochs = 4
    episode_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        states, actions, old_log_probs, rewards = [], [], [], []
        done = False
        
        # Collect trajectory
        while not done:
            action, log_prob, _ = model.get_action_and_value(state)
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob.detach())
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        
        # Compute returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        
        # PPO update
        for _ in range(n_epochs):
            logits, values = model(states)
            new_log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Ratio and clipped loss
            ratio = (new_log_probs - old_log_probs).exp()
            advantages = returns - values.squeeze().detach()
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(sum(rewards))  # Pass REWARD!
        
        episode_rewards.append(sum(rewards))
    
    optimizer.end()
    env.close()
    return np.mean(episode_rewards[-50:])


def train_dqn(env_name, num_episodes, lr, seed, use_soft_algebra):
    """Deep Q-Network"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQNNetwork(state_dim, action_dim)
    target_net = DQNNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    base_opt = torch.optim.Adam(policy_net.parameters(), lr=lr)
    optimizer = MobiuOptimizer(
        base_opt,
        use_soft_algebra=use_soft_algebra,
        maximize=True,
        verbose=False
    )
    
    buffer = ReplayBuffer(10000)
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = num_episodes * 0.8
    target_update = 10
    
    episode_rewards = []
    total_steps = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_start - ep / epsilon_decay)
        
        while not done:
            total_steps += 1
            
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            
            # Train
            if len(buffer) >= batch_size:
                states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)
                
                # Current Q values
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # Target Q values
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards_b + gamma * next_q * (1 - dones)
                
                loss = F.mse_loss(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(episode_reward)  # Pass current episode reward
        
        # Update target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(episode_reward)
    
    optimizer.end()
    env.close()
    return np.mean(episode_rewards[-50:])


# ============================================================
# MAIN
# ============================================================

TRAIN_FUNCS = {
    'REINFORCE': train_reinforce,
    'A2C': train_a2c,
    'PPO': train_ppo,
    'DQN': train_dqn,
}


def main():
    print("=" * 80)
    print("🎮 MOBIU-Q RL BENCHMARK - ALL ALGORITHMS (From Scratch)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {N_SEEDS}")
    print("=" * 80)
    
    all_results = []
    
    for algo_name, config in ALGORITHMS.items():
        print(f"\n{'='*60}")
        print(f"📊 Algorithm: {algo_name}")
        print(f"{'='*60}")
        
        train_func = TRAIN_FUNCS[algo_name]
        
        for env_id in config['envs']:
            print(f"\n  🎯 Environment: {env_id}")
            print(f"  {'-'*50}")
            
            baseline_scores = []
            mobiu_scores = []
            
            for seed in range(N_SEEDS):
                print(f"    Seed {seed+1}/{N_SEEDS}...", end=" ", flush=True)
                
                # Baseline (SA=OFF)
                try:
                    base_score = train_func(
                        env_id, config['episodes'], config['lr'], 
                        seed, use_soft_algebra=False
                    )
                    baseline_scores.append(base_score)
                except Exception as e:
                    print(f"Baseline error: {e}")
                    continue
                
                # Mobiu (SA=ON)
                try:
                    mobiu_score = train_func(
                        env_id, config['episodes'], config['lr'],
                        seed, use_soft_algebra=True
                    )
                    mobiu_scores.append(mobiu_score)
                except Exception as e:
                    print(f"Mobiu error: {e}")
                    continue
                
                diff = ((mobiu_score - base_score) / (abs(base_score) + 1e-8) * 100)
                win = "✅" if mobiu_score > base_score else "❌"
                print(f"Base: {base_score:.1f}, Mobiu: {mobiu_score:.1f}, Δ: {diff:+.1f}% {win}")
            
            if len(baseline_scores) > 0:
                baseline_mean = np.mean(baseline_scores)
                mobiu_mean = np.mean(mobiu_scores)
                improvement = ((mobiu_mean - baseline_mean) / (abs(baseline_mean) + 1e-8) * 100)
                wins = sum(m > b for m, b in zip(mobiu_scores, baseline_scores))
                
                try:
                    _, p_val = stats.ttest_rel(mobiu_scores, baseline_scores)
                except:
                    p_val = 1.0
                
                result = {
                    'algorithm': algo_name,
                    'environment': env_id,
                    'baseline_mean': baseline_mean,
                    'mobiu_mean': mobiu_mean,
                    'improvement': improvement,
                    'wins': wins,
                    'total': len(baseline_scores),
                    'p_value': p_val,
                }
                all_results.append(result)
                
                sig = "✅" if p_val < 0.05 and improvement > 0 else "🔶"
                print(f"\n  📈 Result: {baseline_mean:.1f} → {mobiu_mean:.1f} ({improvement:+.1f}%)")
                print(f"     Wins: {wins}/{len(baseline_scores)}, p={p_val:.4f} {sig}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Algorithm':<12} {'Environment':<20} {'Baseline':<10} {'Mobiu':<10} {'Improve':<10} {'Wins':<8} {'Sig'}")
    print("-" * 80)
    
    for r in all_results:
        sig = "✅" if r['p_value'] < 0.05 and r['improvement'] > 0 else \
              "❌" if r['p_value'] < 0.05 and r['improvement'] < 0 else "🔶"
        print(f"{r['algorithm']:<12} {r['environment']:<20} {r['baseline_mean']:<10.1f} "
              f"{r['mobiu_mean']:<10.1f} {r['improvement']:>+8.1f}% {r['wins']}/{r['total']:<5} {sig}")
    
    # Overall
    if all_results:
        avg_improvement = np.mean([r['improvement'] for r in all_results])
        total_wins = sum(r['wins'] for r in all_results)
        total_tests = sum(r['total'] for r in all_results)
        
        print(f"\n{'='*80}")
        print(f"📈 OVERALL: {avg_improvement:+.1f}% average improvement")
        print(f"   Win Rate: {total_wins}/{total_tests} ({total_wins/total_tests*100:.1f}%)")
    
    # Save
    with open('rl_all_algorithms_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n✅ Results saved to rl_all_algorithms_results.json")


if __name__ == "__main__":
    main()
