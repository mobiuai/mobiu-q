#!/usr/bin/env python3
"""
================================================================================
🎮 ATARI BREAKOUT - CUSTOMER VIEW TEST
================================================================================
This test shows what a CUSTOMER would experience:
- Baseline: Pure Adam optimizer (what customer has BEFORE Mobiu-Q)
- Test: Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)

NOT using use_soft_algebra flag - testing real customer integration!

Requirements:
    pip install mobiu-q ale-py gymnasium torch opencv-python
================================================================================
"""

import gc
import numpy as np
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from collections import deque

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"
METHOD = "adaptive"

NUM_EPISODES = 200
NUM_SEEDS = 3

print("🎮 Atari Breakout - Customer View Test")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 PyTorch: {torch.__version__}")
print(f"🖥️  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CNN ARCHITECTURE
# ============================================================

class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# FRAME PROCESSING
# ============================================================

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, frame):
        frame = preprocess_frame(frame)
        for _ in range(self.k):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)
    
    def step(self, frame):
        frame = preprocess_frame(frame)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []
        self.position = 0


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_atari(use_mobiu=True, num_episodes=200, seed=42, verbose=True):
    """
    Train DQN on Atari Breakout.
    
    use_mobiu=False: Pure Adam (what customer has BEFORE Mobiu-Q)
    use_mobiu=True:  Adam + Mobiu-Q (what customer has AFTER adding Mobiu-Q)
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    env.reset(seed=seed)
    n_actions = env.action_space.n
    
    policy_net = AtariDQN(n_actions).to(device)
    target_net = AtariDQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    base_opt = optim.Adam(policy_net.parameters(), lr=1e-4)
    
    if use_mobiu:
        # Customer adds Mobiu-Q like this:
        optimizer = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method=METHOD,
            maximize=True,
            sync_interval=50,
            verbose=False
        )
    else:
        # Baseline: Pure Adam (no Mobiu at all!)
        optimizer = base_opt
    
    buffer = ReplayBuffer(30000)
    frame_stack = FrameStack(k=4)
    
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 50000
    target_update = 500
    learning_starts = 5000
    
    episode_rewards = []
    total_steps = 0
    epsilon = epsilon_start
    
    mode = "Adam + Mobiu-Q" if use_mobiu else "Pure Adam"
    if verbose:
        print(f"\n🎮 Training {mode}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            epsilon = max(epsilon_end, epsilon_start - total_steps / epsilon_decay)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = frame_stack.step(next_obs)
            
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if total_steps > learning_starts and len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                loss = F.smooth_l1_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                
                if use_mobiu:
                    optimizer.step(episode_reward)
                else:
                    optimizer.step()
            
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"   Ep {episode+1:3d} | Avg: {avg:6.1f} | ε: {epsilon:.2f}")
    
    if use_mobiu:
        optimizer.end()
    env.close()
    
    buffer.clear()
    del policy_net, target_net, buffer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return episode_rewards


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🎮 ATARI BREAKOUT - CUSTOMER VIEW TEST")
    print("=" * 70)
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  • Baseline: Pure Adam optimizer (NO Mobiu)")
    print("  • Test: Adam + Mobiu-Q enhancement")
    print("=" * 70)
    
    results = {'adam': [], 'mobiu': []}
    
    for seed in range(NUM_SEEDS):
        print(f"\n🌱 Seed {seed + 1}/{NUM_SEEDS}")
        
        print("   Pure Adam...")
        r_adam = train_atari(use_mobiu=False, num_episodes=NUM_EPISODES, seed=seed*100)
        avg_adam = np.mean(r_adam[-50:])
        results['adam'].append(avg_adam)
        print(f"   Pure Adam done: {avg_adam:.1f}")
        
        print("   Adam + Mobiu-Q...")
        r_mobiu = train_atari(use_mobiu=True, num_episodes=NUM_EPISODES, seed=seed*100)
        avg_mobiu = np.mean(r_mobiu[-50:])
        results['mobiu'].append(avg_mobiu)
        print(f"   Adam + Mobiu done: {avg_mobiu:.1f}")
    
    adam = np.array(results['adam'])
    mobiu = np.array(results['mobiu'])
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Pure Adam:      {np.mean(adam):.1f} ± {np.std(adam):.1f}")
    print(f"Adam + Mobiu:   {np.mean(mobiu):.1f} ± {np.std(mobiu):.1f}")
    
    improvement = (np.mean(mobiu) - np.mean(adam)) / (abs(np.mean(adam)) + 1e-8) * 100
    wins = np.sum(mobiu > adam)
    
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    
    if improvement > 0 and wins > NUM_SEEDS/2:
        print("\n🏆 Mobiu-Q wins!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
