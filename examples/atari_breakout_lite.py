# atari_breakout_lite.py
"""
בדיקת Atari Breakout - גרסה קלה ל-Colab
- Replay buffer קטן יותר (50K במקום 100K)
- פחות episodes
- ניקוי זיכרון בין ריצות
"""

# %% Cell 1: התקנות
# !pip install gymnasium[atari] ale-py autorom
# !AutoROM --accept-license
# !pip install mobiu-q opencv-python

# %% Cell 2: Imports
import gc
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from datetime import datetime
import cv2
import warnings
warnings.filterwarnings('ignore')

from mobiu_q import MobiuOptimizer

print("🎮 Atari Breakout - Lite Version")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 PyTorch: {torch.__version__}")
print(f"🖥️  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Cell 3: CNN Architecture (smaller)
class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)  # 32->16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # 64->32
        self.fc1 = nn.Linear(32 * 9 * 9, 256)  # 512->256
        self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# %% Cell 4: Frame Processing
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

# %% Cell 5: Compact Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        # Store as uint8 to save memory
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

# %% Cell 6: Training Function
def train_atari(
    use_soft_algebra=True,
    num_episodes=200,
    seed=42,
    license_key="YOUR_KEY",
    verbose=True
):
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
    optimizer = MobiuOptimizer(
        base_opt,
        license_key=license_key,
        method="adaptive",
        use_soft_algebra=use_soft_algebra,
        sync_interval=50,
        verbose=False
    )
    
    # Smaller buffer
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
    
    mode = "Mobiu-Q" if use_soft_algebra else "Adam"
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
                optimizer.step(loss=loss.item())
            
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"   Ep {episode+1:3d} | Avg: {avg:6.1f} | ε: {epsilon:.2f}")
    
    optimizer.end()
    env.close()
    
    # Cleanup
    buffer.clear()
    del policy_net, target_net, buffer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return episode_rewards

# %% Cell 7: Run Comparison
def run_comparison(num_episodes=200, num_seeds=3, license_key="YOUR_KEY"):
    print("=" * 60)
    print("🎮 ATARI BREAKOUT - Mobiu-Q vs Adam")
    print("=" * 60)
    
    results = {'adam': [], 'mobiu': []}
    
    for seed in range(num_seeds):
        print(f"\n🌱 Seed {seed + 1}/{num_seeds}")
        
        # Adam
        print("   Adam...")
        r_adam = train_atari(False, num_episodes, seed*100, license_key)
        avg_adam = np.mean(r_adam[-50:])
        results['adam'].append(avg_adam)
        print(f"   Adam done: {avg_adam:.1f}")
        
        # Mobiu
        print("   Mobiu-Q...")
        r_mobiu = train_atari(True, num_episodes, seed*100, license_key)
        avg_mobiu = np.mean(r_mobiu[-50:])
        results['mobiu'].append(avg_mobiu)
        print(f"   Mobiu done: {avg_mobiu:.1f}")
    
    # Stats
    adam = np.array(results['adam'])
    mobiu = np.array(results['mobiu'])
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Adam:   {np.mean(adam):.1f} ± {np.std(adam):.1f}")
    print(f"Mobiu:  {np.mean(mobiu):.1f} ± {np.std(mobiu):.1f}")
    
    improvement = (np.mean(mobiu) - np.mean(adam)) / (abs(np.mean(adam)) + 1e-8) * 100
    wins = np.sum(mobiu > adam)
    
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"Win rate: {wins}/{num_seeds}")
    
    if improvement > 0 and wins > num_seeds/2:
        print("\n🏆 Mobiu-Q wins!")
    
    return results

# %% Cell 8: הרצה
LICENSE_KEY = "EYOUR_KEY_HERE"

# בדיקה קצרה - 200 episodes, 3 seeds (~1-2 שעות)
results = run_comparison(
    num_episodes=200,
    num_seeds=3,
    license_key=LICENSE_KEY
)
