import gymnasium as gym
import numpy as np
import random
import time
import os
from datetime import datetime
from collections import deque

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
ENV_NAME = "CartPole-v1"
TRAIN_EPISODES = 1200
MAX_STEPS = 500

GAMMA = 0.99
LR = 5e-4

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

REPLAY_SIZE = 50_000
BATCH_SIZE = 64
START_LEARNING = 1000
TRAIN_EVERY = 1
TARGET_UPDATE_EVERY = 1000

SEED = 1
RENDER_HUMAN = False  # needs pygame


# ==========================================
# 📦 UTILITIES
# ==========================================
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plots(rewards, env_name=ENV_NAME):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw Reward")

    window_size = 25
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, linewidth=2, label=f"Moving Avg ({window_size})")

    plt.title(f"Agent Learning Progress ({env_name})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    folder_path = "rl_methods/experiments"
    ensure_dir(folder_path)

    filename = f"{folder_path}/cartpole_dqn_metrics_{timestamp()}.png"
    plt.savefig(filename)
    print(f"📊 Metrics saved to {filename}")
    plt.close()


def make_env(render_mode=None, seed=None):
    env = gym.make(ENV_NAME, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    return env


# ==========================================
# 🧠 DQN MODEL
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def select_action(policy_net, state_np, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy_net(s)
        return int(torch.argmax(q, dim=1).item())


# ==========================================
# ✅ EVALUATION 
# ==========================================
def evaluate_agent(model_path=None, episodes=20, seed=123):
    """
    Returns average reward over N episodes.
    If model_path is None -> random policy.
    """
    env = make_env(render_mode=None, seed=seed)
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = None
    if model_path is not None:
        policy_net = DQN(state_dim=state_dim, n_actions=n_actions).to(device)
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()

    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)
        total = 0.0

        for _ in range(MAX_STEPS):
            if policy_net is None:
                action = env.action_space.sample()
            else:
                action = select_action(policy_net, state, epsilon=0.0, n_actions=n_actions, device=device)

            obs2, reward, terminated, truncated, _ = env.step(action)

            # truncated (time limit) is NOT a terminal failure
            total += float(reward)
            state = np.array(obs2, dtype=np.float32)

            if terminated or truncated:
                break

        scores.append(total)

    env.close()
    return float(np.mean(scores)), float(np.std(scores))


# ==========================================
# 🎬 WATCH (single episode, for fun)
# ==========================================
def watch_agent(model_path=None, delay=0.02):
    render_mode = "human" if RENDER_HUMAN else None

    if model_path is None:
        env = gym.make(ENV_NAME, render_mode=render_mode)
        env.reset()  # no seed
    else:
        env = make_env(render_mode=render_mode, seed=SEED)  # keep deterministic for trained demo

    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = None

    if model_path is not None:
        policy_net = DQN(state_dim=state_dim, n_actions=n_actions).to(device)
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()

    obs, info = env.reset() 
    state = np.array(obs, dtype=np.float32)
    total_reward = 0.0

    print("\n🎬 Simulation Started...")
    for step in range(MAX_STEPS):
        if policy_net is None:
            action = random.randrange(n_actions)
        else:
            action = select_action(policy_net, state, epsilon=0.0, n_actions=n_actions, device=device)

        obs2, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        state = np.array(obs2, dtype=np.float32)

        if not RENDER_HUMAN and (step % 25 == 0 or terminated or truncated):
            print(f"step={step:3d} action={action} reward={reward:.2f} total={total_reward:.2f}")

        time.sleep(delay)
        if terminated or truncated:
            break

    print(f"🏁 Episode finished. Total Score: {total_reward:.2f}\n")
    env.close()


# ==========================================
#  TRAIN
# ==========================================
def train_agent():
    env = make_env(render_mode=None, seed=SEED)
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim=state_dim, n_actions=n_actions).to(device)
    target_net = DQN(state_dim=state_dim, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_SIZE)

    epsilon = EPSILON_START
    rewards_history = []

    grad_steps = 0
    env_steps = 0

    print(f"🔄 Training DQN for {TRAIN_EPISODES} episodes on {ENV_NAME}")

    for ep in tqdm(range(1, TRAIN_EPISODES + 1)):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            action = select_action(policy_net, state, epsilon, n_actions, device)
            obs2, reward, terminated, truncated, _ = env.step(action)

            # Reward shaping: penalize real failure to speed learning
            if terminated:
                reward = -10.0

            next_state = np.array(obs2, dtype=np.float32)

            #  only treat terminated as terminal for bootstrapping
            done_for_learning = 1.0 if terminated else 0.0

            buffer.push(state, action, float(reward), next_state, done_for_learning)
            state = next_state

            episode_reward += float(reward)
            env_steps += 1

            if len(buffer) >= START_LEARNING and env_steps % TRAIN_EVERY == 0:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)

                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_sa = policy_net(s_t).gather(1, a_t)

                with torch.no_grad():
                    max_q_s2 = target_net(s2_t).max(dim=1, keepdim=True)[0]
                    target = r_t + GAMMA * (1.0 - d_t) * max_q_s2

                loss = F.smooth_l1_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                grad_steps += 1
                if grad_steps % TARGET_UPDATE_EVERY == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if terminated or truncated:
                break

        rewards_history.append(episode_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    env.close()

    folder_path = "rl_methods/experiments"
    ensure_dir(folder_path)

    model_path = f"{folder_path}/cartpole_dqn_{timestamp()}.pt"
    torch.save(policy_net.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")

    save_plots(rewards_history, env_name=ENV_NAME)
    return model_path


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"🎯 DQN with Metrics ({ENV_NAME})")

    mean_r, std_r = evaluate_agent(model_path=None, episodes=30, seed=999)
    print(f"🎲 Random policy avg over 30 eps: {mean_r:.1f} ± {std_r:.1f}")

    input("\n❌ Press [Enter] to watch UNTRAINED agent (1 episode)...")
    watch_agent(model_path=None, delay=0.02)

    input("💪 Press [Enter] to TRAIN and generate plots...")
    trained_model_path = train_agent()

    mean_t, std_t = evaluate_agent(model_path=trained_model_path, episodes=30, seed=999)
    print(f"🏆 Trained policy avg over 30 eps: {mean_t:.1f} ± {std_t:.1f}")

    while True:
        input("🏆 Press [Enter] to watch TRAINED agent (1 episode)...")
        watch_agent(model_path=trained_model_path, delay=0.02)
