#!/usr/bin/env python3
"""
Minimal DQN implementation for Gymnasium CartPole-v1 with Rich logging.

Example:
  python dqn_cartpole.py --episodes 400 --learning-rate 1e-3 --gamma 0.99 --batch-size 64 --target-update 10 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 500
"""
from __future__ import annotations
import argparse
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this example. Install a compatible torch wheel or run inside the provided Docker images (CUDA/ROCm)."
    ) from e
from rich.console import Console
from rich.progress import track

console = Console()

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


@dataclass
class DQNConfig:
    episodes: int = 400
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    target_update: int = 10
    buffer_size: int = 10000
    start_training: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    seed: int | None = None
    device: str = "cpu"


def select_action(qnet: QNetwork, state: np.ndarray, steps_done: int, cfg: DQNConfig, action_dim: int) -> int:
    eps_threshold = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1.0 * steps_done / cfg.epsilon_decay)
    if random.random() < eps_threshold:
        return random.randrange(action_dim)
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        q_values = qnet(s)
        return int(q_values.argmax(dim=1).item())


def optimize(qnet: QNetwork, target_net: QNetwork, buffer: ReplayBuffer, optimizer: optim.Optimizer, cfg: DQNConfig):
    if len(buffer) < cfg.start_training or len(buffer) < cfg.batch_size:
        return None
    transitions = buffer.sample(cfg.batch_size)
    state_batch = torch.as_tensor(np.array(transitions.state), dtype=torch.float32, device=cfg.device)
    action_batch = torch.as_tensor(transitions.action, dtype=torch.int64, device=cfg.device).unsqueeze(1)
    reward_batch = torch.as_tensor(transitions.reward, dtype=torch.float32, device=cfg.device).unsqueeze(1)
    next_state_batch = torch.as_tensor(np.array(transitions.next_state), dtype=torch.float32, device=cfg.device)
    done_batch = torch.as_tensor(transitions.done, dtype=torch.float32, device=cfg.device).unsqueeze(1)

    q_values = qnet(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1, keepdim=True)[0]
        target = reward_batch + cfg.gamma * (1.0 - done_batch) * next_q_values

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train(cfg: DQNConfig):
    set_seed(cfg.seed)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    qnet = QNetwork(obs_dim, action_dim).to(cfg.device)
    target_net = QNetwork(obs_dim, action_dim).to(cfg.device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()

    optimizer = optim.Adam(qnet.parameters(), lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_size)

    episode_rewards = []
    global_steps = 0

    for episode in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed)
        done = False
        ep_reward = 0.0
        last_loss = None

        while not done:
            action = select_action(qnet, obs, global_steps, cfg, action_dim)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            global_steps += 1

            loss = optimize(qnet, target_net, buffer, optimizer, cfg)
            if loss is not None:
                last_loss = loss

            if global_steps % cfg.target_update == 0:
                target_net.load_state_dict(qnet.state_dict())

        episode_rewards.append(ep_reward)
        avg_last_20 = np.mean(episode_rewards[-20:])
        console.log(
            f"Episode {episode}/{cfg.episodes} | Reward: {ep_reward:.1f} | Avg(20): {avg_last_20:.1f} | Buffer: {len(buffer)} | Loss: {last_loss if last_loss is not None else 'NA'}"
        )

        if avg_last_20 >= 195.0:
            console.print("[bold green]✓ Environment solved![/bold green]")
            break

    env.close()


def parse_args() -> DQNConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-update", type=int, default=10)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--start-training", type=int, default=1000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    return DQNConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        start_training=args.start_training,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        device=args.device,
    )


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
