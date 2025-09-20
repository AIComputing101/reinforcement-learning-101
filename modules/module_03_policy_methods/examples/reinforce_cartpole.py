#!/usr/bin/env python3
"""
REINFORCE (policy gradient) on CartPole-v1 using PyTorch with Rich logging.

Example:
  python reinforce_cartpole.py --episodes 800 --lr 1e-2 --gamma 0.99 --seed 0 --device cpu

Requires PyTorch; use Docker if PyTorch is not available on your host Python.
"""
from __future__ import annotations
import argparse
import random
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import List
from rich.console import Console

console = Console()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this example. Install a compatible torch wheel or run inside the provided Docker images (CUDA/ROCm)."
    ) from e


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(x), dim=-1)


@dataclass
class Config:
    episodes: int = 800
    gamma: float = 0.99
    lr: float = 1e-2
    seed: int | None = None
    device: str = "cpu"


def run_episode(env, policy: Policy, device: str):
    obs, _ = env.reset()
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logp = policy(obs_t)
        action = torch.distributions.Categorical(logits=logp).sample()
        log_prob = logp[0, action]
        next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(float(reward))
        obs = next_obs
    return log_probs, rewards


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    # normalize for stability
    returns = np.array(returns, dtype=np.float32)
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns.tolist()


def train(cfg: Config):
    set_seed(cfg.seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = Policy(obs_dim, action_dim).to(cfg.device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    best_avg = -1e9
    rewards_hist: List[float] = []

    for ep in range(1, cfg.episodes + 1):
        log_probs, rewards = run_episode(env, policy, cfg.device)
        returns = compute_returns(rewards, cfg.gamma)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=cfg.device)
        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns_t).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        ep_reward = sum(rewards)
        rewards_hist.append(ep_reward)
        avg_last_20 = float(np.mean(rewards_hist[-20:]))
        if avg_last_20 > best_avg:
            best_avg = avg_last_20
        console.log(
            f"Episode {ep}/{cfg.episodes} | Reward: {ep_reward:.1f} | Avg(20): {avg_last_20:.1f} | Loss: {loss.item():.3f}"
        )
        if avg_last_20 >= 195.0:
            console.print("[bold green]✓ Environment solved![/bold green]")
            break

    env.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()
    return Config(episodes=a.episodes, lr=a.lr, gamma=a.gamma, seed=a.seed, device=a.device)


if __name__ == "__main__":
    train(parse_args())
