#!/usr/bin/env python3
"""Minimal Gaussian REINFORCE with value baseline on Pendulum-v1.

Design goals:
 - Keep it short (<150 LOC) and dependency‑light (torch, gymnasium)
 - Safe on CPU; skips gracefully if torch not installed (Python 3.13 host)
 - Demonstrate: policy gradient, baseline (value net), return & advantage normalization, entropy bonus.

Usage:
  python modules/module_03_policy_methods/examples/policy_gradient_pendulum.py --episodes 50 --hidden 128 --lr 3e-4

For faster convergence increase episodes (200+). Rewards in Pendulum are negative; closer to 0 is better.
"""
from __future__ import annotations
import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Tuple, List

from rich.console import Console
from rich.table import Table

console = Console()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    console.print("[red]gymnasium not installed[/red]")
    gym = None


@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    episodes: int = 50
    gamma: float = 0.99
    lr: float = 3e-4
    hidden: int = 128
    entropy_beta: float = 1e-3
    seed: int = 0
    render: bool = False


def set_seed(seed: int):
    random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        # log_std as a parameter (state‑independent for simplicity)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        log_std = self.log_std.clamp(-5, 2)  # numerical stability
        return mean, log_std

    def dist(self, x: torch.Tensor) -> Normal:
        mean, log_std = self.forward(x)
        return Normal(mean, log_std.exp())


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


def normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + eps)


def train(cfg: Config):  # noqa: C901 (complexity okay for small script)
    if not TORCH_AVAILABLE:
        console.print("[yellow]Torch not available; please run inside Docker for this example.[/yellow]")
        return
    if gym is None:
        console.print("[red]gymnasium missing.[/red]")
        return

    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim, cfg.hidden)
    value_fn = ValueNet(obs_dim, cfg.hidden)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=cfg.lr)

    episode_rewards: List[float] = []

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            dist = policy.dist(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()
            value = value_fn(obs_t)

            # Pendulum expects action in [-2,2]; sample is ~[-∞,∞], so clamp scaled tanh
            action_clipped = torch.tanh(action) * 2.0
            next_obs, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy())
            done = terminated or truncated

            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            obs = next_obs
            if cfg.render:
                env.render()

        # Compute returns + advantages
        returns = torch.tensor(compute_returns(rewards, cfg.gamma), dtype=torch.float32)
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()
        advantages = normalize(advantages)
        returns_norm = normalize(returns)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns_norm)
        entropy_loss = -cfg.entropy_beta * entropies_t.mean()
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), 1.0)
        optimizer.step()

        ep_return = sum(rewards)
        episode_rewards.append(ep_return)
        recent = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
        console.print(f"[green]Ep {ep}/{cfg.episodes}[/green] Return {ep_return:.1f} Recent10 {recent:.1f} Loss {loss.item():.3f}")

    # Summary table
    table = Table(title="Pendulum Policy Gradient Summary")
    table.add_column("Episodes", justify="right")
    table.add_column("FinalReturn")
    table.add_column("AvgLast10")
    table.add_row(str(cfg.episodes), f"{episode_rewards[-1]:.1f}", f"{sum(episode_rewards[-10:]) / min(10, len(episode_rewards)):.1f}")
    console.print(table)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--entropy-beta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    return Config(episodes=args.episodes, gamma=args.gamma, lr=args.lr, hidden=args.hidden,
                  entropy_beta=args.entropy_beta, seed=args.seed, render=args.render)


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
