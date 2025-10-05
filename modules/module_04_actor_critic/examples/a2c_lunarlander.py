#!/usr/bin/env python3
"""
A2C (Advantage Actor Critic) on LunarLander-v3 using PyTorch.

Note: LunarLander requires Box2D. If not installed, this script will exit with instructions.

Example:
  python a2c_lunarlander.py --episodes 1000 --lr 3e-4 --gamma 0.99 --gae-lambda 0.95
"""
from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from rich.console import Console

console = Console()

try:
    import gymnasium as gym
except Exception as e:
    raise SystemExit("Gymnasium is required.") from e

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this example. Install torch or use Docker (CUDA/ROCm)."
    ) from e


def check_box2d():
    try:
        import Box2D  # noqa: F401
        return True
    except Exception:
        console.print(
            "[yellow]LunarLander requires Box2D. Install with:[/yellow] pip install box2d-py or pip install gymnasium[box2d]"
        )
        return False


@dataclass
class Config:
    episodes: int = 1000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    seed: int | None = None
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(256, action_dim))
        self.value = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h)
        return logits, value


def gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae_val = 0.0
    # values: V(s_t) for each step plus V(s_{T}) terminal bootstrap; here we keep same length, so shift next value in loop
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1.0 - dones[t]) * values[t + 1] - values[t]
        gae_val = delta + gamma * lam * (1.0 - dones[t]) * gae_val
        advantages.append(gae_val)
    advantages.reverse()
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def train(cfg: Config):
    if not check_box2d():
        sys.exit(1)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = ActorCritic(obs_dim, action_dim).to(cfg.device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    rewards_hist = []

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed)
        done = False
        ep_reward = 0.0

        log_probs = []
        values = []
        rewards = []
        dones = []

        while not done:
            s = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            logits, value = net(s)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value.squeeze(0).item())
            rewards.append(float(reward))
            dones.append(1.0 if done else 0.0)

            obs = next_obs
            ep_reward += reward

        # Bootstrap with final value 0 for terminal
        values.append(0.0)
        adv, ret = gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=cfg.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=cfg.device)
        log_probs_t = torch.stack(log_probs)

        # Normalize advantages
        if adv_t.std().item() > 1e-6:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        policy_loss = -(log_probs_t * adv_t).sum()
        # Value loss: run value on the stored states again for better target fit
        # (simple approximation using stored values as baseline)
        value_loss = 0.5 * ((torch.as_tensor(values[:-1], device=cfg.device) - ret_t) ** 2).sum()
        entropy_bonus = 0.0  # could compute from dist.entropy() per step
        loss = policy_loss + value_loss - 0.0 * entropy_bonus

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        rewards_hist.append(ep_reward)
        avg_last_20 = float(np.mean(rewards_hist[-20:]))
        console.log(
            f"Episode {ep}/{cfg.episodes} | Reward: {ep_reward:.1f} | Avg(20): {avg_last_20:.1f} | Loss: {loss.item():.3f}"
        )

    env.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()
    return Config(
        episodes=a.episodes,
        lr=a.lr,
        gamma=a.gamma,
        gae_lambda=a["gae_lambda"] if isinstance(a, dict) else a.gae_lambda,
        seed=a.seed,
        device=a.device,
    )


if __name__ == "__main__":
    train(parse_args())
