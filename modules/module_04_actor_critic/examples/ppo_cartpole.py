#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) on CartPole-v1 - Simplified Educational Version.

This is a simpler PPO implementation for beginners, using CartPole as a quick
test environment. For a more production-ready version with LunarLander, see
ppo_lunarlander.py.

Key features:
- Single-file implementation (~250 lines)
- Clear variable names and comments
- Fast training on CPU (~2 minutes)
- Minimal dependencies

Example:
  python ppo_cartpole.py --episodes 100 --lr 3e-4 --clip-eps 0.2

Reference:
  Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass

import numpy as np
from rich.console import Console

console = Console()

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    raise SystemExit("Requires: gymnasium, torch") from e


@dataclass
class Config:
    episodes: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int | None = None
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h)

    def get_action_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


def compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def collect_batch(env, model, cfg, steps=2048):
    """Collect experience batch."""
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    episode_rewards = []
    episode_reward = 0.0

    obs, _ = env.reset(seed=cfg.seed)

    for _ in range(steps):
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg.device)

        with torch.no_grad():
            action, log_prob, _, value = model.get_action_value(state_t)

        next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

        states.append(obs)
        actions.append(action.item())
        log_probs.append(log_prob.item())
        values.append(value.item())
        rewards.append(float(reward))
        dones.append(1.0 if done else 0.0)

        episode_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset(seed=cfg.seed)

    return (
        np.array(states), np.array(actions), np.array(log_probs),
        np.array(values), np.array(rewards), np.array(dones),
        episode_rewards
    )


def ppo_update(model, opt, states, actions, old_log_probs, advantages, returns, cfg):
    """PPO policy update."""
    states_t = torch.FloatTensor(states).to(cfg.device)
    actions_t = torch.LongTensor(actions).to(cfg.device)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(cfg.device)
    advantages_t = torch.FloatTensor(advantages).to(cfg.device)
    returns_t = torch.FloatTensor(returns).to(cfg.device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    num_samples = len(states)
    indices = np.arange(num_samples)

    losses = []

    for _ in range(cfg.epochs):
        np.random.shuffle(indices)

        for start in range(0, num_samples, cfg.batch_size):
            idx = indices[start:start + cfg.batch_size]

            _, log_probs, entropy, value = model.get_action_value(
                states_t[idx], actions_t[idx]
            )

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * ((value.squeeze() - returns_t[idx]) ** 2).mean()

            # Total loss
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            losses.append(loss.item())

    return np.mean(losses)


def train(cfg: Config):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    console.print("[bold green]PPO Training on CartPole-v1[/bold green]")
    console.print(f"Episodes: {cfg.episodes}, Clip: {cfg.clip_eps}\n")

    all_rewards = []

    for ep in range(1, cfg.episodes + 1):
        # Collect batch
        states, actions, log_probs, values, rewards, dones, ep_rewards = collect_batch(
            env, model, cfg
        )

        if ep_rewards:
            all_rewards.extend(ep_rewards)

        # Compute advantages
        advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)

        # Update policy
        loss = ppo_update(model, optimizer, states, actions, log_probs, advantages, returns, cfg)

        # Log
        if ep_rewards:
            avg = np.mean(all_rewards[-20:])
            console.log(f"Ep {ep}/{cfg.episodes} | Reward: {np.mean(ep_rewards):.1f} | Avg: {avg:.1f} | Loss: {loss:.3f}")

        # Check solved
        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= 195:
            console.print(f"[green]✓ Solved in {ep} episodes![/green]")
            break

    env.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")

    a = p.parse_args()
    return Config(
        episodes=a.episodes, lr=a.lr, gamma=a.gamma, gae_lambda=a.gae_lambda,
        clip_eps=a.clip_eps, epochs=a.epochs, batch_size=a.batch_size,
        entropy_coef=a.entropy_coef, value_coef=a.value_coef,
        seed=a.seed, device=a.device
    )


if __name__ == "__main__":
    train(parse_args())
