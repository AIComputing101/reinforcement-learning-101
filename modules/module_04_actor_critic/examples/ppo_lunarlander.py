#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) on LunarLander-v2 using PyTorch.

PPO is the industry-standard on-policy RL algorithm (2024-2025), used in:
- OpenAI's RLHF for ChatGPT
- DeepMind's AlphaStar
- Most production RL deployments

Key features:
- Clipped surrogate objective for stable updates
- Multiple epochs per batch for sample efficiency
- GAE for variance reduction
- Entropy bonus for exploration

Note: LunarLander requires Box2D. If not installed, this script will exit with instructions.

Example:
  python ppo_lunarlander.py --episodes 1000 --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 --clip-eps 0.2 --epochs 4

Reference:
  Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
  https://arxiv.org/abs/1707.06347
"""
from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from rich.console import Console
from rich.progress import track

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
    clip_eps: float = 0.2  # PPO clipping parameter
    epochs: int = 4  # Number of policy update epochs per batch
    batch_size: int = 64  # Mini-batch size for updates
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    seed: int | None = None
    device: str = "cpu"


class ActorCritic(nn.Module):
    """Shared network for policy and value function."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h)
        return logits, value

    def get_value(self, x):
        h = self.shared(x)
        return self.value(h)

    def get_action_and_value(self, x, action=None):
        """Get action, log_prob, entropy, and value."""
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h)

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]

    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def collect_rollout(env, net, cfg: Config, max_steps: int = 2048):
    """Collect a batch of experience."""
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []

    obs, _ = env.reset(seed=cfg.seed)
    episode_rewards = []
    episode_reward = 0.0

    for step in range(max_steps):
        state_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = net.get_action_and_value(state_t)

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
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(log_probs, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        episode_rewards
    )


def ppo_update(net, optimizer, states, actions, old_log_probs, advantages, returns, cfg: Config):
    """Perform PPO update with multiple epochs and mini-batches."""

    states_t = torch.as_tensor(states, dtype=torch.float32, device=cfg.device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=cfg.device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=cfg.device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=cfg.device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=cfg.device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    num_samples = len(states)
    indices = np.arange(num_samples)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_clipfrac = 0.0
    num_updates = 0

    for epoch in range(cfg.epochs):
        np.random.shuffle(indices)

        for start in range(0, num_samples, cfg.batch_size):
            end = start + cfg.batch_size
            batch_idx = indices[start:end]

            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]
            batch_returns = returns_t[batch_idx]

            # Get current policy and value
            _, new_log_probs, entropy, values = net.get_action_and_value(batch_states, batch_actions)

            # Policy loss with PPO clipping
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped for stability)
            values = values.squeeze()
            value_loss = 0.5 * ((values - batch_returns) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()

            # Combined loss
            loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

            # Track clipping fraction (diagnostics)
            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                total_clipfrac += clipfrac

            num_updates += 1

    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "clipfrac": total_clipfrac / num_updates
    }


def train(cfg: Config):
    if not check_box2d():
        sys.exit(1)

    # Set seeds for reproducibility
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = ActorCritic(obs_dim, action_dim).to(cfg.device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    console.print(f"[bold green]PPO Training Started[/bold green]")
    console.print(f"Device: {cfg.device}")
    console.print(f"Episodes: {cfg.episodes}")
    console.print(f"Clip epsilon: {cfg.clip_eps}")
    console.print(f"Epochs per update: {cfg.epochs}")
    console.print(f"Batch size: {cfg.batch_size}\n")

    all_episode_rewards = []
    update_count = 0

    for episode in range(1, cfg.episodes + 1):
        # Collect rollout
        states, actions, log_probs, values, rewards, dones, episode_rewards = collect_rollout(
            env, net, cfg, max_steps=2048
        )

        if episode_rewards:
            all_episode_rewards.extend(episode_rewards)

        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)

        # PPO update
        metrics = ppo_update(net, optimizer, states, actions, log_probs, advantages, returns, cfg)

        update_count += 1

        # Logging
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            avg_last_20 = np.mean(all_episode_rewards[-20:]) if len(all_episode_rewards) >= 20 else np.mean(all_episode_rewards)

            console.log(
                f"Episode {episode}/{cfg.episodes} | "
                f"Reward: {avg_reward:.1f} | "
                f"Avg(20): {avg_last_20:.1f} | "
                f"P_Loss: {metrics['policy_loss']:.3f} | "
                f"V_Loss: {metrics['value_loss']:.3f} | "
                f"Entropy: {metrics['entropy']:.3f} | "
                f"ClipFrac: {metrics['clipfrac']:.2%}"
            )

        # Check if solved (LunarLander is solved at 200+)
        if len(all_episode_rewards) >= 100:
            avg_100 = np.mean(all_episode_rewards[-100:])
            if avg_100 >= 200:
                console.print(f"[bold green]✓ Solved! Average reward over last 100 episodes: {avg_100:.1f}[/bold green]")
                break

    env.close()
    console.print(f"\n[bold]Training complete![/bold] Total updates: {update_count}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="PPO on LunarLander-v2")
    p.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    p.add_argument("--epochs", type=int, default=4, help="Number of epochs per update")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = p.parse_args()

    return Config(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
