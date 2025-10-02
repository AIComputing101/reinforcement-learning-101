#!/usr/bin/env python3
"""
PPO with TensorBoard Integration and Advanced Logging.

Enhanced version of PPO CartPole with comprehensive TensorBoard logging:
- Episode rewards and lengths
- Policy and value losses
- Learning rate scheduling
- Gradient norms
- KL divergence tracking
- Advantage and return distributions
- Hyperparameter logging

Example:
  # Basic training with TensorBoard
  python ppo_cartpole_tensorboard.py --episodes 100

  # View logs
  tensorboard --logdir runs/

  # With custom run name
  python ppo_cartpole_tensorboard.py --episodes 100 --run-name my_experiment

  # With learning rate scheduling
  python ppo_cartpole_tensorboard.py --episodes 100 --use-lr-schedule
"""
from __future__ import annotations
import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    console.print("[yellow]TensorBoard not available. Install: pip install tensorboard[/yellow]")


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
    max_grad_norm: float = 0.5
    use_lr_schedule: bool = False  # Cosine annealing LR schedule
    run_name: str | None = None
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
    episode_lengths = []
    episode_reward = 0.0
    episode_length = 0

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
        episode_length += 1
        obs = next_obs

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            obs, _ = env.reset(seed=cfg.seed)

    return (
        np.array(states), np.array(actions), np.array(log_probs),
        np.array(values), np.array(rewards), np.array(dones),
        episode_rewards, episode_lengths
    )


def ppo_update(model, opt, states, actions, old_log_probs, advantages, returns, cfg, writer=None, update_step=0):
    """PPO policy update with detailed logging."""
    states_t = torch.FloatTensor(states).to(cfg.device)
    actions_t = torch.LongTensor(actions).to(cfg.device)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(cfg.device)
    advantages_t = torch.FloatTensor(advantages).to(cfg.device)
    returns_t = torch.FloatTensor(returns).to(cfg.device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    num_samples = len(states)
    indices = np.arange(num_samples)

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clipfrac": [],
        "grad_norm": []
    }

    for epoch in range(cfg.epochs):
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

            # Track gradient norm
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            opt.step()

            # Metrics
            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.mean().item())
            metrics["grad_norm"].append(grad_norm.item())

            # KL divergence and clip fraction
            with torch.no_grad():
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                clipfrac = (torch.abs(ratio - 1) > cfg.clip_eps).float().mean().item()
                metrics["approx_kl"].append(approx_kl)
                metrics["clipfrac"].append(clipfrac)

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar("train/policy_loss", np.mean(metrics["policy_loss"]), update_step)
        writer.add_scalar("train/value_loss", np.mean(metrics["value_loss"]), update_step)
        writer.add_scalar("train/entropy", np.mean(metrics["entropy"]), update_step)
        writer.add_scalar("train/approx_kl", np.mean(metrics["approx_kl"]), update_step)
        writer.add_scalar("train/clipfrac", np.mean(metrics["clipfrac"]), update_step)
        writer.add_scalar("train/grad_norm", np.mean(metrics["grad_norm"]), update_step)
        writer.add_scalar("train/learning_rate", opt.param_groups[0]["lr"], update_step)

        # Distribution statistics
        writer.add_histogram("train/advantages", advantages_t.cpu().numpy(), update_step)
        writer.add_histogram("train/returns", returns_t.cpu().numpy(), update_step)

    return np.mean(metrics["policy_loss"])


def train(cfg: Config):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # TensorBoard setup
    if not TENSORBOARD_AVAILABLE:
        console.print("[yellow]TensorBoard not available. Continuing without logging.[/yellow]")
        writer = None
    else:
        run_name = cfg.run_name or f"ppo_cartpole_{int(time.time())}"
        log_dir = Path("runs") / run_name
        writer = SummaryWriter(log_dir)
        console.print(f"[green]TensorBoard logging to: {log_dir}[/green]")
        console.print(f"[green]View with: tensorboard --logdir runs/[/green]\n")

    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Learning rate scheduler
    scheduler = None
    if cfg.use_lr_schedule:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.episodes)
        console.print("[cyan]Using cosine annealing LR schedule[/cyan]")

    console.print("[bold green]PPO Training with TensorBoard[/bold green]")
    console.print(f"Episodes: {cfg.episodes}, Clip: {cfg.clip_eps}\n")

    all_rewards = []
    update_step = 0
    start_time = time.time()

    for ep in range(1, cfg.episodes + 1):
        # Collect batch
        states, actions, log_probs, values, rewards, dones, ep_rewards, ep_lengths = collect_batch(
            env, model, cfg
        )

        if ep_rewards:
            all_rewards.extend(ep_rewards)

            # Log episode metrics
            if writer is not None:
                for i, (r, l) in enumerate(zip(ep_rewards, ep_lengths)):
                    global_step = len(all_rewards) - len(ep_rewards) + i
                    writer.add_scalar("episode/reward", r, global_step)
                    writer.add_scalar("episode/length", l, global_step)

        # Compute advantages
        advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)

        # Update policy
        loss = ppo_update(model, optimizer, states, actions, log_probs,
                         advantages, returns, cfg, writer, update_step)
        update_step += 1

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Log
        if ep_rewards:
            avg = np.mean(all_rewards[-20:])
            console.log(f"Ep {ep}/{cfg.episodes} | Reward: {np.mean(ep_rewards):.1f} | "
                       f"Avg: {avg:.1f} | Loss: {loss:.3f}")

        # Check solved
        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= 195:
            console.print(f"[green]✓ Solved in {ep} episodes![/green]")
            break

    env.close()

    # Log final hyperparameters
    if writer is not None:
        elapsed = time.time() - start_time
        final_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)

        writer.add_hparams(
            {
                "lr": cfg.lr,
                "clip_eps": cfg.clip_eps,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "entropy_coef": cfg.entropy_coef,
                "value_coef": cfg.value_coef,
                "gamma": cfg.gamma,
                "gae_lambda": cfg.gae_lambda,
            },
            {
                "final_avg_reward": final_avg,
                "total_episodes": len(all_rewards),
                "training_time": elapsed,
            }
        )

        writer.close()
        console.print(f"\n[bold]TensorBoard logs saved to: {log_dir}[/bold]")


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
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--use-lr-schedule", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")

    a = p.parse_args()
    return Config(
        episodes=a.episodes, lr=a.lr, gamma=a.gamma, gae_lambda=a.gae_lambda,
        clip_eps=a.clip_eps, epochs=a.epochs, batch_size=a.batch_size,
        entropy_coef=a.entropy_coef, value_coef=a.value_coef,
        max_grad_norm=a.max_grad_norm, use_lr_schedule=a.use_lr_schedule,
        run_name=a.run_name, seed=a.seed, device=a.device
    )


if __name__ == "__main__":
    train(parse_args())
