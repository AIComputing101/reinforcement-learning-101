#!/usr/bin/env python3
"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) on Pendulum-v1.

TD3 is the industry-standard off-policy algorithm for continuous control (2024-2025).
It improves upon DDPG by addressing overestimation bias and training instability.

Key features:
- Twin Q-networks (take minimum to reduce overestimation)
- Delayed policy updates (update actor less frequently than critic)
- Target policy smoothing (add noise to target actions)
- Deterministic policy with exploration noise

Example:
  python td3_pendulum.py --episodes 200 --lr 3e-4 --buffer-size 100000 --batch-size 256

Reference:
  Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods"
  https://arxiv.org/abs/1802.09477
"""
from __future__ import annotations
import argparse
import random
import sys
from collections import deque, namedtuple
from dataclasses import dataclass

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
    import torch.nn.functional as F
    import torch.optim as optim
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this example. Install torch or use Docker (CUDA/ROCm)."
    ) from e

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


@dataclass
class Config:
    episodes: int = 200
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    buffer_size: int = 100000
    batch_size: int = 256
    start_steps: int = 10000  # Random exploration steps
    policy_delay: int = 2  # Delay actor updates (TD3 key feature)
    noise_std: float = 0.1  # Exploration noise
    target_noise: float = 0.2  # Target policy smoothing noise
    noise_clip: float = 0.5  # Clip target noise
    seed: int | None = None
    device: str = "cpu"


class Actor(nn.Module):
    """Deterministic policy network."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Twin Q-networks (TD3 key feature)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    """TD3 Agent with twin critics and delayed policy updates."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, cfg: Config):
        self.cfg = cfg
        self.max_action = max_action

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(cfg.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(cfg.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)

        # Critic networks (twin Q-networks)
        self.critic = Critic(state_dim, action_dim).to(cfg.device)
        self.critic_target = Critic(state_dim, action_dim).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.replay_buffer = ReplayBuffer(cfg.buffer_size)
        self.total_updates = 0

    def select_action(self, state, add_noise=True):
        """Select action using current policy with optional exploration noise."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(0, self.cfg.noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def update(self):
        """Perform TD3 update."""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return None

        # Sample batch
        transitions = self.replay_buffer.sample(self.cfg.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.cfg.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.cfg.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.cfg.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.cfg.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.cfg.device)

        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = torch.randn_like(action_batch) * self.cfg.target_noise
            noise = noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)

            next_action = self.actor_target(next_state_batch) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-values using twin critics (take minimum)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.cfg.gamma * target_q

        # Update critics
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates (TD3 key feature)
        actor_loss = None
        if self.total_updates % self.cfg.policy_delay == 0:
            # Update actor using Q1 only
            actor_loss = -self.critic.q1_forward(state_batch, self.actor(state_batch)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.total_updates += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None
        }

    def _soft_update(self, source, target):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.cfg.tau) + param.data * self.cfg.tau
            )


def train(cfg: Config):
    # Set seeds
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action, cfg)

    console.print(f"[bold green]TD3 Training Started[/bold green]")
    console.print(f"Device: {cfg.device}")
    console.print(f"Episodes: {cfg.episodes}")
    console.print(f"Policy delay: {cfg.policy_delay}")
    console.print(f"Start steps (random): {cfg.start_steps}\n")

    episode_rewards = []
    total_steps = 0

    for episode in range(1, cfg.episodes + 1):
        state, _ = env.reset(seed=cfg.seed)
        episode_reward = 0.0
        done = False

        while not done:
            # Select action (random during warmup, policy with noise after)
            if total_steps < cfg.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # Update policy
            if total_steps >= cfg.start_steps:
                metrics = agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-20:])

        # Logging
        if episode % 10 == 0 or episode == 1:
            console.log(
                f"Episode {episode}/{cfg.episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg(20): {avg_reward:.1f} | "
                f"Steps: {total_steps} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

        # Check if solved (Pendulum is considered solved at -150 or better)
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 >= -150:
                console.print(f"[bold green]✓ Solved! Average reward over last 100 episodes: {avg_100:.1f}[/bold green]")
                break

    env.close()
    console.print(f"\n[bold]Training complete![/bold] Total steps: {total_steps}")
    console.print(f"Final average reward (last 20): {np.mean(episode_rewards[-20:]):.1f}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="TD3 on Pendulum-v1")
    p.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    p.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
    p.add_argument("--lr-critic", type=float, default=3e-4, help="Critic learning rate")
    p.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--start-steps", type=int, default=10000, help="Random exploration steps")
    p.add_argument("--policy-delay", type=int, default=2, help="Policy update delay")
    p.add_argument("--noise-std", type=float, default=0.1, help="Exploration noise")
    p.add_argument("--target-noise", type=float, default=0.2, help="Target smoothing noise")
    p.add_argument("--noise-clip", type=float, default=0.5, help="Target noise clip")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = p.parse_args()

    return Config(
        episodes=args.episodes,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        policy_delay=args.policy_delay,
        noise_std=args.noise_std,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
