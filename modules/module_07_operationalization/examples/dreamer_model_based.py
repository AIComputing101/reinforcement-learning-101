#!/usr/bin/env python3
"""
Simplified DreamerV3-style Model-Based Reinforcement Learning.

This is an educational implementation inspired by DreamerV3, demonstrating key concepts
of model-based RL: world models, latent imagination, and policy learning in dreams.

Key Concepts:
- World Model: Learn to predict future states and rewards
- Latent Representations: Compress observations into compact latent states
- Imagination: Train policy in imagined trajectories (no environment interaction)
- Actor-Critic: Policy and value learning in latent space

Note: This is a simplified educational version. Production DreamerV3 is significantly
more complex with categorical representations, symlog predictions, and advanced
normalization schemes.

Example:
  # Train world model and policy (CartPole)
  python dreamer_model_based.py --env CartPole-v1 --episodes 200

  # Pendulum (continuous control)
  python dreamer_model_based.py --env Pendulum-v1 --episodes 100

  # With more imagination steps
  python dreamer_model_based.py --env CartPole-v1 --imagine-horizon 20

Reference:
  Hafner et al. (2023) "Mastering Diverse Domains through World Models"
  https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as e:
    raise SystemExit("Requires: gymnasium, torch") from e


@dataclass
class Config:
    # World model
    latent_dim: int = 32
    hidden_dim: int = 128
    imagine_horizon: int = 15  # Imagination rollout length

    # Training
    batch_size: int = 50
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95

    # Data collection
    buffer_size: int = 10000
    collect_interval: int = 100  # Steps between collections

    # Hyperparameters
    episodes: int = 200
    max_steps: int = 1000

    seed: int = 42
    device: str = "cpu"


class WorldModel(nn.Module):
    """
    World model that predicts next latent state and reward.

    Components:
    - Encoder: obs → latent
    - Dynamics: latent + action → next_latent
    - Reward: latent → reward
    - Decoder: latent → obs (for reconstruction)
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int, discrete_actions: bool):
        super().__init__()
        self.discrete_actions = discrete_actions
        self.latent_dim = latent_dim

        # Encoder: observation → latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Dynamics model: (latent, action) → next_latent
        action_input = action_dim if discrete_actions else action_dim
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Reward predictor: latent → reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Decoder: latent → observation (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.encoder(obs)

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Imagine one step: predict next latent and reward.

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action taken [batch, action_dim] or [batch] for discrete

        Returns:
            next_latent: Predicted next latent state
            reward: Predicted reward
        """
        if self.discrete_actions and action.dim() == 1:
            # One-hot encode discrete actions
            action_input = F.one_hot(action.long(), num_classes=self.dynamics[0].in_features - self.latent_dim).float()
        else:
            action_input = action

        # Concatenate latent and action
        x = torch.cat([latent, action_input], dim=-1)

        # Predict next latent
        next_latent = self.dynamics(x)

        # Predict reward
        reward = self.reward_predictor(next_latent).squeeze(-1)

        return next_latent, reward

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation (for reconstruction)."""
        return self.decoder(latent)


class ActorCritic(nn.Module):
    """Actor-Critic that operates in latent space."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, discrete_actions: bool):
        super().__init__()
        self.discrete_actions = discrete_actions

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy)
        if discrete_actions:
            self.actor = nn.Linear(hidden_dim, action_dim)
        else:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_dist: Action distribution
            value: State value
        """
        h = self.shared(latent)

        # Value
        value = self.critic(h).squeeze(-1)

        # Policy
        if self.discrete_actions:
            logits = self.actor(h)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = self.actor_mean(h)
            std = torch.exp(self.actor_logstd)
            action_dist = torch.distributions.Normal(mean, std)

        return action_dist, value


class ReplayBuffer:
    """Simple replay buffer for world model training."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


def compute_gae(rewards, values, gamma, lambda_gae):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_gae * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return np.array(advantages), np.array(returns)


def train_world_model(world_model, buffer, optimizer, cfg):
    """Train world model on replay buffer data."""
    if len(buffer) < cfg.batch_size:
        return {}

    obs, actions, rewards, next_obs, dones = buffer.sample(cfg.batch_size)

    obs_t = torch.FloatTensor(obs).to(cfg.device)
    actions_t = torch.FloatTensor(actions).to(cfg.device) if len(actions.shape) > 1 else torch.LongTensor(actions).to(cfg.device)
    rewards_t = torch.FloatTensor(rewards).to(cfg.device)
    next_obs_t = torch.FloatTensor(next_obs).to(cfg.device)

    # Encode observations
    latent = world_model.encode(obs_t)
    next_latent_true = world_model.encode(next_obs_t)

    # Predict next latent and reward
    next_latent_pred, reward_pred = world_model.imagine_step(latent, actions_t)

    # Losses
    # 1. Dynamics loss (predict next latent)
    dynamics_loss = F.mse_loss(next_latent_pred, next_latent_true.detach())

    # 2. Reward prediction loss
    reward_loss = F.mse_loss(reward_pred, rewards_t)

    # 3. Reconstruction loss (decode latent back to observation)
    obs_recon = world_model.decode(latent)
    recon_loss = F.mse_loss(obs_recon, obs_t)

    # Total loss
    loss = dynamics_loss + reward_loss + 0.1 * recon_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
    optimizer.step()

    return {
        "world_model/dynamics_loss": dynamics_loss.item(),
        "world_model/reward_loss": reward_loss.item(),
        "world_model/recon_loss": recon_loss.item(),
        "world_model/total_loss": loss.item(),
    }


def imagine_rollouts(world_model, actor_critic, start_obs, cfg, num_trajectories=16):
    """
    Generate imagined trajectories using the world model.

    This is the core of Dreamer: we simulate future trajectories in latent space
    without interacting with the real environment.
    """
    # Encode starting observations
    latent = world_model.encode(start_obs)

    # Storage for imagined trajectories
    latents = [latent]
    actions_list = []
    rewards_list = []
    values_list = []
    log_probs_list = []

    # Imagine forward
    for _ in range(cfg.imagine_horizon):
        # Sample action from policy
        action_dist, value = actor_critic(latent)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(-1) if not world_model.discrete_actions else action_dist.log_prob(action)

        # Imagine next state
        next_latent, reward = world_model.imagine_step(latent, action)

        # Store
        actions_list.append(action)
        rewards_list.append(reward)
        values_list.append(value)
        log_probs_list.append(log_prob)
        latents.append(next_latent)

        # Continue from next latent
        latent = next_latent

    return {
        "latents": latents,
        "actions": actions_list,
        "rewards": rewards_list,
        "values": values_list,
        "log_probs": log_probs_list,
    }


def train_actor_critic(actor_critic, imagined_data, optimizer, cfg):
    """Train actor-critic on imagined trajectories."""
    rewards = torch.stack(imagined_data["rewards"]).detach().cpu().numpy()
    values = torch.stack(imagined_data["values"]).detach().cpu().numpy()
    log_probs = torch.stack(imagined_data["log_probs"])

    # Compute advantages using GAE
    advantages, returns = compute_gae(rewards, values, cfg.gamma, cfg.lambda_gae)

    advantages_t = torch.FloatTensor(advantages).to(cfg.device)
    returns_t = torch.FloatTensor(returns).to(cfg.device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # Policy loss (REINFORCE with advantage)
    policy_loss = -(log_probs * advantages_t).mean()

    # Value loss
    values_t = torch.stack(imagined_data["values"])
    value_loss = F.mse_loss(values_t, returns_t)

    # Entropy bonus (for exploration)
    entropy_loss = -log_probs.mean() * 0.01

    # Total loss
    loss = policy_loss + 0.5 * value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
    optimizer.step()

    return {
        "actor_critic/policy_loss": policy_loss.item(),
        "actor_critic/value_loss": value_loss.item(),
        "actor_critic/total_loss": loss.item(),
    }


def evaluate_policy(env, world_model, actor_critic, cfg, num_episodes=5):
    """Evaluate policy in real environment."""
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Encode observation
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg.device)
            latent = world_model.encode(obs_t)

            # Get action from policy
            with torch.no_grad():
                action_dist, _ = actor_critic(latent)
                action = action_dist.mean if not world_model.discrete_actions else action_dist.probs.argmax(dim=-1)

            # Step in real environment
            if world_model.discrete_actions:
                action_np = action.cpu().numpy()[0]
            else:
                action_np = action.cpu().numpy()[0]
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    return np.mean(rewards)


def main():
    parser = argparse.ArgumentParser(description="Dreamer-style Model-Based RL")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    cfg = Config(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        imagine_horizon=args.imagine_horizon,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )

    # Set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    discrete_actions = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete_actions else env.action_space.shape[0]

    console.print(f"[bold green]Dreamer-style Model-Based RL[/bold green]")
    console.print(f"Environment: {args.env}")
    console.print(f"Observation dim: {obs_dim}")
    console.print(f"Action space: {'Discrete' if discrete_actions else 'Continuous'} ({action_dim})")
    console.print(f"Latent dim: {cfg.latent_dim}")
    console.print(f"Imagination horizon: {cfg.imagine_horizon}\n")

    # Models
    world_model = WorldModel(obs_dim, action_dim, cfg.latent_dim, cfg.hidden_dim, discrete_actions).to(cfg.device)
    actor_critic = ActorCritic(cfg.latent_dim, action_dim, cfg.hidden_dim, discrete_actions).to(cfg.device)

    # Optimizers
    world_optimizer = optim.Adam(world_model.parameters(), lr=cfg.learning_rate)
    actor_optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.learning_rate)

    # Replay buffer
    buffer = ReplayBuffer(cfg.buffer_size)

    # Training loop
    total_steps = 0
    best_reward = float('-inf')

    for episode in track(range(1, cfg.episodes + 1), description="Training"):
        obs, _ = env.reset(seed=cfg.seed + episode)
        episode_reward = 0
        done = False

        while not done and total_steps < cfg.max_steps * cfg.episodes:
            # Collect data using current policy
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg.device)
            latent = world_model.encode(obs_t)

            with torch.no_grad():
                action_dist, _ = actor_critic(latent)
                action = action_dist.sample()

            # Execute in environment
            if discrete_actions:
                action_np = action.cpu().numpy()[0]
            else:
                action_np = action.cpu().numpy()[0]
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Store in buffer
            buffer.push(obs, action_np, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            total_steps += 1

            # Train world model and actor-critic periodically
            if total_steps % cfg.collect_interval == 0 and len(buffer) >= cfg.batch_size:
                # Train world model
                world_model_metrics = train_world_model(world_model, buffer, world_optimizer, cfg)

                # Generate imagined trajectories
                start_obs_batch = torch.FloatTensor(buffer.sample(16)[0]).to(cfg.device)
                imagined_data = imagine_rollouts(world_model, actor_critic, start_obs_batch, cfg)

                # Train actor-critic on imagined data
                actor_critic_metrics = train_actor_critic(actor_critic, imagined_data, actor_optimizer, cfg)

        # Evaluation
        if episode % 10 == 0:
            avg_reward = evaluate_policy(env, world_model, actor_critic, cfg, num_episodes=5)
            console.log(f"Episode {episode}/{cfg.episodes} | Reward: {avg_reward:.2f} | Buffer: {len(buffer)}")

            if avg_reward > best_reward:
                best_reward = avg_reward

    # Final evaluation
    console.print(f"\n[bold]Training Complete![/bold]")
    final_reward = evaluate_policy(env, world_model, actor_critic, cfg, num_episodes=20)
    console.print(f"Final evaluation reward: {final_reward:.2f}")
    console.print(f"Best reward during training: {best_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
