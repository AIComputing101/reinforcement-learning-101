#!/usr/bin/env python3
"""
TRPO (Trust Region Policy Optimization) on CartPole-v1.

TRPO is the theoretical foundation for PPO, using a trust region constraint
to ensure stable policy updates. While PPO uses clipping for simplicity,
TRPO uses the exact KL divergence constraint.

Key features:
- Trust region constraint (KL divergence)
- Conjugate gradient for efficient Hessian-vector products
- Line search for step size
- GAE for advantage estimation

Example:
  python trpo_cartpole.py --episodes 300 --max-kl 0.01 --gamma 0.99 --gae-lambda 0.95

Reference:
  Schulman et al. (2015) "Trust Region Policy Optimization"
  https://arxiv.org/abs/1502.05477

Note: This is a simplified educational implementation. Production implementations
      (e.g., Stable-Baselines3) include additional optimizations.
"""
from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

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


@dataclass
class Config:
    episodes: int = 300
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_kl: float = 0.01  # Maximum KL divergence
    damping: float = 0.1  # Conjugate gradient damping
    cg_iters: int = 10  # Conjugate gradient iterations
    line_search_steps: int = 10  # Line search backtracking steps
    value_lr: float = 1e-3  # Value function learning rate
    value_iters: int = 5  # Value function update iterations
    seed: int | None = None
    device: str = "cpu"


class Policy(nn.Module):
    """Policy network for discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x):
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)

    def act(self, x):
        dist = self.get_distribution(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ValueFunction(nn.Module):
    """Value function network."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def collect_rollout(env, policy, value_fn, cfg: Config, max_steps: int = 2048):
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
            action, log_prob = policy.act(state_t)
            value = value_fn(state_t)

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


def flat_grad(grads):
    """Flatten gradients into a single vector."""
    return torch.cat([g.reshape(-1) for g in grads])


def conjugate_gradient(Ax_func, b, cg_iters=10, residual_tol=1e-10):
    """
    Conjugate gradient algorithm for solving Ax = b.

    Args:
        Ax_func: Function that computes matrix-vector product Ax
        b: Target vector
        cg_iters: Maximum iterations
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        Ap = Ax_func(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap

        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break

        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

    return x


def hessian_vector_product(kl, policy, vec, damping):
    """
    Compute Hessian-vector product efficiently using double backprop.

    Fisher Information Matrix approximation:
    H = ∇²_θ KL(π_old || π_new)
    """
    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad_kl = flat_grad(grads)

    # Compute gradient of (gradient · vector)
    grad_v = torch.sum(flat_grad_kl * vec)
    hvp = flat_grad(torch.autograd.grad(grad_v, policy.parameters(), retain_graph=True))

    return hvp + damping * vec


def set_flat_params(model, flat_params):
    """Set model parameters from a flat vector."""
    offset = 0
    for param in model.parameters():
        param_shape = param.shape
        param_size = param.numel()
        param.data = flat_params[offset:offset + param_size].reshape(param_shape)
        offset += param_size


def get_flat_params(model):
    """Get model parameters as a flat vector."""
    return torch.cat([param.reshape(-1) for param in model.parameters()])


def trpo_update(policy, value_fn, value_optimizer, states, actions, advantages, returns, old_log_probs, cfg: Config):
    """Perform TRPO policy update with trust region constraint."""

    states_t = torch.as_tensor(states, dtype=torch.float32, device=cfg.device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=cfg.device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=cfg.device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=cfg.device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=cfg.device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # Update value function (multiple iterations)
    for _ in range(cfg.value_iters):
        values = value_fn(states_t)
        value_loss = ((values - returns_t) ** 2).mean()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

    # Compute policy gradient
    dist = policy.get_distribution(states_t)
    log_probs = dist.log_prob(actions_t)

    # Surrogate loss (policy gradient objective)
    ratio = torch.exp(log_probs - old_log_probs_t)
    surr_loss = -(ratio * advantages_t).mean()

    # Compute policy gradient
    policy.zero_grad()
    surr_loss.backward(retain_graph=True)
    policy_grad = flat_grad([param.grad for param in policy.parameters()])

    # Compute KL divergence with old policy
    with torch.no_grad():
        old_dist = torch.distributions.Categorical(logits=policy(states_t).detach())

    new_dist = policy.get_distribution(states_t)
    kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

    # Define Hessian-vector product function
    def Ax(v):
        return hessian_vector_product(kl, policy, v, cfg.damping)

    # Solve natural gradient: H^(-1) * g using conjugate gradient
    step_direction = conjugate_gradient(Ax, policy_grad, cfg.cg_iters)

    # Compute step size using quadratic approximation
    shs = 0.5 * torch.dot(step_direction, Ax(step_direction))
    step_size = torch.sqrt(2 * cfg.max_kl / (shs + 1e-8))
    full_step = step_size * step_direction

    # Line search for valid step (backtracking)
    old_params = get_flat_params(policy)
    expected_improve = torch.dot(policy_grad, full_step)

    for step_frac in [1.0, 0.5, 0.25, 0.1]:
        new_params = old_params + step_frac * full_step
        set_flat_params(policy, new_params)

        with torch.no_grad():
            new_dist = policy.get_distribution(states_t)
            new_log_probs = new_dist.log_prob(actions_t)
            new_ratio = torch.exp(new_log_probs - old_log_probs_t)
            new_surr_loss = -(new_ratio * advantages_t).mean()

            # Check KL constraint
            new_kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

            improve = new_surr_loss - surr_loss

            if new_kl <= cfg.max_kl and improve < 0:
                # Accept step
                break
    else:
        # No valid step found, revert to old parameters
        set_flat_params(policy, old_params)

    return {
        "surr_loss": surr_loss.item(),
        "value_loss": value_loss.item(),
        "kl": new_kl.item() if 'new_kl' in locals() else kl.item()
    }


def train(cfg: Config):
    # Set seeds
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = Policy(obs_dim, action_dim).to(cfg.device)
    value_fn = ValueFunction(obs_dim).to(cfg.device)
    value_optimizer = optim.Adam(value_fn.parameters(), lr=cfg.value_lr)

    console.print(f"[bold green]TRPO Training Started[/bold green]")
    console.print(f"Device: {cfg.device}")
    console.print(f"Episodes: {cfg.episodes}")
    console.print(f"Max KL divergence: {cfg.max_kl}")
    console.print(f"CG iterations: {cfg.cg_iters}\n")

    all_episode_rewards = []

    for episode in range(1, cfg.episodes + 1):
        # Collect rollout
        states, actions, log_probs, values, rewards, dones, episode_rewards = collect_rollout(
            env, policy, value_fn, cfg, max_steps=2048
        )

        if episode_rewards:
            all_episode_rewards.extend(episode_rewards)

        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)

        # TRPO update
        metrics = trpo_update(policy, value_fn, value_optimizer, states, actions, advantages, returns, log_probs, cfg)

        # Logging
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            avg_last_20 = np.mean(all_episode_rewards[-20:]) if len(all_episode_rewards) >= 20 else np.mean(all_episode_rewards)

            console.log(
                f"Episode {episode}/{cfg.episodes} | "
                f"Reward: {avg_reward:.1f} | "
                f"Avg(20): {avg_last_20:.1f} | "
                f"V_Loss: {metrics['value_loss']:.3f} | "
                f"KL: {metrics['kl']:.4f}"
            )

        # Check if solved (CartPole is solved at 195+)
        if len(all_episode_rewards) >= 100:
            avg_100 = np.mean(all_episode_rewards[-100:])
            if avg_100 >= 195:
                console.print(f"[bold green]✓ Solved! Average reward over last 100 episodes: {avg_100:.1f}[/bold green]")
                break

    env.close()
    console.print(f"\n[bold]Training complete![/bold]")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="TRPO on CartPole-v1")
    p.add_argument("--episodes", type=int, default=300, help="Number of episodes")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    p.add_argument("--max-kl", type=float, default=0.01, help="Maximum KL divergence")
    p.add_argument("--damping", type=float, default=0.1, help="CG damping coefficient")
    p.add_argument("--cg-iters", type=int, default=10, help="Conjugate gradient iterations")
    p.add_argument("--line-search-steps", type=int, default=10, help="Line search steps")
    p.add_argument("--value-lr", type=float, default=1e-3, help="Value function learning rate")
    p.add_argument("--value-iters", type=int, default=5, help="Value function update iterations")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = p.parse_args()

    return Config(
        episodes=args.episodes,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_kl=args.max_kl,
        damping=args.damping,
        cg_iters=args.cg_iters,
        line_search_steps=args.line_search_steps,
        value_lr=args.value_lr,
        value_iters=args.value_iters,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
