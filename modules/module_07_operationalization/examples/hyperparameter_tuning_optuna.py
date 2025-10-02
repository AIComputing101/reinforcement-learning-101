#!/usr/bin/env python3
"""
Hyperparameter Tuning for RL using Optuna.

This example demonstrates automated hyperparameter optimization for PPO on CartPole
using Optuna, a state-of-the-art hyperparameter optimization framework.

Features:
- Automated search space definition
- Parallel trial execution
- Pruning of unpromising trials (MedianPruner)
- Visualization of optimization history
- Best hyperparameters export

Example:
  # Run optimization with 50 trials
  python hyperparameter_tuning_optuna.py --n-trials 50

  # With parallel execution (4 workers)
  python hyperparameter_tuning_optuna.py --n-trials 100 --n-jobs 4

  # Quick test (5 trials)
  python hyperparameter_tuning_optuna.py --n-trials 5 --n-eval-episodes 10

Dependencies:
  pip install optuna optuna-dashboard

View results:
  optuna-dashboard sqlite:///optuna_study.db

Reference:
  Optuna: https://optuna.org/
  Akiba et al. (2019) "Optuna: A Next-generation Hyperparameter Optimization Framework"
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    raise SystemExit("Requires: gymnasium, torch") from e

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as e:
    raise SystemExit("Optuna required. Install: pip install optuna") from e


@dataclass
class Config:
    lr: float
    clip_eps: float
    epochs: int
    batch_size: int
    entropy_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

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
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    episode_rewards = []
    episode_reward = 0.0
    obs, _ = env.reset()

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
            obs, _ = env.reset()

    return (np.array(states), np.array(actions), np.array(log_probs),
            np.array(values), np.array(rewards), np.array(dones), episode_rewards)


def ppo_update(model, opt, states, actions, old_log_probs, advantages, returns, cfg):
    states_t = torch.FloatTensor(states).to(cfg.device)
    actions_t = torch.LongTensor(actions).to(cfg.device)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(cfg.device)
    advantages_t = torch.FloatTensor(advantages).to(cfg.device)
    returns_t = torch.FloatTensor(returns).to(cfg.device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    num_samples = len(states)
    indices = np.arange(num_samples)

    for _ in range(cfg.epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, cfg.batch_size):
            idx = indices[start:start + cfg.batch_size]
            _, log_probs, entropy, value = model.get_action_value(states_t[idx], actions_t[idx])

            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((value.squeeze() - returns_t[idx]) ** 2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


def train_ppo(cfg: Config, n_episodes: int = 100, seed: int = 42, trial=None):
    """Train PPO and return mean reward (for Optuna optimization)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden_dim=64).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    all_rewards = []

    for ep in range(1, n_episodes + 1):
        states, actions, log_probs, values, rewards, dones, ep_rewards = collect_batch(env, model, cfg)

        if ep_rewards:
            all_rewards.extend(ep_rewards)

        advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        ppo_update(model, optimizer, states, actions, log_probs, advantages, returns, cfg)

        # Report intermediate value for pruning
        if trial is not None and ep % 10 == 0 and len(all_rewards) >= 20:
            intermediate_value = np.mean(all_rewards[-20:])
            trial.report(intermediate_value, ep)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

    env.close()

    # Return final performance
    final_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
    return final_reward


def objective(trial, n_train_episodes: int, n_eval_episodes: int):
    """Optuna objective function."""

    # Define hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
    epochs = trial.suggest_int("epochs", 3, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.1)
    value_coef = trial.suggest_float("value_coef", 0.1, 1.0)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    cfg = Config(
        lr=lr,
        clip_eps=clip_eps,
        epochs=epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Train and evaluate
    mean_reward = train_ppo(cfg, n_episodes=n_train_episodes, trial=trial)

    return mean_reward


def optimize_hyperparameters(args):
    """Run Optuna optimization."""

    console.print("[bold green]Hyperparameter Optimization with Optuna[/bold green]")
    console.print(f"Trials: {args.n_trials}")
    console.print(f"Parallel jobs: {args.n_jobs}")
    console.print(f"Training episodes per trial: {args.n_train_episodes}")
    console.print(f"Database: {args.storage}\n")

    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name="ppo_cartpole_tuning",
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, args.n_train_episodes, args.n_eval_episodes),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Results
    console.print("\n[bold]Optimization Complete![/bold]\n")

    # Best trial
    console.print(f"[green]Best trial:[/green]")
    console.print(f"  Value (mean reward): {study.best_trial.value:.2f}")
    console.print(f"  Params:")
    for key, value in study.best_trial.params.items():
        console.print(f"    {key}: {value}")

    # Top 5 trials table
    console.print("\n[bold]Top 5 Trials:[/bold]")
    table = Table(show_header=True)
    table.add_column("Trial", style="cyan")
    table.add_column("Reward", style="green")
    table.add_column("LR", style="yellow")
    table.add_column("Clip ε", style="yellow")
    table.add_column("Epochs", style="yellow")
    table.add_column("Batch Size", style="yellow")

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]

    for trial in sorted_trials:
        if trial.value is not None:
            table.add_row(
                str(trial.number),
                f"{trial.value:.2f}",
                f"{trial.params.get('lr', 'N/A'):.2e}",
                f"{trial.params.get('clip_eps', 'N/A'):.3f}",
                str(trial.params.get('epochs', 'N/A')),
                str(trial.params.get('batch_size', 'N/A')),
            )

    console.print(table)

    # Save best config
    best_config = study.best_trial.params
    console.print(f"\n[bold]Best configuration saved to: best_hyperparameters.txt[/bold]")

    with open("best_hyperparameters.txt", "w") as f:
        f.write("# Best hyperparameters from Optuna optimization\n")
        f.write(f"# Mean reward: {study.best_trial.value:.2f}\n\n")
        for key, value in best_config.items():
            f.write(f"{key} = {value}\n")

    # Optuna dashboard instructions
    if args.storage.startswith("sqlite"):
        console.print(f"\n[cyan]View results with Optuna Dashboard:[/cyan]")
        console.print(f"  optuna-dashboard {args.storage}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RL with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--n-train-episodes", type=int, default=100, help="Episodes per trial")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db", help="Optuna storage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    optimize_hyperparameters(args)


if __name__ == "__main__":
    main()
