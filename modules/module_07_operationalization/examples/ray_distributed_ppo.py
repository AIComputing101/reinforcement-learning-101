#!/usr/bin/env python3
"""
Distributed PPO Training with Ray RLlib.

This example demonstrates production-scale distributed RL training using Ray RLlib:
- Distributed rollout workers for parallel data collection
- Multi-GPU training support
- Automatic hyperparameter tuning with Ray Tune
- TensorBoard integration
- Checkpoint management

Features:
- Scale from 1 GPU to thousands with same code
- Production-ready framework used by industry (Uber, Ant Financial, etc.)
- Built-in algorithms and optimizations

Example:
  # Single machine, multi-core CPU
  python ray_distributed_ppo.py --num-workers 4

  # Multi-GPU training
  python ray_distributed_ppo.py --num-workers 8 --num-gpus 2

  # With Ray Tune for hyperparameter search
  python ray_distributed_ppo.py --tune --num-samples 4

  # Resume from checkpoint
  python ray_distributed_ppo.py --restore path/to/checkpoint

Dependencies:
  pip install "ray[rllib]" "ray[tune]" tensorboard

Note:
  Ray RLlib is actively maintained and production-ready.
  For bleeding-edge features, see: https://docs.ray.io/en/latest/rllib/

Reference:
  Liang et al. (2018) "RLlib: Abstractions for Distributed Reinforcement Learning"
  https://docs.ray.io/en/latest/rllib/index.html
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.logger import pretty_print
except ImportError as e:
    console.print("[red]Ray RLlib not installed.[/red]")
    console.print("\n[yellow]Install with:[/yellow]")
    console.print('  pip install "ray[rllib]" "ray[tune]"')
    console.print("\n[yellow]For GPU support:[/yellow]")
    console.print('  pip install "ray[rllib]" "ray[tune]" torch')
    raise SystemExit(1) from e


def train_single(args):
    """Train PPO with Ray RLlib (single training run)."""

    console.print("[bold green]Distributed PPO Training with Ray RLlib[/bold green]")
    console.print(f"Workers: {args.num_workers}")
    console.print(f"GPUs: {args.num_gpus}")
    console.print(f"Environment: {args.env}\n")

    # Initialize Ray
    ray.init(
        num_cpus=args.num_workers + 1,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True,
    )

    # Configure PPO
    config = (
        PPOConfig()
        .environment(env=args.env)
        .framework("torch")
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=args.num_gpus,
            num_cpus_for_local_worker=1,
        )
        .training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            clip_param=args.clip_param,
            vf_clip_param=args.vf_clip_param,
            entropy_coeff=args.entropy_coeff,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
            },
        )
        .evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_duration=10,
            evaluation_num_workers=1,
        )
        .debugging(
            log_level="INFO",
        )
    )

    # Build algorithm
    algo = config.build()

    # Restore from checkpoint if specified
    if args.restore:
        console.print(f"[cyan]Restoring from checkpoint: {args.restore}[/cyan]")
        algo.restore(args.restore)

    # Training loop
    console.print("\n[bold]Starting Training...[/bold]\n")

    best_reward = float('-inf')
    checkpoint_dir = Path("ray_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        result = algo.train()

        # Extract metrics
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_len_mean = result.get("episode_len_mean", 0)
        timesteps_total = result.get("timesteps_total", 0)

        # Log progress
        if iteration % args.log_interval == 0:
            console.log(
                f"Iteration {iteration}/{args.iterations} | "
                f"Reward: {episode_reward_mean:.2f} | "
                f"Ep Len: {episode_len_mean:.1f} | "
                f"Timesteps: {timesteps_total}"
            )

        # Save checkpoint if improved
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_path = algo.save(checkpoint_dir)
            console.print(f"[green]✓ New best! Saved checkpoint to: {checkpoint_path.checkpoint.path}[/green]")

        # Check if solved (CartPole-v1 is solved at 195+)
        if episode_reward_mean >= 195:
            console.print(f"\n[bold green]✓ Environment solved at iteration {iteration}![/bold green]")
            break

    # Final stats
    console.print(f"\n[bold]Training Complete![/bold]")
    console.print(f"Best reward: {best_reward:.2f}")
    console.print(f"Total timesteps: {result['timesteps_total']}")

    algo.stop()
    ray.shutdown()


def train_with_tune(args):
    """Hyperparameter tuning with Ray Tune."""

    console.print("[bold green]Hyperparameter Tuning with Ray Tune[/bold green]")
    console.print(f"Samples: {args.num_samples}")
    console.print(f"Workers per trial: {args.num_workers}\n")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Define search space
    config = {
        "env": args.env,
        "framework": "torch",
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus / args.num_samples if args.num_gpus > 0 else 0,
        "train_batch_size": tune.choice([2000, 4000, 8000]),
        "sgd_minibatch_size": tune.choice([64, 128, 256]),
        "num_sgd_iter": tune.choice([5, 10, 20]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.9, 0.999),
        "clip_param": tune.uniform(0.1, 0.3),
        "entropy_coeff": tune.uniform(0.0, 0.1),
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh",
        },
    }

    # Run hyperparameter search
    console.print("\n[bold]Starting hyperparameter search...[/bold]\n")

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"episode_reward_mean": 195, "timesteps_total": 100000},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                num_to_keep=3,
            ),
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            metric="episode_reward_mean",
            mode="max",
        ),
    )

    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    console.print("\n[bold green]Tuning Complete![/bold green]\n")
    console.print(f"Best reward: {best_result.metrics['episode_reward_mean']:.2f}")
    console.print(f"\n[bold]Best hyperparameters:[/bold]")

    # Display results table
    table = Table(show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")

    for key, value in best_result.config.items():
        if key not in ["env", "framework", "num_workers", "num_gpus", "model"]:
            table.add_row(key, str(value))

    console.print(table)

    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Distributed PPO with Ray RLlib")

    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")

    # Parallelization
    parser.add_argument("--num-workers", type=int, default=4, help="Number of rollout workers")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs")

    # Training
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--train-batch-size", type=int, default=4000, help="Train batch size")
    parser.add_argument("--sgd-minibatch-size", type=int, default=128, help="SGD minibatch size")
    parser.add_argument("--num-sgd-iter", type=int, default=10, help="SGD iterations")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--vf-clip-param", type=float, default=10.0, help="Value function clip")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient")

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluation interval")

    # Logging
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval")

    # Checkpointing
    parser.add_argument("--restore", type=str, default=None, help="Restore from checkpoint")

    # Ray Tune
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--num-samples", type=int, default=4, help="Tune: number of samples")

    args = parser.parse_args()

    if args.tune:
        train_with_tune(args)
    else:
        train_single(args)


if __name__ == "__main__":
    main()
