#!/usr/bin/env python3
"""
Offline Reinforcement Learning Batch Training

Demonstrates offline RL training using Conservative Q-Learning (CQL) on pre-collected
datasets. This approach trains agents from logged data without environment interaction,
useful for production scenarios where online exploration is costly or risky.

Usage:
    python offline_rl_batch.py --dataset cartpole --algorithm cql --batch-size 256
    python offline_rl_batch.py --generate-dataset --env CartPole-v1 --episodes 1000
"""
from __future__ import annotations
import argparse
import pickle
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import gymnasium as gym
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

console = Console()


@dataclass
class OfflineDataset:
    """Container for offline RL dataset."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    info: Dict[str, Any]

    def __len__(self):
        return len(self.states)

    def save(self, filepath: str):
        """Save dataset to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        console.print(f"Dataset saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'OfflineDataset':
        """Load dataset from disk."""
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        console.print(f"Dataset loaded from {filepath}")
        return dataset

    def get_batch(self, batch_size: int, indices: Optional[np.ndarray] = None) -> Tuple:
        """Get a batch of transitions."""
        if indices is None:
            indices = np.random.choice(len(self), batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "size": len(self),
            "state_dim": self.states.shape[1] if len(self.states.shape) > 1 else 1,
            "action_dim": self.actions.shape[1] if len(self.actions.shape) > 1 else 1,
            "mean_reward": float(np.mean(self.rewards)),
            "std_reward": float(np.std(self.rewards)),
            "mean_episode_length": self.info.get("mean_episode_length", "unknown"),
            "num_episodes": self.info.get("num_episodes", "unknown"),
            "collection_policy": self.info.get("collection_policy", "unknown")
        }


class DatasetGenerator:
    """Generates offline datasets by running policies in environments."""

    def __init__(self, env_name: str, seed: int = 42):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.rng = np.random.RandomState(seed)

    def collect_random_data(self, num_episodes: int) -> OfflineDataset:
        """Collect data using random policy."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_lengths = []

        for episode in track(range(num_episodes), description="Collecting random data"):
            state, _ = self.env.reset()
            episode_length = 0

            while True:
                # Random action
                action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_length += 1

                if done:
                    break

            episode_lengths.append(episode_length)

        return OfflineDataset(
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=bool),
            info={
                "num_episodes": num_episodes,
                "mean_episode_length": np.mean(episode_lengths),
                "collection_policy": "random",
                "environment": self.env_name
            }
        )

    def collect_mixed_data(self, num_episodes: int) -> OfflineDataset:
        """Collect data using a mix of random and semi-optimal policies."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_lengths = []

        for episode in track(range(num_episodes), description="Collecting mixed data"):
            state, _ = self.env.reset()
            episode_length = 0

            # Use different policies for different episodes
            if episode % 3 == 0:
                policy_type = "random"
            elif episode % 3 == 1:
                policy_type = "semi_optimal"
            else:
                policy_type = "noisy_optimal"

            while True:
                # Choose action based on policy type
                if policy_type == "random":
                    action = self.env.action_space.sample()
                elif policy_type == "semi_optimal":
                    # Simple heuristic for CartPole
                    if self.env_name == "CartPole-v1" and len(state) >= 4:
                        # Try to balance pole
                        pole_angle = state[2]
                        action = 1 if pole_angle > 0 else 0
                    else:
                        action = self.env.action_space.sample()
                else:  # noisy_optimal
                    # Semi-optimal with noise
                    if self.env_name == "CartPole-v1" and len(state) >= 4:
                        pole_angle = state[2]
                        pole_velocity = state[3]
                        action = 1 if (pole_angle + 0.1 * pole_velocity) > 0 else 0
                        # Add noise
                        if self.rng.random() < 0.1:
                            action = 1 - action
                    else:
                        action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_length += 1

                if done:
                    break

            episode_lengths.append(episode_length)

        return OfflineDataset(
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=bool),
            info={
                "num_episodes": num_episodes,
                "mean_episode_length": np.mean(episode_lengths),
                "collection_policy": "mixed",
                "environment": self.env_name
            }
        )


class QNetwork(nn.Module):
    """Q-network for offline RL."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CQLAgent:
    """Conservative Q-Learning (CQL) agent for offline RL."""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 3e-4,
                 gamma: float = 0.99, cql_weight: float = 1.0, tau: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_weight = cql_weight
        self.tau = tau

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)

        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def train_batch(self, batch: Tuple[np.ndarray, ...]) -> Dict[str, float]:
        """Train on a batch of transitions using CQL."""
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Bellman error (standard DQN loss)
        bellman_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # CQL regularization term
        # This penalizes Q-values for out-of-distribution actions
        all_q_values = self.q_network(states)
        max_q_values = torch.max(all_q_values, dim=1)[0]
        action_q_values = current_q_values.squeeze()

        # Conservative penalty: maximize the gap between OOD and data actions
        cql_loss = torch.mean(max_q_values - action_q_values)

        # Total loss
        total_loss = bellman_loss + self.cql_weight * cql_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        return {
            "total_loss": total_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "mean_q_value": torch.mean(action_q_values).item()
        }

    def _soft_update(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class OfflineRLTrainer:
    """Manages offline RL training pipeline."""

    def __init__(self, dataset: OfflineDataset, algorithm: str = "cql"):
        self.dataset = dataset
        self.algorithm = algorithm

        # Initialize agent based on dataset
        state_dim = dataset.states.shape[1] if len(dataset.states.shape) > 1 else 1
        if len(dataset.actions.shape) > 1:
            action_dim = dataset.actions.shape[1]
        else:
            # Discrete actions
            action_dim = int(np.max(dataset.actions)) + 1

        if algorithm == "cql":
            self.agent = CQLAgent(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, num_epochs: int, batch_size: int = 256) -> Dict[str, List[float]]:
        """Train the agent on the offline dataset."""
        if not TORCH_AVAILABLE:
            console.print("[red]PyTorch is required for offline RL training.[/red]")
            return {}

        metrics = {"total_loss": [], "bellman_loss": [], "cql_loss": [], "mean_q_value": []}

        dataset_size = len(self.dataset)
        batches_per_epoch = dataset_size // batch_size

        console.print(f"Starting offline training with {self.algorithm.upper()}")
        console.print(f"Dataset size: {dataset_size}, Batch size: {batch_size}")
        console.print(f"Epochs: {num_epochs}, Batches per epoch: {batches_per_epoch}")

        for epoch in track(range(num_epochs), description="Training epochs"):
            epoch_metrics = {"total_loss": [], "bellman_loss": [], "cql_loss": [], "mean_q_value": []}

            # Shuffle dataset indices
            indices = np.random.permutation(dataset_size)

            for batch_idx in range(batches_per_epoch):
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch = self.dataset.get_batch(batch_size, batch_indices)

                # Train on batch
                batch_metrics = self.agent.train_batch(batch)

                # Accumulate metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)

            # Average metrics for this epoch
            for key in epoch_metrics:
                avg_value = np.mean(epoch_metrics[key])
                metrics[key].append(avg_value)

            # Log progress
            if (epoch + 1) % 10 == 0:
                console.print(f"Epoch {epoch + 1}/{num_epochs} - "
                            f"Loss: {metrics['total_loss'][-1]:.4f}, "
                            f"Q-value: {metrics['mean_q_value'][-1]:.4f}")

        return metrics

    def evaluate(self, env_name: str, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent in the environment."""
        env = gym.make(env_name)
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0

            while True:
                action = self.agent.act(state, epsilon=0.0)  # No exploration
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                total_reward += reward
                steps += 1

                if done or steps >= 1000:  # Max episode length
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths)
        }

    def save_agent(self, filepath: str):
        """Save trained agent."""
        if hasattr(self.agent, 'q_network'):
            torch.save(self.agent.q_network.state_dict(), filepath)
            console.print(f"Agent saved to {filepath}")

    def load_agent(self, filepath: str):
        """Load trained agent."""
        if hasattr(self.agent, 'q_network'):
            self.agent.q_network.load_state_dict(torch.load(filepath))
            console.print(f"Agent loaded from {filepath}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline RL Batch Training")
    parser.add_argument("--dataset", default="cartpole_mixed",
                       help="Dataset name or path")
    parser.add_argument("--algorithm", choices=["cql"], default="cql",
                       help="Offline RL algorithm")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--cql-weight", type=float, default=1.0,
                       help="CQL regularization weight")

    # Dataset generation
    parser.add_argument("--generate-dataset", action="store_true",
                       help="Generate new dataset")
    parser.add_argument("--env", default="CartPole-v1",
                       help="Environment for dataset generation")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Episodes for dataset generation")
    parser.add_argument("--policy", choices=["random", "mixed"], default="mixed",
                       help="Policy for data collection")

    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate trained agent")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Episodes for evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    console.print(f"[bold green]Offline RL Batch Training[/bold green]")

    # Generate dataset if requested
    if args.generate_dataset:
        console.print(f"Generating dataset for {args.env} with {args.policy} policy...")

        generator = DatasetGenerator(args.env)

        if args.policy == "random":
            dataset = generator.collect_random_data(args.episodes)
        else:  # mixed
            dataset = generator.collect_mixed_data(args.episodes)

        # Save dataset
        dataset_path = f"datasets/{args.env}_{args.policy}_{args.episodes}.pkl"
        os.makedirs("datasets", exist_ok=True)
        dataset.save(dataset_path)

        # Show statistics
        stats = dataset.get_statistics()
        table = Table(title="Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)
        return

    # Load dataset
    if os.path.exists(args.dataset):
        dataset_path = args.dataset
    else:
        dataset_path = f"datasets/{args.dataset}.pkl"

    if not os.path.exists(dataset_path):
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        console.print("Generate a dataset first with --generate-dataset")
        return

    dataset = OfflineDataset.load(dataset_path)

    # Show dataset info
    stats = dataset.get_statistics()
    console.print(f"Loaded dataset: {stats['size']} transitions from {stats['num_episodes']} episodes")

    # Initialize trainer
    trainer = OfflineRLTrainer(dataset, args.algorithm)

    # Train agent
    console.print(f"Training {args.algorithm.upper()} agent for {args.epochs} epochs...")
    training_metrics = trainer.train(args.epochs, args.batch_size)

    if training_metrics:
        # Show training results
        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Final Value", style="green")

        for key, values in training_metrics.items():
            if values:
                table.add_row(key.replace("_", " ").title(), f"{values[-1]:.4f}")

        console.print(table)

        # Save agent
        agent_path = f"models/{args.algorithm}_{args.dataset.replace('/', '_')}.pth"
        os.makedirs("models", exist_ok=True)
        trainer.save_agent(agent_path)

    # Evaluate if requested
    if args.evaluate:
        env_name = dataset.info.get("environment", "CartPole-v1")
        console.print(f"Evaluating agent on {env_name}...")

        eval_results = trainer.evaluate(env_name, args.eval_episodes)

        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in eval_results.items():
            table.add_row(key.replace("_", " ").title(), f"{value:.2f}")

        console.print(table)

        # Compare with dataset performance
        dataset_reward = stats["mean_reward"]
        improvement = eval_results["mean_reward"] - dataset_reward
        console.print(f"\nDataset mean reward: {dataset_reward:.2f}")
        console.print(f"Agent mean reward: {eval_results['mean_reward']:.2f}")
        console.print(f"Improvement: {improvement:+.2f} ({improvement/abs(dataset_reward)*100:+.1f}%)")


if __name__ == "__main__":
    main()