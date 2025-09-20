#!/usr/bin/env python3
"""
Curiosity-Driven Learning for Super Mario Bros

Implements Intrinsic Curiosity Module (ICM) for exploration in Super Mario Bros.
This example demonstrates how curiosity-driven learning can help agents explore
complex environments with sparse rewards by generating intrinsic rewards based
on prediction errors.

Usage:
    python curiosity_supermario.py --episodes 100 --curiosity-weight 0.1 --lr 3e-4
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table

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
class Args:
    episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    curiosity_weight: float
    forward_loss_weight: float
    inverse_loss_weight: float
    memory_size: int
    batch_size: int
    target_update: int
    seed: int


class MarioEnvironment:
    """
    Simplified Mario-like environment for educational purposes.

    Since gym-super-mario-bros requires complex setup, we'll create a simplified
    2D platformer that demonstrates the same curiosity-driven learning principles.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Environment parameters
        self.grid_width = 20
        self.grid_height = 10
        self.max_steps = 500

        # State: [player_x, player_y, previous_x, previous_y, level_progress, enemies_visible]
        self.state_size = 6
        self.action_size = 4  # right, left, jump, stay

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Player starts at bottom left
        self.player_x = 1
        self.player_y = 1
        self.previous_x = 1
        self.previous_y = 1

        # Generate a random level layout
        self.level = self._generate_level()

        # Reset counters
        self.step_count = 0
        self.level_progress = 0
        self.enemies_defeated = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action and return next state, reward, done, info."""
        self.previous_x = self.player_x
        self.previous_y = self.player_y

        # Action mapping: 0=right, 1=left, 2=jump, 3=stay
        if action == 0:  # Right
            self.player_x = min(self.player_x + 1, self.grid_width - 1)
        elif action == 1:  # Left
            self.player_x = max(self.player_x - 1, 0)
        elif action == 2:  # Jump
            if self.player_y > 0:  # Can only jump if on ground or platform
                self.player_y = min(self.player_y + 1, self.grid_height - 1)
        # action == 3 is stay (no movement)

        # Apply gravity
        if self.player_y > 1 and not self._is_on_platform():
            self.player_y = max(self.player_y - 1, 1)

        # Calculate reward
        reward = self._calculate_reward()

        # Update progress
        self.level_progress = max(self.level_progress, self.player_x)

        # Check if episode is done
        self.step_count += 1
        done = (self.step_count >= self.max_steps or
                self.player_x >= self.grid_width - 1 or  # Reached end
                self._check_collision())  # Hit enemy/obstacle

        info = {
            "level_progress": self.level_progress,
            "enemies_defeated": self.enemies_defeated,
            "position": (self.player_x, self.player_y),
            "level_completed": self.player_x >= self.grid_width - 1
        }

        return self._get_state(), reward, done, info

    def _generate_level(self) -> np.ndarray:
        """Generate a random level layout."""
        level = np.zeros((self.grid_height, self.grid_width), dtype=int)

        # Add platforms randomly
        for x in range(2, self.grid_width - 2):
            if self.rng.random() < 0.3:  # 30% chance of platform
                y = self.rng.randint(2, self.grid_height - 2)
                level[y, x] = 1  # Platform

        # Add enemies randomly
        for x in range(2, self.grid_width - 2):
            if self.rng.random() < 0.2:  # 20% chance of enemy
                y = 1  # Enemies on ground
                level[y, x] = 2  # Enemy

        return level

    def _is_on_platform(self) -> bool:
        """Check if player is standing on a platform."""
        if self.player_y <= 1:
            return True  # Ground level
        below_y = self.player_y - 1
        return (below_y >= 0 and
                self.level[below_y, self.player_x] == 1)

    def _check_collision(self) -> bool:
        """Check if player collided with enemy."""
        return self.level[self.player_y, self.player_x] == 2

    def _calculate_reward(self) -> float:
        """Calculate sparse extrinsic reward."""
        reward = 0.0

        # Small reward for forward progress
        if self.player_x > self.previous_x:
            reward += 1.0

        # Large reward for reaching the end
        if self.player_x >= self.grid_width - 1:
            reward += 100.0

        # Penalty for going backward
        if self.player_x < self.previous_x:
            reward -= 0.5

        # Penalty for collision
        if self._check_collision():
            reward -= 10.0

        return reward

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Count visible enemies in 3x3 area around player
        enemies_visible = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = self.player_x + dx, self.player_y + dy
                if (0 <= x < self.grid_width and 0 <= y < self.grid_height and
                    self.level[y, x] == 2):
                    enemies_visible += 1

        state = np.array([
            self.player_x / self.grid_width,  # Normalized position
            self.player_y / self.grid_height,
            self.previous_x / self.grid_width,  # Previous position
            self.previous_y / self.grid_height,
            self.level_progress / self.grid_width,  # Progress
            enemies_visible / 9.0  # Normalized enemy count
        ], dtype=np.float32)

        return state


class InverseDynamicsModel(nn.Module):
    """Inverse model that predicts action given current and next state."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ForwardDynamicsModel(nn.Module):
    """Forward model that predicts next state features given current state and action."""

    def __init__(self, state_dim: int, action_dim: int, feature_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, state, action):
        state_features = self.feature_encoder(state)

        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=4).float()

        # Predict next state features
        x = torch.cat([state_features, action_onehot], dim=1)
        predicted_next_features = self.forward_model(x)

        return state_features, predicted_next_features

    def encode_state(self, state):
        """Encode state to features."""
        return self.feature_encoder(state)


class DQNWithCuriosity(nn.Module):
    """DQN network for action-value estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CuriosityAgent:
    """DQN agent with Intrinsic Curiosity Module (ICM)."""

    def __init__(self, state_dim: int, action_dim: int, args: Args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.curiosity_weight = args.curiosity_weight
        self.forward_loss_weight = args.forward_loss_weight
        self.inverse_loss_weight = args.inverse_loss_weight
        self.batch_size = args.batch_size
        self.target_update = args.target_update

        # Networks
        self.q_network = DQNWithCuriosity(state_dim, action_dim).to(self.device)
        self.target_network = DQNWithCuriosity(state_dim, action_dim).to(self.device)
        self.inverse_model = InverseDynamicsModel(state_dim, action_dim).to(self.device)
        self.forward_model = ForwardDynamicsModel(state_dim, action_dim).to(self.device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.icm_optimizer = optim.Adam(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()),
            lr=args.learning_rate
        )

        # Experience replay
        self.memory = []
        self.memory_size = args.memory_size

        # Update target network
        self.update_target_network()
        self.update_count = 0

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def calculate_intrinsic_reward(self, state, action, next_state):
        """Calculate intrinsic reward using forward model prediction error."""
        state_tensor = torch.FloatTensor([state]).to(self.device)
        next_state_tensor = torch.FloatTensor([next_state]).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)

        with torch.no_grad():
            # Get current and predicted next state features
            current_features, predicted_features = self.forward_model(state_tensor, action_tensor)
            actual_next_features = self.forward_model.encode_state(next_state_tensor)

            # Calculate prediction error (intrinsic reward)
            prediction_error = F.mse_loss(predicted_features, actual_next_features, reduction='none')
            intrinsic_reward = torch.mean(prediction_error).item()

        return intrinsic_reward

    def learn(self):
        """Train the agent using ICM and DQN."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Calculate intrinsic rewards
        intrinsic_rewards = torch.zeros(self.batch_size).to(self.device)
        for i in range(self.batch_size):
            intrinsic_reward = self.calculate_intrinsic_reward(
                batch[i][0], batch[i][1], batch[i][3]
            )
            intrinsic_rewards[i] = intrinsic_reward

        # Combined reward (extrinsic + intrinsic)
        total_rewards = rewards + self.curiosity_weight * intrinsic_rewards

        # Train ICM
        self._train_icm(states, actions, next_states)

        # Train DQN with combined rewards
        self._train_dqn(states, actions, total_rewards, next_states, dones)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_network()

    def _train_icm(self, states, actions, next_states):
        """Train the Intrinsic Curiosity Module."""
        # Forward model loss
        current_features, predicted_next_features = self.forward_model(states, actions)
        actual_next_features = self.forward_model.encode_state(next_states)
        forward_loss = F.mse_loss(predicted_next_features, actual_next_features.detach())

        # Inverse model loss
        predicted_actions = self.inverse_model(states, next_states)
        inverse_loss = F.cross_entropy(predicted_actions, actions)

        # Combined ICM loss
        icm_loss = (self.forward_loss_weight * forward_loss +
                   self.inverse_loss_weight * inverse_loss)

        # Update ICM
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

    def _train_dqn(self, states, actions, rewards, next_states, dones):
        """Train the DQN with combined rewards."""
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # DQN loss
        dqn_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Update DQN
        self.q_optimizer.zero_grad()
        dqn_loss.backward()
        self.q_optimizer.step()

    def update_target_network(self):
        """Update target network parameters."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Curiosity-Driven Learning for Mario")
    parser.add_argument("--episodes", type=int, default=100,
                      help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0,
                      help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                      help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                      help="Minimum epsilon")
    parser.add_argument("--curiosity-weight", type=float, default=0.1,
                      help="Weight for intrinsic curiosity reward")
    parser.add_argument("--forward-loss-weight", type=float, default=0.8,
                      help="Weight for forward model loss")
    parser.add_argument("--inverse-loss-weight", type=float, default=0.2,
                      help="Weight for inverse model loss")
    parser.add_argument("--memory-size", type=int, default=10000,
                      help="Experience replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--target-update", type=int, default=100,
                      help="Target network update frequency")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    return Args(**vars(parser.parse_args()))


def main():
    if not TORCH_AVAILABLE:
        console.print("[red]PyTorch is required for this example. Please install with:[/red]")
        console.print("pip install torch")
        return

    args = parse_args()

    console.print(f"[bold green]Curiosity-Driven Mario Learning[/bold green]")
    console.print(f"Episodes: {args.episodes}, Curiosity Weight: {args.curiosity_weight}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environment and agent
    env = MarioEnvironment(args.seed)
    agent = CuriosityAgent(env.state_size, env.action_size, args)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    intrinsic_rewards = []
    level_progress = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        total_intrinsic = 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Calculate intrinsic reward for logging
            intrinsic_reward = agent.calculate_intrinsic_reward(state, action, next_state)
            total_intrinsic += intrinsic_reward

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Train agent
            agent.learn()

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        intrinsic_rewards.append(total_intrinsic)
        level_progress.append(info["level_progress"])

        # Logging
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            avg_intrinsic = np.mean(intrinsic_rewards[-20:])
            avg_progress = np.mean(level_progress[-20:])

            console.print(f"Episode {episode + 1:4d} | "
                        f"Avg Reward: {avg_reward:6.2f} | "
                        f"Avg Length: {avg_length:5.1f} | "
                        f"Avg Intrinsic: {avg_intrinsic:5.2f} | "
                        f"Avg Progress: {avg_progress:4.1f} | "
                        f"Epsilon: {agent.epsilon:.3f}")

    # Final results
    console.print(f"\n[bold]Training Complete![/bold]")

    table = Table(title="Curiosity-Driven Learning Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    final_20_rewards = episode_rewards[-20:]
    final_20_lengths = episode_lengths[-20:]
    final_20_intrinsic = intrinsic_rewards[-20:]
    final_20_progress = level_progress[-20:]

    table.add_row("Avg Extrinsic Reward (last 20)", f"{np.mean(final_20_rewards):.2f}")
    table.add_row("Avg Intrinsic Reward (last 20)", f"{np.mean(final_20_intrinsic):.2f}")
    table.add_row("Avg Episode Length (last 20)", f"{np.mean(final_20_lengths):.1f}")
    table.add_row("Avg Level Progress (last 20)", f"{np.mean(final_20_progress):.1f}")
    table.add_row("Final Epsilon", f"{agent.epsilon:.3f}")

    console.print(table)

    # Test the trained agent
    console.print(f"\n[bold]Testing trained agent...[/bold]")

    test_episodes = 5
    test_rewards = []
    test_progress = []

    for test_ep in range(test_episodes):
        state = env.reset()
        total_reward = 0

        # Set epsilon to 0 for deterministic testing
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

        for step in range(500):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        test_rewards.append(total_reward)
        test_progress.append(info["level_progress"])

        console.print(f"Test Episode {test_ep + 1}: "
                    f"Reward = {total_reward:.2f}, "
                    f"Progress = {info['level_progress']:.1f}, "
                    f"Completed = {'Yes' if info['level_completed'] else 'No'}")

        # Restore epsilon
        agent.epsilon = original_epsilon

    console.print(f"\nTest Results: "
                f"Avg Reward = {np.mean(test_rewards):.2f}, "
                f"Avg Progress = {np.mean(test_progress):.1f}")


if __name__ == "__main__":
    main()