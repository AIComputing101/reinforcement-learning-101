#!/usr/bin/env python3
"""
Rainbow DQN on Atari Games

A comprehensive implementation of Rainbow DQN (combining DQN, Double DQN,
Dueling DQN, Prioritized Experience Replay, Multi-step learning, Distributional DQN,
and Noisy Networks) for Atari games. This is an educational implementation
showcasing advanced deep RL techniques.

Usage:
    python rainbow_atari.py --game PongNoFrameskip-v4 --episodes 100 --lr 1e-4
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from rich.console import Console
from rich.table import Table

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import gymnasium as gym
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

console = Console()


@dataclass
class Args:
    game: str
    episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    memory_size: int
    batch_size: int
    target_update: int
    multi_step: int
    atoms: int
    v_min: float
    v_max: float
    seed: int


if TORCH_AVAILABLE:
    class NoisyLinear(nn.Module):
        """Noisy linear layer for exploration."""

        def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init

            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))

            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            mu_range = 1 / np.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

        def reset_noise(self):
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def _scale_noise(self, size: int):
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign().mul_(x.abs().sqrt_())

        def forward(self, input: torch.Tensor):
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(input, weight, bias)


class RainbowDQN(nn.Module):
    """Rainbow DQN with all components."""

    def __init__(self, state_shape: Tuple[int, ...], action_size: int, atoms: int = 51,
                 v_min: float = -10, v_max: float = 10):
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, atoms)
        self.delta_z = (v_max - v_min) / (atoms - 1)

        # Convolutional layers for Atari frames
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(state_shape)

        # Dueling architecture with noisy layers
        self.feature = NoisyLinear(conv_out_size, 512)

        # Value stream
        self.value_hidden = NoisyLinear(512, 512)
        self.value = NoisyLinear(512, atoms)

        # Advantage stream
        self.advantage_hidden = NoisyLinear(512, 512)
        self.advantage = NoisyLinear(512, action_size * atoms)

    def _get_conv_out_size(self, shape):
        o = nn.Sequential(self.conv1, self.conv2, self.conv3)
        return o(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)

        # Feature layer
        feature = F.relu(self.feature(x))

        # Value stream
        value = F.relu(self.value_hidden(feature))
        value = self.value(value).view(batch_size, 1, self.atoms)

        # Advantage stream
        advantage = F.relu(self.advantage_hidden(feature))
        advantage = self.advantage(advantage).view(batch_size, self.action_size, self.atoms)

        # Dueling aggregation
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_dist, dim=-1)

        return q_dist

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for layer in [self.feature, self.value_hidden, self.value,
                     self.advantage_hidden, self.advantage]:
            layer.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        # Sum tree for efficient sampling
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.size = 0
        self.max_priority = 1.0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, experience, priority: float = None):
        if priority is None:
            priority = self.max_priority

        idx = self.size % self.capacity
        data_idx = idx + self.capacity - 1

        self.data[idx] = experience
        self.update(data_idx, priority)

        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, priority)

    def update(self, idx: int, priority: float):
        change = priority ** self.alpha - self.tree[idx]
        self.tree[idx] = priority ** self.alpha
        self._propagate(idx, change)

    def sample(self, batch_size: int):
        batch_indices = []
        batch_data = []
        priorities = []

        segment = self.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1

            batch_indices.append(idx)
            batch_data.append(self.data[data_idx])
            priorities.append(self.tree[idx])

        # Importance sampling weights
        sampling_probs = np.array(priorities) / self.total()
        weights = (self.size * sampling_probs) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch_data, batch_indices, weights


class AtariWrapper:
    """Simplified Atari preprocessing wrapper."""

    def __init__(self, env_name: str):
        try:
            self.env = gym.make(env_name)
            self.action_size = self.env.action_space.n
        except:
            # Fallback to a simple environment for testing
            console.print("[yellow]Atari environment not available, using CartPole for testing[/yellow]")
            self.env = gym.make("CartPole-v1")
            self.action_size = self.env.action_space.n

        self.frame_stack = 4
        self.frames = []

    def reset(self):
        obs, _ = self.env.reset()

        # For Atari, we'd preprocess frames here
        # For simplicity, we'll use a placeholder
        if hasattr(self.env.observation_space, 'shape') and len(self.env.observation_space.shape) == 3:
            # Atari environment
            frame = self._preprocess_frame(obs)
        else:
            # Simple environment - create fake frame stack
            frame = np.zeros((84, 84), dtype=np.uint8)

        self.frames = [frame] * self.frame_stack
        return self._get_state()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if hasattr(self.env.observation_space, 'shape') and len(self.env.observation_space.shape) == 3:
            frame = self._preprocess_frame(obs)
        else:
            frame = np.zeros((84, 84), dtype=np.uint8)

        self.frames.append(frame)
        if len(self.frames) > self.frame_stack:
            self.frames.pop(0)

        return self._get_state(), reward, done, info

    def _preprocess_frame(self, frame):
        # Simplified preprocessing - in practice you'd convert to grayscale,
        # resize to 84x84, etc.
        if len(frame.shape) == 3:
            gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
            return gray.astype(np.uint8)
        return frame.astype(np.uint8)

    def _get_state(self):
        return np.stack(self.frames, axis=0)


class RainbowAgent:
    """Rainbow DQN agent."""

    def __init__(self, state_shape: Tuple[int, ...], action_size: int, args: Args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.gamma = args.gamma
        self.multi_step = args.multi_step
        self.atoms = args.atoms
        self.v_min = args.v_min
        self.v_max = args.v_max

        # Networks
        self.q_network = RainbowDQN(state_shape, action_size, args.atoms, args.v_min, args.v_max).to(self.device)
        self.target_network = RainbowDQN(state_shape, action_size, args.atoms, args.v_min, args.v_max).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        # Experience replay
        self.memory = PrioritizedReplayBuffer(args.memory_size)
        self.batch_size = args.batch_size
        self.target_update = args.target_update

        # Multi-step learning
        self.multi_step_buffer = []

        # Support for distributional RL
        self.support = torch.linspace(args.v_min, args.v_max, args.atoms).to(self.device)
        self.delta_z = (args.v_max - args.v_min) / (args.atoms - 1)

        self.update_target_network()
        self.steps = 0

    def act(self, state: np.ndarray) -> int:
        """Choose action using the network (no epsilon-greedy due to noisy layers)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_dist = self.q_network(state_tensor)

        # Convert distribution to Q-values
        q_values = (q_dist * self.support).sum(dim=-1)
        action = q_values.argmax().item()

        return action

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience with multi-step learning."""
        self.multi_step_buffer.append((state, action, reward, next_state, done))

        if len(self.multi_step_buffer) >= self.multi_step:
            # Calculate n-step return
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.multi_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break

            # Store n-step experience
            s0, a0, _, _, _ = self.multi_step_buffer[0]
            sn, _, _, _, dn = self.multi_step_buffer[-1]

            experience = (s0, a0, n_step_return, sn, dn, len(self.multi_step_buffer))
            self.memory.add(experience)

            self.multi_step_buffer.pop(0)

    def learn(self):
        """Train the network using prioritized experience replay."""
        if self.memory.size < self.batch_size:
            return

        # Sample batch
        batch, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)

        # Unpack batch
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        n_steps = torch.LongTensor([e[5] for e in batch]).to(self.device)

        # Current Q distribution
        q_dist = self.q_network(states)
        action_dist = q_dist[range(self.batch_size), actions]

        # Target Q distribution (Double DQN)
        with torch.no_grad():
            next_q_dist = self.q_network(next_states)
            next_q_values = (next_q_dist * self.support).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=1)

            target_q_dist = self.target_network(next_states)
            target_action_dist = target_q_dist[range(self.batch_size), next_actions]

            # Distributional Bellman operator
            gamma_with_terminal = self.gamma ** n_steps.float() * (~dones).float()
            target_support = rewards.unsqueeze(1) + gamma_with_terminal.unsqueeze(1) * self.support.unsqueeze(0)

            # Clamp to support range
            target_support = target_support.clamp(min=self.v_min, max=self.v_max)

            # Compute projection
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            target_dist = torch.zeros_like(target_action_dist)
            for i in range(self.batch_size):
                for j in range(self.atoms):
                    target_dist[i, l[i, j]] += target_action_dist[i, j] * (u[i, j].float() - b[i, j])
                    target_dist[i, u[i, j]] += target_action_dist[i, j] * (b[i, j] - l[i, j].float())

        # Cross-entropy loss
        log_action_dist = torch.log(action_dist + 1e-8)
        loss = -(target_dist * log_action_dist).sum(dim=1)

        # Weighted loss for prioritized replay
        weighted_loss = (weights * loss).mean()

        # Update priorities
        priorities = loss.detach().cpu().numpy() + 1e-6
        for idx, priority in zip(indices, priorities):
            self.memory.update(idx, priority)

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Reset noise
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target_network()

    def update_target_network(self):
        """Hard update of target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Rainbow DQN on Atari")
    parser.add_argument("--game", default="PongNoFrameskip-v4",
                      help="Atari game environment")
    parser.add_argument("--episodes", type=int, default=100,
                      help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1,
                      help="Exploration rate (unused in noisy networks)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                      help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                      help="Minimum epsilon")
    parser.add_argument("--memory-size", type=int, default=50000,
                      help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--target-update", type=int, default=1000,
                      help="Target network update frequency")
    parser.add_argument("--multi-step", type=int, default=3,
                      help="Multi-step learning steps")
    parser.add_argument("--atoms", type=int, default=51,
                      help="Number of atoms for distributional RL")
    parser.add_argument("--v-min", type=float, default=-10,
                      help="Minimum value for distributional RL")
    parser.add_argument("--v-max", type=float, default=10,
                      help="Maximum value for distributional RL")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    return Args(**vars(parser.parse_args()))


def main():
    if not TORCH_AVAILABLE:
        console.print("[red]PyTorch is required for this example. Please install with:[/red]")
        console.print("pip install torch")
        return

    args = parse_args()

    console.print(f"[bold green]Rainbow DQN Training[/bold green]")
    console.print(f"Game: {args.game}, Episodes: {args.episodes}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environment
    env = AtariWrapper(args.game)

    # Initialize agent
    state_shape = (4, 84, 84)  # Frame stack, height, width
    agent = RainbowAgent(state_shape, env.action_size, args)

    # Training metrics
    episode_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Train agent
            agent.learn()

            if done or steps >= 1000:  # Max episode length
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])

            console.print(f"Episode {episode + 1:4d} | "
                        f"Avg Reward: {avg_reward:7.2f} | "
                        f"Avg Length: {avg_length:5.1f} | "
                        f"Memory Size: {agent.memory.size:5d}")

    # Final results
    console.print(f"\n[bold]Training Complete![/bold]")

    table = Table(title="Rainbow DQN Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    final_10_rewards = episode_rewards[-10:]
    final_10_lengths = episode_lengths[-10:]

    table.add_row("Avg Reward (last 10)", f"{np.mean(final_10_rewards):.2f}")
    table.add_row("Avg Episode Length (last 10)", f"{np.mean(final_10_lengths):.1f}")
    table.add_row("Total Episodes", f"{len(episode_rewards)}")
    table.add_row("Memory Size", f"{agent.memory.size}")

    console.print(table)

    # Test the trained agent
    console.print(f"\n[bold]Testing trained agent...[/bold]")
    state = env.reset()
    total_reward = 0

    for step in range(200):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    console.print(f"Test episode reward: {total_reward}")


if __name__ == "__main__":
    main()