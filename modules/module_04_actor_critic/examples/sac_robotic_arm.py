#!/usr/bin/env python3
"""
SAC for Robotic Arm Control

Soft Actor-Critic (SAC) implementation for continuous control robotic arm tasks.
This example demonstrates off-policy, maximum entropy reinforcement learning
for robotic manipulation. Uses a simplified 2D robotic arm environment
for educational purposes.

Usage:
    python sac_robotic_arm.py --episodes 300 --learning-rate 3e-4 --tau 0.005
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

console = Console()


@dataclass
class Args:
    episodes: int
    learning_rate: float
    gamma: float
    tau: float
    alpha: float
    batch_size: int
    memory_size: int
    hidden_size: int
    seed: int


class RoboticArmEnv:
    """
    Simplified 2D robotic arm environment for educational purposes.

    The arm has 2 joints and must reach a target position while avoiding obstacles.
    State: [joint1_angle, joint2_angle, joint1_velocity, joint2_velocity, target_x, target_y, end_effector_x, end_effector_y]
    Action: [joint1_torque, joint2_torque] (continuous, normalized to [-1, 1])
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Arm parameters
        self.link1_length = 1.0
        self.link2_length = 1.0
        self.max_torque = 2.0
        self.dt = 0.02  # 50 Hz control frequency

        # State limits
        self.max_angle = np.pi
        self.max_velocity = 10.0

        # Environment bounds
        self.workspace_radius = 1.8

        # State and action spaces
        self.state_size = 8
        self.action_size = 2

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Random initial joint angles
        self.joint1_angle = self.rng.uniform(-np.pi/2, np.pi/2)
        self.joint2_angle = self.rng.uniform(-np.pi/2, np.pi/2)

        # Zero initial velocities
        self.joint1_velocity = 0.0
        self.joint2_velocity = 0.0

        # Random target position within workspace
        target_distance = self.rng.uniform(0.5, self.workspace_radius)
        target_angle = self.rng.uniform(0, 2 * np.pi)
        self.target_x = target_distance * np.cos(target_angle)
        self.target_y = target_distance * np.sin(target_angle)

        self.step_count = 0
        self.max_steps = 200

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take an action and return the next state, reward, done flag, and info."""
        # Clip and scale actions
        torque1 = np.clip(action[0], -1, 1) * self.max_torque
        torque2 = np.clip(action[1], -1, 1) * self.max_torque

        # Simple dynamics (mass = 1, damping = 0.1)
        damping = 0.1

        # Update velocities (simplified dynamics)
        self.joint1_velocity += (torque1 - damping * self.joint1_velocity) * self.dt
        self.joint2_velocity += (torque2 - damping * self.joint2_velocity) * self.dt

        # Clip velocities
        self.joint1_velocity = np.clip(self.joint1_velocity, -self.max_velocity, self.max_velocity)
        self.joint2_velocity = np.clip(self.joint2_velocity, -self.max_velocity, self.max_velocity)

        # Update angles
        self.joint1_angle += self.joint1_velocity * self.dt
        self.joint2_angle += self.joint2_velocity * self.dt

        # Wrap angles to [-pi, pi]
        self.joint1_angle = self._wrap_angle(self.joint1_angle)
        self.joint2_angle = self._wrap_angle(self.joint2_angle)

        # Calculate end effector position
        end_effector_x, end_effector_y = self._forward_kinematics()

        # Calculate reward
        reward = self._calculate_reward(end_effector_x, end_effector_y, action)

        # Check if episode is done
        self.step_count += 1
        distance_to_target = np.sqrt((end_effector_x - self.target_x)**2 + (end_effector_y - self.target_y)**2)

        done = (self.step_count >= self.max_steps) or (distance_to_target < 0.1)

        info = {
            "end_effector_x": end_effector_x,
            "end_effector_y": end_effector_y,
            "distance_to_target": distance_to_target,
            "target_reached": distance_to_target < 0.1
        }

        return self._get_state(), reward, done, info

    def _forward_kinematics(self) -> Tuple[float, float]:
        """Calculate end effector position from joint angles."""
        x1 = self.link1_length * np.cos(self.joint1_angle)
        y1 = self.link1_length * np.sin(self.joint1_angle)

        x2 = x1 + self.link2_length * np.cos(self.joint1_angle + self.joint2_angle)
        y2 = y1 + self.link2_length * np.sin(self.joint1_angle + self.joint2_angle)

        return x2, y2

    def _calculate_reward(self, end_x: float, end_y: float, action: np.ndarray) -> float:
        """Calculate reward based on distance to target and action penalty."""
        # Distance to target
        distance = np.sqrt((end_x - self.target_x)**2 + (end_y - self.target_y)**2)

        # Reward components
        distance_reward = -distance * 2.0  # Negative distance (closer is better)

        # Bonus for reaching target
        target_bonus = 10.0 if distance < 0.1 else 0.0

        # Action penalty (encourage smooth control)
        action_penalty = -0.01 * np.sum(action**2)

        # Velocity penalty (encourage stability)
        velocity_penalty = -0.001 * (self.joint1_velocity**2 + self.joint2_velocity**2)

        total_reward = distance_reward + target_bonus + action_penalty + velocity_penalty

        return total_reward

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        end_effector_x, end_effector_y = self._forward_kinematics()

        state = np.array([
            self.joint1_angle / np.pi,  # Normalized joint angles
            self.joint2_angle / np.pi,
            self.joint1_velocity / self.max_velocity,  # Normalized velocities
            self.joint2_velocity / self.max_velocity,
            self.target_x / self.workspace_radius,  # Normalized target position
            self.target_y / self.workspace_radius,
            end_effector_x / self.workspace_radius,  # Normalized end effector position
            end_effector_y / self.workspace_radius
        ], dtype=np.float32)

        return state


class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )


class Actor(nn.Module):
    """SAC Actor network with Gaussian policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Action bounds
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale + self.action_bias

        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """Get action for evaluation (deterministic)."""
        mean, log_std = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class Critic(nn.Module):
    """SAC Critic network (Q-function)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1 forward pass
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2 forward pass
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2


class SACAgent:
    """Soft Actor-Critic agent."""

    def __init__(self, state_dim: int, action_dim: int, args: Args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size

        # Networks
        self.actor = Actor(state_dim, action_dim, args.hidden_size).to(self.device)
        self.critic = Critic(state_dim, action_dim, args.hidden_size).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, args.hidden_size).to(self.device)

        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(args.memory_size, state_dim, action_dim)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if training:
            action, _ = self.actor.sample(state_tensor)
        else:
            action = self.actor.get_action(state_tensor)

        return action.cpu().data.numpy().flatten()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        """Update the agent's networks."""
        if self.replay_buffer.size < self.batch_size:
            return

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update alpha (temperature parameter)
        new_actions, log_probs = self.actor.sample(states)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target network
        self._soft_update(self.critic_target, self.critic, self.tau)

    def _soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="SAC for Robotic Arm Control")
    parser.add_argument("--episodes", type=int, default=300,
                      help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate for all networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                      help="Soft update parameter")
    parser.add_argument("--alpha", type=float, default=0.2,
                      help="Initial temperature parameter")
    parser.add_argument("--batch-size", type=int, default=256,
                      help="Training batch size")
    parser.add_argument("--memory-size", type=int, default=100000,
                      help="Replay buffer size")
    parser.add_argument("--hidden-size", type=int, default=256,
                      help="Hidden layer size")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    return Args(**vars(parser.parse_args()))


def main():
    if not TORCH_AVAILABLE:
        console.print("[red]PyTorch is required for this example. Please install with:[/red]")
        console.print("pip install torch")
        return

    args = parse_args()

    console.print(f"[bold green]SAC Robotic Arm Training[/bold green]")
    console.print(f"Episodes: {args.episodes}, Learning Rate: {args.learning_rate}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environment and agent
    env = RoboticArmEnv(args.seed)
    agent = SACAgent(env.state_size, env.action_size, args)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Update agent
            if agent.replay_buffer.size >= args.batch_size:
                agent.learn()

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(1.0 if info["target_reached"] else 0.0)

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_success = np.mean(success_rate[-50:])

            console.print(f"Episode {episode + 1:4d} | "
                        f"Avg Reward: {avg_reward:7.2f} | "
                        f"Avg Length: {avg_length:5.1f} | "
                        f"Success Rate: {avg_success:.3f} | "
                        f"Buffer Size: {agent.replay_buffer.size:5d}")

    # Final results
    console.print(f"\n[bold]Training Complete![/bold]")

    table = Table(title="SAC Robotic Arm Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    final_50_rewards = episode_rewards[-50:]
    final_50_lengths = episode_lengths[-50:]
    final_50_success = success_rate[-50:]

    table.add_row("Avg Reward (last 50)", f"{np.mean(final_50_rewards):.2f}")
    table.add_row("Avg Episode Length (last 50)", f"{np.mean(final_50_lengths):.1f}")
    table.add_row("Success Rate (last 50)", f"{np.mean(final_50_success):.3f}")
    table.add_row("Total Episodes", f"{len(episode_rewards)}")

    console.print(table)

    # Test the trained agent
    console.print(f"\n[bold]Testing trained agent...[/bold]")

    test_episodes = 5
    test_rewards = []
    test_successes = []

    for test_ep in range(test_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.act(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        test_rewards.append(total_reward)
        test_successes.append(1.0 if info["target_reached"] else 0.0)

        console.print(f"Test Episode {test_ep + 1}: Reward = {total_reward:.2f}, "
                    f"Success = {'Yes' if info['target_reached'] else 'No'}")

    console.print(f"\nTest Results: Avg Reward = {np.mean(test_rewards):.2f}, "
                f"Success Rate = {np.mean(test_successes):.3f}")


if __name__ == "__main__":
    main()