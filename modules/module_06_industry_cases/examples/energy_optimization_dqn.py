#!/usr/bin/env python3
"""
Energy Optimization using DQN

A simplified smart building energy management system that learns to optimize
HVAC and lighting controls to minimize energy consumption while maintaining
occupant comfort. This is an educational example of applying RL to industrial
energy optimization.

Usage:
    python energy_optimization_dqn.py --episodes 500 --building-type office --season winter
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Args:
    building_type: str
    season: str
    episodes: int
    learning_rate: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    gamma: float
    memory_size: int
    batch_size: int
    seed: int


class EnergyEnvironment:
    """Simplified smart building energy management environment."""

    def __init__(self, building_type: str, season: str, seed: int = 42):
        self.building_type = building_type
        self.season = season
        self.rng = np.random.RandomState(seed)

        # Building parameters
        self.building_params = {
            "office": {"base_load": 50, "occupancy_factor": 1.2, "thermal_mass": 0.8},
            "residential": {"base_load": 30, "occupancy_factor": 0.8, "thermal_mass": 1.0},
            "retail": {"base_load": 80, "occupancy_factor": 1.5, "thermal_mass": 0.6}
        }

        # Season parameters (target temps and outdoor temp ranges)
        self.season_params = {
            "winter": {"target_temp": 22, "outdoor_range": (-5, 10), "heating_cost": 1.5},
            "summer": {"target_temp": 24, "outdoor_range": (25, 40), "cooling_cost": 1.2},
            "spring": {"target_temp": 23, "outdoor_range": (15, 25), "heating_cost": 1.0},
            "autumn": {"target_temp": 23, "outdoor_range": (10, 20), "heating_cost": 1.1}
        }

        # State: [indoor_temp, outdoor_temp, hour_of_day, occupancy, energy_price]
        self.state_size = 5
        # Actions: [hvac_setting, lighting_level] - discretized
        self.action_size = 9  # 3x3 grid: hvac (low/med/high) x lighting (low/med/high)

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        params = self.building_params[self.building_type]
        season_params = self.season_params[self.season]

        # Initialize state
        self.indoor_temp = season_params["target_temp"] + self.rng.uniform(-2, 2)
        self.outdoor_temp = self.rng.uniform(*season_params["outdoor_range"])
        self.hour = self.rng.randint(0, 24)
        self.occupancy = self._get_occupancy(self.hour)
        self.energy_price = self._get_energy_price(self.hour)

        self.step_count = 0
        self.total_energy_cost = 0
        self.comfort_violations = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action and return next state, reward, done, info."""
        hvac_action, lighting_action = divmod(action, 3)  # Decode composite action

        # Apply actions
        hvac_power = (hvac_action + 1) * 0.3  # 0.3, 0.6, 0.9
        lighting_power = (lighting_action + 1) * 0.2  # 0.2, 0.4, 0.6

        # Update environment
        self._update_temperature(hvac_power)
        self._update_time()

        # Calculate costs and comfort
        energy_cost = self._calculate_energy_cost(hvac_power, lighting_power)
        comfort_penalty = self._calculate_comfort_penalty()

        # Reward function: minimize cost and discomfort
        reward = -(energy_cost + comfort_penalty * 2.0)  # Weight comfort heavily

        self.total_energy_cost += energy_cost
        if comfort_penalty > 0:
            self.comfort_violations += 1

        self.step_count += 1
        done = self.step_count >= 24  # One day simulation

        info = {
            "energy_cost": energy_cost,
            "comfort_penalty": comfort_penalty,
            "indoor_temp": self.indoor_temp,
            "hvac_power": hvac_power,
            "lighting_power": lighting_power
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.array([
            self.indoor_temp / 30.0,  # Normalize temperature
            self.outdoor_temp / 40.0,
            self.hour / 24.0,
            self.occupancy,
            self.energy_price
        ], dtype=np.float32)

    def _update_temperature(self, hvac_power: float):
        """Update indoor temperature based on HVAC action."""
        params = self.building_params[self.building_type]
        season_params = self.season_params[self.season]

        # Temperature dynamics (simplified)
        thermal_mass = params["thermal_mass"]
        outdoor_influence = 0.1 * (self.outdoor_temp - self.indoor_temp) / thermal_mass

        if self.season == "winter":
            # Heating mode
            hvac_effect = hvac_power * 3.0 * season_params["heating_cost"]
        else:
            # Cooling mode
            hvac_effect = -hvac_power * 2.5 * season_params["cooling_cost"]

        # Occupancy heat generation
        occupancy_heat = self.occupancy * params["occupancy_factor"] * 0.5

        # Update temperature
        temp_change = hvac_effect + outdoor_influence + occupancy_heat
        self.indoor_temp += temp_change * 0.1  # Time step factor

        # Add some noise
        self.indoor_temp += self.rng.normal(0, 0.1)

    def _update_time(self):
        """Update time and dependent variables."""
        self.hour = (self.hour + 1) % 24
        self.occupancy = self._get_occupancy(self.hour)
        self.energy_price = self._get_energy_price(self.hour)

        # Update outdoor temperature (daily cycle)
        outdoor_base = np.mean(self.season_params[self.season]["outdoor_range"])
        daily_variation = 5 * np.sin(2 * np.pi * self.hour / 24)
        self.outdoor_temp = outdoor_base + daily_variation + self.rng.normal(0, 1)

    def _get_occupancy(self, hour: int) -> float:
        """Get occupancy level based on building type and hour."""
        if self.building_type == "office":
            if 9 <= hour <= 17:
                return 0.8 + self.rng.uniform(-0.2, 0.2)
            else:
                return 0.1 + self.rng.uniform(0, 0.1)
        elif self.building_type == "residential":
            if hour <= 8 or hour >= 18:
                return 0.7 + self.rng.uniform(-0.2, 0.2)
            else:
                return 0.3 + self.rng.uniform(-0.1, 0.1)
        else:  # retail
            if 10 <= hour <= 21:
                return 0.6 + self.rng.uniform(-0.3, 0.3)
            else:
                return 0.1 + self.rng.uniform(0, 0.1)

    def _get_energy_price(self, hour: int) -> float:
        """Get energy price based on time-of-use pricing."""
        # Peak hours: 6-9 AM, 5-8 PM
        if (6 <= hour <= 9) or (17 <= hour <= 20):
            return 0.15 + self.rng.uniform(-0.02, 0.02)  # Peak pricing
        else:
            return 0.08 + self.rng.uniform(-0.01, 0.01)  # Off-peak pricing

    def _calculate_energy_cost(self, hvac_power: float, lighting_power: float) -> float:
        """Calculate energy cost for current time step."""
        params = self.building_params[self.building_type]
        total_power = params["base_load"] + hvac_power * 50 + lighting_power * 20
        return total_power * self.energy_price

    def _calculate_comfort_penalty(self) -> float:
        """Calculate comfort penalty based on temperature deviation."""
        target_temp = self.season_params[self.season]["target_temp"]
        temp_deviation = abs(self.indoor_temp - target_temp)

        # Comfort zone: ±2°C
        if temp_deviation <= 2.0:
            return 0
        else:
            return (temp_deviation - 2.0) ** 2


class SimpleDQN:
    """Simplified DQN agent for energy optimization."""

    def __init__(self, state_size: int, action_size: int, args: Args):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = args.learning_rate
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        # Initialize Q-table (simplified - no neural network)
        self.q_table = np.zeros((100, action_size))  # Discretized state space
        self.memory = []
        self.memory_size = args.memory_size

    def _discretize_state(self, state: np.ndarray) -> int:
        """Discretize continuous state into discrete bins."""
        # Simple binning for Q-table lookup
        bins = []
        for i, val in enumerate(state):
            bin_val = int(np.clip(val * 10, 0, 9))  # 10 bins per dimension
            bins.append(bin_val)

        # Convert to single index (simplified)
        return min(sum(bins) * 2 + int(state[0] * 50), 99)

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_idx = self._discretize_state(state)
        return np.argmax(self.q_table[state_idx])

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_idx = self._discretize_state(state)
            next_state_idx = self._discretize_state(next_state)

            target = reward
            if not done:
                target += self.gamma * np.amax(self.q_table[next_state_idx])

            # Q-learning update
            self.q_table[state_idx][action] += self.lr * (target - self.q_table[state_idx][action])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Energy optimization using DQN")
    parser.add_argument("--building-type", default="office",
                      choices=["office", "residential", "retail"],
                      help="Type of building to optimize")
    parser.add_argument("--season", default="winter",
                      choices=["winter", "summer", "spring", "autumn"],
                      help="Season for simulation")
    parser.add_argument("--episodes", type=int, default=300,
                      help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                      help="Learning rate for DQN")
    parser.add_argument("--epsilon", type=float, default=1.0,
                      help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                      help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                      help="Minimum epsilon")
    parser.add_argument("--gamma", type=float, default=0.95,
                      help="Discount factor")
    parser.add_argument("--memory-size", type=int, default=2000,
                      help="Experience replay memory size")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    return Args(**vars(parser.parse_args()))


def main():
    args = parse_args()

    console.print(f"[bold green]Energy Optimization DQN[/bold green]")
    console.print(f"Building: {args.building_type}, Season: {args.season}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment and agent
    env = EnergyEnvironment(args.building_type, args.season, args.seed)
    agent = SimpleDQN(env.state_size, env.action_size, args)

    # Training metrics
    episode_rewards = []
    energy_costs = []
    comfort_violations = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        # Train agent
        agent.replay(args.batch_size)

        # Track metrics
        episode_rewards.append(total_reward)
        energy_costs.append(env.total_energy_cost)
        comfort_violations.append(env.comfort_violations)

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_cost = np.mean(energy_costs[-50:])
            avg_violations = np.mean(comfort_violations[-50:])

            console.print(f"Episode {episode + 1:4d} | "
                        f"Avg Reward: {avg_reward:6.2f} | "
                        f"Avg Cost: ${avg_cost:6.2f} | "
                        f"Avg Violations: {avg_violations:4.1f} | "
                        f"Epsilon: {agent.epsilon:.3f}")

    # Final summary
    console.print("\n[bold]Training Complete![/bold]")

    table = Table(title="Energy Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    final_50_rewards = episode_rewards[-50:]
    final_50_costs = energy_costs[-50:]
    final_50_violations = comfort_violations[-50:]

    table.add_row("Avg Reward (last 50)", f"{np.mean(final_50_rewards):.2f}")
    table.add_row("Avg Energy Cost (last 50)", f"${np.mean(final_50_costs):.2f}")
    table.add_row("Avg Comfort Violations (last 50)", f"{np.mean(final_50_violations):.1f}")
    table.add_row("Final Epsilon", f"{agent.epsilon:.3f}")

    console.print(table)

    # Show learned policy sample
    console.print(f"\n[bold]Testing learned policy...[/bold]")
    state = env.reset()
    day_log = []

    for hour in range(24):
        action = agent.act(state)
        hvac_action, lighting_action = divmod(action, 3)

        state, reward, done, info = env.step(action)
        day_log.append({
            "hour": hour,
            "indoor_temp": info["indoor_temp"],
            "hvac": ["Low", "Med", "High"][hvac_action],
            "lighting": ["Low", "Med", "High"][lighting_action],
            "cost": info["energy_cost"]
        })

        if done:
            break

    # Show sample day
    console.print(f"\n[bold]Sample 24-hour control schedule:[/bold]")
    for entry in day_log[::4]:  # Show every 4 hours
        console.print(f"Hour {entry['hour']:2d}: "
                    f"Temp {entry['indoor_temp']:5.1f}°C | "
                    f"HVAC: {entry['hvac']} | "
                    f"Light: {entry['lighting']} | "
                    f"Cost: ${entry['cost']:.2f}")


if __name__ == "__main__":
    main()