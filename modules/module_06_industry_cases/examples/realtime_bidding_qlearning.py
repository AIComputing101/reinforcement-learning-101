#!/usr/bin/env python3
"""
Real-Time Bidding using Q-Learning

A simplified real-time bidding (RTB) system that learns optimal bidding strategies
for display advertising auctions. The agent learns to bid on ad placements to
maximize ROI while managing budget constraints.

Usage:
    python realtime_bidding_qlearning.py --budget 10000 --campaigns 5 --episodes 500
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Args:
    budget: int
    campaigns: int
    episodes: int
    learning_rate: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    gamma: float
    seed: int


class AdAuction:
    """Represents a single ad auction with features and outcome."""

    def __init__(self, user_features: np.ndarray, context_features: np.ndarray,
                 true_value: float):
        self.user_features = user_features  # age, income, interests, etc.
        self.context_features = context_features  # time, device, location, etc.
        self.true_value = true_value  # True conversion value (unknown to agent)

    def get_features(self) -> np.ndarray:
        """Get combined feature vector for the auction."""
        return np.concatenate([self.user_features, self.context_features])


class RTBEnvironment:
    """Real-time bidding environment with simulated ad auctions."""

    def __init__(self, budget: int, campaigns: int, seed: int = 42):
        self.initial_budget = budget
        self.num_campaigns = campaigns
        self.rng = np.random.RandomState(seed)

        # Feature dimensions
        self.user_feature_dim = 4  # age, income, gender, interests
        self.context_feature_dim = 3  # hour, device_type, location
        self.feature_dim = self.user_feature_dim + self.context_feature_dim

        # Action space: bid amounts (discretized)
        self.bid_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # $ amounts
        self.action_size = len(self.bid_levels)

        # State space: [budget_remaining, hour, campaign_performance, auction_features]
        self.state_size = 2 + campaigns + self.feature_dim

        self.hour = 0  # Initialize hour before reset
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.budget_remaining = self.initial_budget
        self.step_count = 0
        self.total_conversions = 0
        self.total_spend = 0
        self.total_revenue = 0

        # Campaign performance tracking (CTR estimates)
        self.campaign_performance = np.ones(self.num_campaigns) * 0.02  # 2% baseline CTR

        # Generate first auction
        self.current_auction = self._generate_auction()
        self.hour = self.rng.randint(0, 24)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take bidding action and advance to next auction."""
        bid_amount = self.bid_levels[action]

        # Determine if we win the auction
        won_auction, winning_price = self._simulate_auction(bid_amount)

        revenue = 0
        conversion = False

        if won_auction:
            # Pay the winning price
            cost = winning_price
            self.budget_remaining -= cost
            self.total_spend += cost

            # Simulate conversion based on true value
            conversion_prob = min(self.current_auction.true_value / 10.0, 0.2)  # Max 20% conversion
            if self.rng.random() < conversion_prob:
                conversion = True
                revenue = self.current_auction.true_value
                self.total_revenue += revenue
                self.total_conversions += 1

                # Update campaign performance (simplified)
                campaign_id = self.step_count % self.num_campaigns
                self.campaign_performance[campaign_id] = 0.9 * self.campaign_performance[campaign_id] + 0.1 * 0.1

        # Calculate reward (ROI-based)
        if won_auction:
            roi = (revenue - winning_price) / max(winning_price, 0.01)
            reward = roi * 10  # Scale reward
        else:
            reward = 0  # No cost, no reward for not bidding/losing

        # Penalize budget exhaustion
        if self.budget_remaining <= 0:
            reward -= 50
            done = True
        else:
            self.step_count += 1
            done = self.step_count >= 1000  # Episode length

        # Generate next auction
        if not done:
            self.current_auction = self._generate_auction()
            self.hour = (self.hour + 1) % 24

        info = {
            "won_auction": won_auction,
            "bid_amount": bid_amount,
            "winning_price": winning_price if won_auction else 0,
            "conversion": conversion,
            "revenue": revenue,
            "budget_remaining": self.budget_remaining,
            "roi": (revenue - winning_price) / max(winning_price, 0.01) if won_auction else 0
        }

        return self._get_state(), reward, done, info

    def _generate_auction(self) -> AdAuction:
        """Generate a new ad auction with random features."""
        # User features: [age_group, income_level, gender, interests]
        user_features = self.rng.uniform(0, 1, self.user_feature_dim)

        # Context features: [hour_normalized, device_type, location_tier]
        context_features = np.array([
            self.hour / 24.0,
            self.rng.choice([0.2, 0.5, 0.8]),  # device type (mobile, tablet, desktop)
            self.rng.choice([0.3, 0.6, 0.9])   # location tier (rural, suburban, urban)
        ])

        # True value based on features (unknown to agent)
        feature_quality = np.mean(user_features) * np.mean(context_features)
        true_value = 2.0 + feature_quality * 8.0 + self.rng.normal(0, 1)
        true_value = max(0, true_value)

        return AdAuction(user_features, context_features, true_value)

    def _simulate_auction(self, our_bid: float) -> Tuple[bool, float]:
        """Simulate auction outcome with competing bidders."""
        # Simulate competitor bids based on auction quality
        auction_quality = np.mean(self.current_auction.get_features())
        num_competitors = self.rng.poisson(5) + 1  # 1-10 competitors

        competitor_bids = []
        for _ in range(num_competitors):
            # Competitors bid based on auction quality + noise
            competitor_bid = auction_quality * 3.0 + self.rng.exponential(1.0)
            competitor_bids.append(max(0.01, competitor_bid))

        # Second-price auction logic
        all_bids = sorted(competitor_bids + [our_bid], reverse=True)

        if all_bids[0] == our_bid:  # We have the highest bid
            winning_price = all_bids[1] if len(all_bids) > 1 else our_bid * 0.9
            return True, winning_price
        else:
            return False, 0.0

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        budget_norm = self.budget_remaining / self.initial_budget
        hour_norm = self.hour / 24.0

        state = np.concatenate([
            [budget_norm, hour_norm],
            self.campaign_performance,
            self.current_auction.get_features()
        ])

        return state.astype(np.float32)


class QLearningAgent:
    """Q-Learning agent for RTB optimization."""

    def __init__(self, state_size: int, action_size: int, args: Args):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = args.learning_rate
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        # Q-table with discretized states
        self.q_table = np.zeros((1000, action_size))
        self.state_bins = 10  # Bins per state dimension

    def _discretize_state(self, state: np.ndarray) -> int:
        """Discretize continuous state into discrete bins."""
        # Simple state discretization
        state_key = 0
        for i, val in enumerate(state[:5]):  # Use first 5 features for discretization
            bin_val = int(np.clip(val * self.state_bins, 0, self.state_bins - 1))
            state_key += bin_val * (self.state_bins ** i)

        return min(state_key, 999)  # Ensure within table bounds

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if self.epsilon > random.random():
            return random.randrange(self.action_size)

        state_idx = self._discretize_state(state)
        return np.argmax(self.q_table[state_idx])

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Update Q-table using Q-learning."""
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)

        # Q-learning update
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])

        self.q_table[state_idx][action] += self.lr * (target - self.q_table[state_idx][action])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Real-time bidding using Q-Learning")
    parser.add_argument("--budget", type=int, default=10000,
                      help="Total budget for bidding")
    parser.add_argument("--campaigns", type=int, default=5,
                      help="Number of ad campaigns")
    parser.add_argument("--episodes", type=int, default=500,
                      help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                      help="Learning rate for Q-learning")
    parser.add_argument("--epsilon", type=float, default=1.0,
                      help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                      help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                      help="Minimum epsilon")
    parser.add_argument("--gamma", type=float, default=0.95,
                      help="Discount factor")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    return Args(**vars(parser.parse_args()))


def main():
    args = parse_args()

    console.print(f"[bold green]Real-Time Bidding Q-Learning[/bold green]")
    console.print(f"Budget: ${args.budget}, Campaigns: {args.campaigns}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment and agent
    env = RTBEnvironment(args.budget, args.campaigns, args.seed)
    agent = QLearningAgent(env.state_size, env.action_size, args)

    # Training metrics
    episode_rewards = []
    episode_rois = []
    episode_conversions = []
    episode_spend = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        total_spend = 0
        conversions = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            total_spend += info.get("winning_price", 0)
            if info.get("conversion", False):
                conversions += 1

            if done:
                break

        # Calculate episode ROI
        episode_roi = (env.total_revenue - env.total_spend) / max(env.total_spend, 1) if env.total_spend > 0 else 0

        # Track metrics
        episode_rewards.append(total_reward)
        episode_rois.append(episode_roi)
        episode_conversions.append(conversions)
        episode_spend.append(env.total_spend)

        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_roi = np.mean(episode_rois[-100:])
            avg_conversions = np.mean(episode_conversions[-100:])
            avg_spend = np.mean(episode_spend[-100:])

            console.print(f"Episode {episode + 1:4d} | "
                        f"Avg Reward: {avg_reward:7.2f} | "
                        f"Avg ROI: {avg_roi:6.3f} | "
                        f"Avg Conversions: {avg_conversions:4.1f} | "
                        f"Avg Spend: ${avg_spend:7.2f} | "
                        f"Epsilon: {agent.epsilon:.3f}")

    # Final summary
    console.print("\n[bold]Training Complete![/bold]")

    table = Table(title="RTB Performance Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    final_100_rewards = episode_rewards[-100:]
    final_100_rois = episode_rois[-100:]
    final_100_conversions = episode_conversions[-100:]
    final_100_spend = episode_spend[-100:]

    table.add_row("Avg Reward (last 100)", f"{np.mean(final_100_rewards):.2f}")
    table.add_row("Avg ROI (last 100)", f"{np.mean(final_100_rois):.3f}")
    table.add_row("Avg Conversions (last 100)", f"{np.mean(final_100_conversions):.1f}")
    table.add_row("Avg Spend (last 100)", f"${np.mean(final_100_spend):.2f}")
    table.add_row("Final Epsilon", f"{agent.epsilon:.3f}")

    console.print(table)

    # Test learned policy
    console.print(f"\n[bold]Testing learned policy on sample auctions...[/bold]")
    state = env.reset()

    test_results = []
    for i in range(10):
        action = agent.act(state)
        bid_amount = env.bid_levels[action]

        next_state, reward, done, info = env.step(action)

        test_results.append({
            "auction": i + 1,
            "bid": bid_amount,
            "won": info["won_auction"],
            "price": info["winning_price"],
            "conversion": info["conversion"],
            "roi": info["roi"]
        })

        state = next_state
        if done:
            break

    # Show sample auction results
    console.print(f"\n[bold]Sample auction results:[/bold]")
    for result in test_results[:5]:
        status = "WON" if result["won"] else "LOST"
        console.print(f"Auction {result['auction']}: "
                    f"Bid ${result['bid']:.2f} - {status}")
        if result["won"]:
            conversion_text = "CONVERTED" if result["conversion"] else "No conversion"
            console.print(f"  → Paid ${result['price']:.2f}, {conversion_text}, ROI: {result['roi']:.3f}")


if __name__ == "__main__":
    main()