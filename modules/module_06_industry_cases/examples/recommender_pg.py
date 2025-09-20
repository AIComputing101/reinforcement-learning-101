#!/usr/bin/env python3
"""
Recommender System using Policy Gradients

A simplified recommender system that learns to suggest items to users to maximize
both engagement and diversity. Uses policy gradients to handle the complex
multi-objective optimization typical in recommendation systems.

Usage:
    python recommender_pg.py --users 100 --items 50 --episodes 500 --diversity-bonus 0.1
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
    users: int
    items: int
    episodes: int
    learning_rate: float
    gamma: float
    diversity_bonus: float
    max_recommendations: int
    seed: int


class User:
    """Represents a user with preferences and interaction history."""

    def __init__(self, user_id: int, preferences: np.ndarray, rng: np.random.RandomState):
        self.user_id = user_id
        self.preferences = preferences  # Item category preferences
        self.interaction_history = []
        self.satisfaction_score = 0.5  # Dynamic satisfaction
        self.rng = rng

    def interact_with_item(self, item_id: int, item_features: np.ndarray) -> Tuple[float, bool]:
        """User interacts with recommended item, returns (rating, clicked)."""
        # Calculate base interest based on preference alignment
        preference_match = np.dot(self.preferences, item_features)

        # Add novelty bonus (users like some diversity)
        recent_categories = [item_features for _, item_features in self.interaction_history[-5:]]
        if recent_categories:
            avg_recent = np.mean(recent_categories, axis=0)
            novelty_bonus = 1.0 - np.dot(avg_recent, item_features)
        else:
            novelty_bonus = 0.5

        # Calculate engagement probability
        engagement_score = 0.6 * preference_match + 0.3 * novelty_bonus + 0.1 * self.satisfaction_score
        engagement_score = np.clip(engagement_score, 0, 1)

        # Add noise to simulate real user behavior
        engagement_score += self.rng.normal(0, 0.1)
        engagement_score = np.clip(engagement_score, 0, 1)

        clicked = self.rng.random() < engagement_score
        rating = engagement_score + self.rng.normal(0, 0.1) if clicked else 0

        # Update satisfaction (users get tired of similar content)
        if clicked:
            self.satisfaction_score = 0.9 * self.satisfaction_score + 0.1 * engagement_score
            self.interaction_history.append((item_id, item_features))
        else:
            self.satisfaction_score = 0.95 * self.satisfaction_score  # Slight decrease for bad recs

        return max(0, min(1, rating)), clicked


class RecommenderEnvironment:
    """Recommendation environment with users, items, and diversity metrics."""

    def __init__(self, num_users: int, num_items: int, max_recs: int, seed: int = 42):
        self.num_users = num_users
        self.num_items = num_items
        self.max_recommendations = max_recs
        self.rng = np.random.RandomState(seed)

        # Item features (categories: entertainment, education, news, sports, tech)
        self.item_features = self._generate_items()

        # User preferences
        self.users = self._generate_users()

        # State: [user_embedding, recent_interactions, time_context]
        self.state_size = 5 + 5 + 1  # user prefs + recent categories + time

        # Action: item recommendation (simplified)
        self.action_size = num_items

        self.reset()

    def _generate_items(self) -> np.ndarray:
        """Generate item feature matrix."""
        items = []
        for _ in range(self.num_items):
            # Each item has a distribution over 5 categories
            categories = self.rng.dirichlet(np.ones(5))  # Sum to 1
            items.append(categories)
        return np.array(items)

    def _generate_users(self) -> List[User]:
        """Generate users with diverse preferences."""
        users = []
        for i in range(self.num_users):
            # User preferences over categories
            preferences = self.rng.dirichlet(np.ones(5) * 2)  # More concentrated
            users.append(User(i, preferences, self.rng))
        return users

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_user_idx = self.rng.randint(0, self.num_users)
        self.step_count = 0
        self.episode_clicks = 0
        self.episode_ratings = []
        self.recommended_items = []
        self.diversity_score = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Recommend item (action) to current user."""
        item_id = action
        user = self.users[self.current_user_idx]

        # User interacts with recommended item
        rating, clicked = user.interact_with_item(item_id, self.item_features[item_id])

        # Track episode metrics
        self.episode_ratings.append(rating)
        self.recommended_items.append(item_id)
        if clicked:
            self.episode_clicks += 1

        # Calculate diversity (variety in recommended items)
        if len(self.recommended_items) > 1:
            recent_features = self.item_features[self.recommended_items[-5:]]
            diversity = self._calculate_diversity(recent_features)
            self.diversity_score = 0.8 * self.diversity_score + 0.2 * diversity

        # Reward function: engagement + diversity
        engagement_reward = rating * 10  # Scale rating
        diversity_reward = self.diversity_score * 2 if len(self.recommended_items) > 1 else 0

        reward = engagement_reward + diversity_reward

        # Move to next user or end episode
        self.step_count += 1
        if self.step_count >= self.max_recommendations:
            done = True
        else:
            # Switch to different user occasionally
            if self.rng.random() < 0.3:
                self.current_user_idx = self.rng.randint(0, self.num_users)
            done = False

        info = {
            "user_id": self.current_user_idx,
            "item_id": item_id,
            "rating": rating,
            "clicked": clicked,
            "diversity_score": self.diversity_score,
            "engagement_reward": engagement_reward,
            "diversity_reward": diversity_reward
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        user = self.users[self.current_user_idx]

        # User preferences
        user_prefs = user.preferences

        # Recent interaction categories (last 5)
        if len(user.interaction_history) > 0:
            recent_features = [features for _, features in user.interaction_history[-5:]]
            avg_recent = np.mean(recent_features, axis=0) if recent_features else np.zeros(5)
        else:
            avg_recent = np.zeros(5)

        # Time context (normalized step count)
        time_context = [self.step_count / self.max_recommendations]

        state = np.concatenate([user_prefs, avg_recent, time_context])
        return state.astype(np.float32)

    def _calculate_diversity(self, item_features: np.ndarray) -> float:
        """Calculate diversity score for a set of items."""
        if len(item_features) < 2:
            return 0.0

        # Calculate pairwise similarities and return 1 - average similarity
        similarities = []
        for i in range(len(item_features)):
            for j in range(i + 1, len(item_features)):
                sim = np.dot(item_features[i], item_features[j])
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity


class PolicyGradientAgent:
    """Policy gradient agent for recommendation."""

    def __init__(self, state_size: int, action_size: int, args: Args):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = args.learning_rate
        self.gamma = args.gamma

        # Simple linear policy (no neural network)
        self.policy_weights = np.random.normal(0, 0.1, (state_size, action_size))
        self.baseline = 0.0  # Simple baseline for variance reduction

        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []

    def act(self, state: np.ndarray) -> int:
        """Choose action using policy."""
        # Compute action probabilities
        logits = np.dot(state, self.policy_weights)

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Sample action
        action = np.random.choice(self.action_size, p=probs)
        return action

    def remember(self, state: np.ndarray, action: int, reward: float):
        """Store experience for episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.rewards) == 0:
            return

        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        # Update baseline (simple average)
        self.baseline = 0.9 * self.baseline + 0.1 * np.mean(returns)

        # Normalize returns (advantage estimation)
        advantages = returns - self.baseline
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Policy gradient update
        for state, action, advantage in zip(self.states, self.actions, advantages):
            # Compute gradients
            logits = np.dot(state, self.policy_weights)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Gradient for REINFORCE
            grad = np.outer(state, probs)
            grad[:, action] -= state

            # Update weights
            self.policy_weights -= self.lr * advantage * grad

        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Recommender system using Policy Gradients")
    parser.add_argument("--users", type=int, default=100,
                      help="Number of users in the system")
    parser.add_argument("--items", type=int, default=50,
                      help="Number of items to recommend")
    parser.add_argument("--episodes", type=int, default=500,
                      help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="Learning rate for policy gradient")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument("--diversity-bonus", type=float, default=0.1,
                      help="Weight for diversity in reward")
    parser.add_argument("--max-recommendations", type=int, default=20,
                      help="Max recommendations per episode")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    return Args(**vars(parser.parse_args()))


def main():
    args = parse_args()

    console.print(f"[bold green]Recommender System Policy Gradients[/bold green]")
    console.print(f"Users: {args.users}, Items: {args.items}, Diversity bonus: {args.diversity_bonus}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment and agent
    env = RecommenderEnvironment(args.users, args.items, args.max_recommendations, args.seed)
    agent = PolicyGradientAgent(env.state_size, env.action_size, args)

    # Training metrics
    episode_rewards = []
    episode_clicks = []
    episode_diversity = []
    episode_ratings = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward)
            state = next_state
            total_reward += reward

            if done:
                break

        # Train agent after episode
        agent.learn()

        # Track metrics
        episode_rewards.append(total_reward)
        episode_clicks.append(env.episode_clicks)
        episode_diversity.append(env.diversity_score)
        episode_ratings.append(np.mean(env.episode_ratings) if env.episode_ratings else 0)

        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_clicks = np.mean(episode_clicks[-100:])
            avg_diversity = np.mean(episode_diversity[-100:])
            avg_rating = np.mean(episode_ratings[-100:])

            console.print(f"Episode {episode + 1:4d} | "
                        f"Avg Reward: {avg_reward:7.2f} | "
                        f"Avg Clicks: {avg_clicks:4.1f} | "
                        f"Avg Diversity: {avg_diversity:.3f} | "
                        f"Avg Rating: {avg_rating:.3f}")

    # Final summary
    console.print("\n[bold]Training Complete![/bold]")

    table = Table(title="Recommender System Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    final_100_rewards = episode_rewards[-100:]
    final_100_clicks = episode_clicks[-100:]
    final_100_diversity = episode_diversity[-100:]
    final_100_ratings = episode_ratings[-100:]

    table.add_row("Avg Reward (last 100)", f"{np.mean(final_100_rewards):.2f}")
    table.add_row("Avg Clicks (last 100)", f"{np.mean(final_100_clicks):.1f}")
    table.add_row("Avg Diversity (last 100)", f"{np.mean(final_100_diversity):.3f}")
    table.add_row("Avg Rating (last 100)", f"{np.mean(final_100_ratings):.3f}")
    table.add_row("Click Rate", f"{np.mean(final_100_clicks) / args.max_recommendations:.3f}")

    console.print(table)

    # Test learned policy
    console.print(f"\n[bold]Testing learned policy...[/bold]")
    state = env.reset()

    test_recommendations = []
    for i in range(10):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        test_recommendations.append({
            "rec": i + 1,
            "user": info["user_id"],
            "item": info["item_id"],
            "rating": info["rating"],
            "clicked": info["clicked"],
            "diversity": info["diversity_score"]
        })

        state = next_state
        if done:
            break

    # Show sample recommendations
    console.print(f"\n[bold]Sample recommendations:[/bold]")
    for rec in test_recommendations[:5]:
        click_text = "CLICKED" if rec["clicked"] else "Not clicked"
        console.print(f"Rec {rec['rec']}: User {rec['user']} → Item {rec['item']} | "
                    f"Rating: {rec['rating']:.3f} | {click_text}")


if __name__ == "__main__":
    main()