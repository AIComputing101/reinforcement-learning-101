#!/usr/bin/env python3
"""
Tabular Q-learning for discretized CartPole-v1 to demonstrate value iteration via lookup table.
Simple and educational; not optimized.
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import gymnasium as gym
from rich.console import Console

console = Console()


def discretize(obs, bins):
    # Observation: [cart_pos, cart_vel, pole_ang, pole_ang_vel]
    upper_bounds = [2.4, 3.0, 0.21, 3.5]
    lower_bounds = [-2.4, -3.0, -0.21, -3.5]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(4)]
    new_obs = [int(round((bins[i] - 1) * r)) for i, r in enumerate(ratios)]
    new_obs = [min(bins[i] - 1, max(0, new_obs[i])) for i in range(4)]
    return tuple(new_obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--bins", type=int, nargs=4, default=[6, 12, 6, 12])
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    bins = args.bins
    q_table = np.zeros(bins + [env.action_space.n])

    rewards = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        state = discretize(obs, bins)
        done = False
        ep_reward = 0

        while not done:
            if np.random.rand() < args.epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins)
            ep_reward += reward

            best_next = np.max(q_table[next_state])
            td_target = reward + args.gamma * best_next * (0.0 if done else 1.0)
            q_table[state + (action,)] += args.alpha * (td_target - q_table[state + (action,)])

            state = next_state

        rewards.append(ep_reward)
        args.epsilon = max(args.epsilon_min, args.epsilon * args.epsilon_decay)

        if ep % 100 == 0:
            avg = np.mean(rewards[-100:])
            console.log(f"Episode {ep}/{args.episodes} | Avg(100): {avg:.1f} | Epsilon: {args.epsilon:.2f}")

    env.close()


if __name__ == "__main__":
    main()
