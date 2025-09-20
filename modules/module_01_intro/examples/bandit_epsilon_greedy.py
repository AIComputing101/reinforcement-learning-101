#!/usr/bin/env python3
"""
Epsilon-Greedy Multi-Armed Bandit demo.

Usage:
  python bandit_epsilon_greedy.py --arms 10 --steps 2000 --epsilon 0.1 --seed 0

Logs a simple training loop using `rich` and prints summary stats.
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class BanditArm:
    mean: float
    std: float = 1.0

    def pull(self) -> float:
        return float(np.random.normal(self.mean, self.std))


class EpsilonGreedyAgent:
    def __init__(self, n_arms: int, epsilon: float, init_value: float = 0.0):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.q_values = np.full(n_arms, init_value, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)

    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_arms))
        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        # incremental mean update
        self.q_values[action] += (reward - self.q_values[action]) / n


def build_bandit(n_arms: int, seed: int | None) -> List[BanditArm]:
    set_seed(seed)
    # Random means around N(0,1)
    means = np.random.normal(0, 1, size=n_arms)
    return [BanditArm(mean=float(m), std=1.0) for m in means]


def run(arms: int, steps: int, epsilon: float, seed: int | None):
    bandit = build_bandit(arms, seed)
    agent = EpsilonGreedyAgent(n_arms=arms, epsilon=epsilon)

    rewards = []
    optimal_action_counts = 0
    optimal_arm = int(np.argmax([b.mean for b in bandit]))

    with console.status("[bold green]Training bandit..."):
        for t in range(1, steps + 1):
            a = agent.select_action()
            r = bandit[a].pull()
            agent.update(a, r)
            rewards.append(r)
            if a == optimal_arm:
                optimal_action_counts += 1
            if t % max(steps // 10, 1) == 0:
                console.log(
                    f"Step {t}/{steps} | AvgReward: {np.mean(rewards):.3f} | BestArmPct: {optimal_action_counts/t:.2%} | Epsilon: {epsilon}"
                )

    table = Table(title="Bandit Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Steps", str(steps))
    table.add_row("Arms", str(arms))
    table.add_row("Epsilon", f"{epsilon}")
    table.add_row("Average Reward", f"{np.mean(rewards):.3f}")
    table.add_row("Best Arm %", f"{optimal_action_counts/steps:.2%}")
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run(args.arms, args.steps, args.epsilon, args.seed)


if __name__ == "__main__":
    main()
