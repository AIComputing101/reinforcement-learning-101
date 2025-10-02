#!/usr/bin/env python3
"""Upper Confidence Bound (UCB1) Multi-Armed Bandit demo.

Usage:
  python bandit_ucb.py --arms 10 --steps 2000 --c 2.0 --seed 0

Implements the UCB1 exploration strategy and logs rich metrics during play.
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


@dataclass
class BanditArm:
    mean: float
    std: float = 1.0

    def pull(self) -> float:
        return float(np.random.normal(self.mean, self.std))


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


class UCBAgent:
    def __init__(self, n_arms: int, c: float = 2.0, init_value: float = 0.0):
        self.c = c
        self.n_arms = n_arms
        self.q_values = np.full(n_arms, init_value, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.total_steps = 0

    def select_action(self) -> int:
        self.total_steps += 1
        # Ensure each arm is tried at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_scores = self.q_values + self.c * np.sqrt(
            2.0 * np.log(self.total_steps) / self.counts
        )
        return int(np.argmax(ucb_scores))

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


def build_bandit(n_arms: int, seed: int | None) -> List[BanditArm]:
    set_seed(seed)
    means = np.random.normal(0, 1, size=n_arms)
    return [BanditArm(mean=float(m), std=1.0) for m in means]


def run(arms: int, steps: int, c: float, seed: int | None):
    bandit = build_bandit(arms, seed)
    agent = UCBAgent(n_arms=arms, c=c)

    rewards: list[float] = []
    optimal_action_counts = 0
    optimal_arm = int(np.argmax([b.mean for b in bandit]))

    with console.status("[bold blue]Running UCB bandit..."):
        for t in range(1, steps + 1):
            action = agent.select_action()
            reward = bandit[action].pull()
            agent.update(action, reward)
            rewards.append(reward)
            if action == optimal_arm:
                optimal_action_counts += 1

            if t % max(steps // 10, 1) == 0:
                console.log(
                    "Step {}/{} | AvgReward: {:.3f} | BestArmPct: {:.2%} | c: {}".format(
                        t,
                        steps,
                        float(np.mean(rewards)),
                        optimal_action_counts / t,
                        c,
                    )
                )

    table = Table(title="Bandit UCB Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Steps", str(steps))
    table.add_row("Arms", str(arms))
    table.add_row("c", f"{c}")
    table.add_row("Average Reward", f"{float(np.mean(rewards)):.3f}")
    table.add_row("Best Arm %", f"{optimal_action_counts / steps:.2%}")
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--c", type=float, default=2.0, help="Exploration constant")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.arms, args.steps, args.c, args.seed)


if __name__ == "__main__":
    main()
