#!/usr/bin/env python3
"""
Simple Ad Placement as Multi-Armed Bandit using epsilon-greedy.

Simulates N ads with unknown Bernoulli CTRs. At each step, choose an ad and
observe a click (1) or no click (0). The agent learns average CTR per ad and
trades off exploration via epsilon-greedy.

Usage:
  python ad_placement.py --ads 5 --steps 5000 --epsilon 0.1 --seed 42
"""
from __future__ import annotations
import argparse
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
class AdArm:
    ctr: float  # true click-through rate in [0,1]

    def click(self) -> int:
        return int(np.random.rand() < self.ctr)


class EpsilonGreedy:
    def __init__(self, n_arms: int, epsilon: float):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q = np.zeros(n_arms, dtype=np.float64)
        self.n = np.zeros(n_arms, dtype=np.int64)

    def select(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_arms))
        return int(np.argmax(self.q))

    def update(self, a: int, r: float):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]


def simulate(n_ads: int, steps: int, epsilon: float, seed: int | None):
    set_seed(seed)
    # Sample true CTRs from Beta(2,8) (mostly low CTRs, as in ads)
    ctrs = np.random.beta(2.0, 8.0, size=n_ads)
    ads = [AdArm(float(c)) for c in ctrs]
    optimal = int(np.argmax(ctrs))

    agent = EpsilonGreedy(n_ads, epsilon)

    clicks = 0
    rewards = []
    optimal_picks = 0

    for t in range(1, steps + 1):
        a = agent.select()
        r = ads[a].click()
        agent.update(a, r)
        clicks += r
        rewards.append(r)
        if a == optimal:
            optimal_picks += 1
        if t % max(steps // 10, 1) == 0:
            console.log(
                f"Step {t}/{steps} | CTR_est_best: {agent.q.max():.3f} | Empirical CTR: {clicks/t:.3f} | BestArm%: {optimal_picks/t:.2%}"
            )

    table = Table(title="Ad Placement Bandit Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Steps", str(steps))
    table.add_row("Ads", str(n_ads))
    table.add_row("Epsilon", f"{epsilon}")
    table.add_row("Empirical CTR", f"{clicks/steps:.3f}")
    table.add_row("Best Arm %", f"{optimal_picks/steps:.2%}")
    table.add_row("True Best CTR", f"{ctrs[optimal]:.3f}")
    console.print(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ads", type=int, default=5)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    simulate(args.ads, args.steps, args.epsilon, args.seed)


if __name__ == "__main__":
    main()
