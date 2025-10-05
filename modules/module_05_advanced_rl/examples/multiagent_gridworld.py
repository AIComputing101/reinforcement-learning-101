#!/usr/bin/env python3
"""Minimal cooperative Multi-Agent GridWorld with independent Q-Learning.

Scenario:
  - Grid of size N x N (default 5x5)
  - Two agents start in opposite corners (0,0) and (N-1,N-1)
  - Shared goal cell in center; episode ends when BOTH agents have visited goal (not necessarily simultaneously—each must visit at least once)
  - Each step: -0.01 shared shaping penalty; when an agent first reaches goal: +1 (shared); if both completed: +2 bonus and episode terminates.
  - Independent tabular Q-learners (each agent ignores the other except via shared reward). Joint coordination emerges through shared reward shaping.

Focus: Demonstrate non-stationarity & coordination with minimal code.

Run:
  python multiagent_gridworld.py --episodes 200

Dependencies: only NumPy + rich.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


Action = int  # 0: up,1: right,2: down,3: left, 4: stay
ACTION_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (0, 0),
}


@dataclass
class Config:
    size: int = 5
    episodes: int = 200
    max_steps: int = 100
    alpha: float = 0.3
    gamma: float = 0.95
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 0


class GridWorldCoop:
    def __init__(self, size: int, seed: int = 0):
        self.size = size
        self.rng = random.Random(seed)
        self.n_agents = 2
        self.goal = (size // 2, size // 2)
        self.reset()

    def reset(self):
        self.pos = [ (0,0), (self.size-1, self.size-1) ]
        self.visited_goal = [False, False]
        return self.observe()

    def observe(self) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        # explicitly return 2-tuple for static type checkers
        return self.pos[0], self.pos[1]

    def step(self, actions: List[Action]):
        reward = -0.01  # small step penalty
        for i, a in enumerate(actions):
            dr, dc = ACTION_DELTAS.get(a, (0,0))
            r, c = self.pos[i]
            nr, nc = max(0, min(self.size-1, r + dr)), max(0, min(self.size-1, c + dc))
            self.pos[i] = (nr, nc)
            if self.pos[i] == self.goal and not self.visited_goal[i]:
                self.visited_goal[i] = True
                reward += 1.0  # first time arrival
        done = all(self.visited_goal)
        if done:
            reward += 2.0  # completion bonus
        return self.observe(), reward, done, {}


class IndependentQLearner:
    def __init__(self, cfg: Config, agent_id: int):
        self.cfg = cfg
        self.agent_id = agent_id
        # State: agent row, agent col, other row, other col, goal visited flag (self)
        self.q = np.zeros((cfg.size, cfg.size, cfg.size, cfg.size, 2, 5), dtype=np.float32)

    def state_index(self, obs: Tuple[Tuple[int,int], Tuple[int,int]], visited: List[bool]):
        (r1,c1),(r2,c2)=obs
        if self.agent_id == 0:
            r,c = r1,c1; or_,oc = r2,c2; flag = int(visited[0])
        else:
            r,c = r2,c2; or_,oc = r1,c1; flag = int(visited[1])
        return r,c,or_,oc,flag

    def select_action(self, state_idx, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0,4)
        return int(np.argmax(self.q[state_idx]))

    def update(self, s_idx, a, r, ns_idx, done: bool, alpha: float, gamma: float):
        qsa = self.q[s_idx][a]
        target = r if done else r + gamma * np.max(self.q[ns_idx])
        self.q[s_idx][a] = qsa + alpha * (target - qsa)


def run(cfg: Config):
    random.seed(cfg.seed)
    env = GridWorldCoop(cfg.size, cfg.seed)
    agents = [IndependentQLearner(cfg, 0), IndependentQLearner(cfg, 1)]
    epsilon = cfg.epsilon_start
    episode_returns: List[float] = []
    coop_success = 0

    for ep in range(1, cfg.episodes+1):
        obs = env.reset()
        total_r = 0.0
        for step in range(cfg.max_steps):
            s_indices = [ag.state_index(obs, env.visited_goal) for ag in agents]
            actions = [ag.select_action(s_indices[i], epsilon) for i,ag in enumerate(agents)]
            next_obs, reward, done, _ = env.step(actions)
            total_r += reward
            ns_indices = [ag.state_index(next_obs, env.visited_goal) for ag in agents]
            for i, ag in enumerate(agents):
                ag.update(s_indices[i], actions[i], reward, ns_indices[i], done, cfg.alpha, cfg.gamma)
            obs = next_obs
            if done:
                coop_success += 1
                break
        episode_returns.append(total_r)
        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
        if ep % 20 == 0 or ep == 1:
            avg_last = sum(episode_returns[-20:]) / min(len(episode_returns), 20)
            console.print(f"Ep {ep:4d} Return {total_r:5.2f} Avg(20) {avg_last:5.2f} Eps {epsilon:.3f} Success {coop_success/ep:.2f}")

    table = Table(title="Multi-Agent GridWorld Summary")
    table.add_column("Episodes")
    table.add_column("AvgReturn")
    table.add_column("CoopSuccessRate")
    table.add_row(str(cfg.episodes), f"{sum(episode_returns)/len(episode_returns):.2f}", f"{coop_success/cfg.episodes:.2f}")
    console.print(table)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=0.9)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    return Config(size=a.size, episodes=a.episodes, max_steps=a.max_steps, alpha=a.alpha, gamma=a.gamma,
                  epsilon_start=a.epsilon_start, epsilon_end=a.epsilon_end, epsilon_decay=a.epsilon_decay, seed=a.seed)


def main():
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
