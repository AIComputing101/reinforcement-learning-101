#!/usr/bin/env python3
"""
Simple evolutionary strategy (ES) for CartPole-v1 using NumPy only.

We evolve a linear policy: action = 1 if w^T s > 0 else 0. Mutate weights over generations.
This runs without PyTorch and is suitable for CPU-only environments.
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import gymnasium as gym
from rich.console import Console

console = Console()


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    population: int = 50
    elite_frac: float = 0.2
    noise_std: float = 0.1
    generations: int = 50
    episodes_per_eval: int = 3
    seed: int | None = None


def evaluate(env, w: np.ndarray, episodes: int) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = 1 if float(np.dot(w, obs)) > 0 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total += ep_reward
    return total / episodes


def train(cfg: Config):
    set_seed(cfg.seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]

    mu = np.zeros(obs_dim, dtype=np.float32)
    best_score = -1e9
    elite_k = max(1, int(cfg.elite_frac * cfg.population))

    for gen in range(1, cfg.generations + 1):
        population = np.random.randn(cfg.population, obs_dim).astype(np.float32) * cfg.noise_std + mu
        scores = np.zeros(cfg.population, dtype=np.float32)
        for i in range(cfg.population):
            scores[i] = evaluate(env, population[i], cfg.episodes_per_eval)
        elite_idx = np.argsort(scores)[-elite_k:]
        elites = population[elite_idx]
        mu = elites.mean(axis=0)
        gen_best = float(scores[elite_idx[-1]])
        best_score = max(best_score, gen_best)
        console.log(
            f"Gen {gen}/{cfg.generations} | Best: {gen_best:.1f} | Mean Elite: {float(scores[elite_idx].mean()):.1f} | GlobalBest: {best_score:.1f}"
        )
        if gen_best >= 195.0:
            console.print("[bold green]✓ Environment solved by ES![/bold green]")
            break

    env.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--episodes-per-eval", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    return Config(
        population=a.population,
        elite_frac=a.elite_frac,
        noise_std=a.noise_std,
        generations=a.generations,
        episodes_per_eval=a.episodes_per_eval,
        seed=a.seed,
    )


if __name__ == "__main__":
    train(parse_args())
