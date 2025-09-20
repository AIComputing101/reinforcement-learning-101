# Module 05: Advanced RL Topics (Curiosity, ES, Multi‑Agent)

## Overview
Explore intrinsic motivation (ICM/RND), Evolution Strategies for black‑box optimization, and fundamentals of multi‑agent RL and coordination.

## Learning Objectives
- Understand curiosity‑driven exploration (ICM/RND)
- Implement a simple Evolution Strategies optimizer
- Recognize multi‑agent settings and CTDE patterns
- Know when to consider hierarchical/meta/imitation/model‑based RL

## Key Concepts
- Curiosity: reward shaping via prediction error (ICM) or RND features
- ES: Gaussian perturbations, fitness shaping, gradient‑free search
- Multi‑agent: non‑stationarity, self‑play, centralized critic, credit assignment
- Extras (when applicable): hierarchical options, meta‑learning, imitation, model‑based planning

## Run the Examples
Evolution Strategies (NumPy‑only):
```bash
python modules/module_05_advanced_rl/examples/evolutionary_cartpole.py --generations 5 --population 32 --seed 0
```

Curiosity (stub):
```bash
python modules/module_05_advanced_rl/examples/curiosity_supermario.py --env SuperMarioBros-1-1-v0
```

Multi‑agent (stub):
```bash
python modules/module_05_advanced_rl/examples/multiagent_gridworld.py
```

## Exercises
1) ES: tune `--sigma`, `--alpha`, and population size
2) Add fitness normalization or rank‑based shaping
3) Extend gridworld with cooperative rewards; look for emergent behavior
4) Sketch an options‑based hierarchy for a long‑horizon task (waypoints + local controller)

## Debugging & Best Practices
- ES: ensure symmetric perturbations; verify gradient‑estimate sign
- Curiosity: balance intrinsic vs extrinsic reward; normalize signals
- Multi‑agent: mitigate non‑stationarity with CTDE; evaluate with self‑play
- Prefer reproducible seeds; log returns, exploration metrics, and stability indicators

## Further Reading
- ICM (Pathak et al., 2017); RND (Burda et al., 2018)
- OpenAI ES (Salimans et al., 2017)
- MADDPG (Lowe et al., 2017)
- Options framework (Sutton et al., 1999)
