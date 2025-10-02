# Reinforcement Learning 101

**Progressive, Hands‑On Reinforcement Learning Project (CLI-First)**  
Built for clarity, reproducibility, and production awareness.

![python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![logging](https://img.shields.io/badge/logging-rich%20console-purple)
![status](https://img.shields.io/badge/status-active-success)

An opinionated, end‑to‑end tutorial project for learning Reinforcement Learning (RL) from first principles to deployment. **No notebooks.** Everything is an explicit, inspectable Python script you can diff, profile, containerize, and ship.

---

## Table of Contents
1. [Who Is This For?](#1-who-is-this-for)
2. [Learning Outcomes](#2-learning-outcomes)
3. [Quick Start](#3-quick-start)
4. [Prerequisites](#4-prerequisites)
5. [Module Path](#5-module-path-progressive-difficulty)
6. [Project Layout](#6-project-layout)
7. [Running Examples](#7-running-examples-more-highlights)
8. [Environment & Reproducibility](#8-environment--reproducibility)
9. [GPU & Docker](#9-gpu--docker)
10. [Dependencies](#10-dependencies)
11. [Testing & Fast Validation](#11-testing--fast-validation)
12. [Troubleshooting](#12-troubleshooting)
13. [Extending the Project](#13-extending-the-project)
14. [Contributing](#14-contributing)
15. [Roadmap Snapshot](#15-roadmap-snapshot)
16. [License](#16-license)
17. [Citation](#17-citation-optional)
18. [FAQ](#18-faq)

---

## 1. Who Is This For?
Learners who want a structured, hands-on path:
* You know basic Python & NumPy, maybe a little PyTorch.
* You want to understand RL algorithms by reading and running minimal reference implementations.
* You prefer reproducible scripts over exploratory notebooks.
* You eventually want to operationalize RL (serving, batch/offline, containers, Kubernetes).

If you just want a black‑box library, this project intentionally is not that. It shows the scaffolding explicitly.

## 2. Learning Outcomes
By completing all 7 modules you will be able to:
* Implement and compare exploration strategies in multi‑armed bandits.
* Derive and code tabular Q-Learning; extend to deep value methods (DQN family, Rainbow components).
* Train policy gradient and REINFORCE baselines; reason about variance & baselines.
* Build actor‑critic agents (A2C, SAC) and understand stability trade‑offs.
* Experiment with advanced ideas (evolutionary strategies, curiosity, multi-agent coordination).
* Apply RL framing to industry‑style scenarios (bidding, recommendation, energy control).
* Package, serve, and batch‑evaluate trained agents (TorchServe, Offline RL, Kubernetes jobs).

## 3. Quick Start

### Automated Setup (Recommended)
One command with GPU auto-detection:
```bash
./setup.sh
```

Or choose your backend directly:
```bash
./setup.sh native          # Native Python (auto-detect GPU)
./setup.sh docker cuda     # Docker with NVIDIA GPU
./setup.sh native cpu      # CPU-only native setup
```

### Manual Setup (CPU)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/requirements-base.txt
pip install -r requirements/requirements-torch-cpu.txt
```

**Verify installation:**
```bash
python scripts/smoke_test.py
```

Run your first bandit:
```bash
python modules/module_01_intro/examples/bandit_epsilon_greedy.py --arms 10 --steps 2000 --epsilon 0.1
```

Run a value method (needs PyTorch):
```bash
python modules/module_02_value_methods/examples/dqn_cartpole.py --episodes 400 --learning-rate 1e-3
```

Discover flags for any script:
```bash
python path/to/script.py --help
```

## 4. Prerequisites
* Python 3.11+ (minimum supported: 3.11; newer versions generally work—use Docker if ecosystem wheels lag)
* Basic linear algebra & probability familiarity
* Optional GPU (CUDA or ROCm) for heavier experiments

## 5. Module Path (Progressive Difficulty)
| Module | Theme | Core Topics | Sample Command |
|--------|-------|------------|----------------|
| 01 Intro | Bandits | Epsilon-greedy, exploration vs exploitation | `bandit_epsilon_greedy.py --arms 10 --steps 2000` |
| 02 Value Methods | Q / DQN / Rainbow | Replay, target nets, prioritized, distributional | `dqn_cartpole.py --episodes 400` |
| 03 Policy Methods | REINFORCE / PG | Return estimation, baselines, continuous actions | `policy_gradient_pendulum.py --episodes 100` |
| 04 Actor-Critic | A2C / SAC | Advantage estimates, entropy, stochastic policies | `a2c_lunarlander.py --episodes 300` |
| 05 Advanced RL | Evolution, Curiosity, Multi-agent | Exploration bonuses, population search | `evolutionary_cartpole.py --generations 5` |
| 06 Industry Cases | Applied RL | Energy, bidding, recommendation framing | `realtime_bidding_qlearning.py --episodes 500` |
| 07 Operationalization | Deployment & Offline | K8s jobs, TorchServe, batch eval | `offline_rl_batch.py --dataset path` |

Each module folder includes `content.md` (theory + checklist) and an `examples/` directory of runnable scripts.

## 6. Project Layout
```
modules/                # Module theory + examples (core learning path)
docker/                 # Dockerfiles, docker-compose.yml, run scripts
scripts/                # Smoke tests & utilities
docs/                   # Roadmap, references, best practices
requirements/           # Modular requirements (base, CPU, CUDA, ROCm)
setup.sh               # Smart setup script with GPU auto-detection
SETUP.md               # Comprehensive setup guide
```

## 7. Running Examples (More Highlights)
Bandits / Intro (NumPy only):
```bash
python modules/module_01_intro/examples/bandit_epsilon_greedy.py --arms 5 --steps 1000 --epsilon 0.05
```

Value Methods:
```bash
python modules/module_02_value_methods/examples/q_learning_cartpole.py --episodes 300
python modules/module_02_value_methods/examples/rainbow_atari.py --episodes 100  # Needs Atari env & ROM legality
```

Policy & Actor-Critic:
```bash
python modules/module_03_policy_methods/examples/reinforce_cartpole.py --episodes 300
python modules/module_04_actor_critic/examples/sac_robotic_arm.py --episodes 200
```

Advanced / Exploration:
```bash
python modules/module_05_advanced_rl/examples/curiosity_supermario.py --episodes 50  # External assets may be required
python modules/module_05_advanced_rl/examples/multiagent_gridworld.py --episodes 200
```

Industry & Ops:
```bash
python modules/module_06_industry_cases/examples/energy_optimization_dqn.py --episodes 300
python modules/module_07_operationalization/examples/torchserve_inference.py --model-path ./models/
```

Use smaller numbers (`--episodes 5`, `--generations 1`, tiny populations) for dry runs.

## 8. Environment & Reproducibility
Set seeds (many scripts expose `--seed`):
```bash
python modules/module_02_value_methods/examples/dqn_cartpole.py --episodes 50 --seed 42
```
Design tenets (see `docs/best_practices.md`):
* Deterministic where feasible (seeding PyTorch, NumPy, env wrappers)
* Structured logging with `rich` for human scan + copy/paste
* Explicit CLI flags over hidden config files
* Separate environment creation from learning logic
* Incremental complexity; minimal runnable baseline first

## 9. GPU & Docker

### Docker Setup
Prefer Docker for reproducible environments and hassle-free GPU support:

```bash
# Automated (auto-detects GPU)
./setup.sh docker

# Manual selection
bash docker/run.sh cpu     # CPU-only (lightweight ~500MB)
bash docker/run.sh cuda    # NVIDIA CUDA 12.9 + PyTorch 2.8
bash docker/run.sh rocm    # AMD ROCm 6.x + PyTorch
```

### Native GPU Setup
For native installations with GPU support:

**NVIDIA CUDA:**
```bash
pip install -r requirements/requirements-base.txt
pip install -r requirements/requirements-torch-cuda.txt
```

**AMD ROCm:**
```bash
pip install -r requirements/requirements-base.txt
pip install -r requirements/requirements-torch-rocm.txt
```

**Verify GPU:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Inside Docker containers, the repo is mounted at `/workspace`. Run scripts directly without additional setup.

## 10. Dependencies

### Requirements Structure
```
requirements/
├── requirements-base.txt        # Core: NumPy, Rich, Gymnasium, TensorBoard
├── requirements-torch-cpu.txt   # PyTorch CPU-only (~500MB)
├── requirements-torch-cuda.txt  # PyTorch with CUDA support
└── requirements-torch-rocm.txt  # PyTorch with ROCm (AMD GPU)
```

**Core (always installed):** `numpy>=1.24`, `rich>=13.7`, `gymnasium[classic-control]>=1.0`, `tensorboard>=2.16`

**PyTorch (choose one):**
- CPU-only: Lightweight, works everywhere
- CUDA: NVIDIA GPUs (requires CUDA 12.x drivers)
- ROCm: AMD GPUs (requires ROCm 6.x drivers)

### Python Version Policy
* **Recommended:** Python 3.11 (best PyTorch wheel availability)
* **Supported:** 3.11 and later (tested on 3.11 & 3.12)
* **Minimum:** 3.11 (earlier versions not supported)
* **Note:** Python 3.13+ may have limited PyTorch wheels—use Docker for consistent environment

If a script requires PyTorch and it's missing, it exits with clear guidance.

## 11. Testing & Fast Validation
Smoke test:
```bash
python scripts/smoke_test.py
```
Quick algorithm sanity runs:
```bash
python modules/module_04_actor_critic/examples/a2c_lunarlander.py --episodes 5
python modules/module_05_advanced_rl/examples/evolutionary_cartpole.py --generations 1 --population 8
```

## 12. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| ImportError: torch | Not installed / wrong Python version | `pip install torch` or use Docker |
| Extremely slow training | Running on CPU with large model | Reduce network size / episodes; try GPU container |
| Atari env fails | ROM / ALE dependency missing | Install appropriate gymnasium extras; ensure legal ROM acquisition |
| Non-deterministic returns | Env stochasticity | Set `--seed`, limit parallelism, check gymnasium version |

## 13. Extending the Project
Add a new example script under the appropriate module's `examples/` and follow existing patterns:
* Top docstring: purpose + minimal usage
* `argparse` flags with sane defaults
* Seed handling (`--seed`)
* Clear separation: model definition, experience gathering, update step
* Log episodic reward + key diagnostics (loss, epsilon, entropy, etc.)

## 14. Contributing
See `CONTRIBUTING.md`.
Principles:
* Keep runtimes short by default (fast smoke params)
* Avoid heavy hidden dependencies; guard imports
* Favor clarity over cleverness—this is a teaching repo
* Update `docs/roadmap.md` if feature scope changes

## 15. Roadmap Snapshot
See `docs/roadmap.md` for full list. Upcoming: PPO / TRPO / TD3, more multi-agent diversity, industry verticals, visualization & benchmarking, integration with SB3 & RLlib.

## 16. License
Licensed under the **Apache License, Version 2.0** (see `LICENSE`). You may use, modify, and distribute this project under the terms of that license. Please retain instructional comments where practical to preserve educational value.

## 17. Citation
If this helped your study or project, consider citing:
```
@misc{rl101tutorial,
	title  = {Reinforcement Learning 101: A Progressive Hands-On Project},
    author = {Stephen Shao}
	year   = {2025},
	howpublished = {GitHub repository},
	url    = {https://github.com/AIComputing101/reinforcement-learning-101}
}
```

## 18. FAQ
Q: Why no notebooks?  
A: Scripts enforce explicit structure, easier diffing, and production parity. You can still adapt them into notebooks if desired.

Q: Where are pretrained weights?  
A: Intentionally omitted to nudge you to train; add caching if you extend.

Q: How long do examples take?  
A: Baselines aim for a few minutes on CPU; scale episodes upward only after verifying flow.

---
Happy learning & experimenting. PRs welcome!