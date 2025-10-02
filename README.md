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
   - [5.1 Algorithm Selection Guide](#51-algorithm-selection-guide---which-algorithm-should-i-use) ⭐ **NEW**
6. [Project Layout](#6-project-layout)
7. [Running Examples](#7-running-examples-more-highlights)
8. [Environment & Reproducibility](#8-environment--reproducibility)
9. [Best Practices](#9-best-practices)
10. [GPU & Docker](#10-gpu--docker)
11. [Dependencies](#11-dependencies)
12. [Testing & Fast Validation](#12-testing--fast-validation)
13. [Troubleshooting](#13-troubleshooting)
14. [Extending the Project](#14-extending-the-project)
15. [Contributing](#15-contributing)
16. [Roadmap](#16-roadmap)
17. [References](#17-references)
18. [License](#18-license)
19. [Citation (Optional)](#19-citation-optional)
20. [FAQ](#20-faq)

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
* Build actor‑critic agents (A2C, **PPO**, **TD3**, SAC, TRPO) and understand stability trade‑offs.
* Master **industry-standard algorithms** used in production (ChatGPT's RLHF uses PPO).
* **Apply cutting-edge algorithms (2024-2025)**: Offline RL (CQL, IQL), Model-Based (Dreamer), RLHF for LLMs.
* Experiment with advanced ideas (evolutionary strategies, curiosity, multi-agent coordination).
* Apply RL framing to industry‑style scenarios (bidding, recommendation, energy control).
* Package, serve, and batch‑evaluate trained agents (TorchServe, Ray RLlib, Kubernetes jobs).
* **Optimize training infrastructure**: GPU acceleration, distributed training, hyperparameter tuning.

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
# Run all tests (comprehensive)
python scripts/smoke_test.py

# Quick test - core examples only (no PyTorch)
python scripts/smoke_test.py --core-only

# Skip optional/slow tests
python scripts/smoke_test.py --skip-optional

# Test specific group
python scripts/smoke_test.py --group deep-rl
```

The runner now inspects your environment up front: it reports detected versions of PyTorch, Ray, and Optuna, and will automatically skip optional checks that require missing extras (for example, `ppo_lunarlander.py` is skipped if `Box2D` is unavailable). Longer-running jobs such as `sac_robotic_arm.py` live in the optional Infrastructure group—use `--skip-optional` for the fastest required pass.

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
* Python 3.11 or 3.12 (3.13 is not yet supported by official PyTorch wheels—use Docker if you’re on a newer interpreter)
* Basic linear algebra & probability familiarity
* Optional GPU (CUDA or ROCm) for heavier experiments

## 5. Module Path (Progressive Difficulty)
| Module | Theme | Core Topics | Sample Command |
|--------|-------|------------|----------------|
| 01 Intro | Bandits | Epsilon-greedy, exploration vs exploitation | `bandit_epsilon_greedy.py --arms 10 --steps 2000` |
| 02 Value Methods | Q / DQN / Rainbow | Replay, target nets, prioritized, distributional | `dqn_cartpole.py --episodes 400` |
| 03 Policy Methods | REINFORCE / PG | Return estimation, baselines, continuous actions | `policy_gradient_pendulum.py --episodes 100` |
| 04 Actor-Critic | **PPO** / **TD3** / A2C / SAC / TRPO | Industry-standard algorithms, trust regions | `ppo_cartpole.py --episodes 100` ⭐ |
| 05 Advanced RL | Evolution, Curiosity, Multi-agent | Exploration bonuses, population search | `evolutionary_cartpole.py --generations 5` |
| 06 Industry Cases | Applied RL | Energy, bidding, recommendation framing | `realtime_bidding_qlearning.py --episodes 500` |
| 07 Operationalization | **Offline RL**, **RLHF**, Deployment | CQL, IQL, Dreamer, distributed training, TorchServe | `cql_offline_rl.py --mode compare` ⭐ **NEW** |

Each module folder includes `content.md` (theory + checklist) and an `examples/` directory of runnable scripts.

---

## 5.1. Algorithm Selection Guide - Which Algorithm Should I Use?

Use this decision tree to quickly find the right algorithm for your problem:

```
┌─────────────────────────────────────────────────────┐
│  What type of ACTION SPACE do you have?            │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    DISCRETE             CONTINUOUS
    (e.g., 4 actions)    (e.g., torque, velocity)
         │                   │
         │                   │
┌────────┴────────┐    ┌─────┴────────┐
│                 │    │              │
│  Do you have    │    │ Do you need  │
│  offline data?  │    │ exploration? │
│                 │    │              │
└───┬─────────┬───┘    └───┬──────┬───┘
    │         │            │      │
   YES       NO           YES     NO
    │         │            │      │
    │         │            │      │
    │    ┌────┴─────┐      │      │
    │    │          │      │      │
    │  On-Policy Off-Policy│      │
    │    │          │      │      │
    ▼    ▼          ▼      ▼      ▼
  ┌─────────────────────────────────────────┐
  │  RECOMMENDED ALGORITHMS                 │
  ├─────────────────────────────────────────┤
  │  Discrete + Offline    → CQL/IQL        │
  │  Discrete + On-Policy  → PPO ⭐          │
  │  Discrete + Off-Policy → DQN/Rainbow    │
  │  Continuous + Explore  → SAC ⭐          │
  │  Continuous + Exploit  → TD3 ⭐          │
  │  Multi-Armed Bandits   → ε-Greedy       │
  │  Model-Based          → DreamerV3       │
  └─────────────────────────────────────────┘
```

### Quick Reference Table

| Your Situation | Best Algorithm | File to Run | Why? |
|----------------|----------------|-------------|------|
| **Just starting RL** | PPO | `ppo_cartpole.py` | Most stable, widely used, fast training |
| **Production deployment** | PPO or TD3 | `ppo_lunarlander.py` or `td3_pendulum.py` | Industry standards (ChatGPT uses PPO) |
| **Continuous control (robotics)** | TD3 or SAC | `td3_pendulum.py` | State-of-the-art for continuous actions |
| **Discrete actions (games)** | PPO or DQN | `ppo_cartpole.py` or `dqn_cartpole.py` | PPO for stability, DQN for sample efficiency |
| **Limited data (offline RL)** | CQL or IQL | Module 07 examples | Learn from fixed datasets |
| **Exploration needed** | SAC or Curiosity | `sac_robotic_arm.py` | Maximum entropy or intrinsic rewards |
| **Sample efficiency critical** | Model-based (DreamerV3) | Future implementation | Learns world model, imagines trajectories |
| **Learning the theory** | Start with Bandits → Q-Learning → Policy Gradient → PPO | Follow Module 01-04 | Progressive difficulty |

### Algorithm Family Tree

```
Reinforcement Learning Algorithms
│
├── Value-Based (Learn Q(s,a))
│   ├── Tabular
│   │   └── Q-Learning ..................... Module 02
│   └── Deep
│       ├── DQN ............................ Module 02 ⭐
│       ├── Double DQN ..................... Module 02
│       ├── Dueling DQN .................... Module 02
│       └── Rainbow DQN .................... Module 02
│
├── Policy-Based (Learn π(a|s))
│   ├── REINFORCE .......................... Module 03
│   ├── Policy Gradient .................... Module 03
│   └── Evolutionary Strategies ............ Module 05
│
├── Actor-Critic (Learn both π and V/Q)
│   ├── On-Policy
│   │   ├── A2C ............................ Module 04
│   │   ├── PPO ............................ Module 04 ⭐⭐⭐ [INDUSTRY STANDARD]
│   │   └── TRPO ........................... Module 04
│   └── Off-Policy
│       ├── DDPG ........................... (TD3 is better)
│       ├── TD3 ............................ Module 04 ⭐⭐⭐ [INDUSTRY STANDARD]
│       └── SAC ............................ Module 04 ⭐⭐
│
├── Model-Based (Learn environment model)
│   ├── DreamerV3 .......................... Module 07 ⭐⭐ **NEW**
│   └── MuZero ............................. (Future)
│
├── Offline RL (Learn from fixed datasets)
│   ├── CQL (Conservative Q-Learning) ...... Module 07 ⭐⭐⭐ **NEW**
│   ├── IQL (Implicit Q-Learning) .......... Module 07 ⭐⭐⭐ **NEW**
│   └── Behavioral Cloning ................. Module 07
│
└── Exploration & Advanced
    ├── Multi-Armed Bandits ................ Module 01 ⭐ [START HERE]
    ├── Curiosity-Driven ................... Module 05
    ├── Multi-Agent ........................ Module 05
    └── RLHF (for LLMs) .................... Module 07 ⭐⭐⭐ **NEW**

⭐ = Beginner-friendly
⭐⭐ = Production-ready
⭐⭐⭐ = Industry standard (2024-2025)
```

### When to Use Each Algorithm (Practical Decision Guide)

**Use PPO when:**
- ✅ You want the safest, most reliable choice
- ✅ You're deploying to production (ChatGPT, AlphaStar use this)
- ✅ You have either discrete OR continuous actions
- ✅ You can afford to collect fresh data for each update
- ✅ Training stability > sample efficiency

**Use TD3 when:**
- ✅ You have continuous action spaces (robotics, control)
- ✅ Sample efficiency matters (expensive simulations)
- ✅ You can use a replay buffer (store past experiences)
- ✅ You need deterministic policies
- ✅ You're benchmarking against research papers

**Use SAC when:**
- ✅ You have continuous actions + need exploration
- ✅ Maximum sample efficiency is critical
- ✅ Environment is stochastic (benefits from entropy)
- ✅ You want automatic temperature tuning
- ✅ Robustness to hyperparameters is important

**Use DQN when:**
- ✅ You have discrete actions (simple games)
- ✅ You want to learn from a replay buffer
- ✅ You understand value-based methods
- ✅ PPO is overkill for your simple problem

**Use Multi-Armed Bandits when:**
- ✅ You're just starting with RL (great introduction!)
- ✅ You have a stateless decision problem
- ✅ You need exploration strategies (ε-greedy, UCB)
- ✅ A/B testing, recommendation systems

**Use Model-Based (DreamerV3) when:**
- ✅ Sample efficiency is CRITICAL (very expensive data)
- ✅ You can learn an accurate world model
- ✅ You want to plan ahead via imagination
- ✅ You have access to GPU resources

---

## 6. Project Layout
```
modules/                # Module theory + examples (core learning path)
docker/                 # Dockerfiles, docker-compose.yml, run scripts
scripts/                # Smoke tests & utilities
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

**⭐ NEW: Advanced Algorithms (2024-2025)**
```bash
# Offline RL - Learn from fixed datasets (no environment interaction!)
python modules/module_07_operationalization/examples/cql_offline_rl.py --mode compare --dataset-path data/cartpole.pkl
python modules/module_07_operationalization/examples/iql_offline_rl.py --mode compare --dataset-path data/cartpole.pkl

# Model-Based RL - Train policy in imagination
python modules/module_07_operationalization/examples/dreamer_model_based.py --env CartPole-v1 --episodes 200

# RLHF - Language model alignment (like ChatGPT)
python modules/module_07_operationalization/examples/rlhf_text_generation.py --task sentiment --iterations 100

# Infrastructure - Distributed training & hyperparameter tuning
python modules/module_07_operationalization/examples/ray_distributed_ppo.py --num-workers 4
python modules/module_07_operationalization/examples/hyperparameter_tuning_optuna.py --n-trials 50

# Benchmark Suite - Compare algorithms
python modules/module_07_operationalization/examples/benchmark_suite.py --env CartPole-v1 --algorithms dqn ppo
```

Use smaller numbers (`--episodes 5`, `--generations 1`, tiny populations) for dry runs.

## 8. Environment & Reproducibility
Set seeds (many scripts expose `--seed`):
```bash
python modules/module_02_value_methods/examples/dqn_cartpole.py --episodes 50 --seed 42
```
Design tenets (consolidated below in [Best Practices](#9-best-practices)):
* Deterministic where feasible (seeding PyTorch, NumPy, env wrappers)
* Structured logging with `rich` for human scan + copy/paste
* Explicit CLI flags over hidden config files
* Separate environment creation from learning logic
* Incremental complexity; minimal runnable baseline first

## 9. Best Practices

Guidelines distilled from maintaining RL101 in production-like environments.

### Development Environment

- **Use `setup.sh`** for automated provisioning with GPU auto-detection.
- **Prefer `.venv`** when working natively to isolate dependencies.
- **Containerize for consistency** using the curated Dockerfiles when collaborating or deploying.
- **Install modular requirements**: start with `requirements-base.txt`, then add the PyTorch variant that matches your hardware.

```bash
# Base dependencies (always needed)
pip install -r requirements/requirements-base.txt

# Choose a PyTorch flavor
pip install -r requirements/requirements-torch-cpu.txt   # CPU-only
pip install -r requirements/requirements-torch-cuda.txt  # NVIDIA GPU
pip install -r requirements/requirements-torch-rocm.txt  # AMD GPU
```

### Code Quality

**Reproducibility**

- Seed NumPy, PyTorch, and Gymnasium environments whenever deterministic comparisons matter.
- Expose a `--seed` flag on new scripts; document stochastic behavior when determinism is infeasible.

**Logging & Output**

- Use `rich` for structured console output instead of plain `print`.
- Capture core metrics (reward, loss, epsilon/temperature, entropy) and surface them per episode.
- Employ `rich.progress` for long-running loops to track momentum without spamming logs.

**Configuration Management**

- Stick to a CLI-first design with `argparse`.
- Favor explicit flags over hidden configuration files; ship scripts with sane defaults.
- Craft comprehensive `--help` text so users can discover knobs quickly.

**Code Organization**

- Separate environment setup, learning logic, and evaluation/serving into distinct functions or modules.
- Keep functions focused; add type hints where it aids readability.
- Include top-of-file docstrings that state the goal and show a sample command.

**Dependency Management**

- Guard optional imports and fail gracefully with actionable guidance:

```python
try:
    import torch
except ImportError:
    console.print("[red]PyTorch required. Install: pip install torch[/red]")
    sys.exit(1)
```

**Incremental Complexity**

- Start from a minimal working agent before layering advanced tricks.
- Add enhancements one at a time, validating behavior (and performance) at each step.
- Capture the “why” in comments—especially when you diverge from textbook algorithms.

### Testing & Validation

- Ship defaults that finish within minutes so contributors can iterate quickly.
- Support tiny dry-run parameters (e.g., `--episodes 5`, `--generations 1`).
- Run smoke tests before commits touching shared code:

```bash
# Run all tests (comprehensive)
python scripts/smoke_test.py

# Quick validation (core only, ~30 seconds)
python scripts/smoke_test.py --core-only

# Skip optional tests (infrastructure, advanced)
python scripts/smoke_test.py --skip-optional

# Test specific groups
python scripts/smoke_test.py --group core
python scripts/smoke_test.py --group deep-rl
python scripts/smoke_test.py --group infrastructure
python scripts/smoke_test.py --group advanced

# Verbose output for debugging
python scripts/smoke_test.py --verbose
```

- When possible, verify changes across CPU, CUDA, and ROCm configurations—Docker images help here.

### Performance Tips

**Native Environment**

- Match your dependency footprint to your hardware; CPU-only wheels keep things lean.
- Isolate work in a virtual environment to dodge global site-packages conflicts.

**Docker Environment**

- Optimize Dockerfiles for layer caching (requirements before code) to speed rebuilds.
- Mount a pip cache volume when rebuilding frequently.
- Stick with minimal base images unless you absolutely need a heavier stack.

**Runtime Optimization**

- Profile before prematurely optimizing; use PyTorch/TensorBoard profilers to find hot spots.
- Prefer vectorized NumPy/PyTorch ops over Python loops.
- Manage GPU memory proactively (`torch.cuda.empty_cache()`) when experimenting with large models.

## 10. GPU & Docker

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

## 11. Dependencies

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
* **Supported:** 3.11 and 3.12 (PyTorch wheels available and tested)
* **Minimum:** 3.11 (earlier versions not supported)
* **Note:** Python 3.13+ is currently unsupported by PyTorch wheels—use Docker or downgrade your interpreter

If a script requires PyTorch and it's missing, it exits with clear guidance.

## 12. Testing & Fast Validation

### Comprehensive Smoke Tests
The project includes an intelligent test suite organized by dependency requirements:

```bash
# Run all tests (4 groups: core, deep-rl, infrastructure, advanced)
python scripts/smoke_test.py

# Quick test - core examples only (no PyTorch, ~30 seconds)
python scripts/smoke_test.py --core-only

# Skip optional tests (faster CI/CD)
python scripts/smoke_test.py --skip-optional

# Test specific groups
python scripts/smoke_test.py --group core           # NumPy-based examples
python scripts/smoke_test.py --group deep-rl        # PyTorch-based RL
python scripts/smoke_test.py --group infrastructure # GPU, distributed, tracking
python scripts/smoke_test.py --group advanced       # Offline RL, RLHF, Dreamer

# Verbose output for debugging
python scripts/smoke_test.py --verbose
```

**Test Groups:**
- **Core Examples**: Multi-armed bandits, tabular RL (no PyTorch required)
- **Deep RL Examples**: DQN, PPO, TD3, SAC, TRPO (requires PyTorch)
- **Infrastructure**: GPU optimization, Ray RLlib, Optuna (optional dependencies; includes the slower `sac_robotic_arm.py` quick-run variant)
- **Advanced Algorithms**: CQL, IQL, Dreamer, RLHF (cutting-edge research)

When you launch the suite it announces which optional dependencies are available (PyTorch, Ray, Optuna) and marks missing ones clearly. Tests that depend on unavailable extras are counted as skipped rather than failed, so you can still get a green run on slim environments. Box2D-driven workloads (e.g., `ppo_lunarlander.py`) auto-skip when `Box2D` isn’t installed, keeping core validation snappy.

### Quick Algorithm Validation
```bash
# Test Phase 1: Core algorithms (2 minutes each)
python modules/module_04_actor_critic/examples/ppo_cartpole.py --episodes 5
python modules/module_04_actor_critic/examples/td3_pendulum.py --episodes 5

# Test Phase 3: Advanced algorithms (1 minute each)
python modules/module_07_operationalization/examples/cql_offline_rl.py --mode generate --dataset-size 1000
python modules/module_07_operationalization/examples/benchmark_suite.py --trials 1 --episodes 2
```

## 13. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| ImportError: torch | Not installed / wrong Python version | `pip install torch` or use Docker |
| Extremely slow training | Running on CPU with large model | Reduce network size / episodes; try GPU container |
| Atari env fails | ROM / ALE dependency missing | Install appropriate gymnasium extras; ensure legal ROM acquisition |
| Non-deterministic returns | Env stochasticity | Set `--seed`, limit parallelism, check gymnasium version |

## 14. Extending the Project
Add a new example script under the appropriate module's `examples/` and follow existing patterns:
* Top docstring: purpose + minimal usage
* `argparse` flags with sane defaults
* Seed handling (`--seed`)
* Clear separation: model definition, experience gathering, update step
* Log episodic reward + key diagnostics (loss, epsilon, entropy, etc.)

## 15. Contributing
See `CONTRIBUTING.md`.
Principles:
* Keep runtimes short by default (fast smoke params)
* Avoid heavy hidden dependencies; guard imports
* Favor clarity over cleverness—this is a teaching repo
* Log roadmap-impacting ideas via GitHub Issues or Discussions so the community can weigh in

## 16. Roadmap

### Completed ✅

#### Core Algorithms
- ✅ Expand DQN to Double/Dueling/Prioritized Replay
- ✅ Add Rainbow Atari example
- ✅ Add policy gradient examples (REINFORCE, Pendulum)
- ✅ Add A2C and SAC examples
- ✅ **Add industry-standard algorithms (PPO, TD3, TRPO)** ⭐ **Phase 1 Complete**
- ✅ Add advanced topics (curiosity, multi-agent)
- ✅ **Add cutting-edge algorithms (CQL, IQL, Dreamer, RLHF)** ⭐ **Phase 3 Complete**

#### Infrastructure & Setup
- ✅ Modular requirements structure (base, CPU, CUDA, ROCm)
- ✅ Automated setup script with GPU auto-detection (`setup.sh`)
- ✅ Optimized Dockerfiles (CPU: python:3.11-slim, CUDA/ROCm: official bases)
- ✅ Enhanced docker-compose.yml with pip caching and proper GPU configs
- ✅ Comprehensive setup documentation (`SETUP.md`)
- ✅ Updated `CONTRIBUTING.md` with development guidelines
- ✅ **Intelligent smoke test suite with dependency-based grouping** ⭐ **Phase 2 Complete**
- ✅ **GPU optimization (vectorized envs, mixed precision)** ⭐ **Phase 2 Complete**
- ✅ **Distributed training (Ray RLlib)** ⭐ **Phase 2 Complete**
- ✅ **Hyperparameter tuning (Optuna)** ⭐ **Phase 2 Complete**
- ✅ **TensorBoard integration** ⭐ **Phase 2 Complete**

### In Progress 🔨
- Docker multi-platform builds (ARM64 support)
- Automated Docker image publishing to registry
- Pre-commit hooks for code quality

### Future Enhancements 🚀

#### Algorithms
- ~~Add more advanced algorithms (PPO, TRPO, TD3)~~ ✅ **DONE (Phase 1)**
- ~~Model-based RL examples (Dreamer)~~ ✅ **DONE (Phase 3)**
- ~~Offline RL (CQL, IQL)~~ ✅ **DONE (Phase 3)**
- ~~RLHF for LLMs~~ ✅ **DONE (Phase 3)**
- MuZero implementation
- Meta-RL and few-shot learning examples
- Hierarchical RL implementations

#### Multi-Agent & Competition
- Expand multi-agent scenarios (competitive environments)
- Self-play implementations
- Communication protocols in multi-agent systems
- Tournament and leaderboard systems

#### Industry Applications
- Add more industry case studies (finance, healthcare, robotics)
- Real-world deployment patterns
- A/B testing frameworks for RL policies
- Cost optimization and budget constraints

#### Tools & Infrastructure
- Enhanced visualization and debugging tools
- ~~Integration with popular RL frameworks (Ray RLlib)~~ ✅ **DONE (Phase 2)**
- ~~Performance benchmarking suite~~ ✅ **DONE (Phase 3)**
- ~~Hyperparameter optimization examples (Optuna, Ray Tune)~~ ✅ **DONE (Phase 2)**
- ~~Distributed training examples~~ ✅ **DONE (Phase 2)**
- Cloud deployment guides (AWS, GCP, Azure)
- Kubernetes production deployment examples

#### Documentation & Learning
- Interactive tutorials and exercises
- Video walkthroughs for each module
- Jupyter notebook variants (optional, for exploratory learning)
- Algorithm comparison benchmarks
- Common pitfalls and debugging guide

## 17. References

- Sutton & Barto: *Reinforcement Learning: An Introduction*
- OpenAI: *Spinning Up in Deep RL*
- Gymnasium documentation
- Stable-Baselines3 documentation
- CleanRL

## 18. License
Licensed under the **Apache License, Version 2.0** (see `LICENSE`). You may use, modify, and distribute this project under the terms of that license. Please retain instructional comments where practical to preserve educational value.

## 19. Citation (Optional)
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

## 20. FAQ
Q: Why no notebooks?  
A: Scripts enforce explicit structure, easier diffing, and production parity. You can still adapt them into notebooks if desired.

Q: Where are pretrained weights?  
A: Intentionally omitted to nudge you to train; add caching if you extend.

Q: How long do examples take?  
A: Baselines aim for a few minutes on CPU; scale episodes upward only after verifying flow.

---
Happy learning & experimenting. PRs welcome!