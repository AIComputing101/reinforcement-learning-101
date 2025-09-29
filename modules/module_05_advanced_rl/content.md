# Module 05: Advanced RL Topics (Curiosity, ES, Multi‑Agent)

## Overview
Advanced RL topics extend beyond basic algorithms to address fundamental challenges: exploration in sparse-reward environments, gradient-free optimization, and coordination among multiple learning agents. This module introduces cutting-edge techniques that enable RL to tackle complex, real-world problems where traditional methods struggle.

## Learning Objectives
- Master intrinsic motivation and curiosity-driven exploration techniques
- Understand Evolution Strategies as an alternative to gradient-based optimization
- Explore multi-agent RL dynamics, coordination, and emergent behavior
- Survey hierarchical RL, meta-learning, imitation learning, and model-based approaches
- Recognize when and how to apply advanced techniques to specific problem domains

## Key Concepts

### 🔍 Curiosity-Driven Exploration
**Problem**: Traditional RL struggles with sparse rewards and large state spaces
**Solution**: Generate intrinsic rewards based on novelty or surprise

#### Intrinsic Curiosity Module (ICM)
**Core Idea**: Reward the agent for reaching states that are hard to predict
```python
# Forward model: predict next state from current state and action
s_next_pred = forward_model(s_t, a_t)
intrinsic_reward = ||s_{t+1} - s_next_pred||²  # Prediction error
```

#### Random Network Distillation (RND)
**Core Idea**: Use fixed random network to measure state novelty
```python
# Fixed random network and learnable predictor
target_output = random_network(s_t)  # Fixed
predictor_output = predictor_network(s_t)  # Learned
intrinsic_reward = ||target_output - predictor_output||²
```

**Why Curiosity Works**:
- **Automatic exploration**: No need to manually design exploration bonuses
- **Handles sparse rewards**: Provides dense learning signal
- **Scales to complex environments**: Works with high-dimensional observations

### 🧬 Evolution Strategies (ES)
**Philosophy**: Treat policy optimization as black-box optimization problem

#### Basic ES Algorithm:
```python
# Population-based search
for generation in range(num_generations):
    # Generate population by perturbing best policy
    population = [θ + σ * ε_i for ε_i in sample_noise(pop_size)]

    # Evaluate fitness (expected reward) for each candidate
    fitness = [evaluate_policy(θ_i) for θ_i in population]

    # Update toward high-fitness directions
    θ = θ + α * (1/σ) * Σ fitness_i * ε_i
```

**Advantages of ES**:
- **Gradient-free**: Works with non-differentiable objectives
- **Parallelizable**: Each policy evaluation is independent
- **Robust to hyperparameters**: Often easier to tune than gradient methods
- **Exploration in parameter space**: Natural regularization effect

### 🤝 Multi-Agent Reinforcement Learning
**Challenge**: Environment becomes non-stationary when multiple agents learn simultaneously

#### Key Multi-Agent Concepts:
- **Non-stationarity**: Other agents' changing policies affect your environment
- **Credit assignment**: Which agent contributed to team success/failure?
- **Coordination**: How do agents learn to cooperate effectively?
- **Communication**: Should agents share information? How?

#### Centralized Training, Decentralized Execution (CTDE)
**Training phase**: Agents can access global information and other agents' policies
**Execution phase**: Each agent acts independently with local observations

**Example**: Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
- **Centralized critic**: Q(s₁,s₂,...,a₁,a₂,...) uses global state-action
- **Decentralized actor**: π_i(a_i|s_i) uses only local observation

### 🏗️ Advanced RL Paradigms

#### Hierarchical Reinforcement Learning
**Goal**: Learn temporal abstractions and hierarchical policies
- **Options/Skills**: Reusable sub-policies for achieving sub-goals
- **Meta-controller**: High-level policy that selects which skill to use
- **Applications**: Long-horizon tasks, transfer learning

#### Model-Based Reinforcement Learning
**Approach**: Learn environment model, then plan using the model
- **Advantages**: Sample efficiency, can plan ahead, interpretable
- **Challenges**: Model errors compound, complex environments hard to model
- **Modern approaches**: Learn latent dynamics, robust planning

#### Meta-Learning (Learning to Learn)
**Goal**: Learn algorithms that can quickly adapt to new tasks
- **Few-shot learning**: Adapt to new environments with minimal data
- **Algorithm learning**: Learn better RL algorithms themselves
- **Transfer**: Leverage experience across related tasks

## 📋 Run the Examples
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

## 🔬 Exercises
1. **ES tuning**: tune `--sigma`, `--alpha`, and population size
2. **Fitness shaping**: Add fitness normalization or rank‑based shaping
3. **Multi-agent cooperation**: Extend gridworld with cooperative rewards; look for emergent behavior
4. **Hierarchical design**: Sketch an options‑based hierarchy for a long‑horizon task (waypoints + local controller)
5. **Curiosity experimentation**: Compare ICM vs RND on sparse reward environments
6. **Population analysis**: Visualize ES population diversity over generations

## 🔍 Deep Dive Questions
1. **Theoretical**: Why might ES be more robust to hyperparameters than gradient methods?
2. **Practical**: How would you design intrinsic rewards for real-world robotics?
3. **Design**: What are the scaling challenges of multi-agent RL with 100+ agents?
4. **Applied**: How might you apply meta-learning to few-shot robotics tasks?

## 🛠️ Debugging & Best Practices
- **ES**: ensure symmetric perturbations; verify gradient‑estimate sign
- **Curiosity**: balance intrinsic vs extrinsic reward; normalize signals
- **Multi‑agent**: mitigate non‑stationarity with CTDE; evaluate with self‑play
- Prefer reproducible seeds; log returns, exploration metrics, and stability indicators

## 📖 Further Reading
- ICM (Pathak et al., 2017); RND (Burda et al., 2018)
- OpenAI ES (Salimans et al., 2017)
- MADDPG (Lowe et al., 2017)
- Options framework (Sutton et al., 1999)

---

**Ready for the cutting edge?** Advanced RL opens doors to solving the most challenging problems in AI! 🚀
