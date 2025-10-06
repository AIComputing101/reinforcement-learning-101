# Module 01: Introduction to Reinforcement Learning

## Overview
Learn the RL interaction loop, key terminology, the exploration–exploitation tradeoff via multi‑armed bandits, and how to use Gymnasium.

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand what makes RL different from supervised and unsupervised learning
- Grasp the agent-environment interaction loop and key RL terminology
- Master the exploration-exploitation tradeoff through multi-armed bandits
- Know how to set up and use Gymnasium environments
- Implement your first RL agent from scratch

## 📚 What is Reinforcement Learning?

### The Big Picture
Reinforcement Learning (RL) is **a machine learning technique that enables agents to make intelligent decisions by learning from experience**. Unlike supervised learning where we learn from labeled examples, or unsupervised learning where we find patterns in data, RL agents learn by taking actions in an environment and receiving feedback through a trial-and-error process.

**Key Distinguishing Features of RL:**
- **No labeled training data required** - agents discover optimal behaviors through interaction
- **Sequential decision making** - actions affect future states and opportunities
- **Delayed rewards** - consequences of actions may not be immediately apparent
- **Long-term strategic planning** - agents must balance immediate and future rewards
- **Adaptability** - can adjust to changing environments and requirements

Think of it like learning to ride a bike:
- **Agent**: You (the learner)
- **Environment**: The world around you (bike, ground, gravity)
- **Actions**: Steering, pedaling, braking
- **Observations**: Your balance, speed, direction
- **Rewards**: Staying upright (+1), falling (-1)

### The RL Framework: Markov Decision Process (MDP)
```
Agent ←→ Environment
   ↑         ↓
Action    Observation + Reward
```

RL operates through a **Markov Decision Process (MDP)**, which provides the mathematical framework for sequential decision making. The MDP interaction cycle:

1. **Agent starts in initial state** `s_0`
2. **Agent observes current state** `s_t`
3. **Agent chooses action** `a_t` based on current state
4. **Agent interacts with environment** by executing the action
5. **Environment transitions to new state** `s_{t+1}`
6. **Environment provides reward** `r_{t+1}` as feedback
7. **Agent updates policy** through experience
8. **Repeat** until episode termination

**Why MDP Matters:**
- **Markov Property**: Future depends only on current state, not history
- **Mathematical Foundation**: Enables theoretical analysis and guarantees
- **Flexible Framework**: Applies to diverse problem domains
- **Optimal Control**: Provides tools for finding optimal policies

## Key Concepts

### Key RL Terminology
- **Agent–environment loop**: observe `s_t`, act `a_t ~ π(·|s_t)`, receive `r_{t+1}`, transition to `s_{t+1}`
- **Return**: `G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + …`, with `γ ∈ [0,1)`
- **Policy (π)**: The agent's strategy for choosing actions given states
- **Value Functions**: `V^π(s) = E[G_t|s_t=s]`, `Q^π(s,a) = E[G_t|s_t=s,a_t=a]`
- **Episode**: A complete sequence from start to terminal state
- **Discount Factor (γ)**: How much we value future vs immediate rewards

### Bandits: exploration vs exploitation; ε‑greedy baseline

## 🤔 RL vs Other ML Paradigms

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|--------|--------------------|-----------------------|------------------------|
| **Data** | Labeled examples | Unlabeled data | Interaction sequences |
| **Goal** | Predict labels | Find patterns | Maximize rewards |
| **Feedback** | Immediate, correct | None | Delayed, scalar |
| **Learning** | From examples | From structure | From experience |

### When to Use RL?
✅ **Use RL when:**
- Actions affect future observations
- No direct supervision available
- Need to balance immediate vs long-term rewards
- Sequential decision making required

❌ **Don't use RL when:**
- You have abundant labeled data for your exact task
- Actions don't affect environment
- Problem can be solved with simpler methods

## 🎰 The Multi-Armed Bandit Problem

Before diving into full RL, let's start with a simpler problem that captures the core challenge: **exploration vs exploitation**.

### The Setup
Imagine you're in a casino with `k` slot machines (arms). Each machine has an unknown probability of paying out. You have `n` coins to spend. **Goal**: Maximize your total winnings.

**The Dilemma:**
- **Exploit**: Play the machine that seems best so far
- **Explore**: Try other machines to see if they're better

This is the **exploration-exploitation tradeoff** - fundamental to all RL!

### Epsilon-Greedy Strategy
A simple but effective approach:
```python
if random() < epsilon:
    action = random_action()    # Explore
else:
    action = best_known_action() # Exploit
```

**Key Insights:**
- High ε: More exploration, slower convergence to best arm
- Low ε: Less exploration, might miss the optimal arm
- Common strategy: Start high, decay ε over time

### Why Bandits Matter
Multi-armed bandits teach us:
1. **Exploration is essential** - even when you think you know the best action
2. **Data efficiency matters** - we can't try everything infinitely
3. **Uncertainty quantification** - confidence in our estimates drives decisions
4. **Online learning** - we learn and act simultaneously

## 🏋️ Working with Gymnasium

Gymnasium is the standard interface for RL environments. Think of it as the "dataset" for RL.

### Basic Environment Interface
```python
import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
```

### Key Concepts
- **Observation Space**: What the agent can perceive
- **Action Space**: What actions the agent can take
- **Reward Signal**: Feedback from environment
- **Episode Termination**: When to reset (goal reached, failure, timeout)

### Popular Environment Types
1. **Classic Control**: CartPole, MountainCar, Pendulum
2. **Atari**: Video games with pixel observations
3. **Robotics**: Manipulation and locomotion tasks
4. **Custom**: Define your own problem domain

## 🌍 Real-World Applications of RL

RL's ability to learn through interaction makes it exceptionally suited for complex, dynamic environments:

### 🤖 Robotics
- **Manipulation**: Learning to grasp and manipulate objects
- **Locomotion**: Walking, running, navigating complex terrain
- **Industrial automation**: Assembly, welding, quality control
- **Autonomous vehicles**: Path planning, obstacle avoidance

### 🎮 Game Strategy & AI
- **Classic games**: Chess, Go, poker (strategic planning)
- **Real-time strategy**: Resource management, tactical decisions
- **Video games**: NPC behavior, procedural content generation

### 💼 Business & Industry
- **Marketing personalization**: Dynamic ad placement, recommendation systems
- **Finance**: Algorithmic trading, portfolio optimization, risk management
- **Supply chain**: Inventory management, logistics optimization
- **Energy**: Grid optimization, demand response, renewable integration

### 🏥 Healthcare & Science
- **Drug discovery**: Molecular design, treatment protocols
- **Personalized medicine**: Treatment recommendations, dosing optimization
- **Medical imaging**: Automated diagnosis, surgical planning

### 🏭 Industrial Control
- **Manufacturing**: Process optimization, quality control, predictive maintenance
- **Chemical processing**: Reaction optimization, safety protocols
- **HVAC systems**: Energy-efficient climate control

**Why RL Excels in These Domains:**
- **Adaptability**: Environments change, RL agents adapt
- **Strategic planning**: Long-term optimization, not just immediate rewards
- **No expert demonstrations needed**: Discovers strategies through exploration
- **Handles uncertainty**: Robust to noisy, unpredictable environments

## 🧠 Mental Models for RL

### The Credit Assignment Problem
When you win a chess game, which moves were actually good? RL faces this challenge constantly - **which past actions led to current rewards?**

### The Bias-Variance Tradeoff in RL
- **High bias, low variance**: Simple policies that are consistently okay
- **Low bias, high variance**: Complex policies that might be great or terrible
- **The sweet spot**: Policies that balance both effectively

### Sample Efficiency
RL is notoriously sample-hungry because:
- Each action only gives one data point
- Exploration can lead to poor rewards
- Credit assignment requires seeing full consequences

## 🚀 The Three Main RL Approaches

Understanding the fundamental approaches helps navigate the RL landscape:

### 1. Model-Free Methods
Learn optimal behavior **without modeling the environment**:

**Value-Based Methods:**
- Learn value functions (Q-learning, DQN)
- **Advantages**: Sample efficient through experience replay, stable
- **Best for**: Discrete action spaces, off-policy learning

**Policy Gradient Methods:**
- Directly learn action policies (REINFORCE, PPO)
- **Advantages**: Handle continuous actions, stochastic policies
- **Best for**: Continuous control, complex action spaces

**Actor-Critic Methods:**
- Combine value and policy learning (A2C, SAC)
- **Advantages**: Lower variance than pure policy methods
- **Best for**: Balancing stability and flexibility

### 2. Model-Based Methods
**Learn environmental model** to predict next state and reward:
- **Advantages**: Can plan ahead, sample efficient
- **Challenges**: Model errors compound, complex environments
- **Applications**: Robotics, resource planning

### 3. Reinforcement Learning From Human Feedback (RLHF)
**Incorporates human input** into the learning process:
- **Use cases**: Language models, complex preference learning
- **Benefits**: Aligns AI behavior with human values
- **Challenges**: Scalability, consistency of human feedback

## 📋 Run the Examples
```bash
# Epsilon-greedy exploration
python modules/module_01_intro/examples/bandit_epsilon_greedy.py --arms 10 --steps 2000 --epsilon 0.1 --seed 0

# Upper Confidence Bound (UCB) exploration
python modules/module_01_intro/examples/bandit_ucb.py --arms 10 --steps 2000 --c 2.0 --seed 0

# Ad placement scenario
python modules/module_01_intro/examples/ad_placement.py --ads 5 --steps 5000 --epsilon 0.1 --seed 42
```

Observe: average reward, % optimal‑arm, effect of ε (epsilon-greedy) vs c (UCB exploration constant).

## 🔬 Exercises
1. **Epsilon sweep**: try `--epsilon {0.01,0.1,0.3}`; compare optimal‑arm %.
2. **Non‑stationary bandits**: add drifting arm means; use ε‑decay.
3. **CartPole random rollouts**: measure average reward vs solved threshold (195 over last 100 eps).
4. **Environment exploration**: Create a simple script to explore CartPole:
   ```python
   import gymnasium as gym
   env = gym.make("CartPole-v1")
   # Observe action/observation spaces, try random actions
   ```

## 🔍 Deep Dive Questions
1. **Philosophical**: Is RL the right model for how humans learn?
2. **Practical**: When might random exploration be better than ε-greedy?
3. **Mathematical**: How would you prove ε-greedy converges to optimal?
4. **Applied**: Design a bandit algorithm for A/B testing

## 🛠️ Debugging & Best Practices
- Set seeds for reproducibility where possible
- Log metrics (reward, optimal‑arm %, ε)
- Prefer clear CLI flags and defaults

## 📖 Further Reading
- Sutton & Barto, ch. 1–2
- "Algorithms for the Multi-Armed Bandit Problem" (Kuleshov & Precup)
- OpenAI Spinning Up: "Introduction to RL"

---

**Next Steps**: After mastering bandits, you'll explore:
- **Module 2**: Value-based methods (Q-learning, DQN)
- **Module 3**: Policy methods (REINFORCE, policy gradients)
- **Module 4**: Actor-critic (A2C, PPO, SAC)
- **Module 5**: Advanced topics including model-based and multi-agent RL

**Ready to explore?** Run the examples and experiment with different parameters. The journey of a thousand algorithms begins with a single epsilon! 🎯