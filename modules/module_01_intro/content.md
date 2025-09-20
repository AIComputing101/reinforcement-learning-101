# Module 01: Introduction to Reinforcement Learning

## Overview
Learn the RL interaction loop, key terminology, the exploration–exploitation tradeoff via multi‑armed bandits, and how to use Gymnasium.

## Learning Objectives
- Explain state, action, reward, policy, value
- Describe and solve multi‑armed bandits
- Interact with Gymnasium environments
- Run and modify CLI examples

## Key Concepts
- Agent–environment loop: observe `s_t`, act `a_t ~ π(·|s_t)`, receive `r_{t+1}`, transition to `s_{t+1}`
- Return: `G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + …`, with `γ ∈ [0,1)`
- Values: `V^π(s) = E[G_t|s_t=s]`, `Q^π(s,a) = E[G_t|s_t=s,a_t=a]`
- Bandits: exploration vs exploitation; ε‑greedy baseline

## Run the Examples
```bash
python modules/module_01_intro/examples/bandit_epsilon_greedy.py --arms 10 --steps 2000 --epsilon 0.1 --seed 0
python modules/module_01_intro/examples/ad_placement.py --ads 5 --steps 5000 --epsilon 0.1 --seed 42
```

Observe: average reward, % optimal‑arm, effect of ε.

## Exercises
1) Epsilon sweep: try `--epsilon {0.01,0.1,0.3}`; compare optimal‑arm %.
2) Non‑stationary bandits: add drifting arm means; use ε‑decay.
3) CartPole random rollouts: measure average reward vs solved threshold (195 over last 100 eps).

## Debugging & Best Practices
- Set seeds for reproducibility where possible
- Log metrics (reward, optimal‑arm %, ε)
- Prefer clear CLI flags and defaults

## Further Reading
- Sutton & Barto, ch. 1–2
- Grokking Deep RL (Manning)
- Deep RL in Action (Manning)

# Module 01: Introduction to Reinforcement Learning

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand what makes RL different from supervised and unsupervised learning
- Grasp the agent-environment interaction loop and key RL terminology
- Master the exploration-exploitation tradeoff through multi-armed bandits
- Know how to set up and use Gymnasium environments
- Implement your first RL agent from scratch

## 📚 What is Reinforcement Learning?

### The Big Picture
Reinforcement Learning is learning through **interaction**. Unlike supervised learning where we learn from labeled examples, or unsupervised learning where we find patterns in data, RL agents learn by taking actions in an environment and receiving feedback.

Think of it like learning to ride a bike:
- **Agent**: You (the learner)
- **Environment**: The world around you (bike, ground, gravity)
- **Actions**: Steering, pedaling, braking
- **Observations**: Your balance, speed, direction
- **Rewards**: Staying upright (+1), falling (-1)

### The RL Framework
```
Agent ←→ Environment
   ↑         ↓
Action    Observation + Reward
```

At each time step `t`:
1. Agent observes state `s_t`
2. Agent takes action `a_t`
3. Environment transitions to new state `s_{t+1}`
4. Environment gives reward `r_{t+1}`
5. Repeat...

### Key RL Terminology
- **Policy (π)**: The agent's strategy for choosing actions given states
- **Value Function (V)**: Expected future rewards from a state
- **Action-Value Function (Q)**: Expected future rewards from a state-action pair
- **Episode**: A complete sequence from start to terminal state
- **Discount Factor (γ)**: How much we value future vs immediate rewards

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

## 🚀 Your First RL Journey

### Start Here
1. **Run the bandit examples** - understand exploration-exploitation
2. **Experiment with ε values** - see how it affects learning
3. **Try different reward distributions** - what happens with more/fewer arms?
4. **Implement your own strategy** - can you beat ε-greedy?

### Think About
- How does the number of arms affect learning time?
- What if rewards are non-stationary (change over time)?
- How would you adapt ε-greedy for real-world applications?

### Next Steps
After mastering bandits, you'll be ready for:
- **Value-based methods**: Learning which states/actions are valuable
- **Policy methods**: Directly learning action strategies
- **Actor-critic**: Combining the best of both worlds

## 📋 Practical Exercises

### Exercise 1: Bandit Experiments
Run the epsilon-greedy bandit with different parameters:
```bash
python examples/bandit_epsilon_greedy.py --arms 5 --epsilon 0.1 --steps 1000
python examples/bandit_epsilon_greedy.py --arms 5 --epsilon 0.3 --steps 1000
```
**Question**: How does epsilon affect the percentage of optimal arm selections?

### Exercise 2: Ad Placement Application
```bash
python examples/ad_placement.py --ads 3 --epsilon 0.1 --steps 2000
```
**Question**: How does this relate to real-world recommendation systems?

### Exercise 3: Environment Exploration
Create a simple script to explore CartPole:
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

## 📖 Further Reading
- Chapter 2 of Sutton & Barto: "Multi-arm Bandits"
- "Algorithms for the Multi-Armed Bandit Problem" (Kuleshov & Precup)
- OpenAI Spinning Up: "Introduction to RL"

---

**Ready to explore?** Run the examples and experiment with different parameters. The journey of a thousand algorithms begins with a single epsilon! 🎯
