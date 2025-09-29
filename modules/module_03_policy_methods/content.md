# Module 03: Policy Methods (REINFORCE)

## Overview
Policy gradient methods directly optimize the agent's decision-making strategy (policy) rather than learning value functions first. This approach enables learning in continuous action spaces and can handle stochastic optimal policies, making it particularly powerful for complex control tasks and environments where the optimal strategy involves randomization.

## Learning Objectives
- Understand policy parameterization and optimization
- Master the policy gradient theorem and its derivation
- Implement REINFORCE algorithm for discrete and continuous control
- Apply variance reduction techniques for stable learning
- Design entropy regularization for exploration-exploitation balance

## Key Concepts

### Policy Parameterization
Unlike value methods that learn "which actions are good," policy methods directly learn "what to do":

#### Stochastic Policy π_θ(a|s)
**Definition**: Probability distribution over actions given state, parameterized by θ
```
π_θ(a|s) = P(A_t = a | S_t = s, θ)
```

**For discrete actions** (e.g., CartPole):
```python
# Softmax policy
π_θ(a|s) = exp(h_θ(s,a)) / ∑_a' exp(h_θ(s,a'))
```

**For continuous actions** (e.g., Pendulum):
```python
# Gaussian policy
π_θ(a|s) = N(μ_θ(s), σ_θ(s))  # Mean and std from neural network
```

### Policy Gradient Theorem: The Foundation
The policy gradient theorem shows how to compute gradients of policy performance:

#### Objective Function
**Goal**: Maximize expected cumulative reward
```
J(θ) = E_π_θ[G_t] = E_π_θ[∑_{k=0}^∞ γ^k R_{t+k+1}]
```

#### The Gradient
**Policy Gradient Theorem**:
```
∇J(θ) = E_π_θ[∇ log π_θ(a|s) · Q^π_θ(s,a)]
```

**Intuitive Interpretation**:
- `∇ log π_θ(a|s)`: Direction that increases probability of action `a` in state `s`
- `Q^π_θ(s,a)`: How good that action actually was
- **Result**: Push policy toward actions that yielded high rewards

### REINFORCE Algorithm
**Strategy**: Use Monte Carlo returns as unbiased estimates of Q^π(s,a)

```python
# REINFORCE pseudocode
for episode in episodes:
    trajectory = collect_episode(π_θ)  # [(s_t, a_t, r_t), ...]

    for t in range(len(trajectory)):
        s_t, a_t = trajectory[t][0], trajectory[t][1]

        # Monte Carlo return from time t
        G_t = sum(γ^k * r_{t+k} for k in range(len(trajectory)-t))

        # Policy gradient step
        grad = ∇ log π_θ(a_t|s_t) * G_t
        θ += α * grad
```

**Properties**:
- **Unbiased**: E[G_t] = Q^π(s,a) exactly
- **High variance**: Single trajectory estimates can be noisy
- **Model-free**: No need for environment dynamics
- **On-policy**: Must use policy π_θ for both action selection and learning

## 🎯 Variance Reduction Techniques

The main challenge with REINFORCE is high variance in gradient estimates. Several techniques help:

### 1. Baseline Subtraction
**Idea**: Subtract a state-dependent baseline b(s) that doesn't change the expected gradient:

```python
# Instead of: grad = ∇ log π_θ(a|s) * G_t
# Use: grad = ∇ log π_θ(a|s) * (G_t - b(s))
```

**Common baselines**:
- **Constant**: b(s) = average return across episodes
- **State-dependent**: b(s) = V^π(s) learned by separate network
- **Moving average**: b(s) = exponentially weighted average of past returns

### 2. Return Normalization
**Standardize returns** within each batch:
```python
returns = (returns - returns.mean()) / (returns.std() + ε)
```

### 3. Entropy Regularization
**Encourage exploration** by adding entropy bonus:
```python
loss = -policy_loss - β * entropy_bonus
entropy_bonus = -∑_a π_θ(a|s) log π_θ(a|s)
```

**Benefits**: Prevents premature convergence to deterministic policies

## 🚀 Advantages of Policy Methods

### ✅ Strengths:
- **Continuous action spaces**: Natural handling of continuous control
- **Stochastic policies**: Can learn randomized strategies when optimal
- **Direct policy optimization**: No need for value function approximation
- **Convergence guarantees**: Local optima guaranteed under mild conditions

### ⚠️ Challenges:
- **Sample efficiency**: Typically requires more samples than value methods
- **High variance**: Gradient estimates can be noisy
- **Local optima**: May converge to suboptimal policies
- **On-policy constraint**: Must collect fresh data for each update

## 🎯 When to Use Policy Methods

### ✅ Ideal For:
- **Continuous action spaces**: Natural parameterization
- **Stochastic optimal policies**: When randomization is beneficial
- **High-dimensional action spaces**: Better scaling than value methods
- **Direct policy optimization**: When you want interpretable policies

### ⚠️ Consider Alternatives When:
- **Sample efficiency critical**: Value methods might be better
- **Discrete actions with small spaces**: Q-learning variants might suffice
- **Need off-policy learning**: Actor-critic or pure value methods

## 📋 Run the Examples
REINFORCE (requires PyTorch; use Docker on Python 3.13):
```bash
python modules/module_03_policy_methods/examples/reinforce_cartpole.py --episodes 500 --seed 0
```
Gaussian policy (stub):
```bash
python modules/module_03_policy_methods/examples/policy_gradient_pendulum.py --env Pendulum-v1
```

## 🔬 Exercises
1. **Variance reduction**: Add reward normalization and a running‑mean baseline; compare variance and speed
2. **Hyperparameter tuning**: Tune entropy coefficient `--entropy-beta` and learning rate
3. **Visualization**: Enable rendering and log returns to TensorBoard
4. **Baseline comparison**: Implement REINFORCE with and without baseline. Compare:
   - Convergence speed
   - Gradient variance
   - Final performance
5. **Policy visualization**: For CartPole, visualize how the policy changes:
   - Plot action probabilities over different pole angles
   - Show entropy over training time
   - Compare stochastic vs deterministic final policies
6. **Entropy analysis**: Try different entropy coefficients. How does exploration change?

## 🧩 Design Patterns

### The Policy Gradient Pattern
1. **Collect episodes** using current policy
2. **Calculate returns** for each step
3. **Compute policy gradients** using REINFORCE
4. **Apply variance reduction** (baselines, normalization)
5. **Update policy parameters**
6. **Repeat** until convergence

### The Continuous Control Pattern
1. **Parameterize policy** as Gaussian distribution
2. **Sample actions** from policy distribution
3. **Compute log-probabilities** for gradient calculation
4. **Use entropy regularization** to maintain exploration
5. **Clip gradients** to prevent instability

## 🔍 Deep Dive Questions
1. **Theoretical**: Why is the policy gradient theorem unbiased but high-variance?
2. **Practical**: How would you adapt REINFORCE for partially observable environments?
3. **Design**: What are the trade-offs between on-policy vs off-policy policy gradients?
4. **Applied**: How might you use policy gradients for automated trading strategies?

## 🛠️ Debugging & Best Practices
- Ensure `log_prob` is used correctly and gradients flow
- Normalize returns; clip gradients if unstable
- Verify episode boundaries; don't leak trajectories across episodes

## 📖 Further Reading
- Sutton & Barto Chapter 13: "Policy Gradient Methods"
- Williams (1992): "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (original REINFORCE)
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms" (PPO)

---

**Ready to learn policies directly?** Start with REINFORCE to understand the fundamentals, then explore continuous control! 🎪