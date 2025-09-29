# Module 04: Actor–Critic Methods (A2C, PPO, SAC)
## Overview
Actor-Critic methods represent the best of both worlds, combining the advantages of value-based and policy-based approaches. The **actor** (policy network) learns what actions to take, while the **critic** (value network) evaluates how good those actions were. This combination significantly reduces the variance problems of pure policy gradient methods while maintaining the ability to handle continuous action spaces.

## Learning Objectives
- Master the actor-critic framework and understand variance reduction mechanisms
- Implement Advantage Actor-Critic (A2C) for discrete control problems
- Understand advanced techniques: GAE, PPO's clipped objective, and trust regions
- Apply Soft Actor-Critic (SAC) for continuous control with maximum entropy RL
- Compare on-policy vs off-policy actor-critic methods

## Key Concepts

### The Actor-Critic Framework
**Core Idea**: Use a critic to provide better estimates of action values, reducing the variance of policy gradient estimates.

#### Traditional REINFORCE Problem:
```python
# High variance gradient estimate
grad = ∇ log π_θ(a|s) * G_t  # G_t has high variance
```

#### Actor-Critic Solution:
```python
# Lower variance using critic's estimate
grad = ∇ log π_θ(a|s) * A_ψ(s,a)  # Advantage from critic
```

### The Advantage Function
**Definition**: How much better an action is compared to the average:
```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

**Intuitive Meaning**:
- **A(s,a) > 0**: Action `a` is better than average in state `s`
- **A(s,a) < 0**: Action `a` is worse than average in state `s`
- **A(s,a) = 0**: Action `a` is exactly average in state `s`

**Why Advantages Reduce Variance**:
- **Baseline effect**: Subtracting V^π(s) reduces magnitude of updates
- **Relative comparison**: Focus on action quality, not absolute rewards
- **Zero mean**: Reduces gradient variance without bias

### Temporal Difference Advantage Estimation
Since we don't know true Q^π and V^π, we estimate advantages using TD methods:

#### One-Step TD Advantage:
```python
A_t = r_t + γ V(s_{t+1}) - V(s_t)
```

#### Generalized Advantage Estimation (GAE):
```python
# Exponentially weighted combination of n-step advantages
A_t^{GAE(λ)} = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
```
where `δ_t = r_t + γ V(s_{t+1}) - V(s_t)` is the TD error.

**GAE Parameters**:
- **λ = 0**: Low variance, high bias (1-step TD)
- **λ = 1**: High variance, low bias (Monte Carlo)
- **λ ∈ (0,1)**: Balanced trade-off

## 🎭 Advanced Actor-Critic Methods

### Proximal Policy Optimization (PPO)
**Problem**: Large policy updates can destabilize training
**Solution**: Clip the objective to prevent destructive updates

#### PPO-Clip Objective:
```python
# Standard policy gradient
ratio = π_new(a|s) / π_old(a|s)
surr1 = ratio * advantage

# Clipped version
surr2 = clip(ratio, 1-ε, 1+ε) * advantage

# Take minimum to be conservative
loss = -min(surr1, surr2)
```

**Benefits**:
- **Stable updates**: Prevents policy collapse from large changes
- **Sample efficiency**: Can reuse data for multiple updates
- **Practical**: Widely used in practice, easy to implement

### Soft Actor-Critic (SAC)
**Philosophy**: Maximize rewards **AND** entropy for robust policies

#### Maximum Entropy Objective:
```python
J(π) = E[∑ γ^t (r_t + α H(π(·|s_t)))]
```
where `H(π(·|s))` is the entropy of the policy.

**Key Properties**:
- **Off-policy**: Can learn from replay buffer
- **Automatic temperature tuning**: Adapts exploration vs exploitation
- **Continuous control**: Designed for continuous action spaces
- **State-of-the-art**: Often achieves best sample efficiency

## 🔄 On-Policy vs Off-Policy Actor-Critic

### On-Policy Methods (A2C, PPO)
- **Data usage**: Must collect fresh data for each update
- **Stability**: More stable, easier to tune
- **Sample efficiency**: Generally less sample efficient
- **Examples**: A2C, PPO, TRPO

### Off-Policy Methods (SAC, DDPG)
- **Data usage**: Can reuse old experiences via replay buffer
- **Complexity**: More complex, harder to tune
- **Sample efficiency**: Generally more sample efficient
- **Examples**: SAC, TD3, DDPG

## 🎯 When to Use Actor-Critic Methods

### ✅ Ideal For:
- **Continuous control**: Natural handling of continuous actions
- **Balanced variance-bias**: Better than pure policy or value methods
- **Complex environments**: Where both policy and value learning help
- **Stable training**: Modern variants (PPO, SAC) are quite robust

### ⚠️ Consider Alternatives When:
- **Simple environments**: Might be overkill for basic problems
- **Discrete actions with large replay buffer**: DQN might be simpler
- **Very sample-limited**: Model-based methods might be better
## 📋 Run the Examples
A2C (requires PyTorch and Box2D; use Docker or install Box2D locally):
```bash
python modules/module_04_actor_critic/examples/a2c_lunarlander.py --episodes 1000 --seed 0
```
SAC (stub with SB3/PyBullet notes):
```bash
python modules/module_04_actor_critic/examples/sac_robotic_arm.py --env ReacherBulletEnv-v0
```

## 🔬 Exercises
1. **Hyperparameter tuning**: Tune `--gamma`, `--gae-lambda`, `--entropy-beta`, `--lr`
2. **Method comparison**: Compare on‑policy (A2C/PPO) vs off‑policy (DQN/SAC)
3. **Library integration**: Try PPO via Stable‑Baselines3; compare stability and sample efficiency
4. **GAE analysis**: Experiment with different λ values in GAE and observe variance-bias tradeoff

## 🔍 Deep Dive Questions
1. **Theoretical**: Why do actor-critic methods typically have lower variance than pure policy gradients?
2. **Practical**: How would you handle very sparse rewards in actor-critic methods?
3. **Design**: What are the trade-offs between synchronous vs asynchronous actor-critic?
4. **Applied**: How might you adapt SAC for discrete action spaces?

## 🛠️ Debugging & Best Practices
- Normalize advantages; clip gradients
- Balance policy and value loss magnitudes
- Entropy too low → collapse; too high → slow learning
- Track episodic returns and value error

## 📖 Further Reading
- A2C/A3C (Mnih et al., 2016)
- PPO (Schulman et al., 2017)
- SAC (Haarnoja et al., 2018)

---

**Ready to balance acting and critiquing?** Actor-critic methods provide the stability and efficiency needed for complex control tasks! 🎭
