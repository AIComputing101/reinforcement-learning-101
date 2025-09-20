# Module 03: Policy Methods (REINFORCE)

## Overview
Learn policy‑based RL and the policy gradient theorem. Implement REINFORCE for CartPole and apply variance‑reduction and entropy regularization.
## Learning Objectives
- Explain policy gradients and their objective
- Implement REINFORCE on CartPole
- Apply variance reduction (baselines, normalization)
## Key Concepts
- Stochastic policy `π_θ(a|s)` with objective `J(θ) = E_π[G_t]`
- Policy gradient theorem: `∇J(θ) = E_π[∇ log π_θ(a|s) · Q^π(s,a)]`
- REINFORCE: Monte‑Carlo return as `Q` estimate (unbiased, high variance)
- Variance reduction: subtract baseline `b(s)≈V^π(s)`; normalize rewards/returns
- Entropy bonus: encourage exploration `L = J − β H(π)`

## Run the Examples
REINFORCE (requires PyTorch; use Docker on Python 3.13):
```bash
python modules/module_03_policy_methods/examples/reinforce_cartpole.py --episodes 500 --seed 0
```
Gaussian policy (stub):
```bash
python modules/module_03_policy_methods/examples/policy_gradient_pendulum.py --env Pendulum-v1
```
## Exercises
1) Add reward normalization and a running‑mean baseline; compare variance and speed
2) Tune entropy coefficient `--entropy-beta` and learning rate
3) Enable rendering and log returns to TensorBoard

## Debugging & Best Practices
- Ensure `log_prob` is used correctly and gradients flow
- Normalize returns; clip gradients if unstable
- Verify episode boundaries; don’t leak trajectories across episodes

## Further Reading
- Sutton & Barto, ch. 13
- Williams (1992) REINFORCE
- PPO (Schulman et al., 2017) for more stable policy gradients
**Experiment**: Try different entropy coefficients. How does exploration change?

### Exercise 3: Baseline Comparison
Implement REINFORCE with and without baseline. Compare:
- Convergence speed
- Gradient variance
- Final performance

### Exercise 4: Policy Visualization
For CartPole, visualize how the policy changes:
- Plot action probabilities over different pole angles
- Show entropy over training time
- Compare stochastic vs deterministic final policies

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

## 📖 Further Reading
- Sutton & Barto Chapter 13: "Policy Gradient Methods"
- Williams (1992): "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (original REINFORCE)
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms" (PPO)

---

**Ready to learn policies directly?** Start with REINFORCE to understand the fundamentals, then explore continuous control! 🎪
