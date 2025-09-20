# Module 02: Value Methods (Q-Learning, DQN)

## Overview
Learn value functions and Bellman equations, implement tabular Q‑learning, and understand DQN with core stabilization tricks and common extensions.
## Learning Objectives
- Define value functions and Bellman equations
- Implement tabular Q‑learning on CartPole (discretized)
- Understand DQN stabilization (target net, replay, ε‑greedy)
- Recognize DQN variants (Double, Dueling, PER, Rainbow)
## Key Concepts
- Bellman expectation: `V^π(s) = E[r + γ V^π(s')]`
- Bellman optimality: `Q*(s,a) = E[r + γ max_{a'} Q*(s',a')]`
# Sample random batch for learning
batch = replay_buffer.sample(batch_size)
# Breaks correlation between consecutive samples!
```

**Why it helps**:
- Breaks temporal correlations in data
- More data-efficient (reuse experiences)
- Stabilizes learning

#### 2. Target Network
Use a separate, slowly-updated network for targets:
```python
# Main network: Q(s,a; θ)
# Target network: Q(s,a; θ⁻) where θ⁻ is updated less frequently

td_target = r + gamma * max_a Q(s_next, a; θ⁻)  # Use target network
loss = (Q(s, a; θ) - td_target)²
```

**Why it helps**:
- Targets change slowly, providing stability
- Reduces correlation between predictions and targets

### DQN Architecture
```
State (4D vector) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2) → Q-values
```

For Atari: `Image → Conv layers → Flatten → Linear layers → Q-values`

### DQN Training Loop
```python
# Initialize networks
q_network = QNetwork()
target_network = QNetwork()
replay_buffer = ReplayBuffer()

for episode in episodes:
    state = env.reset()
    while not done:
        # Epsilon-greedy action selection
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)

        # Store experience
        replay_buffer.add(state, action, reward, next_state, done)

        # Train on batch
        if len(replay_buffer) > min_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_td_loss(batch, q_network, target_network)
            optimize(loss)

        # Update target network periodically
        if step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
```

## 🚀 DQN Improvements (Preview)

### Double DQN
**Problem**: DQN overestimates Q-values due to max operator bias.
**Solution**: Use main network to select action, target network to evaluate.

### Dueling DQN
**Problem**: Not all states need precise action-value estimates.
**Solution**: Separate value and advantage streams: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

### Prioritized Experience Replay
**Problem**: All experiences aren't equally important.
**Solution**: Sample experiences proportional to their TD error.

### Rainbow DQN
Combines multiple improvements: Double DQN + Dueling + Prioritized Replay + Distributional RL + Noisy Networks + Multi-step learning.

## 🎯 When to Use Value-Based Methods

### ✅ Good for:
- **Discrete action spaces** (or easily discretizable)
- **Sample efficiency** important (reuse data via replay)
- **Off-policy learning** needed
- **Interpretable action selection** (argmax Q-values)

### ❌ Challenging for:
- **Continuous action spaces** (exponential complexity)
- **Very large action spaces** (curse of dimensionality)
- **Stochastic optimal policies** (value methods learn deterministic policies)

## 🔬 Understanding Your Q-Network

### Debugging Questions
1. **Are Q-values reasonable?** (Not too high/low, ordered sensibly)
2. **Is the network learning?** (Loss decreasing, Q-values changing)
3. **Exploration vs exploitation balance?** (Not too much/little exploration)
4. **Target network updates?** (Not too frequent/infrequent)

### Common Failure Modes
- **Overestimation bias**: Q-values grow unrealistically high
- **Instability**: Q-values oscillate wildly
- **Poor exploration**: Agent gets stuck in suboptimal behavior
- **Catastrophic forgetting**: Performance drops suddenly

## 📋 Practical Exercises

### Exercise 1: Tabular Q-Learning
```bash
python examples/q_learning_cartpole.py --episodes 5000 --alpha 0.1 --gamma 0.99
```
**Experiment**: Try different discretization schemes. How does it affect learning?

### Exercise 2: DQN CartPole
```bash
python examples/dqn_cartpole.py --episodes 400 --learning-rate 1e-3 --batch-size 64
```
**Questions**:
- What happens without experience replay?
- How does target network update frequency affect learning?

### Exercise 3: Hyperparameter Sensitivity
Try different values for:
- Learning rate: [1e-4, 1e-3, 1e-2]
- Epsilon decay: [200, 500, 1000]
- Batch size: [32, 64, 128]

Which combinations work best?

### Exercise 4: Visualizing Q-Values
Modify the DQN code to log and plot Q-values over time. Do they stabilize? Grow? Oscillate?

## 🧩 Design Patterns

### The Q-Learning Pattern
1. **Initialize** Q-function (table or network)
2. **Interact** with environment using ε-greedy
3. **Store** experiences (if using replay)
4. **Update** Q-function using Bellman equation
5. **Repeat** until convergence

### The Experience Replay Pattern
1. **Circular buffer** for storage efficiency
2. **Random sampling** to break correlations
3. **Prioritized sampling** for important experiences
4. **Batch updates** for computational efficiency

## 🔍 Deep Dive Questions
1. **Theoretical**: Why does Q-learning converge in the tabular case but not always with function approximation?
2. **Practical**: How would you adapt DQN for a problem with 1000 discrete actions?
3. **Design**: What other ways could you stabilize Q-learning besides target networks?
4. **Applied**: How might you use Q-learning for resource allocation in cloud computing?

## 📖 Further Reading
- Sutton & Barto Chapter 6: "Temporal-Difference Learning"
- Original DQN paper: Mnih et al. "Human-level control through deep reinforcement learning"
- Rainbow paper: Hessel et al. "Rainbow: Combining Improvements in Deep Reinforcement Learning"

---

**Ready to value the future?** Start with tabular Q-learning to build intuition, then dive into the deep end with DQN! 🌊
