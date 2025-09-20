# Module 04: Actor–Critic Methods (A2C, PPO, SAC)
## Overview
Learn actor–critic methods that pair a policy (actor) with a value estimator (critic) to reduce variance. Run A2C on LunarLander and understand GAE, PPO, and SAC.
## Learning Objectives
- Explain actor–critic and variance reduction vs REINFORCE
- Run A2C on LunarLander (discrete control)
- Understand GAE and PPO’s clipped objective
- Recognize SAC for continuous control with entropy maximization
## Key Concepts
- Actor–critic: actor improves using critic’s advantage/value
- Advantage: `A(s,a) = Q(s,a) − V(s)` reduces variance
- GAE(λ): exponentially‑weighted advantage estimator
- PPO: clipped surrogate objective stabilizes updates
- SAC: optimizes return + entropy for robustness
## Run the Examples
A2C (requires PyTorch and Box2D; use Docker or install Box2D locally):
```bash
python modules/module_04_actor_critic/examples/a2c_lunarlander.py --episodes 1000 --seed 0
```
SAC (stub with SB3/PyBullet notes):
```bash
python modules/module_04_actor_critic/examples/sac_robotic_arm.py --env ReacherBulletEnv-v0
```
## Exercises
1) Tune `--gamma`, `--gae-lambda`, `--entropy-beta`, `--lr`
2) Compare on‑policy (A2C/PPO) vs off‑policy (DQN/SAC)
3) Try PPO via Stable‑Baselines3; compare stability and sample efficiency
## Debugging & Best Practices
- Normalize advantages; clip gradients
- Balance policy and value loss magnitudes
- Entropy too low → collapse; too high → slow learning
- Track episodic returns and value error
## Further Reading
- A2C/A3C (Mnih et al., 2016)
- PPO (Schulman et al., 2017)
- SAC (Haarnoja et al., 2018)
# Module 04: Actor-Critic Methods
