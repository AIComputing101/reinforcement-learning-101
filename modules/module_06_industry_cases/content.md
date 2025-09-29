# Module 06: Industry Case Studies

## Overview
Real-world RL deployment requires bridging the gap between research algorithms and production systems. This module examines how leading companies apply RL to solve complex business problems while navigating constraints like safety, interpretability, and regulatory compliance. We explore successful industry applications and the engineering patterns that make them work at scale.

## Learning Objectives
- Map RL algorithms to specific industry use cases and ROI metrics
- Navigate the lab-to-production gap with practical deployment strategies
- Design hybrid systems combining RL with traditional control methods
- Understand offline/batch RL for learning from logged data safely
- Implement gradual rollout and A/B testing strategies for RL systems

## Key Industry Applications

### 🏭 Industrial Control & Automation
**Real-World Examples**:
- **Google DeepMind**: Data center cooling optimization (40% energy reduction)
- **Facebook**: Server load balancing and resource allocation
- **Manufacturing**: Predictive maintenance, quality control, supply chain optimization

**Key Challenges**:
- **Safety constraints**: Hard limits that cannot be violated
- **Interpretability**: Need to explain decisions to human operators
- **Non-stationarity**: Equipment degrades, demand patterns change
- **Sample efficiency**: Limited ability to experiment with real systems

### 📊 Marketing & Personalization
**Applications**:
- **Dynamic ad placement**: Real-time bidding optimization (RTB)
- **Recommendation systems**: Sequential recommendation with long-term user engagement
- **Content optimization**: A/B testing with multi-armed bandits
- **Personalized pricing**: Dynamic pricing strategies

**Why RL Excels**:
- **Sequential decisions**: Actions affect future user behavior
- **Personalization**: Adapt to individual user preferences over time
- **Long-term optimization**: Maximize lifetime customer value, not just immediate clicks

### 💰 Finance & Trading
**Use Cases**:
- **Algorithmic trading**: Portfolio optimization with transaction costs
- **Risk management**: Dynamic hedging strategies
- **Fraud detection**: Sequential anomaly detection
- **Credit scoring**: Dynamic limit adjustment

**Production Considerations**:
- **Regulatory compliance**: Must explain trading decisions
- **Risk management**: Hard constraints on maximum loss
- **Latency requirements**: Microsecond decision making
- **Market impact**: Actions affect future market conditions

### 🎮 Gaming & Entertainment
**Breakthrough Applications**:
- **Game AI**: OpenAI Five (Dota 2), AlphaStar (StarCraft II)
- **Procedural content**: Dynamic difficulty adjustment, level generation
- **Player engagement**: Retention optimization through adaptive experiences

## 🏗️ Production RL Architecture Patterns

### Hybrid Control Systems
**Philosophy**: Combine RL with traditional controllers for safety and reliability

```python
class HybridController:
    def __init__(self, rl_agent, fallback_controller, safety_checker):
        self.rl_agent = rl_agent
        self.fallback = fallback_controller
        self.safety_checker = safety_checker

    def act(self, state):
        rl_action = self.rl_agent.act(state)

        # Safety override
        if not self.safety_checker.is_safe(state, rl_action):
            return self.fallback.act(state)

        return rl_action
```

### Offline/Batch RL for Logged Data
**Use Case**: Learn from historical data without online experimentation
**Benefits**: Safe exploration, leverage existing datasets
**Challenges**: Distribution shift, out-of-distribution actions

### Contextual Bandits vs Full RL
**Contextual Bandits**: Single-step decisions with immediate feedback
- **Examples**: Ad placement, content recommendation
- **Advantages**: Simpler, faster, more interpretable

**Full RL**: Sequential decisions with delayed rewards
- **Examples**: Trading strategies, game playing
- **Advantages**: Long-term optimization, temporal credit assignment

## Run the Examples
Energy optimization (stub):
```bash
python modules/module_06_industry_cases/examples/energy_optimization_dqn.py --building-type office --season winter
```

RTB simulation (stub):
```bash
python modules/module_06_industry_cases/examples/realtime_bidding_qlearning.py --budget 10000 --campaigns 5
```

Recommender (stub):
```bash
python modules/module_06_industry_cases/examples/recommender_pg.py --users 1000 --items 500 --diversity-bonus 0.1
```

## Exercises
1) Define clear KPIs and safety constraints for one case study
2) Sketch an offline RL pipeline (dataset, OPE, safe policy improvement)
3) Propose a hybrid controller with guardrails and rollback triggers
4) Design an A/B testing plan and monitoring metrics

## Debugging & Best Practices
- Validate with offline evaluation before shadow/A/B phases
- Enforce constraints and safe defaults; set blast‑radius limits
- Monitor RL‑specific and business metrics; alert on drift
- Favor simple baselines first; add complexity incrementally

## Further Reading
- Real‑world RL challenges (Dulac‑Arnold et al., 2020)
- Offline RL tutorial/review (Levine et al., 2020)
- "Deep RL Doesn’t Work Yet" (Irpan, 2018)
