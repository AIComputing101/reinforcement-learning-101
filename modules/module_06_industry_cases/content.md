# Module 06: Industry Case Studies

## Overview
Map RL techniques to real business problems and production constraints across energy optimization, RTB, recommenders, and UAV coordination.

## Learning Objectives
- Identify where RL adds ROI in industry scenarios
- Translate domain problems to RL formulations
- Understand production constraints and safe deployment patterns
- Design hybrid controllers and gradual rollout strategies

## Key Concepts
- Lab vs production gap: noise, cost, non‑stationarity, safety, latency
- Batch/Offline RL for logged data, off‑policy evaluation, safe improvement
- Contextual bandits for single‑step decisions; RL for sequential value
- Hybrid control: safe fallbacks, constraints, and business rules

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
