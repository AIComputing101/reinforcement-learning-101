# Module 07: Operationalization & Deployment

## Overview
Operationalizing RL systems requires mastering the infrastructure, tooling, and practices that enable reliable deployment at scale. This module covers the complete MLOps lifecycle for RL: from distributed training and model serving to monitoring and continuous improvement. You'll learn the engineering practices that separate successful production RL from academic experiments.

## Learning Objectives
- Design and implement distributed RL training systems using modern orchestration
- Master offline RL techniques for learning from historical data safely
- Deploy RL models as production APIs with proper versioning and rollback
- Implement comprehensive monitoring, alerting, and debugging for RL systems
- Understand the unique operational challenges of RL compared to supervised learning

## 🏗️ RL Production Infrastructure

### Distributed Training Architecture
**Why Distribute RL Training?**
- **Sample collection**: Parallel environment rollouts speed up data gathering
- **Computation**: Large neural networks require distributed compute
- **Exploration**: Multiple workers can explore different parts of state space

#### Key Patterns:
1. **Asynchronous Actor-Critic (A3C)**: Workers independently update shared parameters
2. **Synchronous Training**: Collect batches from multiple workers, update in sync
3. **Parameter Servers**: Centralized parameter storage with distributed workers
4. **Ray RLlib**: Modern distributed RL framework with fault tolerance

```python
# Ray RLlib distributed training example
import ray
from ray import tune

config = {
    "env": "CartPole-v1",
    "num_workers": 8,  # Parallel rollout workers
    "num_gpus": 1,     # GPUs for neural network training
    "train_batch_size": 4000,
}

tune.run("PPO", config=config)
```

### Model Serving & Deployment
**RL-Specific Serving Challenges**:
- **Stateful policies**: Some policies maintain internal state across actions
- **Latency requirements**: Real-time decision making
- **Action masking**: Invalid actions must be filtered in some environments
- **Exploration vs exploitation**: Production policies are typically deterministic

#### Serving Architecture:
```python
# FastAPI serving example
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load("policy_model.pt")

@app.post("/action")
def get_action(state: Dict):
    with torch.no_grad():
        state_tensor = preprocess(state)
        action_probs = model(state_tensor)
        action = torch.argmax(action_probs).item()
    return {"action": action, "confidence": float(torch.max(action_probs))}
```

## 📊 Monitoring & Observability

### RL-Specific Metrics
Unlike supervised learning, RL requires monitoring the **closed-loop system**:

#### Performance Metrics:
- **Episode rewards**: Primary success metric
- **Episode length**: Efficiency of learned behavior
- **Success rate**: Task completion percentage
- **Policy entropy**: Exploration vs exploitation balance

#### System Health Metrics:
- **Sample efficiency**: Rewards per environment step
- **Training stability**: Value function loss, policy updates
- **Distribution shift**: How much the data distribution changes
- **Action distribution**: Are policies becoming too deterministic?

#### Business Metrics:
- **ROI**: Revenue impact of RL vs baseline
- **User engagement**: Click-through rates, session length
- **Operational costs**: Energy usage, compute resources
- **Safety violations**: Constraint violations, fallback usage

### Continuous Learning & Model Updates
**Challenge**: RL policies can degrade as environments change
**Solutions**:
- **Online learning**: Continuously update policies with new data
- **Periodic retraining**: Retrain on fresh data at regular intervals
- **Distribution monitoring**: Detect when environment shifts occur
- **Gradual rollouts**: Test new policies on small traffic percentages

## 🔒 Safety & Risk Management

### Safe Deployment Patterns
1. **Shadow mode**: Run RL policy alongside production system, compare decisions
2. **Canary deployments**: Gradually increase traffic to new policy
3. **Fallback systems**: Always have a safe baseline to fall back to
4. **Human oversight**: Allow human operators to override RL decisions

### Offline RL for Safe Learning
**Problem**: Online learning can be risky in production
**Solution**: Learn from logged historical data

#### Phase 3 Implementations

**Conservative Q-Learning (CQL)** - Prevents overestimation on out-of-distribution actions:
```python
# CQL penalty: encourage lower Q-values for unseen actions
logsumexp_q = torch.logsumexp(all_q_values, dim=1).mean()  # All actions
dataset_q = q_values.mean()                                 # Dataset actions only
cql_loss = logsumexp_q - dataset_q                         # Conservative penalty

# Total loss with conservative penalty
loss = bellman_loss + alpha * cql_loss
```

**Implicit Q-Learning (IQL)** - Simpler approach using expectile regression:
```python
# 1. Update Q-network (standard Bellman backup)
q_loss = MSE(Q(s,a), r + γ * V(s'))

# 2. Update V-network (expectile regression - learns upper quantile)
v_loss = expectile_loss(V(s) - Q(s, a_dataset), τ=0.7)

# 3. Update policy (advantage-weighted behavioral cloning)
advantage = Q(s,a) - V(s)
weights = exp(advantage / β)
policy_loss = -weights * log π(a|s)
```

**IQL vs CQL:**
- **IQL**: Simpler, fewer hyperparameters, more stable
- **CQL**: More explicit conservatism, better on very large datasets
- **Both**: State-of-the-art offline RL (2024-2025)

**Industry Applications:**
- Autonomous Driving (Waymo): Learn from human driving logs
- Healthcare: Learn treatment policies from patient records
- Recommender Systems: Learn from user interaction logs
- Robotics: Learn from teleoperation demonstrations

### Model-Based RL: World Models

**Problem**: Environment interaction is expensive
**Solution**: Learn a world model, train policy in imagination

**Dreamer Architecture** (inspired by DreamerV3):
```python
# 1. World Model Components
encoder: observation → latent_state
dynamics: (latent_state, action) → next_latent_state
reward_predictor: latent_state → reward
decoder: latent_state → observation  # For reconstruction

# 2. Collect Real Data
states, actions, rewards = collect_from_real_environment()

# 3. Train World Model
latent = encoder(observation)
next_latent_pred, reward_pred = dynamics(latent, action)
next_latent_true = encoder(next_observation)

dynamics_loss = MSE(next_latent_pred, next_latent_true)
reward_loss = MSE(reward_pred, reward)
recon_loss = MSE(decoder(latent), observation)

# 4. Imagine Future Trajectories (Dreaming)
for horizon in range(H):
    action = policy(latent)
    latent, reward = dynamics(latent, action)  # No real environment!
    imagined_rewards.append(reward)

# 5. Train Policy on Imagined Data
train_policy(imagined_rewards)  # Pure model-based learning
```

**Benefits:**
- **Sample Efficiency**: Learn from fewer real interactions
- **Planning**: Simulate futures without environment
- **Transfer**: World model generalizes across tasks
- **Safety**: Test policies in simulation first

**Industry Examples:**
- DeepMind: MuZero (board games), DreamerV3 (continuous control)
- Google: Model-based planning for robotics
- Tesla: World models for autonomous driving prediction

### RLHF: Aligning Language Models

**Problem**: Language models need to follow human preferences
**Solution**: 3-stage RLHF pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**
```python
# Train on human demonstrations
for demo in demonstrations:
    tokens = tokenize(demo)
    loss = CrossEntropy(model(tokens[:-1]), tokens[1:])  # Next token prediction
```

**Stage 2: Reward Model Training**
```python
# Learn from human preferences (A vs B comparisons)
r_A = reward_model(text_A)
r_B = reward_model(text_B)

# Bradley-Terry model: P(B > A) = σ(r_B - r_A)
loss = -log_sigmoid(r_preferred - r_rejected)
```

**Stage 3: PPO with KL Penalty**
```python
# Generate text with policy
text = policy.generate(prompt)
reward = reward_model(text)

# KL penalty prevents reward hacking
kl_penalty = KL(policy || reference_policy)
final_reward = reward - β * kl_penalty

# Optimize with PPO
ppo_loss = PPO_objective(policy, final_reward)
```

**Why KL Penalty?**
- Without: Policy exploits reward model (generates nonsense with high reward)
- With: Policy stays close to reference (only conservative improvements)

**Industry Examples:**
- OpenAI: ChatGPT alignment
- Anthropic: Claude Constitutional AI
- Meta: LLaMA2-Chat
- Google: Bard RLHF

## Run the Examples

### Phase 3: Advanced Algorithms (2024-2025)

**Offline RL - CQL (Conservative Q-Learning)**
```bash
# Generate offline dataset
python modules/module_07_operationalization/examples/cql_offline_rl.py \
    --mode generate --dataset-path data/cartpole_medium.pkl

# Train CQL on offline dataset
python modules/module_07_operationalization/examples/cql_offline_rl.py \
    --mode train --dataset-path data/cartpole_medium.pkl

# Compare CQL vs Behavioral Cloning
python modules/module_07_operationalization/examples/cql_offline_rl.py \
    --mode compare --dataset-path data/cartpole_medium.pkl
```

**Offline RL - IQL (Implicit Q-Learning)**
```bash
# Train IQL (simpler and more stable than CQL)
python modules/module_07_operationalization/examples/iql_offline_rl.py \
    --mode train --dataset-path data/cartpole_medium.pkl

# Compare IQL vs CQL vs BC
python modules/module_07_operationalization/examples/iql_offline_rl.py \
    --mode compare --dataset-path data/cartpole_medium.pkl
```

**Model-Based RL - Dreamer**
```bash
# Train world model and policy (CartPole)
python modules/module_07_operationalization/examples/dreamer_model_based.py \
    --env CartPole-v1 --episodes 200

# Continuous control (Pendulum)
python modules/module_07_operationalization/examples/dreamer_model_based.py \
    --env Pendulum-v1 --episodes 100 --imagine-horizon 20
```

**RLHF for Language Models**
```bash
# Train RLHF pipeline (educational character-level example)
python modules/module_07_operationalization/examples/rlhf_text_generation.py \
    --task sentiment --iterations 100
```

**Benchmark Suite**
```bash
# Compare algorithms on CartPole
python modules/module_07_operationalization/examples/benchmark_suite.py \
    --env CartPole-v1 --algorithms dqn ppo random --trials 5

# Save results to JSON
python modules/module_07_operationalization/examples/benchmark_suite.py \
    --env CartPole-v1 --output results.json
```

### Phase 2: Infrastructure (Production-Ready)

**Distributed Training with Ray RLlib**
```bash
# Distributed PPO training
python modules/module_07_operationalization/examples/ray_distributed_ppo.py \
    --num-workers 4 --iterations 100

# With hyperparameter tuning
python modules/module_07_operationalization/examples/ray_distributed_ppo.py \
    --tune --num-samples 4
```

**Hyperparameter Tuning with Optuna**
```bash
# Automated hyperparameter optimization
python modules/module_07_operationalization/examples/hyperparameter_tuning_optuna.py \
    --n-trials 50 --n-jobs 4
```

### Legacy Examples

```bash
# Kubernetes job (apply the Job manifest in the examples folder)
kubectl apply -f modules/module_07_operationalization/examples/rl-training-job.yaml

# TorchServe (handler/client demo)
python modules/module_07_operationalization/examples/torchserve_inference.py --create-archive
```

## Exercises
1) Author a Job YAML for your trainer and set GPU requests/limits
2) Implement a minimal CQL penalty term in the offline stub
3) Package a scripted policy and serve via TorchServe; test latency
4) Define production SLAs and alerts for an RL service

## Debugging & Best Practices
- Set resource requests/limits; prefer GPU quota isolation
- Validate with off‑policy evaluation before live traffic
- Use canary/A‑B deploys with automatic rollback on KPI regressions
- Monitor latency, action distributions, reward drift, and error rates

## Further Reading
- Kubernetes docs for ML workloads
- TorchServe docs
- Levine et al. (2020): Offline RL tutorial

---

## 🚀 From Research to Production

### The Deployment Challenge
Moving RL from research to production involves unique challenges:

| Research Phase | Production Phase |
|----------------|------------------|
| **Single machine training** | **Distributed, scalable training** |
| **Perfect simulation** | **Real-world data and constraints** |
| **Unlimited exploration** | **Safe, controlled exploration** |
| **Static environments** | **Dynamic, evolving systems** |
| **Academic metrics** | **Business KPIs and SLAs** |

### Production RL Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Training Infra │───▶│ Model Serving   │
│ • Logs          │    │ • Kubernetes    │    │ • TorchServe    │
│ • Simulators    │    │ • Ray Cluster   │    │ • API Gateway   │
│ • A/B Tests     │    │ • GPUs/TPUs     │    │ • Load Balancer │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │ Experiment      │    │ Feature Store   │
│ • Metrics       │    │ Tracking        │    │ • Real-time     │
│ • Alerting      │    │ • MLflow        │    │ • Batch         │
│ • Dashboards    │    │ • Weights&Biases│    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ☸️ Distributed Training with Kubernetes

### Kubernetes for RL Workloads
**Why Kubernetes for RL?**
- **Resource management**: Dynamic GPU allocation
- **Scalability**: Scale workers based on training needs
- **Fault tolerance**: Restart failed training jobs
- **Multi-tenancy**: Share cluster resources efficiently

### RL Training Job Manifest
```yaml
# rl-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: dqn-training-job
spec:
  parallelism: 4  # Number of parallel workers
  template:
    spec:
      containers:
      - name: rl-trainer
        image: your-registry/rl-trainer:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        ---

        ## 🚀 From Research to Production
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc
      restartPolicy: OnFailure
```

### Distributed A3C Implementation
```python
# distributed_a3c.py
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group
import os

class DistributedA3C:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.setup_distributed()

    def setup_distributed(self):
        # Initialize process group for distributed training
        init_process_group(
            backend='nccl',  # Use NCCL for GPU communication
            init_method='env://',  # Use environment variables
            rank=self.rank,
            world_size=self.world_size
        )

    def train_worker(self, global_model, optimizer, env_name):
        # Each worker trains on its own environment instance
        local_model = copy.deepcopy(global_model)
        env = gym.make(env_name)

        for episode in range(self.max_episodes):
            # Collect experience
            states, actions, rewards = self.collect_experience(env, local_model)

            # Compute gradients
            loss = self.compute_loss(states, actions, rewards)
            gradients = torch.autograd.grad(loss, local_model.parameters())

            # Synchronize gradients across workers
            for i, grad in enumerate(gradients):
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                grad /= self.world_size

            # Update global model
            optimizer.step()

            # Periodically sync local model with global
            if episode % self.sync_freq == 0:
                self.sync_models(local_model, global_model)
```

### Ray for RL Workloads
```python
# ray_distributed_training.py
import ray
from ray.rllib.algorithms.ppo import PPO

# Initialize Ray cluster
ray.init(address="ray://head-node:10001")

# Configure distributed training
config = {
    "env": "CartPole-v1",
    "num_workers": 8,          # Number of parallel workers
    "num_gpus": 1,             # GPUs for the trainer
    "num_gpus_per_worker": 0,  # GPUs per worker
    "train_batch_size": 8000,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
}

# Create and train the algorithm
trainer = PPO(config=config)

for i in range(1000):
    result = trainer.train()

    # Save checkpoint every 100 iterations
    if i % 100 == 0:
        checkpoint = trainer.save("/models/ppo_checkpoint")
        print(f"Checkpoint saved at {checkpoint}")
```

## 🎮 Offline Reinforcement Learning

### Learning from Fixed Datasets
**When to use Offline RL:**
- Historical data available but no environment interaction
- Safety-critical domains (healthcare, autonomous vehicles)
- Expensive data collection (robotics, real-world experiments)
- Legal/ethical constraints on exploration

### Offline RL Challenges
1. **Distribution shift**: Training data ≠ policy data
2. **Extrapolation error**: Q-function overestimates unseen actions
3. **Coverage**: Dataset might not cover important states
4. **Evaluation**: Can't easily test policy performance

### Conservative Q-Learning (CQL)
```python
class CQL:
    def __init__(self, state_dim, action_dim, alpha=1.0):
        self.q_network = QNetwork(state_dim, action_dim)
        self.alpha = alpha  # Conservatism coefficient

    def cql_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Standard Q-learning loss
        q_values = self.q_network(states, actions)
        targets = rewards + self.gamma * self.target_q(next_states) * (1 - dones)
        bellman_loss = F.mse_loss(q_values, targets.detach())

        # CQL penalty: encourage lower Q-values for out-of-distribution actions
        all_actions = self.sample_actions(states)  # Sample many actions
        q_all = self.q_network(states, all_actions)
        q_dataset = self.q_network(states, actions)

        cql_penalty = torch.logsumexp(q_all, dim=1) - q_dataset
        cql_loss = self.alpha * cql_penalty.mean()

        return bellman_loss + cql_loss

    def train_on_dataset(self, dataset):
        for batch in dataset:
            loss = self.cql_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

### Behavioral Cloning Plus
```python
def offline_training_pipeline(dataset_path, model_path):
    # Load offline dataset
    dataset = load_dataset(dataset_path)

    # Train behavioral cloning baseline
    bc_policy = BehavioralCloning()
    bc_policy.train(dataset)

    # Improve with offline RL
    offline_agent = CQL()
    offline_agent.train_on_dataset(dataset)

    # Evaluate both approaches
    bc_performance = evaluate_policy(bc_policy, simulator)
    cql_performance = evaluate_policy(offline_agent, simulator)

    # Save best model
    if cql_performance > bc_performance:
        torch.save(offline_agent, model_path)
    else:
        torch.save(bc_policy, model_path)
```

## 🚢 Model Serving with TorchServe (API-oriented deployment)

### TorchServe for RL Models
**Why TorchServe?**
- **Production-ready**: Built for high-throughput serving
- **Model versioning**: A/B test different policies
- **Monitoring**: Built-in metrics and logging
- **Scalability**: Auto-scaling based on load

### Model Archive Creation
```python
# model_handler.py
import torch
from ts.torch_handler.base_handler import BaseHandler

class RLPolicyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, context):
        """Initialize the model for inference"""
        manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load the RL policy
        self.model = torch.jit.load(f"{model_dir}/policy.pt")
        self.model.eval()

    def preprocess(self, data):
        """Preprocess input data"""
        # Convert input to tensor
        observations = torch.tensor([item["observation"] for item in data])
        return observations

    def inference(self, data):
        """Run inference on preprocessed data"""
        with torch.no_grad():
            # Get action probabilities or values
            outputs = self.model(data)

            # Sample actions for stochastic policies
            if hasattr(outputs, 'sample'):
                actions = outputs.sample()
            else:
                actions = outputs.argmax(dim=-1)

        return actions

    def postprocess(self, data):
        """Postprocess inference results"""
        return [{"action": int(action.item())} for action in data]
```

### Deployment Configuration
```bash
# Create model archive
torch-model-archiver \
    --model-name rl_policy \
    --version 1.0 \
    --model-file policy_model.py \
    --serialized-file policy.pt \
    --handler model_handler.py \
    --export-path model_store/

# Start TorchServe
torchserve \
    --start \
    --model-store model_store \
    --models rl_policy=rl_policy.mar \
    --ts-config config.properties
```

### Production Inference API
```python
# inference_client.py
import requests
import numpy as np

class RLInferenceClient:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def get_action(self, observation):
        """Get action from deployed RL policy"""
        payload = {
            "observation": observation.tolist()
        }

        response = requests.post(
            f"{self.endpoint_url}/predictions/rl_policy",
            json=payload,
            timeout=0.1  # 100ms timeout for real-time systems
        )

        if response.status_code == 200:
            return response.json()["action"]
        else:
            # Fallback to safe default action
            return self.safe_default_action()

    def batch_inference(self, observations):
        """Batch inference for higher throughput"""
        payload = [{"observation": obs.tolist()} for obs in observations]

        response = requests.post(
            f"{self.endpoint_url}/predictions/rl_policy",
            json=payload
        )

        return [item["action"] for item in response.json()]
```

## 📊 Monitoring & Observability

### RL-Specific Metrics
```python
# rl_monitoring.py
import wandb
from prometheus_client import Counter, Histogram, Gauge

class RLMonitoring:
    def __init__(self):
        # Prometheus metrics
        self.episode_reward = Histogram('rl_episode_reward', 'Episode reward distribution')
        self.episode_length = Histogram('rl_episode_length', 'Episode length distribution')
        self.exploration_rate = Gauge('rl_exploration_rate', 'Current exploration rate')
        self.policy_entropy = Gauge('rl_policy_entropy', 'Policy entropy')
        self.q_value_estimates = Histogram('rl_q_values', 'Q-value estimates')

        # Training metrics
        self.training_loss = Histogram('rl_training_loss', 'Training loss')
        self.gradient_norm = Histogram('rl_gradient_norm', 'Gradient norm')

        # Production metrics
        self.inference_latency = Histogram('rl_inference_latency_seconds', 'Inference latency')
        self.action_distribution = Counter('rl_actions_total', 'Action distribution', ['action'])

    def log_training_metrics(self, episode_data):
        """Log training-specific metrics"""
        reward = sum(episode_data['rewards'])
        length = len(episode_data['rewards'])

        self.episode_reward.observe(reward)
        self.episode_length.observe(length)

        # Log to Weights & Biases
        wandb.log({
            'episode_reward': reward,
            'episode_length': length,
            'exploration_rate': episode_data['epsilon'],
            'policy_entropy': episode_data['entropy']
        })

    def log_inference_metrics(self, action, latency):
        """Log production inference metrics"""
        self.inference_latency.observe(latency)
        self.action_distribution.labels(action=str(action)).inc()

    def check_policy_degradation(self, current_metrics, baseline_metrics):
        """Alert on significant policy performance degradation"""
        reward_drop = (baseline_metrics['reward'] - current_metrics['reward']) / baseline_metrics['reward']

        if reward_drop > 0.1:  # 10% drop threshold
            self.send_alert(f"Policy performance dropped by {reward_drop:.2%}")

    def send_alert(self, message):
        """Send alert to monitoring system"""
        # Integration with alerting systems (PagerDuty, Slack, etc.)
        pass
```

### Production Monitoring Dashboard
```python
# dashboard_config.py
monitoring_dashboard = {
    "training_metrics": [
        "episode_reward_mean",
        "episode_length_mean",
        "training_loss",
        "exploration_rate",
        "policy_entropy"
    ],
    "production_metrics": [
        "inference_latency_p99",
        "action_distribution",
        "request_rate",
        "error_rate"
    ],
    "alerts": [
        {
            "name": "High inference latency",
            "condition": "inference_latency_p99 > 100ms",
            "action": "scale_up_serving"
        },
        {
            "name": "Policy performance degradation",
            "condition": "episode_reward_mean < baseline * 0.9",
            "action": "rollback_model"
        }
    ]
}
```

## 🔄 Continuous Learning Pipeline

### MLOps for RL
```python
# continuous_learning.py
class RLMLOpsPipeline:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.data_pipeline = DataPipeline()
        self.training_pipeline = TrainingPipeline()
        self.evaluation_pipeline = EvaluationPipeline()

    def run_pipeline(self):
        """Run the complete ML pipeline"""
        # 1. Data collection and preprocessing
        new_data = self.data_pipeline.collect_recent_data()
        processed_data = self.data_pipeline.preprocess(new_data)

        # 2. Trigger training if enough new data
        if len(processed_data) > self.retrain_threshold:
            new_model = self.training_pipeline.train(processed_data)

            # 3. Evaluate new model
            performance = self.evaluation_pipeline.evaluate(new_model)

            # 4. A/B test new model
            if performance > self.current_baseline:
                self.deploy_ab_test(new_model, traffic_split=0.1)

            # 5. Full deployment if A/B test succeeds
            if self.ab_test_results(new_model).success:
                self.deploy_production(new_model)
                self.model_registry.register(new_model)

    def deploy_ab_test(self, model, traffic_split):
        """Deploy model for A/B testing"""
        # Deploy to subset of traffic
        pass

    def deploy_production(self, model):
        """Deploy model to full production traffic"""
        # Blue-green deployment
        pass
```

## 📋 Practical Exercises

### Exercise 1: Kubernetes Training Job
```bash
# Deploy distributed training job
kubectl apply -f examples/kubernetes_training.yaml

# Monitor job progress
kubectl logs -f job/rl-training-job

# Scale workers dynamically
kubectl scale job rl-training-job --replicas=8
```

### Exercise 2: Offline RL Dataset
```bash
python examples/offline_rl_batch.py \
    --dataset /data/d4rl_dataset.pkl \
    --algorithm cql \
    --eval-episodes 100
```

### Exercise 3: Model Serving Setup
```bash
# Create model archive
python examples/torchserve_inference.py --create-archive

# Start serving
torchserve --start --model-store ./model_store

# Test inference
curl -X POST http://localhost:8080/predictions/rl_policy \
     -H "Content-Type: application/json" \
     -d '{"observation": [0.1, 0.2, 0.3, 0.4]}'
```

## 🔍 Deep Dive Questions
1. **Infrastructure**: How would you handle GPU resource allocation for 100+ concurrent RL training jobs?
2. **Reliability**: What strategies ensure RL model serving maintains <100ms latency during traffic spikes?
3. **Safety**: How do you implement safe rollback mechanisms for RL policies in production?
4. **Economics**: How would you optimize the cost of distributed RL training while maintaining performance?

## 📖 Further Reading
- Kubernetes documentation: "Running ML Workloads"
- TorchServe documentation: "Model Serving Best Practices"
- Levine et al. (2020): "Offline Reinforcement Learning: Tutorial, Review, and Perspectives"
- Google: "MLOps: Continuous delivery and automation pipelines in machine learning"

---

**Ready to ship RL to production?** This module bridges the gap between research breakthroughs and real-world impact! 🌍
