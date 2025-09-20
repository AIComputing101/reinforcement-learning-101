# Module 07: Operationalization & Deployment

## Overview
Bridge research to production: distributed training, offline RL, serving with TorchServe, and robust monitoring in real environments.

## Learning Objectives
- Run distributed RL on Kubernetes/Ray
- Apply offline RL to fixed datasets
- Serve RL policies with TorchServe (API‑oriented)
- Monitor and operate RL systems in production

## Key Concepts
- Distributed training patterns (A3C, Ray)
- Offline RL pitfalls (distribution shift, CQL)
- Model serving concerns (latency, versioning, rollback)
- Monitoring: RL metrics, alerting, continuous learning

## Run the Examples
```bash
# Kubernetes job (apply the Job manifest in the examples folder)
kubectl apply -f modules/module_07_operationalization/examples/rl-training-job.yaml

# Offline RL (stub)
python modules/module_07_operationalization/examples/offline_rl_batch.py --dataset /data/d4rl_dataset.pkl --algorithm cql

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
