#!/usr/bin/env python3
"""
Kubernetes Distributed RL Training

Demonstrates how to run distributed reinforcement learning training on Kubernetes
with GPU nodes. This example generates Kubernetes Job manifests for distributed
training and provides utilities for job management and monitoring.

Usage:
    python kubernetes_training.py --config small --nodes 2 --generate-manifests
    python kubernetes_training.py --monitor-job rl-training-job-123
"""
from __future__ import annotations
import argparse
import yaml
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess
import os

console = Console()


@dataclass
class TrainingConfig:
    """Configuration for distributed RL training."""
    name: str
    nodes: int
    gpus_per_node: int
    cpu_cores: int
    memory_gb: int
    algorithm: str
    environment: str
    episodes: int
    learning_rate: float
    image: str
    storage_size: str


class KubernetesManager:
    """Manages Kubernetes jobs for RL training."""

    def __init__(self):
        self.namespace = "rl-training"
        self.configs = {
            "small": TrainingConfig(
                name="small-training",
                nodes=2,
                gpus_per_node=1,
                cpu_cores=4,
                memory_gb=16,
                algorithm="dqn",
                environment="CartPole-v1",
                episodes=1000,
                learning_rate=1e-3,
                image="rl-training:cuda",
                storage_size="10Gi"
            ),
            "medium": TrainingConfig(
                name="medium-training",
                nodes=4,
                gpus_per_node=1,
                cpu_cores=8,
                memory_gb=32,
                algorithm="ppo",
                environment="LunarLander-v3",
                episodes=5000,
                learning_rate=3e-4,
                image="rl-training:cuda",
                storage_size="50Gi"
            ),
            "large": TrainingConfig(
                name="large-training",
                nodes=8,
                gpus_per_node=2,
                cpu_cores=16,
                memory_gb=64,
                algorithm="sac",
                environment="BipedalWalker-v3",
                episodes=10000,
                learning_rate=1e-4,
                image="rl-training:cuda",
                storage_size="100Gi"
            )
        }

    def generate_namespace_manifest(self) -> Dict[str, Any]:
        """Generate namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace,
                "labels": {
                    "name": self.namespace,
                    "purpose": "reinforcement-learning"
                }
            }
        }

    def generate_storage_manifest(self, config: TrainingConfig) -> Dict[str, Any]:
        """Generate PersistentVolumeClaim manifest."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{config.name}-storage",
                "namespace": self.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {
                    "requests": {
                        "storage": config.storage_size
                    }
                },
                "storageClassName": "fast-ssd"
            }
        }

    def generate_configmap_manifest(self, config: TrainingConfig) -> Dict[str, Any]:
        """Generate ConfigMap with training parameters."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{config.name}-config",
                "namespace": self.namespace
            },
            "data": {
                "algorithm": config.algorithm,
                "environment": config.environment,
                "episodes": str(config.episodes),
                "learning_rate": str(config.learning_rate),
                "nodes": str(config.nodes),
                "distributed": "true"
            }
        }

    def generate_worker_job_manifest(self, config: TrainingConfig, rank: int) -> Dict[str, Any]:
        """Generate Job manifest for a training worker."""
        job_name = f"{config.name}-worker-{rank}"

        # Environment variables for distributed training
        env_vars = [
            {"name": "RANK", "value": str(rank)},
            {"name": "WORLD_SIZE", "value": str(config.nodes)},
            {"name": "MASTER_ADDR", "value": f"{config.name}-master"},
            {"name": "MASTER_PORT", "value": "23456"},
            {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
            {"name": "ALGORITHM", "valueFrom": {"configMapKeyRef": {"name": f"{config.name}-config", "key": "algorithm"}}},
            {"name": "ENVIRONMENT", "valueFrom": {"configMapKeyRef": {"name": f"{config.name}-config", "key": "environment"}}},
            {"name": "EPISODES", "valueFrom": {"configMapKeyRef": {"name": f"{config.name}-config", "key": "episodes"}}},
            {"name": "LEARNING_RATE", "valueFrom": {"configMapKeyRef": {"name": f"{config.name}-config", "key": "learning_rate"}}}
        ]

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "rl-training",
                    "config": config.name,
                    "rank": str(rank),
                    "component": "worker" if rank > 0 else "master"
                }
            },
            "spec": {
                "backoffLimit": 3,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "rl-training",
                            "config": config.name,
                            "rank": str(rank)
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "rl-trainer",
                            "image": config.image,
                            "command": ["python", "/workspace/scripts/distributed_train.py"],
                            "env": env_vars,
                            "resources": {
                                "requests": {
                                    "nvidia.com/gpu": str(config.gpus_per_node),
                                    "cpu": str(config.cpu_cores),
                                    "memory": f"{config.memory_gb}Gi"
                                },
                                "limits": {
                                    "nvidia.com/gpu": str(config.gpus_per_node),
                                    "cpu": str(config.cpu_cores),
                                    "memory": f"{config.memory_gb}Gi"
                                }
                            },
                            "volumeMounts": [{
                                "name": "training-storage",
                                "mountPath": "/workspace/data"
                            }, {
                                "name": "training-storage",
                                "mountPath": "/workspace/logs",
                                "subPath": "logs"
                            }, {
                                "name": "training-storage",
                                "mountPath": "/workspace/models",
                                "subPath": "models"
                            }]
                        }],
                        "volumes": [{
                            "name": "training-storage",
                            "persistentVolumeClaim": {
                                "claimName": f"{config.name}-storage"
                            }
                        }],
                        "nodeSelector": {
                            "accelerator": "nvidia-tesla-v100"
                        },
                        "tolerations": [{
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule"
                        }]
                    }
                }
            }
        }

    def generate_service_manifest(self, config: TrainingConfig) -> Dict[str, Any]:
        """Generate Service manifest for master node communication."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.name}-master",
                "namespace": self.namespace
            },
            "spec": {
                "selector": {
                    "app": "rl-training",
                    "config": config.name,
                    "rank": "0"
                },
                "ports": [{
                    "port": 23456,
                    "targetPort": 23456,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }

    def generate_all_manifests(self, config_name: str) -> List[Dict[str, Any]]:
        """Generate all required manifests for distributed training."""
        config = self.configs[config_name]
        manifests = []

        # Add namespace
        manifests.append(self.generate_namespace_manifest())

        # Add storage
        manifests.append(self.generate_storage_manifest(config))

        # Add config map
        manifests.append(self.generate_configmap_manifest(config))

        # Add service for master communication
        manifests.append(self.generate_service_manifest(config))

        # Add jobs for each worker (including master at rank 0)
        for rank in range(config.nodes):
            manifests.append(self.generate_worker_job_manifest(config, rank))

        return manifests

    def save_manifests(self, manifests: List[Dict[str, Any]], output_dir: str = "k8s-manifests"):
        """Save manifests to YAML files."""
        os.makedirs(output_dir, exist_ok=True)

        for i, manifest in enumerate(manifests):
            kind = manifest["kind"].lower()
            name = manifest["metadata"]["name"]
            filename = f"{i:02d}-{kind}-{name}.yaml"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)

            console.print(f"Generated: {filepath}")

    def check_kubectl(self) -> bool:
        """Check if kubectl is available."""
        try:
            result = subprocess.run(["kubectl", "version", "--client"],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def apply_manifests(self, manifest_dir: str):
        """Apply manifests to Kubernetes cluster."""
        if not self.check_kubectl():
            console.print("[red]kubectl not found. Please install kubectl and configure cluster access.[/red]")
            return

        try:
            # Apply all manifests
            cmd = ["kubectl", "apply", "-f", manifest_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                console.print(f"[green]Successfully applied manifests from {manifest_dir}[/green]")
                console.print(result.stdout)
            else:
                console.print(f"[red]Failed to apply manifests:[/red]")
                console.print(result.stderr)

        except Exception as e:
            console.print(f"[red]Error applying manifests: {e}[/red]")

    def monitor_jobs(self, config_name: str):
        """Monitor training jobs."""
        if not self.check_kubectl():
            console.print("[red]kubectl not found.[/red]")
            return

        config = self.configs[config_name]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Monitoring training jobs...", total=None)

            while True:
                try:
                    # Get job status
                    cmd = ["kubectl", "get", "jobs", "-n", self.namespace,
                          "-l", f"config={config.name}", "-o", "json"]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        jobs_data = json.loads(result.stdout)

                        table = Table(title=f"Training Jobs Status - {config.name}")
                        table.add_column("Job Name", style="cyan")
                        table.add_column("Status", style="green")
                        table.add_column("Completions", style="yellow")
                        table.add_column("Duration", style="blue")

                        all_completed = True
                        for job in jobs_data["items"]:
                            name = job["metadata"]["name"]
                            status = job.get("status", {})

                            if status.get("succeeded", 0) > 0:
                                job_status = "Completed"
                            elif status.get("failed", 0) > 0:
                                job_status = "Failed"
                                all_completed = False
                            elif status.get("active", 0) > 0:
                                job_status = "Running"
                                all_completed = False
                            else:
                                job_status = "Pending"
                                all_completed = False

                            completions = f"{status.get('succeeded', 0)}/{status.get('completions', 1)}"

                            # Calculate duration
                            start_time = job["status"].get("startTime")
                            completion_time = job["status"].get("completionTime")
                            if start_time and completion_time:
                                duration = "Completed"
                            elif start_time:
                                duration = "Running"
                            else:
                                duration = "Not started"

                            table.add_row(name, job_status, completions, duration)

                        console.clear()
                        console.print(table)

                        if all_completed:
                            console.print("[green]All jobs completed![/green]")
                            break

                    time.sleep(10)  # Check every 10 seconds

                except KeyboardInterrupt:
                    console.print("\n[yellow]Monitoring stopped.[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error monitoring jobs: {e}[/red]")
                    break

    def get_logs(self, config_name: str, rank: int = 0):
        """Get logs from a specific training job."""
        if not self.check_kubectl():
            console.print("[red]kubectl not found.[/red]")
            return

        config = self.configs[config_name]
        job_name = f"{config.name}-worker-{rank}"

        try:
            # Get pod name for the job
            cmd = ["kubectl", "get", "pods", "-n", self.namespace,
                  "-l", f"job-name={job_name}", "-o", "jsonpath={.items[0].metadata.name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout:
                pod_name = result.stdout.strip()

                # Get logs
                cmd = ["kubectl", "logs", "-n", self.namespace, pod_name]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    console.print(f"[green]Logs for {job_name} (pod: {pod_name}):[/green]")
                    console.print(result.stdout)
                else:
                    console.print(f"[red]Failed to get logs: {result.stderr}[/red]")
            else:
                console.print(f"[yellow]No pod found for job {job_name}[/yellow]")

        except Exception as e:
            console.print(f"[red]Error getting logs: {e}[/red]")

    def cleanup_jobs(self, config_name: str):
        """Clean up training jobs and resources."""
        if not self.check_kubectl():
            console.print("[red]kubectl not found.[/red]")
            return

        config = self.configs[config_name]

        try:
            # Delete jobs
            cmd = ["kubectl", "delete", "jobs", "-n", self.namespace,
                  "-l", f"config={config.name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                console.print(f"[green]Cleaned up jobs for {config.name}[/green]")
            else:
                console.print(f"[red]Failed to cleanup jobs: {result.stderr}[/red]")

        except Exception as e:
            console.print(f"[red]Error during cleanup: {e}[/red]")


def create_distributed_training_script():
    """Create a sample distributed training script."""
    script_content = '''#!/usr/bin/env python3
"""
Distributed RL Training Script for Kubernetes

This script runs distributed reinforcement learning training using PyTorch
distributed backend. It's designed to work with the Kubernetes Job manifests.
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import gymnasium as gym
import numpy as np
from rich.console import Console

console = Console()

def init_distributed():
    """Initialize distributed training."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    # Set CUDA device
    torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size

def train_distributed():
    """Main distributed training function."""
    rank, world_size = init_distributed()

    console.print(f"[green]Worker {rank}/{world_size} started[/green]")

    # Get training parameters from environment
    algorithm = os.environ.get("ALGORITHM", "dqn")
    environment = os.environ.get("ENVIRONMENT", "CartPole-v1")
    episodes = int(os.environ.get("EPISODES", "1000"))
    lr = float(os.environ.get("LEARNING_RATE", "1e-3"))

    console.print(f"Training {algorithm} on {environment} for {episodes} episodes")

    # Create environment
    env = gym.make(environment)

    # Simple training loop (placeholder)
    for episode in range(episodes // world_size):
        # Each worker trains on a subset of episodes
        if episode % 100 == 0:
            console.print(f"Worker {rank}: Episode {episode}")

    # Cleanup
    dist.destroy_process_group()
    console.print(f"[green]Worker {rank} completed training[/green]")

if __name__ == "__main__":
    train_distributed()
'''

    # Create scripts directory if it doesn't exist
    os.makedirs("scripts", exist_ok=True)

    with open("scripts/distributed_train.py", "w") as f:
        f.write(script_content)

    console.print("Created distributed training script: scripts/distributed_train.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Kubernetes RL Training Manager")
    parser.add_argument("--config", choices=["small", "medium", "large"],
                       default="small", help="Training configuration")
    parser.add_argument("--generate-manifests", action="store_true",
                       help="Generate Kubernetes manifests")
    parser.add_argument("--apply-manifests", action="store_true",
                       help="Apply manifests to cluster")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor training jobs")
    parser.add_argument("--logs", type=int, metavar="RANK",
                       help="Get logs from specific worker rank")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up training resources")
    parser.add_argument("--create-script", action="store_true",
                       help="Create distributed training script")

    return parser.parse_args()


def main():
    args = parse_args()

    console.print(f"[bold green]Kubernetes RL Training Manager[/bold green]")

    manager = KubernetesManager()

    if args.create_script:
        create_distributed_training_script()
        return

    if args.generate_manifests:
        console.print(f"Generating manifests for {args.config} configuration...")
        manifests = manager.generate_all_manifests(args.config)
        manager.save_manifests(manifests)

        # Show summary
        config = manager.configs[args.config]
        table = Table(title=f"Training Configuration: {args.config}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Nodes", str(config.nodes))
        table.add_row("GPUs per Node", str(config.gpus_per_node))
        table.add_row("CPU Cores", str(config.cpu_cores))
        table.add_row("Memory", f"{config.memory_gb}GB")
        table.add_row("Algorithm", config.algorithm)
        table.add_row("Environment", config.environment)
        table.add_row("Episodes", str(config.episodes))
        table.add_row("Learning Rate", str(config.learning_rate))

        console.print(table)

        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Build and push Docker image:")
        console.print("   docker build -f docker/Dockerfile.cuda -t rl-training:cuda .")
        console.print("   docker tag rl-training:cuda your-registry/rl-training:cuda")
        console.print("   docker push your-registry/rl-training:cuda")
        console.print("\n2. Apply manifests:")
        console.print(f"   python {__file__} --config {args.config} --apply-manifests")
        console.print("\n3. Monitor training:")
        console.print(f"   python {__file__} --config {args.config} --monitor")

    elif args.apply_manifests:
        console.print(f"Applying manifests for {args.config} configuration...")
        manager.apply_manifests("k8s-manifests")

    elif args.monitor:
        console.print(f"Monitoring training jobs for {args.config} configuration...")
        manager.monitor_jobs(args.config)

    elif args.logs is not None:
        console.print(f"Getting logs for worker rank {args.logs}...")
        manager.get_logs(args.config, args.logs)

    elif args.cleanup:
        console.print(f"Cleaning up resources for {args.config} configuration...")
        manager.cleanup_jobs(args.config)

    else:
        console.print("[yellow]No action specified. Use --help for available options.[/yellow]")

        # Show available configurations
        table = Table(title="Available Training Configurations")
        table.add_column("Config", style="cyan")
        table.add_column("Nodes", style="green")
        table.add_column("GPUs", style="yellow")
        table.add_column("Algorithm", style="blue")
        table.add_column("Environment", style="magenta")

        for name, config in manager.configs.items():
            table.add_row(
                name,
                str(config.nodes),
                f"{config.gpus_per_node}/node",
                config.algorithm.upper(),
                config.environment
            )

        console.print(table)


if __name__ == "__main__":
    main()