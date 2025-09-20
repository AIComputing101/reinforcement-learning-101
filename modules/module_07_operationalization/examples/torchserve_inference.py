#!/usr/bin/env python3
"""
TorchServe RL Model Deployment

Demonstrates how to deploy trained reinforcement learning models using TorchServe
for production inference. This example shows how to export PyTorch models to
TorchScript, package them as model archives, and serve them via REST API.

Usage:
    python torchserve_inference.py --export-model --model-path models/dqn_cartpole.pth
    python torchserve_inference.py --create-handler --model-type dqn
    python torchserve_inference.py --package-model --model-name dqn-cartpole
    python torchserve_inference.py --start-server --model-store model_store
    python torchserve_inference.py --test-inference --endpoint http://localhost:8080
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Dict, Any, List, Optional
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import track
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

console = Console()


class DQNNetwork(nn.Module):
    """Sample DQN network for demonstration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    """Sample policy network for demonstration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x)
        return mean, log_std


class ModelExporter:
    """Handles model export to TorchScript and TorchServe packaging."""

    def __init__(self):
        self.model_configs = {
            "dqn": {
                "network_class": DQNNetwork,
                "state_dim": 4,
                "action_dim": 2,
                "input_type": "state",
                "output_type": "q_values"
            },
            "policy": {
                "network_class": PolicyNetwork,
                "state_dim": 8,
                "action_dim": 2,
                "input_type": "state",
                "output_type": "action_distribution"
            }
        }

    def export_to_torchscript(self, model_path: str, model_type: str, output_path: str):
        """Export PyTorch model to TorchScript."""
        if not TORCH_AVAILABLE:
            console.print("[red]PyTorch is required for model export.[/red]")
            return False

        if model_type not in self.model_configs:
            console.print(f"[red]Unknown model type: {model_type}[/red]")
            return False

        config = self.model_configs[model_type]

        try:
            # Create network instance
            network = config["network_class"](config["state_dim"], config["action_dim"])

            # Load trained weights
            if os.path.exists(model_path):
                network.load_state_dict(torch.load(model_path, map_location='cpu'))
                console.print(f"Loaded model weights from {model_path}")
            else:
                console.print(f"[yellow]Model file not found, using random weights for demo[/yellow]")

            # Set to evaluation mode
            network.eval()

            # Create example input
            example_input = torch.randn(1, config["state_dim"])

            # Export to TorchScript
            traced_model = torch.jit.trace(network, example_input)

            # Save TorchScript model
            traced_model.save(output_path)
            console.print(f"[green]Model exported to TorchScript: {output_path}[/green]")

            # Verify the exported model
            loaded_model = torch.jit.load(output_path)
            test_output = loaded_model(example_input)
            console.print(f"Verification successful. Output shape: {test_output.shape}")

            return True

        except Exception as e:
            console.print(f"[red]Error exporting model: {e}[/red]")
            return False

    def create_model_handler(self, model_type: str, handler_path: str):
        """Create TorchServe model handler."""
        config = self.model_configs.get(model_type, self.model_configs["dqn"])

        handler_code = f'''
import torch
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
import json
import logging

logger = logging.getLogger(__name__)

class RLModelHandler(BaseHandler):
    """
    Custom handler for RL models
    """

    def __init__(self):
        super(RLModelHandler, self).__init__()
        self.model = None
        self.device = None
        self.model_type = "{model_type}"
        self.state_dim = {config["state_dim"]}
        self.action_dim = {config["action_dim"]}

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        logger.info("Initializing RL model handler")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_file = context.manifest["model"]["modelFile"]
        model_path = f"{{model_dir}}/{{model_file}}"

        logger.info(f"Loading model from {{model_path}}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param data: List of raw inputs from client
        :return: List of preprocessed inputs
        """
        logger.info(f"Preprocessing data: {{data}}")

        preprocessed_data = []
        for raw_data in data:
            # Parse JSON input
            if isinstance(raw_data, dict):
                input_data = raw_data
            else:
                input_data = json.loads(raw_data.decode('utf-8'))

            # Extract state
            if "state" in input_data:
                state = input_data["state"]
            elif "observation" in input_data:
                state = input_data["observation"]
            else:
                raise ValueError("Input must contain 'state' or 'observation' field")

            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            preprocessed_data.append(state_tensor)

        return preprocessed_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        logger.info("Running inference")

        inference_output = []
        with torch.no_grad():
            for data in model_input:
                output = self.model(data)

                if self.model_type == "dqn":
                    # For DQN, return Q-values and best action
                    q_values = output.cpu().numpy()
                    best_action = int(torch.argmax(output, dim=1).item())
                    result = {{
                        "q_values": q_values.tolist(),
                        "action": best_action,
                        "confidence": float(torch.softmax(output, dim=1).max().item())
                    }}
                elif self.model_type == "policy":
                    # For policy networks, return action distribution
                    mean, log_std = output
                    action = torch.tanh(mean).cpu().numpy()
                    std = torch.exp(log_std).cpu().numpy()
                    result = {{
                        "action": action.tolist(),
                        "mean": mean.cpu().numpy().tolist(),
                        "std": std.tolist()
                    }}
                else:
                    # Generic output
                    result = {{"output": output.cpu().numpy().tolist()}}

                inference_output.append(result)

        return inference_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        logger.info("Postprocessing results")
        return inference_output
'''

        # Write handler to file
        os.makedirs(os.path.dirname(handler_path), exist_ok=True)
        with open(handler_path, 'w') as f:
            f.write(handler_code)

        console.print(f"[green]Model handler created: {handler_path}[/green]")

    def create_model_config(self, model_name: str, config_path: str):
        """Create model configuration file."""
        config = {
            "modelName": model_name,
            "modelVersion": "1.0",
            "modelFile": f"{model_name}.pt",
            "handler": "model_handler.py",
            "runtime": "python",
            "batchSize": 1,
            "responseTimeout": 120,
            "deviceType": "cpu",
            "asyncCommunication": True
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        console.print(f"[green]Model config created: {config_path}[/green]")

    def package_model(self, model_name: str, torchscript_path: str, handler_path: str,
                     output_dir: str = "model_store") -> str:
        """Package model as TorchServe model archive (.mar)."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create temporary directory for packaging
            temp_dir = f"temp_{model_name}"
            os.makedirs(temp_dir, exist_ok=True)

            # Copy files to temp directory
            model_file = f"{model_name}.pt"
            shutil.copy(torchscript_path, os.path.join(temp_dir, model_file))
            shutil.copy(handler_path, os.path.join(temp_dir, "model_handler.py"))

            # Create model config
            config_path = os.path.join(temp_dir, "config.json")
            self.create_model_config(model_name, config_path)

            # Create model archive using torch-model-archiver
            mar_file = f"{model_name}.mar"
            mar_path = os.path.join(output_dir, mar_file)

            cmd = [
                "torch-model-archiver",
                "--model-name", model_name,
                "--version", "1.0",
                "--model-file", model_file,
                "--serialized-file", model_file,
                "--handler", "model_handler.py",
                "--export-path", output_dir,
                "--force"
            ]

            # Run in temp directory
            result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)

            if result.returncode == 0:
                console.print(f"[green]Model archive created: {mar_path}[/green]")

                # Cleanup temp directory
                shutil.rmtree(temp_dir)
                return mar_path
            else:
                console.print(f"[red]Failed to create model archive:[/red]")
                console.print(result.stderr)
                return ""

        except FileNotFoundError:
            console.print("[red]torch-model-archiver not found. Please install TorchServe:[/red]")
            console.print("pip install torchserve torch-model-archiver")
            return ""
        except Exception as e:
            console.print(f"[red]Error packaging model: {e}[/red]")
            return ""


class TorchServeManager:
    """Manages TorchServe server operations."""

    def __init__(self, model_store: str = "model_store", port: int = 8080):
        self.model_store = model_store
        self.port = port
        self.management_port = 8081
        self.server_process = None

    def check_torchserve(self) -> bool:
        """Check if TorchServe is installed."""
        try:
            result = subprocess.run(["torchserve", "--help"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def start_server(self, models: Optional[List[str]] = None) -> bool:
        """Start TorchServe server."""
        if not self.check_torchserve():
            console.print("[red]TorchServe not found. Please install:[/red]")
            console.print("pip install torchserve torch-model-archiver")
            return False

        # Create model store directory
        os.makedirs(self.model_store, exist_ok=True)

        # Build command
        cmd = [
            "torchserve",
            "--start",
            "--model-store", self.model_store,
            "--ts-config", "config.properties"
        ]

        # Add initial models if specified
        if models:
            cmd.extend(["--models"] + models)

        # Create TorchServe config
        self._create_config()

        try:
            console.print("Starting TorchServe server...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                console.print(f"[green]TorchServe started on port {self.port}[/green]")

                # Wait for server to be ready
                if self._wait_for_server():
                    return True
                else:
                    console.print("[red]Server failed to start properly[/red]")
                    return False
            else:
                console.print(f"[red]Failed to start TorchServe:[/red]")
                console.print(result.stderr)
                return False

        except Exception as e:
            console.print(f"[red]Error starting server: {e}[/red]")
            return False

    def stop_server(self):
        """Stop TorchServe server."""
        try:
            subprocess.run(["torchserve", "--stop"], capture_output=True)
            console.print("[green]TorchServe stopped[/green]")
        except Exception as e:
            console.print(f"[red]Error stopping server: {e}[/red]")

    def _create_config(self):
        """Create TorchServe configuration file."""
        config = f"""
inference_address=http://0.0.0.0:{self.port}
management_address=http://0.0.0.0:{self.management_port}
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
job_queue_size=10
number_of_gpu=0
batch_size=1
max_batch_delay=5000
response_timeout=120
unregister_model_timeout=120
decode_input_request=true
"""
        with open("config.properties", 'w') as f:
            f.write(config.strip())

    def _wait_for_server(self, timeout: int = 60) -> bool:
        """Wait for server to be ready."""
        for _ in range(timeout):
            try:
                response = requests.get(f"http://localhost:{self.management_port}/ping")
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False

    def register_model(self, model_name: str) -> bool:
        """Register a model with the running server."""
        try:
            url = f"http://localhost:{self.management_port}/models"
            params = {
                "url": f"{model_name}.mar",
                "initial_workers": 1
            }

            response = requests.post(url, params=params)

            if response.status_code == 200:
                console.print(f"[green]Model {model_name} registered successfully[/green]")
                return True
            else:
                console.print(f"[red]Failed to register model: {response.text}[/red]")
                return False

        except Exception as e:
            console.print(f"[red]Error registering model: {e}[/red]")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List registered models."""
        try:
            response = requests.get(f"http://localhost:{self.management_port}/models")
            if response.status_code == 200:
                return response.json()["models"]
            else:
                console.print(f"[red]Failed to list models: {response.text}[/red]")
                return []
        except Exception as e:
            console.print(f"[red]Error listing models: {e}[/red]")
            return []

    def predict(self, model_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make prediction using deployed model."""
        try:
            url = f"http://localhost:{self.port}/predictions/{model_name}"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                console.print(f"[red]Prediction failed: {response.text}[/red]")
                return None

        except Exception as e:
            console.print(f"[red]Error making prediction: {e}[/red]")
            return None


def test_inference_endpoint(endpoint: str, model_name: str):
    """Test the inference endpoint with sample data."""
    console.print(f"Testing inference endpoint: {endpoint}")

    # Sample test data for different model types
    test_cases = {
        "dqn": {
            "state": [0.1, 0.2, 0.3, 0.4]
        },
        "policy": {
            "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        }
    }

    manager = TorchServeManager()

    # Try different test cases
    for model_type, test_data in test_cases.items():
        console.print(f"\n[bold]Testing {model_type} model:[/bold]")

        result = manager.predict(model_name, test_data)

        if result:
            console.print("[green]Prediction successful:[/green]")
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[red]Failed to get prediction for {model_type}[/red]")


def parse_args():
    parser = argparse.ArgumentParser(description="TorchServe RL Model Deployment")

    # Model export
    parser.add_argument("--export-model", action="store_true",
                       help="Export PyTorch model to TorchScript")
    parser.add_argument("--model-path", default="models/dqn_cartpole.pth",
                       help="Path to trained PyTorch model")
    parser.add_argument("--model-type", choices=["dqn", "policy"], default="dqn",
                       help="Type of RL model")
    parser.add_argument("--output-path", default="models/model.pt",
                       help="Output path for TorchScript model")

    # Handler creation
    parser.add_argument("--create-handler", action="store_true",
                       help="Create TorchServe model handler")
    parser.add_argument("--handler-path", default="handlers/model_handler.py",
                       help="Output path for model handler")

    # Model packaging
    parser.add_argument("--package-model", action="store_true",
                       help="Package model as .mar archive")
    parser.add_argument("--model-name", default="rl-model",
                       help="Name for the model")
    parser.add_argument("--torchscript-path", default="models/model.pt",
                       help="Path to TorchScript model")

    # Server management
    parser.add_argument("--start-server", action="store_true",
                       help="Start TorchServe server")
    parser.add_argument("--stop-server", action="store_true",
                       help="Stop TorchServe server")
    parser.add_argument("--model-store", default="model_store",
                       help="Model store directory")
    parser.add_argument("--port", type=int, default=8080,
                       help="Server port")

    # Model management
    parser.add_argument("--register-model", action="store_true",
                       help="Register model with running server")
    parser.add_argument("--list-models", action="store_true",
                       help="List registered models")

    # Testing
    parser.add_argument("--test-inference", action="store_true",
                       help="Test inference endpoint")
    parser.add_argument("--endpoint", default="http://localhost:8080",
                       help="Inference endpoint URL")

    return parser.parse_args()


def main():
    args = parse_args()

    console.print("[bold green]TorchServe RL Model Deployment[/bold green]")

    exporter = ModelExporter()
    manager = TorchServeManager(args.model_store, args.port)

    if args.export_model:
        console.print("Exporting model to TorchScript...")
        success = exporter.export_to_torchscript(
            args.model_path, args.model_type, args.output_path
        )
        if not success:
            return

    if args.create_handler:
        console.print("Creating model handler...")
        exporter.create_model_handler(args.model_type, args.handler_path)

    if args.package_model:
        console.print("Packaging model archive...")
        handler_path = args.handler_path
        if not os.path.exists(handler_path):
            console.print("Creating handler first...")
            exporter.create_model_handler(args.model_type, handler_path)

        mar_path = exporter.package_model(
            args.model_name, args.torchscript_path, handler_path, args.model_store
        )
        if mar_path:
            console.print(f"Model archive ready: {mar_path}")

    if args.start_server:
        console.print("Starting TorchServe server...")
        success = manager.start_server()
        if success:
            console.print(f"Server running at http://localhost:{args.port}")
            console.print(f"Management API at http://localhost:{manager.management_port}")

    if args.stop_server:
        console.print("Stopping TorchServe server...")
        manager.stop_server()

    if args.register_model:
        console.print(f"Registering model {args.model_name}...")
        manager.register_model(args.model_name)

    if args.list_models:
        console.print("Listing registered models...")
        models = manager.list_models()

        if models:
            table = Table(title="Registered Models")
            table.add_column("Model Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Workers", style="blue")

            for model in models:
                table.add_row(
                    model.get("modelName", "Unknown"),
                    model.get("modelVersion", "Unknown"),
                    model.get("status", "Unknown"),
                    str(model.get("workers", 0))
                )

            console.print(table)
        else:
            console.print("[yellow]No models registered[/yellow]")

    if args.test_inference:
        console.print("Testing inference...")
        test_inference_endpoint(args.endpoint, args.model_name)

    # Show usage examples if no action specified
    if not any([args.export_model, args.create_handler, args.package_model,
               args.start_server, args.stop_server, args.register_model,
               args.list_models, args.test_inference]):

        console.print("\n[bold]Example Usage:[/bold]")
        console.print("1. Export model:")
        console.print("   python torchserve_inference.py --export-model --model-type dqn")

        console.print("\n2. Package model:")
        console.print("   python torchserve_inference.py --package-model --model-name dqn-cartpole")

        console.print("\n3. Start server:")
        console.print("   python torchserve_inference.py --start-server")

        console.print("\n4. Register model:")
        console.print("   python torchserve_inference.py --register-model --model-name dqn-cartpole")

        console.print("\n5. Test inference:")
        console.print("   python torchserve_inference.py --test-inference --model-name dqn-cartpole")

        console.print("\n[bold]Requirements:[/bold]")
        console.print("pip install torchserve torch-model-archiver")


if __name__ == "__main__":
    main()