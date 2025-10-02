# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

### Quick Start (Recommended)
Use the automated setup script with GPU auto-detection:
```bash
# Interactive mode (auto-detects GPU, lets you choose native/docker)
./setup.sh

# Direct modes
./setup.sh native          # Native Python environment (auto-detect GPU)
./setup.sh docker          # Docker container (auto-detect GPU)
./setup.sh native cpu      # Force CPU-only native setup
./setup.sh docker cuda     # Force CUDA Docker setup
```

### Manual Setup

#### Native Python Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements/requirements-base.txt

# Choose PyTorch backend:
# CPU-only (lightweight, ~500MB)
pip install -r requirements/requirements-torch-cpu.txt

# CUDA (NVIDIA GPUs)
pip install -r requirements/requirements-torch-cuda.txt

# ROCm (AMD GPUs)
pip install -r requirements/requirements-torch-rocm.txt
```

#### Docker Environment
```bash
# CPU-only (lightweight ~500MB base)
bash docker/run.sh cpu

# NVIDIA GPUs (requires nvidia-docker)
bash docker/run.sh cuda

# AMD GPUs (requires ROCm drivers)
bash docker/run.sh rocm
```

## Common Commands

### Running Examples
All examples are CLI-based Python scripts in `modules/*/examples/`:
```bash
# Multi-armed bandit (NumPy only)
python modules/module_01_intro/examples/bandit_epsilon_greedy.py --arms 10 --steps 2000 --epsilon 0.1

# DQN CartPole (requires PyTorch)
python modules/module_02_value_methods/examples/dqn_cartpole.py --episodes 400 --learning-rate 1e-3

# Policy gradient (requires PyTorch)
python modules/module_03_policy_methods/examples/policy_gradient_pendulum.py --episodes 100 --lr 3e-4

# Any example script
python modules/module_XX_name/examples/script_name.py --help  # View available options
```

### Testing & Validation
```bash
# Quick smoke test (lightweight configurations, skips torch if missing)
python scripts/smoke_test.py

# Basic unit tests
python -m unittest tests/test_examples.py

# Run specific examples with minimal parameters for testing
python modules/module_05_advanced_rl/examples/evolutionary_cartpole.py --generations 1 --population 8
```

### Environment Verification
```bash
# Verify PyTorch installation and GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Architecture Overview

**Educational RL Project Design:**
This is a module-based project progressing from RL fundamentals to production deployment:

1. **Module 01**: Introduction & Multi-armed Bandits
2. **Module 02**: Value Methods (Q-learning, DQN)
3. **Module 03**: Policy Methods (REINFORCE)
4. **Module 04**: Actor-Critic (A2C, PPO, SAC)
5. **Module 05**: Advanced Topics (Curiosity, ES, Multi-agent)
6. **Module 06**: Industry Case Studies
7. **Module 07**: Operationalization & Deployment

**Code Structure:**
- Each module has `content.md` (comprehensive theory) + `examples/` (runnable scripts)
- Examples are standalone CLI scripts with Rich logging and argparse
- No notebooks - everything runs from terminal for reproducibility
- Dependency management: Core examples use NumPy only, advanced examples require PyTorch

**Key Implementation Patterns:**
- CLI-first with clear flags and help text
- Rich console logging for structured output
- Seed setting for reproducibility where possible
- Examples guard heavy dependencies with try/except imports
- Minimal runtime configurations for smoke testing

## Module Content Structure

Each module follows a standardized format:
- **Overview**: Brief description and scope
- **Learning Objectives**: Specific goals and outcomes
- **Key Concepts**: Core algorithmic and theoretical concepts
- **Run the Examples**: Command examples with key parameters
- **Exercises**: Hands-on modifications and experiments
- **Debugging & Best Practices**: Common issues and solutions

## Dependencies & Compatibility

**Requirements Structure:**
- `requirements/requirements-base.txt`: Core deps (NumPy, Rich, Gymnasium, TensorBoard)
- `requirements/requirements-torch-cpu.txt`: PyTorch CPU-only (~500MB)
- `requirements/requirements-torch-cuda.txt`: PyTorch with CUDA support
- `requirements/requirements-torch-rocm.txt`: PyTorch with ROCm support (AMD GPUs)

**Core Requirements (all modules):**
- `numpy>=1.24`, `rich>=13.7`, `gymnasium[classic-control]>=1.0`

**Deep Learning Requirements:**
- `torch>=2.3` (for modules 2-4, 6-7)
- Backend-specific installations handled by requirements files

**Python Version Compatibility:**
- Recommended: Python 3.11 (best PyTorch wheel availability)
- Python 3.13 may have limited PyTorch wheel support on PyPI
- Docker containers provide consistent environment across all platforms

## Code Conventions

- CLI-first design with argparse and comprehensive help text
- Include docstring with usage examples at top of each script
- Rich console output for structured, readable logs
- Set random seeds when feasible for reproducibility
- Keep configurations explicit via command-line flags
- Guard optional dependencies with try/except and clear error messages
- Maintain short runtime for smoke tests (use minimal episodes/steps)