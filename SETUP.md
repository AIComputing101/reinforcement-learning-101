# Environment Setup Guide

This guide covers setting up the reinforcement learning development environment using either native Python or Docker.

## Quick Start

### Automated Setup (Recommended)

The `setup.sh` script automatically detects your GPU and guides you through setup:

```bash
./setup.sh
```

**Direct modes:**
```bash
./setup.sh native          # Native Python (auto-detect GPU)
./setup.sh docker          # Docker container (auto-detect GPU)
./setup.sh native cpu      # Force CPU-only
./setup.sh docker cuda     # Force CUDA
./setup.sh docker rocm     # Force ROCm (AMD GPUs)
```

---

## Native Python Environment

### Prerequisites
- Python 3.11 or later
- pip and venv
- (Optional) NVIDIA CUDA drivers or AMD ROCm drivers

### Setup Steps

1. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install base dependencies:**
   ```bash
   pip install -r requirements/requirements-base.txt
   ```

3. **Install PyTorch (choose one):**

   **CPU-only (lightweight, ~500MB):**
   ```bash
   pip install -r requirements/requirements-torch-cpu.txt
   ```

   **NVIDIA CUDA (requires CUDA 12.x drivers):**
   ```bash
   pip install -r requirements/requirements-torch-cuda.txt
   ```

   **AMD ROCm (requires ROCm 6.x drivers):**
   ```bash
   pip install -r requirements/requirements-torch-rocm.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Docker Environment

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2
- (Optional) NVIDIA Container Toolkit for CUDA
- (Optional) ROCm drivers for AMD GPUs

### Setup Steps

1. **CPU-only (lightweight, ~500MB base):**
   ```bash
   bash docker/run.sh cpu
   ```

2. **NVIDIA CUDA (requires nvidia-docker):**
   ```bash
   bash docker/run.sh cuda
   ```

3. **AMD ROCm (requires ROCm drivers):**
   ```bash
   bash docker/run.sh rocm
   ```

### Manual Docker Commands

Build and run specific containers:

```bash
# Build
docker compose -f docker/docker-compose.yml build drl-cpu

# Run interactively
docker compose -f docker/docker-compose.yml run --rm drl-cpu

# Run specific example
docker compose -f docker/docker-compose.yml run --rm drl-cpu \
  python modules/module_01_intro/examples/bandit_epsilon_greedy.py
```

---

## Verify Setup

### Check Python Environment
```bash
python --version
pip list | grep -E "(torch|numpy|gymnasium|rich)"
```

### Check GPU Availability

**NVIDIA CUDA:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**AMD ROCm:**
```bash
rocminfo
python -c "import torch; print(torch.cuda.is_available())"  # ROCm uses CUDA API
```

### Run Sample Example
```bash
# Simple bandit (NumPy only, works everywhere)
python modules/module_01_intro/examples/bandit_epsilon_greedy.py --arms 5 --steps 1000

# Deep RL example (requires PyTorch)
python modules/module_02_value_methods/examples/dqn_cartpole.py --episodes 50
```

---

## Troubleshooting

### Common Issues

**1. PyTorch not using GPU:**
- Verify drivers: `nvidia-smi` or `rocminfo`
- Check CUDA/ROCm versions match PyTorch requirements
- Ensure correct PyTorch installation (CUDA/ROCm variant)

**2. Import errors in examples:**
- Activate virtual environment: `source .venv/bin/activate`
- Install missing packages: `pip install -r requirements/requirements-base.txt`

**3. Docker GPU not accessible:**
- NVIDIA: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- AMD: Ensure ROCm drivers installed and `/dev/kfd`, `/dev/dri` accessible

**4. Python 3.13 compatibility:**
- PyTorch wheels may not be available for Python 3.13
- Use Python 3.11 or Docker containers instead

### Performance Tips

**Native environment:**
- Use virtual environment to avoid conflicts
- Install only needed PyTorch variant (CPU/CUDA/ROCm)
- Consider using `mamba` instead of `pip` for faster installs

**Docker environment:**
- Use volume caching for faster pip installs (already configured)
- Keep containers running with `docker compose up -d` for repeated use
- Limit GPU visibility: `NVIDIA_VISIBLE_DEVICES=0 docker compose run drl-cuda`

---

## Requirements Files Structure

```
requirements/
├── requirements-base.txt        # Core dependencies (NumPy, Rich, Gymnasium)
├── requirements-torch-cpu.txt   # PyTorch CPU-only
├── requirements-torch-cuda.txt  # PyTorch with CUDA support
└── requirements-torch-rocm.txt  # PyTorch with ROCm support
```

**Base dependencies (all environments):**
- `numpy>=1.24` - Numerical computing
- `rich>=13.7` - CLI formatting
- `gymnasium[classic-control]>=1.0` - RL environments
- `tensorboard>=2.16` - Logging and metrics

**PyTorch variants:**
- CPU: Lightweight, ~500MB, works everywhere
- CUDA: NVIDIA GPUs, requires CUDA 12.x drivers
- ROCm: AMD GPUs, requires ROCm 6.x drivers

---

## Next Steps

After setup is complete:

1. **Run smoke test:**
   ```bash
   python scripts/smoke_test.py
   ```
   The runner reports detected versions of PyTorch, Ray, and Optuna up front, and will gracefully skip optional checks (for example, Box2D-based environments) if those extras aren’t installed. Use `--core-only` or `--skip-optional` for the quickest validation loops.

2. **Try examples:**
   ```bash
   # Module 1: Multi-armed bandits
   python modules/module_01_intro/examples/bandit_epsilon_greedy.py

   # Module 2: Deep Q-Learning
   python modules/module_02_value_methods/examples/dqn_cartpole.py
   ```

3. **Read module content:**
   - Start with `modules/module_01_intro/content.md`
   - Each module has theory + runnable examples

4. **Experiment:**
   - Modify hyperparameters using CLI flags
   - Add `--help` to any example for options
