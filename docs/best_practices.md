# Best Practices

## Development Environment

### Setup
- **Use `setup.sh`** for automated environment setup with GPU auto-detection
- **Virtual environments**: Always use `.venv` for native Python development
- **Docker for consistency**: Use Docker containers for reproducible environments and GPU support
- **Modular requirements**: Install only what you need (base + specific PyTorch variant)

### Requirements Management
```bash
# Base dependencies (always needed)
pip install -r requirements/requirements-base.txt

# Choose PyTorch variant based on hardware
pip install -r requirements/requirements-torch-cpu.txt   # CPU-only
pip install -r requirements/requirements-torch-cuda.txt  # NVIDIA GPU
pip install -r requirements/requirements-torch-rocm.txt  # AMD GPU
```

## Code Quality

### Reproducibility
- **Set seeds** for NumPy, PyTorch, and environment when feasible
- **Expose `--seed` flag** in all scripts for user control
- **Document stochastic behavior** when determinism isn't possible

### Logging & Output
- **Use `rich` console** for structured, readable outputs
- **Log key metrics**: Episode rewards, losses, epsilon/temperature, entropy
- **Avoid print()**: Use `console.print()` with appropriate formatting
- **Progress tracking**: Use `rich.progress` for long-running operations

### Configuration Management
- **CLI-first design**: All configs via `argparse` flags
- **Explicit over implicit**: No hidden config files
- **Sane defaults**: Scripts should run with minimal flags
- **Help text**: Comprehensive `--help` for every script

### Code Organization
- **Separate concerns**: Environment logic ≠ learning logic ≠ evaluation logic
- **Small functions**: Each function does one thing well
- **Type hints**: Use for function signatures (recommended)
- **Docstrings**: Include usage examples at top of scripts

### Dependency Management
- **Guard imports**: Wrap optional dependencies with try/except
- **Clear error messages**: Tell users what to install

```python
try:
    import torch
except ImportError:
    console.print("[red]PyTorch required. Install: pip install torch[/red]")
    sys.exit(1)
```

### Incremental Complexity
- **Start simple**: Minimal working baseline first
- **Add features gradually**: One enhancement at a time
- **Test at each step**: Verify before adding more complexity
- **Document why**: Explain algorithmic choices in comments

## Testing & Validation

### Smoke Testing
- **Fast defaults**: Examples should complete in minutes by default
- **Minimal configs**: Support `--episodes 5` for quick validation
- **Run smoke tests**: `python scripts/smoke_test.py` before commits

### GPU Testing
- **Test all backends**: CPU, CUDA, ROCm when modifying core code
- **Docker validation**: Test Docker builds don't break
- **Verify GPU usage**: Check tensors are on correct device

## Performance

### Native Environment
- **Use appropriate backend**: Don't use CUDA requirements for CPU-only work
- **Lightweight when possible**: CPU-only PyTorch is ~500MB vs multi-GB CUDA
- **Virtual environment isolation**: Prevent dependency conflicts

### Docker Environment
- **Layer caching**: Order Dockerfile commands for maximum cache reuse
- **Shared pip cache**: Use volume mounts to speed up rebuilds
- **Minimal base images**: CPU Dockerfile uses `python:3.11-slim` not CUDA base

### Runtime Optimization
- **Profile before optimizing**: Don't guess bottlenecks
- **Batch operations**: Vectorize NumPy/PyTorch operations
- **GPU memory management**: Clear cache when needed (`torch.cuda.empty_cache()`)

## Documentation

### Code Documentation
- **Top-level docstrings**: Purpose + usage example
- **Inline comments**: Explain "why" not "what"
- **Algorithm references**: Link to papers/resources

### User Documentation
- **Update SETUP.md**: When changing dependencies or setup
- **Update CONTRIBUTING.md**: When changing development workflow
- **Update module content.md**: When adding examples to modules
