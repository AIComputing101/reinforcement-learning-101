# Contributing

Thanks for your interest in contributing to Reinforcement Learning 101!

## Getting Started

### Setup Development Environment

**Quick start:**
```bash
./setup.sh native          # Recommended for development
```

**Manual setup:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements/requirements-base.txt
pip install -r requirements/requirements-torch-cpu.txt  # Or cuda/rocm

# Run smoke tests
python scripts/smoke_test.py
```

See [SETUP.md](SETUP.md) for detailed environment setup instructions.

## Contribution Guidelines

### Code Style
- **CLI-first design**: All examples must be runnable Python scripts with `argparse`
- **No notebooks**: Maintain reproducibility and version control friendliness
- **Rich logging**: Use `rich` console for structured, readable output
- **Docstrings**: Include usage examples at the top of each script
- **Type hints**: Recommended for function signatures
- **Seed handling**: Expose `--seed` flag for reproducibility

### Script Structure
Every example script should follow this pattern:

```python
#!/usr/bin/env python3
"""
Brief description of what this script does.

Usage:
    python script_name.py --episodes 100 --lr 1e-3
"""
import argparse
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--seed", type=int, default=42)
    # ... more arguments
    args = parser.parse_args()

    # Set seeds
    # Run algorithm
    # Log results with rich.console

if __name__ == "__main__":
    main()
```

### Testing
- **Smoke test compatibility**: Ensure examples run with minimal parameters
- **Fast defaults**: Default configs should complete in minutes, not hours
- **Guard imports**: Wrap optional dependencies (e.g., PyTorch) with try/except

```python
try:
    import torch
except ImportError:
    console.print("[red]PyTorch required. Install with: pip install torch[/red]")
    sys.exit(1)
```

### Documentation
When adding new features:
- Update `docs/roadmap.md` if adding new capabilities
- Add entry to relevant module's `content.md`
- Update `SETUP.md` if changing dependencies or setup process
- Include usage examples in docstrings

### Docker Changes
If modifying Docker setup:
- Test all three backends: `cpu`, `cuda`, `rocm`
- Ensure image builds are optimized (use layer caching)
- Update `docker-compose.yml` if changing service configurations
- Document any new environment variables in `SETUP.md`

### Submitting Changes

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** your changes with smoke tests
4. **Commit** with clear messages (`git commit -m 'Add amazing feature'`)
5. **Push** to your fork (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request with:
   - Clear description of changes
   - How to run/reproduce
   - Expected output or behavior
   - Any new dependencies

### What We Look For
- **Clarity over cleverness**: This is a teaching repository
- **Minimal dependencies**: Avoid adding heavy libraries unless necessary
- **Reproducibility**: Seed handling and deterministic behavior where possible
- **Short runtimes**: Examples should validate quickly by default
- **Educational value**: Code should be readable and well-commented

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- Questions about contributing

Thank you for helping make RL more accessible! 🚀
