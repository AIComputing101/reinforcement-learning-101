#!/usr/bin/env python3
"""Lightweight smoke tests for core examples.

Goal: quickly validate that scripts import and execute a tiny run without dependency-heavy workloads.
Use minimal episodes/steps and skip torch-dependent examples if torch missing.
"""
from __future__ import annotations
import subprocess
import sys
import shutil

EXAMPLES = [
    [sys.executable, "modules/module_01_intro/examples/bandit_epsilon_greedy.py", "--steps", "50", "--epsilon", "0.1"],
    [sys.executable, "modules/module_02_value_methods/examples/q_learning_cartpole.py", "--episodes", "5"],
    [sys.executable, "modules/module_05_advanced_rl/examples/evolutionary_cartpole.py", "--generations", "1", "--population", "8"],
    [sys.executable, "modules/module_05_advanced_rl/examples/multiagent_gridworld.py", "--episodes", "5"],
]

TORCH_EXAMPLES = [
    [sys.executable, "modules/module_03_policy_methods/examples/policy_gradient_pendulum.py", "--episodes", "2"],
]

def run(cmd):
    print(f"\n=== Running: {' '.join(cmd)} ===")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {cmd} -> {e}")
        return False
    return True


def main():
    overall_ok = True
    for c in EXAMPLES:
        ok = run(c)
        overall_ok = overall_ok and ok

    # Torch examples (optional)
    try:
        import torch  # noqa: F401
    except Exception:
        print("Skipping torch-dependent examples (torch not available)")
    else:
        for c in TORCH_EXAMPLES:
            ok = run(c)
            overall_ok = overall_ok and ok

    if not overall_ok:
        print("One or more smoke tests failed.")
        sys.exit(1)
    print("All smoke tests passed.")

if __name__ == "__main__":
    main()
