#!/usr/bin/env python3
"""
Lightweight smoke tests for all RL examples.

Goal: Quickly validate that examples import and execute successfully with minimal runtime.
Organized by dependency requirements for easier debugging and selective testing.

Usage:
    python scripts/smoke_test.py                    # Run all tests
    python scripts/smoke_test.py --core-only        # Core examples only (no torch)
    python scripts/smoke_test.py --skip-optional    # Skip optional/slow examples
    python scripts/smoke_test.py --module MODULE    # Test specific module
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestGroup:
    """Group of related tests with metadata."""
    name: str
    description: str
    tests: list[list[str]]
    required: bool = True  # If False, failures don't affect overall status
    requires_torch: bool = False
    requires_optional_deps: bool = False


# =============================================================================
# Test Definitions (organized by module and dependencies)
# =============================================================================

CORE_TESTS = TestGroup(
    name="Core Examples",
    description="Basic examples with minimal dependencies (numpy, gymnasium)",
    required=True,
    tests=[
        # Module 01: Multi-armed Bandits
        [sys.executable, "modules/module_01_intro/examples/bandit_epsilon_greedy.py",
         "--steps", "50", "--epsilon", "0.1"],
        [sys.executable, "modules/module_01_intro/examples/bandit_ucb.py",
         "--steps", "50", "--c", "2.0"],

        # Module 02: Value Methods (tabular)
        [sys.executable, "modules/module_02_value_methods/examples/q_learning_cartpole.py",
         "--episodes", "5"],

        # Module 05: Evolutionary & Multi-agent
        [sys.executable, "modules/module_05_advanced_rl/examples/evolutionary_cartpole.py",
         "--generations", "1", "--population", "8"],
        [sys.executable, "modules/module_05_advanced_rl/examples/multiagent_gridworld.py",
         "--episodes", "5"],
    ]
)

DEEP_RL_TESTS = TestGroup(
    name="Deep RL Examples",
    description="Examples requiring PyTorch for neural network training",
    required=True,
    requires_torch=True,
    tests=[
        # Module 02: Value Methods (DQN)
        [sys.executable, "modules/module_02_value_methods/examples/dqn_cartpole.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_02_value_methods/examples/dqn_atari_pong.py",
         "--episodes", "1", "--max-steps", "100"],

        # Module 03: Policy Gradient
        [sys.executable, "modules/module_03_policy_methods/examples/policy_gradient_cartpole.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_03_policy_methods/examples/policy_gradient_pendulum.py",
         "--episodes", "2"],

        # Module 04: Actor-Critic (Core algorithms)
        [sys.executable, "modules/module_04_actor_critic/examples/a2c_cartpole.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_04_actor_critic/examples/ppo_cartpole.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_04_actor_critic/examples/ppo_lunarlander.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_04_actor_critic/examples/sac_pendulum.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_04_actor_critic/examples/td3_pendulum.py",
         "--episodes", "2"],
        [sys.executable, "modules/module_04_actor_critic/examples/trpo_cartpole.py",
         "--episodes", "2"],
    ]
)

INFRASTRUCTURE_TESTS = TestGroup(
    name="Infrastructure & Production",
    description="GPU optimization, distributed training, experiment tracking",
    required=False,  # Optional infrastructure features
    requires_torch=True,
    tests=[
        # Vectorized environments (GPU-optimized)
        [sys.executable, "modules/module_02_value_methods/examples/dqn_cartpole_vectorized.py",
         "--episodes", "2", "--num-envs", "2"],

        # TensorBoard integration
        [sys.executable, "modules/module_04_actor_critic/examples/ppo_cartpole_tensorboard.py",
         "--episodes", "2"],

        # Distributed training (Ray RLlib)
        [sys.executable, "modules/module_07_operationalization/examples/ray_distributed_ppo.py",
         "--num-workers", "2", "--iterations", "2"],

        # Hyperparameter tuning (Optuna)
        [sys.executable, "modules/module_07_operationalization/examples/hyperparameter_tuning_optuna.py",
         "--n-trials", "2", "--n-train-episodes", "5", "--n-eval-episodes", "2"],
    ]
)

ADVANCED_ALGORITHMS_TESTS = TestGroup(
    name="Advanced Algorithms",
    description="Cutting-edge RL: Offline RL, Model-Based, RLHF",
    required=False,  # Advanced research features
    requires_torch=True,
    tests=[
        # Offline RL - CQL (Conservative Q-Learning)
        [sys.executable, "modules/module_07_operationalization/examples/cql_offline_rl.py",
         "--mode", "generate", "--dataset-path", "data/test_cartpole.pkl",
         "--dataset-size", "1000", "--seed", "42"],
        [sys.executable, "modules/module_07_operationalization/examples/cql_offline_rl.py",
         "--mode", "train", "--dataset-path", "data/test_cartpole.pkl",
         "--num-updates", "100", "--seed", "42"],

        # Offline RL - IQL (Implicit Q-Learning)
        [sys.executable, "modules/module_07_operationalization/examples/iql_offline_rl.py",
         "--mode", "train", "--dataset-path", "data/test_cartpole.pkl",
         "--num-updates", "100", "--seed", "42"],

        # Model-Based RL (Dreamer)
        [sys.executable, "modules/module_07_operationalization/examples/dreamer_model_based.py",
         "--env", "CartPole-v1", "--episodes", "2", "--batch-size", "16",
         "--imagine-horizon", "5"],

        # RLHF (Language Model Alignment)
        [sys.executable, "modules/module_07_operationalization/examples/rlhf_text_generation.py",
         "--task", "sentiment", "--iterations", "5"],

        # Benchmark Suite
        [sys.executable, "modules/module_07_operationalization/examples/benchmark_suite.py",
         "--env", "CartPole-v1", "--algorithms", "random",
         "--trials", "1", "--episodes", "2"],
    ]
)


# =============================================================================
# Test Runner
# =============================================================================

class SmokeTestRunner:
    """Orchestrates smoke test execution with reporting."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}

    def check_dependencies(self) -> dict:
        """Check which dependencies are available."""
        deps = {
            'torch': False,
            'ray': False,
            'optuna': False,
        }

        try:
            import torch
            deps['torch'] = True
            if self.verbose:
                print(f"✓ PyTorch {torch.__version__} available")
        except ImportError:
            if self.verbose:
                print("✗ PyTorch not available")

        try:
            import ray
            deps['ray'] = True
            if self.verbose:
                print(f"✓ Ray {ray.__version__} available")
        except ImportError:
            if self.verbose:
                print("✗ Ray not available (optional)")

        try:
            import optuna
            deps['optuna'] = True
            if self.verbose:
                print(f"✓ Optuna {optuna.__version__} available")
        except ImportError:
            if self.verbose:
                print("✗ Optuna not available (optional)")

        return deps

    def run_test(self, cmd: list[str]) -> bool:
        """Run a single test command."""
        cmd_str = ' '.join(cmd)
        print(f"  Testing: {Path(cmd[1]).name}")

        if self.verbose:
            print(f"    Command: {cmd_str}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not self.verbose,
                text=True,
                timeout=120  # 2 minute timeout per test
            )
            print(f"    ✓ Passed")
            return True
        except subprocess.TimeoutExpired:
            print(f"    ✗ TIMEOUT (>120s)")
            return False
        except subprocess.CalledProcessError as e:
            print(f"    ✗ FAILED")
            if self.verbose and e.stderr:
                print(f"    Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            return False

    def run_test_group(self, group: TestGroup, deps: dict) -> dict:
        """Run a group of tests."""
        print(f"\n{'='*70}")
        print(f"{group.name}")
        print(f"{'='*70}")
        print(f"Description: {group.description}")

        # Check dependencies
        if group.requires_torch and not deps['torch']:
            print("⊘ SKIPPED: Requires PyTorch (not available)")
            return {"skipped": len(group.tests), "reason": "missing PyTorch"}

        # Run tests
        passed = 0
        failed = 0

        for test_cmd in group.tests:
            if self.run_test(test_cmd):
                passed += 1
            else:
                failed += 1

        # Summary
        total = passed + failed
        print(f"\n{group.name} Summary: {passed}/{total} passed")

        if failed > 0 and not group.required:
            print(f"  (Optional group - failures don't affect overall status)")

        return {
            "passed": passed,
            "failed": failed,
            "total": total,
            "required": group.required
        }

    def run_all(self, test_groups: list[TestGroup], skip_optional: bool = False) -> bool:
        """Run all test groups."""
        print("="*70)
        print("RL SMOKE TEST SUITE")
        print("="*70)

        # Check dependencies
        deps = self.check_dependencies()
        print()

        # Run test groups
        all_passed = True
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for group in test_groups:
            if skip_optional and not group.required:
                print(f"\n⊘ Skipping optional group: {group.name}")
                total_skipped += len(group.tests)
                continue

            result = self.run_test_group(group, deps)

            if "skipped" in result:
                total_skipped += result["skipped"]
            else:
                total_passed += result["passed"]
                total_failed += result["failed"]

                # Only count required group failures
                if result["required"] and result["failed"] > 0:
                    all_passed = False

        # Final summary
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total Passed:  {total_passed}")
        print(f"Total Failed:  {total_failed}")
        print(f"Total Skipped: {total_skipped}")

        if all_passed:
            print(f"\n✓ All required tests passed!")
            return True
        else:
            print(f"\n✗ Some required tests failed.")
            return False


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run smoke tests for RL examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/smoke_test.py                    # Run all tests
  python scripts/smoke_test.py --core-only        # Core examples only
  python scripts/smoke_test.py --skip-optional    # Skip optional tests
  python scripts/smoke_test.py --verbose          # Verbose output
        """
    )

    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only core examples (no PyTorch required)"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional test groups (infrastructure, advanced algorithms)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output (show all command output)"
    )
    parser.add_argument(
        "--group",
        choices=["core", "deep-rl", "infrastructure", "advanced"],
        help="Run specific test group only"
    )

    args = parser.parse_args()

    # Determine which test groups to run
    if args.core_only:
        test_groups = [CORE_TESTS]
    elif args.group == "core":
        test_groups = [CORE_TESTS]
    elif args.group == "deep-rl":
        test_groups = [DEEP_RL_TESTS]
    elif args.group == "infrastructure":
        test_groups = [INFRASTRUCTURE_TESTS]
    elif args.group == "advanced":
        test_groups = [ADVANCED_ALGORITHMS_TESTS]
    else:
        test_groups = [
            CORE_TESTS,
            DEEP_RL_TESTS,
            INFRASTRUCTURE_TESTS,
            ADVANCED_ALGORITHMS_TESTS,
        ]

    # Run tests
    runner = SmokeTestRunner(verbose=args.verbose)
    success = runner.run_all(test_groups, skip_optional=args.skip_optional)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
