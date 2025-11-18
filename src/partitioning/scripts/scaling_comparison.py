"""
scaling_comparison.py

Runs a generation sweep to study synthetic scaling.

Models:
    - resnet18_hybrid
    - mobilebert_hybrid
    - bert_hybrid

Generations:
    0, 2, 4, 6

For every (model, generation) pair we:
- run dreamer.py to get the lower-bound / “dream” configuration
- run partition.py to optimize the actual multi-chiplet configuration

Generation > 0 denotes synthetically widened/deepened models; memory and d2d
links are assumed to scale linearly as documented in the paper.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def base_args(model_name: str, generation: int) -> List[str]:
    return [
        "--model", model_name,
        "--generation", str(generation),
        "--preloading", "0",
        "--solution", "0",
    ]


def run_dreamer(model_name: str, generation: int) -> None:
    cmd = ["python", "dreamer.py"]
    cmd.extend(base_args(model_name, generation))
    run_command(cmd)


def run_partition(model_name: str, generation: int) -> None:
    cmd = ["python", "partition.py"]
    cmd.extend(base_args(model_name, generation))
    run_command(cmd)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    "resnet18_hybrid",
    "mobilebert_hybrid",
    "bert_hybrid",
]

GENERATIONS = [0, 2, 4, 6]


def main() -> None:
    print("\n=== Generation sweep for scaled hybrid models ===")
    for model in MODELS:
        for gen in GENERATIONS:
            # dreamer: lower bound for this synthetic size
            run_dreamer(model, gen)
            # partition: actual chiplet optimization
            run_partition(model, gen)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
