"""
heuristic_speedup.py

Showcases the effect of enabling the partitioning heuristic.

Procedure:
- run dreamer.py once for the chosen model/generation to establish the lower bound
- run partition.py for the same model/generation with:
    1) --heuristic none
    2) --heuristic all

This way we can compare solve/setup/runtime behavior under otherwise identical
conditions (same model, same generation, same preloading/solution settings).
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


def run_partition(model_name: str, generation: int, heuristic: str) -> None:
    cmd = ["python", "partition.py"]
    args = base_args(model_name, generation)
    args.extend(["--heuristic", heuristic])
    args.extend(["-et", "0"])
    cmd.extend(args)
    run_command(cmd)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "resnet18_hybrid"
GENERATION = 2
HEURISTICS = ["all", "none"]   # order matters for readability in the logs


def main() -> None:
    print("\n=== Heuristic speedup comparison ===")
    # lower bound first
    run_dreamer(MODEL, GENERATION)

    # same problem, different heuristic settings
    for h in HEURISTICS:
        run_partition(MODEL, GENERATION, heuristic=h)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
