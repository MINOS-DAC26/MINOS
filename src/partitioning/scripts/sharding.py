#!/usr/bin/env python3
"""
sharding.py

Runs partition.py for different sharding degrees and collects
energy–latency results into a CSV file.

It executes commands of the form:

    python partition.py --model <MODEL_NAME> --generation 0 --solution 0 \
                        --chips 16 --preloading 0

For the model name, it starts with the base model (no prefix) and then
adds prefixes 2_, 4_, 8_:

    0  -> llama1b_hybrid"
    2  -> "2_llama1b_hybrid"
    4  -> "4_llama1b_hybrid"
    8  -> "8_llama1b_hybrid"

At the end of each run, we assume partition.py prints a final line with
three comma-separated values:

    energy, latency, <something_else>

We take the first two (energy, latency), compute EDP = energy * latency,
and write them into a CSV with columns:

    ShardingDegree, EDP, FactorToPrevious

FactorToPrevious is the ratio of EDP in this row to the EDP of
the previous row; for the first row this is left empty.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import csv
import subprocess
from pathlib import Path
from typing import List, Tuple

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# Base model name (without sharding prefix)
BASE_MODEL = "llama1b_hybrid"

# Path to the partition script
PARTITION_SCRIPT = "partition.py"

# Output CSV file
CSV_PATH = Path("results/figures/sharding_results.csv")

# Sharding degrees to test; 0 means "no prefix"
SHARDING_DEGREES = [0, 2, 4, 8]


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def build_model_name(sharding_degree: int) -> str:
    """
    Build the model name for a given sharding degree.

    For degree 0, we use the base model name directly.
    For degree d > 0, we prepend "d_" to the base model.
    """
    if sharding_degree == 0:
        return BASE_MODEL
    return f"{sharding_degree}_{BASE_MODEL}"


def run_partition(model_name: str) -> Tuple[float, float]:
    cmd = [
        "python",
        PARTITION_SCRIPT,
        "--model", model_name,
        "--generation", "0",
        "--solution", "0",
        "--chips", "16",
        "--preloading", "0",
        "-com", "0.25",
        "-coe", "0.25"
    ]

    # Start process and stream its combined stdout/stderr
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_line = None
    assert proc.stdout is not None  # for type checkers
    for line in proc.stdout:
        print(line, end="")          # show live output
        stripped = line.strip()
        if stripped:
            last_line = stripped     # remember last non-empty line

    retcode = proc.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd)

    if last_line is None:
        raise RuntimeError(f"No output captured from command: {' '.join(cmd)}")

    parts = [p.strip() for p in last_line.split(",")]
    if len(parts) < 2:
        raise ValueError(
            f"Expected at least two comma-separated values in last line, "
            f"got: '{last_line}'"
        )

    energy = float(parts[0])
    latency = float(parts[1])
    return energy, latency


def compute_edp(energy: float, latency: float) -> float:
    """Energy–Delay Product."""
    return energy * latency


# --------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: run partition.py for all sharding degrees,
    compute EDPs, and write them into a CSV file.
    """
    results: List[Tuple[int, float]] = []

    # Run experiments for each sharding degree
    for degree in SHARDING_DEGREES:
        model_name = build_model_name(degree)
        print(f"Running partition.py for sharding degree {degree} (model='{model_name}')...")

        energy, latency = run_partition(model_name)
        edp = compute_edp(energy, latency)

        print(f"  -> energy={energy}, latency={latency}, EDP={edp}")
        results.append((degree, edp))

    # Write results to CSV with factor column
    print(f"\nWriting results to {CSV_PATH}...")
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ShardingDegree", "EDP", "FactorToPrevious"])

        prev_edp = None
        for degree, edp in results:
            if prev_edp is None:
                # First row: no previous value to compare to
                factor = ""
            else:
                factor = edp / prev_edp
            writer.writerow([degree, edp, factor])
            prev_edp = edp

    print("Done.")


if __name__ == "__main__":
    main()
