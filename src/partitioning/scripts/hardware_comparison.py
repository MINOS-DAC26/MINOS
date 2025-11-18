"""
hardware_comparison.py

Runs dreamer/partition for bert_deeper_encoder over generations 0..4
with fixed link settings (com=5.7, coe=13.3) and compares EDP.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd: List[str]) -> Tuple[Optional[float], Optional[float]]:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    try:
        e_str, d_str = last.split(",")[:2]
        return float(e_str), float(d_str)
    except Exception:
        print(f"[WARN] could not parse: '{last}'")
        return None, None


def dreamer_args(generation: int) -> List[str]:
    return [
        "python", "dreamer.py",
        "--model", "bert_deeper_encoder",
        "--generation", str(generation),
        "--preloading", "0",
        "--solution", "0",
        "--chips", "1",
        # "-com", "5.7",
        # "-coe", "13.3",
        "-wts", "2",
        "-wcs", "2",
        "--estimates", "hw",
    ]


def partition_args(generation: int, chips: int) -> List[str]:
    return [
        "python", "partition.py",
        "--model", "bert_deeper_encoder",
        "--generation", str(generation),
        "--preloading", "0",
        "--solution", "0",
        "--target", "Energy",
        "--chips", str(chips),
        "-tp", "1D_RING",
        # "-com", "5.7",
        # "-coe", "13.3",
        "-wts", "2",
        "-wcs", "2",
        "--estimates", "hw",
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENERATIONS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== bert_deeper_encoder hardware sweep ===")

    for gen in GENERATIONS:
        # baseline
        d_energy, d_delay = run_command(dreamer_args(gen))
        if not (d_energy and d_delay):
            print(f"[ERR] dreamer failed for gen={gen}")
            continue
        d_edp = d_energy * d_delay

        # scaled partition
        chips = 1 if gen == 0 else 2 * gen
        p_energy, p_delay = run_command(partition_args(gen, chips))
        if not (p_energy and p_delay):
            print(f"[ERR] partition failed for gen={gen}")
            continue
        p_edp = p_energy * p_delay

        edp_factor = (p_edp / d_edp - 1) * 100 if d_edp else None
        print(f"[RES] gen={gen}  dreamer_EDP={d_edp:.3f}  partition_EDP={p_edp:.3f}  overhead={edp_factor:.2f}%")

    print("\n=== done ===")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
