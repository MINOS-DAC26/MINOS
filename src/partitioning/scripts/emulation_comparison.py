"""
emulation_comparison.py

Runs dreamer/partition for resnet18_hybrid over chip counts
to compare EDP for emulation-style settings (com=5.7, coe=13.3).
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


def dreamer_args(chips: int) -> List[str]:
    return [
        "python", "dreamer.py",
        "--model", "resnet18_hybrid",
        "--generation", "0",
        "--preloading", "0",
        "--solution", "0",
        "--chips", str(chips),
        # "-com", "5.7",
        # "-coe", "13.3",
        "--estimates", "hw",
        "-wts", "2",
        "-wcs", "2",
    ]


def partition_args(chips: int) -> List[str]:
    return [
        "python", "partition.py",
        "--model", "resnet18_hybrid",
        "--generation", "0",
        "--preloading", "0",
        "--solution", "0",
        "--chips", str(chips),
        # "-com", "5.7",
        # "-coe", "13.3",
        "--estimates", "hw",
        "-wts", "2",
        "-wcs", "2",
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHIP_COUNTS = [1, 2, 4, 8, 12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== resnet18_hybrid emulation comparison ===")

    for chips in CHIP_COUNTS:
        d_energy, d_delay = run_command(dreamer_args(chips))
        if not (d_energy and d_delay):
            print(f"[ERR] dreamer failed for chips={chips}")
            continue
        d_edp = d_energy * d_delay

        p_energy, p_delay = run_command(partition_args(chips))
        if not (p_energy and p_delay):
            print(f"[ERR] partition failed for chips={chips}")
            continue
        p_edp = p_energy * p_delay

        edp_factor = (p_edp / d_edp - 1) * 100 if d_edp else None
        print(f"[RES] chips={chips}  dreamer_EDP={d_edp:.3f}  partition_EDP={p_edp:.3f}  overhead={edp_factor:.2f}%")

    print("\n=== done ===")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
