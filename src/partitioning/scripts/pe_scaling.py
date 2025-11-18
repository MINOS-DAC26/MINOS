"""
pe_scaling.py

Runs link-scaling experiments for the PE configurations listed in the paper
(table with 8x8, 16x16, 32x32, 64x64, 128x128 PEs).

For each PE size (encoded in the model names `llama1B_zigzag_0` ... `llama1B_zigzag_4`)
we do:

1) run dreamer.py with the PE’s corresponding d2d energy (coe) and bandwidth (com) setting
2) run partition.py with the same settings
3) report energy, delay, and EDP overhead of partition vs. dreamer

Per-index target link settings (from the table):
    idx 0 (≈ 8x8)   -> coe = 5.7,  bw = 6
    idx 1 (≈ 16x16) -> coe = 2.2,  bw = 12
    idx 2 (≈ 32x32) -> coe = 0.78, bw = 33
    idx 3 (≈ 64x64) -> coe = 0.38, bw = 66
    idx 4 (≈ 128x128)-> coe = 0.21, bw = 140

Base link is 4 Gbps, so:
    com_flag = 4 / target_bw
and we pass the target coe directly as -coe.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
import csv
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/figures")
RESULTS_CSV = RESULTS_DIR / "pe_scaling_overhead.csv"


def ensure_results_csv() -> None:
    """
    Ensure results directory exists and CSV has a header if it's a new file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        with RESULTS_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pe_idx", "overhead_percent"])


def append_result_row(pe_idx: int, overhead_percent: float) -> None:
    """
    Append a single row to the CSV immediately after computing results.
    """
    ensure_results_csv()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([pe_idx, f"{overhead_percent:.6f}"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Run a command, stream its output live to the terminal, and return
    (energy, delay, throughput) parsed from the final non-empty stdout line,
    which must be CSV: energy,delay[,throughput].
    """
    print("[RUN]", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
    )

    last_nonempty = ""
    assert proc.stdout is not None  # for type checkers

    for line in proc.stdout:
        # Stream child output directly
        print(line, end="")
        stripped = line.strip()
        if stripped:
            last_nonempty = stripped

    proc.wait()
    if proc.returncode != 0:
        # behave like subprocess.run(..., check=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    if not last_nonempty:
        print("[WARN] no stdout to parse")
        return None, None, None

    parts = [p.strip() for p in last_nonempty.split(",")]
    if len(parts) < 2:
        print(f"[WARN] could not parse CSV from last line: '{last_nonempty}'")
        return None, None, None

    try:
        energy = float(parts[0])
        delay = float(parts[1])
        throughput = float(parts[2]) if len(parts) > 2 else float("nan")
        return energy, delay, throughput
    except Exception:
        print(f"[WARN] could not parse floats from last line: '{last_nonempty}'")
        return None, None, None


def base_args(model: str, generation: int) -> List[str]:
    return [
        "--model", model,
        "--generation", str(generation),
        "--preloading", "0",
        "--solution", "0",
    ]


def dreamer_args(model: str, generation: int, coe: float, com: float) -> List[str]:
    return [
        "python", "dreamer.py",
        *base_args(model, generation),
        "-coe", str(coe),
        "-com", str(com),
    ]


def partition_args(model: str, generation: int, coe: float, com: float) -> List[str]:
    return [
        "python", "partition.py",
        *base_args(model, generation),
        "-coe", str(coe),
        "-com", str(com),
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENERATION = 0

# per-PE-size targets, ordered as in the table (smallest → largest)
COE_TARGETS = [5.7, 2.2, 0.78, 0.38, 0.21]
BW_TARGETS  = [6,   12,  33,   66,   140]   # target bandwidths in Gbps
BASE_COM = 4.0  # base bandwidth in Gbps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PE_SIZES = [8, 16, 32, 64, 128]

def main() -> None:
    print("\n=== PE-size / link-scaling sweep ===")
    ensure_results_csv()

    for idx, pe_size in enumerate(PE_SIZES):
        model_name = f"llama1B_pe{pe_size}"   # or llama1B_zigzag_{idx} if that’s the real name

        target_coe = COE_TARGETS[idx]
        target_bw  = BW_TARGETS[idx]
        com_factor = BASE_COM / target_bw

        print(f"\n--- PE config #{pe_size} ---")
        print(f"[INFO] coe={target_coe}, com={com_factor} (to reach {target_bw} Gbps)")

        d_energy, d_delay, _ = run_command(
            dreamer_args(model_name, GENERATION, target_coe, com_factor)
        )
        if d_energy is None or d_delay is None:
            print("[ERROR] dreamer failed, skipping this PE config")
            continue
        d_edp = d_energy * d_delay
        print(f"[BASE] energy={d_energy:.6f}, delay={d_delay:.6f}, EDP={d_edp:.6f}")

        # 2) partition with the same link settings
        p_energy, p_delay, _ = run_command(
            partition_args(model_name, GENERATION, target_coe, com_factor)
        )
        if p_energy is None or p_delay is None:
            print("[ERROR] partition failed for this PE config")
            continue
        p_edp = p_energy * p_delay

        # overheads (x = ratio), edp_x also converted to percent overhead for CSV
        if d_energy == 0 or d_delay == 0 or d_edp == 0:
            print("[ERROR] baseline has zero value; cannot compute overheads safely")
            continue

        energy_x = p_energy / d_energy
        delay_x = p_delay / d_delay
        edp_x = p_edp / d_edp
        overhead_pct = (edp_x - 1.0) * 100.0

        print(f"[RES] energy_x={energy_x:.3f}  delay_x={delay_x:.3f}  edp_x={edp_x:.3f}")
        print(f"[RES] EDP overhead = {overhead_pct:.4f}%")

        # Write incremental result
        append_result_row(PE_SIZES[idx], overhead_pct)
        print(f"[saved] {RESULTS_CSV}  (+ row for PE idx={PE_SIZES[idx]})")

    print("\n=== Done ===")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
