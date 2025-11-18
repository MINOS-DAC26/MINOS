"""
kv_scaling.py

Runs dreamer.py and partition.py for zigzag LLaMA models with increasing KV sizes.
Model naming follows: llama1B_zigzag_<KV>

KV sizes swept:
    128, 256, 512, ..., 8192

For each KV we:
    - run dreamer.py with the model
    - run partition.py with the same model
Both runs use the same generation and chip count.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
import csv
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_TEMPLATE = "llama1B_kv{}"
GENERATION = 0
CHIPS = 10
KV_SIZES = [2 ** i for i in range(7, 17)] 

# Results CSV (written incrementally)
RESULTS_DIR = Path("results/figures")
RESULTS_CSV = RESULTS_DIR / f"kv_scaling_overhead({CHIPS}).csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd: List[str]) -> str:
    """
    Run a command, stream its output live to the terminal, and return stdout as text.
    We still need the full stdout so parse_last_metrics() can read the last CSV line.
    Raises CalledProcessError on non-zero return code.
    """
    print("[RUN]", " ".join(str(c) for c in cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr for a single stream
        text=True,
    )

    assert proc.stdout is not None  # for type checkers
    lines: List[str] = []

    for line in proc.stdout:
        # Stream line immediately
        print(line, end="")
        lines.append(line)

    proc.wait()

    if proc.returncode != 0:
        # Mimic subprocess.run(..., check=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output="".join(lines))

    stdout_text = "".join(lines)
    return stdout_text


def dreamer_cmd(model_name: str) -> List[str]:
    return [
        "python", "dreamer.py",
        "--model", model_name,
        "--preloading", "0",
        "--solution", "0",
        "--generation", str(GENERATION),
        "-c", str(CHIPS),
    ]


def partition_cmd(model_name: str, kv: int) -> List[str]:
    cmd = [
        "python", "partition.py",
        "--model", model_name,
        "--preloading", "0",
        "--solution", "0",
        "--generation", str(GENERATION),
        "-c", str(CHIPS),
    ]
    if kv == 8192:
        cmd += ["-com", "0.05", "-coe", "0.05"]
    return cmd


def parse_last_metrics(stdout_text: str) -> Tuple[float, float, float]:
    """
    Parse the final non-empty line of stdout as CSV: energy, latency, throughput.
    Returns (energy, latency, throughput) as floats.
    """
    lines = [ln.strip() for ln in stdout_text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("No output lines to parse.")
    last = lines[-1]
    parts = [p.strip() for p in last.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Last line is not CSV with at least energy,latency: {last!r}")
    try:
        energy = float(parts[0])
        latency = float(parts[1])
        throughput = float(parts[2]) if len(parts) > 2 else float("nan")
    except ValueError as e:
        raise ValueError(f"Could not parse floats from last line: {last!r}") from e
    return energy, latency, throughput


def edp(energy: float, latency: float) -> float:
    return energy * latency


def ensure_results_csv():
    """
    Ensure results directory exists and CSV has a header if it's a new file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        with RESULTS_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kv", "overhead_percent"])


def append_result_row(kv: int, overhead_percent: float) -> None:
    """
    Append a single row to the CSV immediately after computing results.
    """
    ensure_results_csv()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([kv, f"{overhead_percent:.6f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== KV scaling sweep ===")
    ensure_results_csv()

    for kv in KV_SIZES:
        model_name = MODEL_TEMPLATE.format(kv)
        print(f"\n--- KV = {kv}  (model={model_name}) ---")

        # 1) dreamer
        dream_stdout = run_command(dreamer_cmd(model_name))
        d_energy, d_latency, _ = parse_last_metrics(dream_stdout)
        d_edp = edp(d_energy, d_latency)
        print(f"[dream] energy={d_energy} latency={d_latency} -> EDP={d_edp}")

        # 2) partition
        part_stdout = run_command(partition_cmd(model_name, kv))
        p_energy, p_latency, _ = parse_last_metrics(part_stdout)
        p_edp = edp(p_energy, p_latency)
        print(f"[partition] energy={p_energy} latency={p_latency} -> EDP={p_edp}")

        # Overhead in percent: (partition/dream - 1) * 100
        if d_edp == 0:
            raise ZeroDivisionError("Dream EDP is zero; cannot compute overhead.")
        overhead_pct = (p_edp / d_edp - 1.0) * 100.0
        print(f"[result] overhead = {overhead_pct:.4f}%")

        # Write incremental result
        append_result_row(kv, overhead_pct)
        print(f"[saved] {RESULTS_CSV}  (+ row for KV={kv})")

    print("\n=== done ===")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        # Keep one extra guard so a parsing error doesn't hide the root cause silently
        print(f"\nERROR: {e}")
        sys.exit(1)
