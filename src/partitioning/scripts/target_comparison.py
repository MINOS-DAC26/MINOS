"""
target_comparison.py

Runs dreamer/partition for different optimization targets and compares
energy/latency (and throughput) against the dreamer baseline.

Steps
-----
1) Run dreamer for target=EDP
2) For target=EDP run partition with:
       -l 0 and --target EDP
       -l 0 and --target Energy
       -l 0 and --target Latency
       -l 1 and same 3 targets
   Compare energy/latency vs. dreamer(EDP)

3) Run dreamer for target=EPT
4) For target=EPT run partition with:
       -l 0 and --target EPT
       -l 0 and --target Energy
       -l 0 and --target Throughput
       -l 1 and same 3 targets
   Compare energy/throughput vs. dreamer(EPT)

Assumes last line of each tool is:
    energy,delay
or
    energy,delay,throughput
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import subprocess
from typing import List, Optional, Tuple

def run_command(cmd: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    print("[RUN]", " ".join(cmd), flush=True)

    # Start process, capture stdout but stream it line by line
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_nonempty = ""
    assert proc.stdout is not None  # for type checkers

    for line in proc.stdout:
        # Stream line directly to our terminal
        print(line, end="")  # `line` already has newline
        stripped = line.strip()
        if stripped:
            last_nonempty = stripped

    proc.wait()
    if proc.returncode != 0:
        # Make it behave like subprocess.run(..., check=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    if not last_nonempty:
        print("[WARN] no output captured from command")
        return None, None, None

    parts = last_nonempty.split(",")
    try:
        energy = float(parts[0])
        delay = float(parts[1]) if len(parts) > 1 else None
        throughput = float(parts[2]) if len(parts) > 2 else None
        return energy, delay, throughput
    except Exception:
        print(f"[WARN] could not parse: '{last_nonempty}'")
        return None, None, None


def base_args(model: str, generation: int) -> List[str]:
    return [
        "--model", model,
        "--generation", str(generation),
        "--preloading", "0",
        "--solution", "0",
    ]


def run_dreamer(model: str, generation: int, target: str):
    cmd = ["python", "dreamer.py"]
    cmd.extend(base_args(model, generation))
    cmd.extend(["--target", target])
    return run_command(cmd)


def run_partition(model: str, generation: int, target: str, l_flag: int):
    cmd = ["python", "partition.py"]
    cmd.extend(base_args(model, generation))
    cmd.extend(["--target", target, "-l", str(l_flag)])
    return run_command(cmd)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

MODEL = "llama1b_hybrid"
GENERATION = 0

EDP_PARTITION_TARGETS = ["EDP", "Energy", "Latency"]
EPT_PARTITION_TARGETS = ["EPT", "Energy", "Throughput"]


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    # ------------------------------------------------------------
    # 1) EDP block
    # ------------------------------------------------------------
    print_header("EDP baseline")
    edp_energy, edp_delay, edp_throughput = run_dreamer(MODEL, GENERATION, "EDP")
    if edp_energy is None or edp_delay is None:
        print("ERROR: dreamer EDP baseline failed")
        sys.exit(1)
    print(f"[BASE-EDP] energy={edp_energy}, delay={edp_delay}")

    print_header("EDP: partition comparison")
    for l_flag in (0, 1):
        print(f"\n[l={l_flag}]")
        for tgt in EDP_PARTITION_TARGETS:
            p_energy, p_delay, _ = run_partition(MODEL, GENERATION, tgt, l_flag)
            if p_energy is None or p_delay is None:
                print(f"{tgt:10s} -> failed")
                continue
            energy_ov = p_energy / edp_energy if edp_energy else None
            delay_ov = p_delay / edp_delay if edp_delay else None
            print(f"{tgt:10s}  energy_x={energy_ov:.3f}  delay_x={delay_ov:.3f}")

    # ------------------------------------------------------------
    # 2) EPT block
    # ------------------------------------------------------------
    print_header("EPT baseline")
    ept_energy, ept_delay, ept_throughput = run_dreamer(MODEL, GENERATION, "EPT")
    if ept_energy is None or ept_throughput is None:
        # delay may or may not matter here, but we need energy + throughput
        print("ERROR: dreamer EPT baseline failed")
        sys.exit(1)
    print(f"[BASE-EPT] energy={ept_energy}, throughput={ept_throughput}")

    print_header("EPT: partition comparison")
    for l_flag in (0, 1):
        print(f"\n[l={l_flag}]")
        for tgt in EPT_PARTITION_TARGETS:
            p_energy, _, p_throughput = run_partition(MODEL, GENERATION, tgt, l_flag)
            if p_energy is None:
                print(f"{tgt:10s} -> failed")
                continue
            energy_ov = p_energy / ept_energy if ept_energy else None
            # for throughput we want ratio vs. baseline (>=1 is better),
            # but to keep it consistent we still show factor
            thr_ov = p_throughput / ept_throughput if (p_throughput and ept_throughput) else None
            if thr_ov is not None:
                print(f"{tgt:10s}  energy_x={energy_ov:.3f}  throughput_x={thr_ov:.3f}")
            else:
                print(f"{tgt:10s}  energy_x={energy_ov:.3f}  throughput_x=?")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
