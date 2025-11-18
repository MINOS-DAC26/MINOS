"""
chip_scaling_heatmap.py

Runs a chip/com-coe sweep to study EDP overhead vs. a baseline.

Baseline:
    - run dreamer.py once for the smallest chip count

For every (chips, com_coe) pair we:
    - run partition.py
    - read "energy,delay" from the last line
    - compute EDP and EDP overhead vs. baseline in percent
    - plot a heatmap of the overheads
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIG_DIR = "results/figures"
FIG_PATH = os.path.join(FIG_DIR, "chip_scaling_heatmap.png")
os.makedirs(FIG_DIR, exist_ok=True)


def run_command(cmd: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Run a command, stream its output live to the terminal, and return
    (energy, delay) parsed from the final non-empty stdout line
    (CSV: energy,delay[,...]). On failure, log and return (None, None).
    """
    print("[RUN]", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout for a single stream
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
        # Mirror your previous behavior: do not crash the whole script
        print(f"[ERROR] command failed with exit code {proc.returncode}")
        return None, None

    if not last_nonempty:
        print("[WARN] no stdout to parse")
        return None, None

    try:
        e_str, d_str = last_nonempty.split(",")[:2]
        return float(e_str), float(d_str)
    except Exception:
        print(f"[WARN] could not parse: '{last_nonempty}'")
        return None, None


def dreamer_args(model: str, generation: int, preloading: int,
                 chips: int, num_subaccs: int) -> List[str]:
    # -lo: reuse previously loaded / adjacent cells to bound the next optimization
    return [
        "python", "dreamer.py",
        "--model", model,
        "--generation", str(generation),
        "--preloading", str(preloading),
        "--solution", "0",
        "--chips", str(chips),
        "--subAccs", str(num_subaccs),
    ]


def partition_args(model: str, generation: int, preloading: int,
                   chips: int, num_subaccs: int, com_coe: float) -> List[str]:
    # same idea: keep -lo, other flags fall back to defaults
    return [
        "python", "partition.py",
        "--model", model,
        "--generation", str(generation),
        "--preloading", str(preloading),
        "--solution", "0",
        "--chips", str(chips),
        "--subAccs", str(num_subaccs),
        "-com", str(com_coe),
        "-coe", str(com_coe),
        "-lo", "1",
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "bert_hybrid"
GENERATION = 0
PRELOADING = 0

CHIPS_VALUES = [2, 4, 8, 16, 32]          # x-axis
COM_COE_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]  # underlying sweep values

# Just for nicer y tick labels in the figure (matches the screenshot style)
D2D_LABELS = ["(0.5, 8)", "(1, 4)", "(2, 2)", "(4, 1)", "(8, 0.5)"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Chip scaling sweep ===")

    # Slightly nicer default fonts (similar to your paper plots)
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    max_chips = CHIPS_VALUES[-1]
    base_chips = CHIPS_VALUES[0]
    base_num_subaccs = max_chips // base_chips

    # Baseline run – if THIS fails, we really cannot compute overheads.
    e_base, d_base = run_command(
        dreamer_args(MODEL, GENERATION, PRELOADING, base_chips, base_num_subaccs)
    )
    if e_base is None or d_base is None:
        print("ERROR: baseline run failed")
        sys.exit(1)
    edp_base = e_base * d_base
    print(f"[INFO] baseline EDP = {edp_base:.4f}")

    # Collect overheads in percent
    # rows = COM_COE_VALUES (y-axis), cols = CHIPS_VALUES (x-axis)
    heat = []
    for com_coe in COM_COE_VALUES:
        row = []
        for chips in CHIPS_VALUES:
            num_subaccs = max_chips // chips
            e, d = run_command(
                partition_args(MODEL, GENERATION, PRELOADING, chips, num_subaccs, com_coe)
            )
            if e is not None and d is not None:
                edp = e * d
                overhead_pct = (edp / edp_base - 1.0) * 100.0
            else:
                overhead_pct = np.nan  # failed run -> NaN in heatmap
            print(
                f"[RES] chips={chips}, com_coe={com_coe} "
                f"-> overhead={overhead_pct:.1f}%"
            )
            row.append(overhead_pct)
        heat.append(row)

    heat = np.array(heat, dtype=float)
    mask = np.isnan(heat)

    # Determine color scale bounds from available data
    vmin = np.nanmin(heat)
    vmax = np.nanmax(heat)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    hm = sns.heatmap(
        heat,
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        xticklabels=CHIPS_VALUES,
        yticklabels=D2D_LABELS,
        cbar_kws={"label": "EDP Overhead vs. Dream(%)"},
    )

    ax.set_xlabel("Number of Chips")
    ax.set_ylabel("D2D Metrics (Gb/s, pJ/b)")

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    print(f"[INFO] saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
