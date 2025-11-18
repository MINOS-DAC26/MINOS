"""
id_wcs_heatmap.py

Sweeps idle-power (ID) and wakeup-cost (WCS) settings for power gating
and measures EDP overhead vs. a single baseline.

Baseline:
    - run dreamer.py once for the first (ID, WCS) pair

For every (ID, WCS) pair we:
    - run partition.py
    - read "energy,delay" from the last line
    - compute EDP and EDP overhead vs. baseline in percent
    - show a heatmap of the overheads
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
FIG_PATH = os.path.join(FIG_DIR, "id_wcs_heatmap.png")
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
        # Keep sweep running, just mark this run as failed
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
                 idle: float, wcs: float) -> List[str]:
    # -lo: reuse adjacent cells for faster bounding
    return [
        "python", "dreamer.py",
        "--model", model,
        "--generation", str(generation),
        "--preloading", str(preloading),
        "--solution", "0",
        "-id", str(idle),
        "-wcs", str(wcs),
        "--subAccs", "2",
        "-lo", "2",
    ]


def partition_args(model: str, generation: int, preloading: int,
                   idle: float, wcs: float) -> List[str]:
    return [
        "python", "partition.py",
        "--model", model,
        "--generation", str(generation),
        "--preloading", str(preloading),
        "--solution", "0",
        "-id", str(idle),
        "-wcs", str(wcs),
        "--subAccs", "2",
        "-lo", "2",
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "bert_hybrid"
GENERATION = 0
PRELOADING = 0

# ID will be the x-axis; WCS the y-axis
ID_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
WCS_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== ID/WCS power-gating sweep ===")

    # Slightly nicer default fonts
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    # baseline: first entry in both lists
    base_id = ID_VALUES[0]
    base_wcs = WCS_VALUES[0]

    e_base, d_base = run_command(
        dreamer_args(MODEL, GENERATION, PRELOADING, base_id, base_wcs)
    )
    if e_base is None or d_base is None:
        print("ERROR: baseline run failed")
        sys.exit(1)

    edp_base = e_base * d_base
    print(f"[INFO] baseline EDP = {edp_base:.4f}")

    # collect overheads in percent
    # rows = WCS (y), cols = ID (x)
    heat = []
    for wcs_val in WCS_VALUES:
        row = []
        for id_val in ID_VALUES:
            e, d = run_command(
                partition_args(MODEL, GENERATION, PRELOADING, id_val, wcs_val)
            )
            if e is not None and d is not None:
                edp = e * d
                # overhead in percent, baseline cell -> 0.0
                overhead_pct = (edp / edp_base - 1.0) * 100.0
            else:
                overhead_pct = np.nan  # failed run
            print(
                f"[RES] id={id_val}, wcs={wcs_val} "
                f"-> overhead={overhead_pct:.1f}%"
            )
            row.append(overhead_pct)
        heat.append(row)

    heat = np.array(heat, dtype=float)

    # Plot heatmap: ID on x-axis, WCS on y-axis
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    # Mask NaNs so they appear blank
    mask = np.isnan(heat)

    # Use a red-ish sequential colormap similar to the screenshot
    vmin = np.nanmin(heat)
    vmax = np.nanmax(heat)

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
        cbar_kws={"label": "EDP Overhead vs. Dream (%)"},
        xticklabels=[f"{val * 100:g}%" for val in ID_VALUES],
        yticklabels=[f"{val:g}" for val in WCS_VALUES],
    )

    # Axis labels similar to the paper-style figure
    ax.set_xlabel("Idle vs. Avg. Power (%)")
    ax.set_ylabel(r"$E_{S2W}$ ($\mu$J)")

    # Put x-axis labels at the bottom with a bit more space
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    print(f"[INFO] saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
