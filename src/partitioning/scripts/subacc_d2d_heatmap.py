"""
subacc_d2d_heatmap.py

Sweeps the number of sub-accelerators vs. d2d (com/coe) settings and
measures EDP overhead vs. a per-sub-accelerator baseline.

For every sub-accelerator count:
    - run dreamer.py once to get the baseline EDP
    - for every d2d (com/coe) value run partition.py
    - compute EDP overhead vs. baseline in percent

Result: heatmap with
    x-axis: #sub-accelerators
    y-axis: D2D metrics (Gb/s, pJ/b)
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
FIG_PATH = os.path.join(FIG_DIR, "subacc_d2d_heatmap.png")
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
        # Keep sweep running, just mark failure
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
                 num_subacc: int) -> List[str]:
    # keep CLI param name as --subAccs until codebase is renamed
    return [
        "python", "dreamer.py",
        "--model", model,
        "--generation", str(generation),
        "--preloading", str(preloading),
        "--solution", "0",
        "--subAccs", str(num_subacc),
        "-lo", "3",
    ]


def partition_args(model: str, generation: int, preloading: int,
                   num_subacc: int, com_coe: float) -> List[str]:
    return [
        "python", "partition.py",
        "--model", model,
        "--generation", str(generation),
        "--preloading", str(preloading),
        "--solution", "0",
        "--subAccs", str(num_subacc),
        "-com", str(com_coe),
        "-coe", str(com_coe),
        "-lo", "3",
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "bert_hybrid"
GENERATION = 0
PRELOADING = 0

SUBACC_VALUES = [1, 2, 4, 8, 16]          # x-axis: SAs per chiplet
D2D_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]    # underlying sweep values

# Pretty labels for the y-axis (matching the paper style)
D2D_LABELS = ["(0.5, 8)", "(1, 4)", "(2, 2)", "(4, 1)", "(8, 0.5)"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== sub-accelerator vs. d2d sweep ===")

    # Paper-like font sizes
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    # 1) baselines per sub-accelerator count
    baselines = {}
    for subacc in SUBACC_VALUES:
        e_base, d_base = run_command(
            dreamer_args(MODEL, GENERATION, PRELOADING, subacc)
        )
        if e_base is None or d_base is None:
            print(
                f"[WARN] baseline run failed for sub-accelerators={subacc}, "
                f"marking its column as NaN"
            )
            baselines[subacc] = np.nan
        else:
            baselines[subacc] = e_base * d_base
            print(
                f"[INFO] baseline EDP for {subacc} sub-accelerators = "
                f"{baselines[subacc]:.4f}"
            )

    # 2) sweep D2D for each sub-accelerator count
    # rows = D2D_VALUES (y-axis), cols = SUBACC_VALUES (x-axis)
    heat = []
    for d2d in D2D_VALUES:
        row = []
        for subacc in SUBACC_VALUES:
            base_edp = baselines.get(subacc, np.nan)
            e, d = run_command(
                partition_args(MODEL, GENERATION, PRELOADING, subacc, d2d)
            )
            if (
                e is not None and d is not None
                and base_edp is not None
                and not np.isnan(base_edp)
            ):
                edp = e * d
                overhead_pct = (edp / base_edp - 1.0) * 100.0
            else:
                overhead_pct = np.nan
            print(
                f"[RES] subacc={subacc}, d2d={d2d} "
                f"-> overhead={overhead_pct:.1f}%"
            )
            row.append(overhead_pct)
        heat.append(row)

    heat = np.array(heat, dtype=float)
    mask = np.isnan(heat)

    vmin = np.nanmin(heat)
    vmax = np.nanmax(heat)

    # 3) plot
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    sns.heatmap(
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
        xticklabels=SUBACC_VALUES,
        yticklabels=D2D_LABELS,
        cbar_kws={"label": "EDP Overhead vs. Dream(%)"},
    )

    ax.set_xlabel("SAs per Chiplet")
    ax.set_ylabel("D2D Metrics (Gb/s, pJ/b)")

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    print(f"[INFO] saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
