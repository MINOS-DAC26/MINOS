"""
topology_overhead_plot.py

Runs a topology sweep for a fixed model/chip count and compares
EDP vs. a dreamer baseline, then plots stacked EDP overhead
bars vs. radix (Mesh / Tori / A2A).
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FIG_DIR = "results/figures"
FIG_PATH = os.path.join(FIG_DIR, "topology_overhead.png")
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
        # Keep sweep running; caller will treat this as a failed run
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "resnet18_hybrid"
GENERATION = 6
CHIPS = 64

# radix -> list of (topology-name, category)
# category controls which stack (Mesh / Tori / A2A) it contributes to
TOPOLOGY_GROUPS = {
    2: [("1D_MESH", "Mesh"), ("1D_RING", "Tori")],
    4: [("2D_MESH", "Mesh"), ("2D_TORUS", "Tori")],
    8: [("3D_MESH", "Mesh"), ("3D_TORUS", "Tori")],
    64: [("ALL_TO_ALL", "A2A")],
}

CATEGORIES = ["Mesh", "Tori", "A2A"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Topology sweep (EDP overhead) ===")

    # Figure / font style similar to the paper
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    # baseline (dreamer)
    e_base, d_base = run_command(
        [
            "python", "dreamer.py",
            "--model", MODEL,
            "--preloading", "0",
            "--solution", "0",
            "--generation", str(GENERATION),
            "--chips", str(CHIPS),
        ]
    )
    if e_base is None or d_base is None:
        print("ERROR: baseline run failed")
        sys.exit(1)
    edp_base = e_base * d_base
    print(f"[INFO] baseline EDP = {edp_base:.4f}")

    # results[radix][category] = overhead_pct
    results = {radix: {cat: None for cat in CATEGORIES}
               for radix in TOPOLOGY_GROUPS.keys()}

    for radix, topo_list in TOPOLOGY_GROUPS.items():
        for topo_name, category in topo_list:
            print(f"[INFO] running partition for radix={radix}, topo={topo_name}")
            e, d = run_command(
                [
                    "python", "partition.py",
                    "--model", MODEL,
                    "--preloading", "0",
                    "--solution", "0",
                    "--generation", str(GENERATION),
                    "--chips", str(CHIPS),
                    "-tp", topo_name,
                ]
            )
            if e is not None and d is not None:
                edp = e * d
                overhead_pct = (edp / edp_base - 1.0) * 100.0
            else:
                overhead_pct = None
            print(f"[RES] radix={radix}, topo={topo_name} -> overhead={overhead_pct}")
            results[radix][category] = overhead_pct

    # --- stacked bar plot ---
    radices = sorted(results.keys())
    x = np.arange(len(radices))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # fixed colors to mimic the paper
    mesh_color = "#4575b4"   # blue-ish
    tori_color = "#fdae61"   # orange-ish
    a2a_color = "#4daf4a"    # green-ish

    mesh_drawn = tori_drawn = a2a_drawn = False

    for i, radix in enumerate(radices):
        bottom = 0.0

        mesh_val = results[radix].get("Mesh")
        if mesh_val is not None:
            ax.bar(
                x[i],
                mesh_val,
                width=bar_width,
                color=mesh_color,
                label="Mesh" if not mesh_drawn else None,
            )
            mesh_drawn = True
            bottom += mesh_val

        tori_val = results[radix].get("Tori")
        if tori_val is not None:
            ax.bar(
                x[i],
                tori_val,
                width=bar_width,
                bottom=bottom,
                color=tori_color,
                label="Tori" if not tori_drawn else None,
            )
            tori_drawn = True
            bottom += tori_val

        a2a_val = results[radix].get("A2A")
        if a2a_val is not None:
            ax.bar(
                x[i],
                a2a_val,
                width=bar_width,
                bottom=bottom,
                color=a2a_color,
                label="A2A" if not a2a_drawn else None,
            )
            a2a_drawn = True

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in radices])
    ax.set_xlabel("Radix")
    ax.set_ylabel("EDP Overhead vs. Dream(%)")

    # match clean style: no grid, thicker frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    print(f"[INFO] saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
