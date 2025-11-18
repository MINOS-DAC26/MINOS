"""
run_cost_sweep_homogeneous_only.py

Runs a homogeneous cost sweep for llama1b_hybrid.

For a fixed total SA and memory budget we:
- load a hardware cost table from CSV (columns: SAs, Memories, Cost)
- consider only chip counts k that divide TOTAL_SAS
- for each such k, construct a homogeneous layout (same Subaccs/chip, same memories/chip)
- if the layout exists in the CSV, run cost.py once for that k
- parse the last line of the log to get EDP
- multiply EDP with the total chip cost to get EDCP
- store results in homogeneous-cost-fixed.json

Usage:
    python run_cost_sweep_homogeneous_only.py
    python run_cost_sweep_homogeneous_only.py --overview_only 1
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import os
import json
import subprocess
import argparse
import math
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOTAL_SAS    = 12
BANKS_TOTAL  = 20
MAX_PARALLEL = 2

CSV_PATH = "scripts/12-SA_20-MEM_cost.csv"  # expects: SAs, Memories, Cost
OUTPUT_JSON = "results/figures/homogeneous-cost-fixed.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_to_csv(xs):
    return ",".join(str(x) for x in xs)


def last_nonempty_line(path):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0:
            step = min(4096, pos)
            pos -= step
            f.seek(pos)
            chunk = f.read(step)
            buf = chunk + buf
            if b"\n" in buf:
                lines = buf.splitlines()
                for line in reversed(lines):
                    if line.strip():
                        return line.decode("utf-8", errors="replace").strip()
                buf = lines[0]
        s = buf.decode("utf-8", errors="replace").strip()
        return s if s else None


def parse_edp(line):
    # last comma-separated field is EDP
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        return float(parts[-1])
    except Exception:
        return None


def load_cost_table(csv_path):
    df = pd.read_csv(csv_path)
    required = {"SAs", "Memories", "Cost"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must have columns: SAs, Memories, Cost")

    df["SAs"] = df["SAs"].astype(int)
    df["Memories"] = df["Memories"].astype(int)

    lookup = {
        (int(sa), int(mem)): float(cost)
        for sa, mem, cost in df[["SAs", "Memories", "Cost"]].itertuples(index=False, name=None)
    }
    valid_sas = sorted(df["SAs"].unique().tolist())
    valid_mems = sorted(df["Memories"].unique().tolist())
    return lookup, set(valid_sas), set(valid_mems)


def save_json(path, obj):
    with open(path, "w") as jf:
        json.dump(obj, jf, indent=2, sort_keys=True)
        jf.flush()
        os.fsync(jf.fileno())


def already_done(store, k, subaccs_csv, banks_csv):
    return store.get(str(k), {}).get(subaccs_csv, {}).get(banks_csv) is not None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--overview_only", "-ov",
        type=int,
        choices=[0, 1],
        default=0,
        help="1: only print planned homogeneous runs, then exit.")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)

    base_cmd = [
        "python", "-u", "cost.py",
        "--model", "llama1b_hybrid",
        "--generation", "0",
        "--target", "EDP",
        "--preloading", "1",
        "--solution", "0",
    ]

    # chip counts that divide TOTAL_SAS
    preferred_k = [k for k in (1, 2, 3, 4, 6, 12) if k <= TOTAL_SAS]
    order = [k for k in preferred_k if TOTAL_SAS % k == 0]

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as jf:
                costs = json.load(jf)
        except Exception:
            costs = {}
    else:
        costs = {}
        save_json(OUTPUT_JSON, costs)

    try:
        cost_lookup, valid_sas, valid_mems = load_cost_table(CSV_PATH)
        print(
            f"Loaded cost table: {CSV_PATH} | entries={len(cost_lookup)} "
            f"| SAs≈[{min(valid_sas)},{max(valid_sas)}] | Memories≈[{min(valid_mems)},{max(valid_mems)}]"
        )
    except Exception as e:
        print(f"ERROR loading cost table: {e}")
        return

    print("\nPlanning overview (homogeneous only):")
    header = "k | Subaccs/chip | Banks/chip | Cost/Chip   | TotalCost   | CoveredInCSV"
    print(header)
    print("-" * len(header))

    planned = []
    covered_runs = 0
    for k in order:
        sa_h = TOTAL_SAS // k
        banks_per_chip_h = math.ceil(BANKS_TOTAL / k)
        per_chip_cost = cost_lookup.get((sa_h, banks_per_chip_h))
        covered = per_chip_cost is not None
        total_cost = (per_chip_cost * k) if covered else None
        planned.append((k, sa_h, banks_per_chip_h, per_chip_cost, total_cost, covered))
        if covered:
            covered_runs += 1

        cchip = f"{per_chip_cost:.6f}" if per_chip_cost is not None else "None"
        ctot = f"{total_cost:.6f}" if total_cost is not None else "None"
        print(f"{k:>1} | {sa_h:>8} | {banks_per_chip_h:>10} | {cchip:>11} | {ctot:>11} | {str(covered):>12}")

    print(f"\nPlanned homogeneous runs (CSV-covered): {covered_runs}")
    if args.overview_only:
        return

    procs = []
    meta = []

    for (k, sa_h, banks_per_chip_h, per_chip_cost, total_cost, covered) in planned:
        if not covered:
            print(f"\n[k={k}] skipping, no cost entry for (SAs={sa_h}, Memories={banks_per_chip_h})")
            continue

        subaccs_h = [sa_h] * k
        banks_h = [banks_per_chip_h] * k

        subaccs_csv = list_to_csv(subaccs_h)
        banks_csv = list_to_csv(banks_h)

        if already_done(costs, k, subaccs_csv, banks_csv):
            print(f"\n[k={k}] already in {OUTPUT_JSON}, skipping")
            continue

        sizes_csv = list_to_csv(f"{(1.0 / k):.6f}" for _ in range(k))
        max_per_chip = sa_h

        cmd = base_cmd + [
            "--chips", str(k),
            "--subaccs_per_chip", subaccs_csv,
            "--chip_sizes", sizes_csv,
            "--subAccs", str(max_per_chip),
            "--max_subacc", str(TOTAL_SAS),
        ]
        logname = (
            f"logs/chips{k}_homogeneous_subaccs{subaccs_csv.replace(',','-')}"
            f"_mem{banks_csv.replace(',','-')}.log"
        )

        print("\n->", " ".join(cmd), "| log:", logname)
        f = open(logname, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        procs.append((p, f))
        meta.append((k, subaccs_csv, banks_csv, logname, total_cost))

        # throttle
        if len(procs) >= MAX_PARALLEL:
            p0, f0 = procs.pop(0)
            k0, sas0, mem0, log0, total_cost0 = meta.pop(0)
            p0.wait()
            f0.close()
            line = last_nonempty_line(log0)
            edp = parse_edp(line)
            if edp is not None:
                edcp = edp * total_cost0
                entry = {"EDP": edp, "Cost": total_cost0, "EDCP": edcp}
                costs.setdefault(str(k0), {}).setdefault(sas0, {})[mem0] = entry
                save_json(OUTPUT_JSON, costs)

    # drain remaining
    for (p, f), (k, subaccs_csv, banks_csv, logname, total_cost) in zip(procs, meta):
        p.wait()
        f.close()
        line = last_nonempty_line(logname)
        edp = parse_edp(line)
        if edp is not None:
            edcp = edp * total_cost
            entry = {"EDP": edp, "Cost": total_cost, "EDCP": edcp}
            costs.setdefault(str(k), {}).setdefault(subaccs_csv, {})[banks_csv] = entry
            save_json(OUTPUT_JSON, costs)

    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
