# run_cost_sweep_homogeneous_only.py
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import os, json, subprocess, argparse, math
import pandas as pd

# Hard-coded constants
TOTAL_SAS    = 96
BANKS_TOTAL  = 96
MAX_PARALLEL = 2

CSV_PATH = "scripts/96-SA_96-MEM_cost.csv"  # expects columns: SAs, Memories, Cost

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
    """
    Extract EDP from the last line of the log.
    We take the last comma-separated field and parse it as float.
    This is robust whether the line is 'EDCP,Cost,EDP' or just '...,EDP'.
    """
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        return float(parts[-1])
    except:
        return None

def load_cost_table(csv_path):
    """
    Load the CSV with columns: SAs, Memories, Cost.
    Return dict lookup[(sa, mem)] = cost and sets of valid SAs/Mems.
    """
    df = pd.read_csv(csv_path)
    required = {"SAs","Memories","Cost"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must have columns: SAs, Memories, Cost")
    df["SAs"] = df["SAs"].astype(int)
    df["Memories"] = df["Memories"].astype(int)
    lookup = {(int(sa), int(mem)): float(cost)
              for sa, mem, cost in df[["SAs","Memories","Cost"]].itertuples(index=False, name=None)}
    valid_sas  = sorted(df["SAs"].unique().tolist())
    valid_mems = sorted(df["Memories"].unique().tolist())
    return lookup, set(valid_sas), set(valid_mems)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overview_only", "-ov", type=int, choices=[0,1], default=0,
        help="1: only print per-k overview and planned runs, then exit without running anything.")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)

    # Switch optimization target to EDP
    base_cmd = [
        "python","-u","cost.py",
        "--model","llama1b_hybrid",
        "--generation","0",
        "--target","EDP",
        "--preloading", "0",
        "--solution","0",
        "--banks", str(BANKS_TOTAL),
    ]

    # Only support homogeneous runs; consider k that divide TOTAL_SAS
    preferred_k = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96) if k <= TOTAL_SAS]
    order = [k for k in preferred_k if TOTAL_SAS % k == 0]

    output_file = "results/figures/homogeneous-cost-96.json"
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as jf:
                costs = json.load(jf)
        except:
            costs = {}
    else:
        costs = {}
        with open(output_file, "w") as jf:
            json.dump(costs, jf, indent=2, sort_keys=True)

    def save_now(costs_dict):
        with open(output_file, "w") as jf:
            json.dump(costs_dict, jf, indent=2, sort_keys=True)
            jf.flush()
            os.fsync(jf.fileno())

    def already_done(costs_dict, k, subaccs_csv, banks_csv):
        return costs_dict.get(str(k), {}).get(subaccs_csv, {}).get(banks_csv) is not None

    try:
        cost_lookup, valid_sas, valid_mems = load_cost_table(CSV_PATH)
        print(f"Loaded cost table: {CSV_PATH} | entries={len(cost_lookup)} "
              f"| SAs≈[{min(valid_sas)},{max(valid_sas)}] | Memories≈[{min(valid_mems)},{max(valid_mems)}]")
    except Exception as e:
        print(f"ERROR loading cost table: {e}")
        return

    print("\nPlanning overview (homogeneous only, no runs yet):")
    header = "k | Subaccs/chip | Banks/chip | Cost/Chip   | TotalCost   | CoveredInCSV"
    print(header)
    print("-" * len(header))

    planned = []
    total_runs = 0
    for k in order:
        sa_h = TOTAL_SAS // k
        banks_per_chip_h = math.ceil(BANKS_TOTAL / k)
        per_chip_cost = cost_lookup.get((sa_h, banks_per_chip_h))
        covered = per_chip_cost is not None
        total_cost = (per_chip_cost * k) if covered else None
        planned.append((k, sa_h, banks_per_chip_h, per_chip_cost, total_cost, covered))
        total_runs += 1 if covered else 0
        cchip = f"{per_chip_cost:.6f}" if per_chip_cost is not None else "   None"
        ctot  = f"{total_cost:.6f}"     if total_cost is not None     else "   None"
        print(f"{k:>1} | {sa_h:>8} | {banks_per_chip_h:>10} | {cchip:>11} | {ctot:>11} | {str(covered):>12}")

    print(f"\nPlanned homogeneous runs (CSV-covered): {total_runs}")
    if args.overview_only:
        return

    procs, meta = [], []

    for (k, sa_h, banks_per_chip_h, per_chip_cost, total_cost, covered) in planned:
        if not covered:
            print(f"\n[k={k}] Skipping: per-chip cost missing in CSV for (SAs={sa_h}, Memories={banks_per_chip_h}).")
            continue

        subaccs_h = [sa_h] * k
        banks_h = [banks_per_chip_h] * k

        subaccs_csv   = list_to_csv(subaccs_h)
        banks_csv = list_to_csv(banks_h)
        if already_done(costs, k, subaccs_csv, banks_csv):
            print(f"\n[k={k}] -> already in JSON, skipping")
            continue

        sizes_csv = list_to_csv(f"{(1.1/k):.6f}" for _ in range(k))
        max_per_chip = sa_h

        cmd = base_cmd + [
            "--chips", str(k),
            "--subaccs_per_chip", subaccs_csv,
            "--chip_sizes", sizes_csv,
            "--subAccs", str(max_per_chip),
            "--max_subacc", str(TOTAL_SAS),
        ]
        logname = f"logs/chips{k}_homogeneous_subaccs{subaccs_h[0]}_mem{banks_h[0]}.log"
        f = open(logname, "w")
        print("\n->", " ".join(cmd), "| log:", logname)
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        procs.append((p, f))
        meta.append((k, subaccs_csv, banks_csv, logname, total_cost))  # stash total_cost for EDCP later

        if len(procs) >= MAX_PARALLEL:
            p0, f0 = procs.pop(0)
            k0, sas0, mem0, log0, total_cost0 = meta.pop(0)
            p0.wait(); f0.close()
            line = last_nonempty_line(log0)
            edp = parse_edp(line) if line else None
            if edp is not None:
                edcp = edp * total_cost0
                entry = {"EDP": edp, "Cost": total_cost0, "EDCP": edcp}
                costs.setdefault(str(k0), {}).setdefault(sas0, {})[mem0] = entry
                save_now(costs)

    for (p, f), (k, subaccs_csv, banks_csv, logname, total_cost) in zip(procs, meta):
        p.wait(); f.close()
        line = last_nonempty_line(logname)
        edp = parse_edp(line) if line else None
        if edp is not None:
            edcp = edp * total_cost
            entry = {"EDP": edp, "Cost": total_cost, "EDCP": edcp}
            costs.setdefault(str(k), {}).setdefault(subaccs_csv, {})[banks_csv] = entry
            save_now(costs)

    print(f"\nSaved results to homogeneous-cost-96.json")

if __name__ == "__main__":
    main()
