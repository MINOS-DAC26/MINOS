# run_cost_sweep_24pes.py
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import os, json, subprocess, argparse, random, math
import pandas as pd
import numpy as np
import heapq

# Hard-coded constants
TOTAL_SAS   = 12
BANKS_TOTAL = 20
MAX_PARALLEL = 2

# Cost filtering factor (was 10.0 before)
COST_FILTER_FACTOR = 1.5

CSV_PATH = "scripts/12-SA_20-MEM_cost.csv"  # expects columns: SAs, Memories, Cost

def partitions_exact(n, k, min_part=1):
    """
    All non-increasing k-part compositions of n with parts >= min_part.
    Returns a list of tuples in non-increasing order.
    """
    def _rec(remaining, parts_left, max_allowed):
        if parts_left == 1:
            if min_part <= remaining <= max_allowed:
                yield (remaining,)
            return
        start = min(max_allowed, remaining - (parts_left - 1)*min_part)
        for first in range(start, min_part-1, -1):
            for tail in _rec(remaining-first, parts_left-1, first):
                yield (first,) + tail
    return list(_rec(n, k, n))

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

def parse_metrics(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3: return None
    try:
        return {"EDCP": float(parts[0]), "Cost": float(parts[1]), "EDP": float(parts[2])}
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

def total_config_cost_from_vectors(subaccs_parts, banks, cost_lookup):
    """
    Sum per-chip costs for aligned vectors (non-increasing). If any (sa,mem) missing, return None.
    """
    total = 0.0
    gl = cost_lookup.get
    for sa,mem in zip(subaccs_parts, banks):
        c = gl((int(sa), int(mem)))
        if c is None:
            return None
        total += c
    return total

def canonical_pair_key(subaccs_parts, banks):
    """
    Canonical multiset key for a setup, ignoring chip permutations:
      - create (sa, mem) pairs
      - sort descending by (sa, mem)
      - return as a tuple
    This guarantees that e.g. (3,3,1) with banks (4,3,2) is the same
    regardless of which '3' got 4 vs 3 banks.
    """
    return tuple(sorted(zip(subaccs_parts, banks), key=lambda x: (x[0], x[1]), reverse=True))

# ----- Planning helper: compute counts for k without launching runs -----
def plan_counts_for_k(k, samples, homogeneous_only, cost_lookup):
    have_homog = (TOTAL_SAS % k == 0)
    homog_runs = 1 if have_homog else 0

    if homogeneous_only:
        return {
            "k": k,
            "sa_parts": 0,
            "mem_parts": 0,
            "cartesian": 0,
            "feasible_unique": 0,
            "valid_csv": 0,
            "min_cost": None,
            "max_cost": None,
            "range_factor": None,
            "after_fx": 0,
            "hetero_scheduled": 0,
            "homog_runs": homog_runs,
            "total_runs": homog_runs,
        }

    sa_parts_list   = partitions_exact(TOTAL_SAS, k, 1)
    bank_parts_list = partitions_exact(BANKS_TOTAL, k, 1)

    cartesian = len(sa_parts_list) * len(bank_parts_list)

    unique_keys = set()
    feasible_unique_pairs = []
    for subaccs_parts in sa_parts_list:
        for banks in bank_parts_list:
            ok = True
            for sa,mem in zip(subaccs_parts, banks):
                if mem < sa:
                    ok = False
                    break
            if not ok:
                continue
            key = canonical_pair_key(subaccs_parts, banks)
            if key in unique_keys:
                continue
            unique_keys.add(key)
            feasible_unique_pairs.append((subaccs_parts, banks))

    feasible_unique = len(feasible_unique_pairs)

    gl = cost_lookup.get
    min_cost = float("inf")
    max_cost = -float("inf")
    valid_csv = 0
    per_pair_costs = []
    for subaccs_parts, banks in feasible_unique_pairs:
        total = 0.0
        missing = False
        for sa,mem in zip(subaccs_parts, banks):
            c = gl((int(sa), int(mem)))
            if c is None:
                missing = True
                break
            total += c
        if not missing:
            valid_csv += 1
            per_pair_costs.append(total)
            if total < min_cost:
                min_cost = total
            if total > max_cost:
                max_cost = total

    if valid_csv == 0:
        after_fx = 0
        hetero_scheduled = 0
        min_cost_out = None
        max_cost_out = None
        range_factor = None
    else:
        thr = COST_FILTER_FACTOR * min_cost
        after_fx = sum(1 for c in per_pair_costs if c <= thr)
        hetero_scheduled = min(after_fx, samples)
        min_cost_out = min_cost
        max_cost_out = max_cost
        range_factor = (max_cost / min_cost) if min_cost > 0 else None

    total_runs = homog_runs + hetero_scheduled

    return {
        "k": k,
        "sa_parts": len(sa_parts_list),
        "mem_parts": len(bank_parts_list),
        "cartesian": cartesian,
        "feasible_unique": feasible_unique,
        "valid_csv": valid_csv,
        "min_cost": min_cost_out,
        "max_cost": max_cost_out,
        "range_factor": range_factor,
        "after_fx": after_fx,
        "hetero_scheduled": hetero_scheduled,
        "homog_runs": homog_runs,
        "total_runs": total_runs,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", "-sa", type=int, default=1000,
        help=f"Max number of cheapest heterogeneous configs per chip count to run (after {COST_FILTER_FACTOR}x-min filter).")
    ap.add_argument("--homogeneous", "-ho", type=int, choices=[0,1], default=0,
        help=("1: only equal-split cases (k in {1,2,3,4,6,8,12,24} dividing 24), "
              "Subaccs equal, chip_sizes=1/k, banks/chip=ceil(39/k) (over-provision). "
              "0: also enumerate ALL heterogeneous configs."))
    ap.add_argument("--overview_only", "-ov", type=int, choices=[0,1], default=0,
        help="1: only print per-k counts and total planned runs, then exit without running anything.")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)

    base_cmd = [
        "python","-u","cost.py",
        "--model","llama1b_hybrid",
        "--generation","0",
        "--target","EDCP",
        "--preloading","1",
        "--solution","0",
        "--banks", str(BANKS_TOTAL),
    ]

    preferred_k = [k for k in (1,2,3,4,6,12) if k <= TOTAL_SAS]
    if args.homogeneous:
        order = [k for k in preferred_k if TOTAL_SAS % k == 0]
    else:
        order = preferred_k + [k for k in range(1, TOTAL_SAS+1) if k not in preferred_k]

    output_file = "results/figures/homogenous-cost.json" if args.homogeneous else "results/figures/heterogeneous-cost.json"
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

    print("\nPlanning overview (no runs yet):")
    header = (f"k  | SAparts | MemParts | Cartesian | Feasible* | ValidCSV | MinCost   | MaxCost   | RangeX | After{int(COST_FILTER_FACTOR)}x | Hetero | Homog | Total")
    print(header)
    print("-" * len(header))

    grand_total = 0
    for k in order:
        row = plan_counts_for_k(
            k=k,
            samples=args.samples,
            homogeneous_only=bool(args.homogeneous),
            cost_lookup=cost_lookup
        )
        grand_total += row["total_runs"]
        min_str = f"{row['min_cost']:.6f}" if row["min_cost"] is not None else "   None"
        max_str = f"{row['max_cost']:.6f}" if row["max_cost"] is not None else "   None"
        rng_str = f"{row['range_factor']:.3f}x" if row["range_factor"] is not None else "  N/A"
        print(f"{row['k']:>2} | {row['sa_parts']:>7} | {row['mem_parts']:>8} | {row['cartesian']:>9} | "
              f"{row['feasible_unique']:>9} | {row['valid_csv']:>8} | {min_str:>9} | {max_str:>9} | {rng_str:>6} | "
              f"{row['after_fx']:>8} | {row['hetero_scheduled']:>6} | {row['homog_runs']:>5} | {row['total_runs']:>5}")

    print(f"\n(*) 'Feasible*' counts unique setups only (no permutations).")
    print(f"Planned total runs across all k: {grand_total}")

    if args.overview_only:
        return

    procs, meta = [], []

    for k in order:
        print("\n" + "="*100)
        print(f"Chip count k={k}")

        have_homog = (TOTAL_SAS % k == 0)
        if have_homog:
            sa_h = TOTAL_SAS // k
            banks_per_chip_h = math.ceil(BANKS_TOTAL / k)
            subaccs_h = [sa_h] * k
            banks_h = [banks_per_chip_h] * k
            print(f"  Homogeneous config: Subaccs/chip={sa_h}, banks/chip=ceil(39/{k})={banks_per_chip_h}")
        else:
            print(f"  Homogeneous config not possible (k does not divide {TOTAL_SAS}).")

        if args.homogeneous:
            if not have_homog:
                continue
            subaccs_csv   = list_to_csv(subaccs_h)
            banks_csv = list_to_csv(banks_h)
            if already_done(costs, k, subaccs_csv, banks_csv):
                print("-> already in JSON, skipping")
                continue
            sizes_csv = list_to_csv(f"{(1.0/k):.6f}" for _ in range(k))
            max_per_chip = sa_h
            cmd = base_cmd + [
                "--chips", str(k),
                "--subaccs_per_chip", subaccs_csv,
                "--chip_sizes", sizes_csv,
                "--subAccs", str(max_per_chip),
                "--max_subacc", str(TOTAL_SAS),
            ]
            logname = f"logs/chips{k}_homogeneous_subaccs{subaccs_csv.replace(',','-')}_mem{banks_csv.replace(',','-')}.log"
            f = open(logname, "w")
            print("->", " ".join(cmd), "| log:", logname)
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            procs.append((p, f))
            meta.append((k, subaccs_csv, banks_csv, logname))

            if len(procs) >= MAX_PARALLEL:
                p0, f0 = procs.pop(0)
                k0, sas0, mem0, log0 = meta.pop(0)
                p0.wait(); f0.close()
                line = last_nonempty_line(log0)
                m = parse_metrics(line) if line else None
                if m:
                    costs.setdefault(str(k0), {}).setdefault(sas0, {})[mem0] = m
                    save_now(costs)
            continue

        sa_parts_list   = partitions_exact(TOTAL_SAS, k, 1)
        bank_parts_list = partitions_exact(BANKS_TOTAL, k, 1)

        total_cartesian = len(sa_parts_list) * len(bank_parts_list)
        print(f"  SA partitions (sum=24): {len(sa_parts_list)}")
        print(f"  Memory partitions (sum=39, >=1 each): {len(bank_parts_list)}")
        print(f"  Total candidate pairs (cartesian): {total_cartesian}")

        feasible_pairs = []
        unique_keys = set()
        for subaccs_parts in sa_parts_list:
            for banks in bank_parts_list:
                feasible = True
                for sa,mem in zip(subaccs_parts, banks):
                    if mem < sa:
                        feasible = False
                        break
                if not feasible:
                    continue
                key = canonical_pair_key(subaccs_parts, banks)
                if key in unique_keys:
                    continue
                unique_keys.add(key)
                feasible_pairs.append((subaccs_parts, banks))

        print(f"  Feasible under banks_i >= subaccs_i (unique setups): {len(feasible_pairs)}")

        if not feasible_pairs:
            print("  [warn] No feasible configs for this k. Skipping heterogeneous runs.")
            cheapest = []
        else:
            min_cost = float("inf")
            max_cost = -float("inf")
            valid_candidates = 0
            for subaccs_parts, banks in feasible_pairs:
                c = total_config_cost_from_vectors(subaccs_parts, banks, cost_lookup)
                if c is None:
                    continue
                valid_candidates += 1
                if c < min_cost:
                    min_cost = c
                if c > max_cost:
                    max_cost = c
            print(f"  Valid configs with CSV cost: {valid_candidates}")
            if valid_candidates == 0:
                print("  [warn] No valid CSV-covered configs; skipping heterogeneous runs for this k.")
                cheapest = []
            else:
                filtered_count = 0
                topN = args.samples
                heap = []
                thr = COST_FILTER_FACTOR * min_cost
                for subaccs_parts, banks in feasible_pairs:
                    c = total_config_cost_from_vectors(subaccs_parts, banks, cost_lookup)
                    if c is None or c > thr:
                        continue
                    filtered_count += 1
                    item = (-c, subaccs_parts, banks)
                    if len(heap) < topN:
                        heapq.heappush(heap, item)
                    else:
                        if c < -heap[0][0]:
                            heapq.heapreplace(heap, item)

                range_factor = (max_cost / min_cost) if min_cost > 0 else float("inf")
                print(f"  Min total cost: {min_cost:.6f}")
                print(f"  Max total cost: {max_cost:.6f}")
                print(f"  Cost range factor (max/min): {range_factor:.3f}x")
                print(f"  After {COST_FILTER_FACTOR}x-min filter: {filtered_count} remain")
                cheapest = []
                while heap:
                    negc, subaccs_parts, banks = heapq.heappop(heap)
                    cheapest.append((-negc, subaccs_parts, banks))
                cheapest.sort(key=lambda x: x[0])
                print(f"  Scheduled heterogeneous candidates (cap={args.samples}): {len(cheapest)}")

        if have_homog:
            subaccs_csv   = list_to_csv(subaccs_h)
            banks_csv = list_to_csv(banks_h)
            if not already_done(costs, k, subaccs_csv, banks_csv):
                sizes_csv = list_to_csv(f"{(1.0/k):.6f}" for _ in range(k))
                max_per_chip = max(subaccs_h)
                cmd = base_cmd + [
                    "--chips", str(k),
                    "--subaccs_per_chip", subaccs_csv,
                    "--chip_sizes", sizes_csv,
                    "--subAccs", str(max_per_chip),
                    "--max_subacc", str(TOTAL_SAS),
                ]
                logname = f"logs/chips{k}_homogeneous_subaccs{subaccs_csv.replace(',','-')}_mem{banks_csv.replace(',','-')}.log"
                f = open(logname, "w")
                print("->", " ".join(cmd), "| log:", logname)
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                procs.append((p, f))
                meta.append((k, subaccs_csv, banks_csv, logname))
                if len(procs) >= MAX_PARALLEL:
                    p0, f0 = procs.pop(0)
                    k0, sas0, mem0, log0 = meta.pop(0)
                    p0.wait(); f0.close()
                    line = last_nonempty_line(log0)
                    m = parse_metrics(line) if line else None
                    if m:
                        costs.setdefault(str(k0), {}).setdefault(sas0, {})[mem0] = m
                        save_now(costs)
            else:
                print("-> already in JSON, skipping")

        if cheapest:
            for (c, subaccs_parts, banks) in cheapest:
                subaccs_csv   = list_to_csv(subaccs_parts)
                banks_csv = list_to_csv(banks)
                if already_done(costs, k, subaccs_csv, banks_csv):
                    print("-> already in JSON, skipping")
                    continue
                chip_sizes = [b / BANKS_TOTAL for b in banks]
                sizes_csv = list_to_csv(f"{s:.6f}" for s in chip_sizes)
                max_per_chip = max(subaccs_parts)

                cmd = base_cmd + [
                    "--chips", str(k),
                    "--subaccs_per_chip", subaccs_csv,
                    "--chip_sizes", sizes_csv,
                    "--subAccs", str(max_per_chip),
                    "--max_subacc", str(TOTAL_SAS),
                ]
                logname = f"logs/chips{k}_subaccs{subaccs_csv.replace(',','-')}_mem{banks_csv.replace(',','-')}.log"
                f = open(logname, "w")
                print("->", " ".join(cmd), "| log:", logname)
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                procs.append((p, f))
                meta.append((k, subaccs_csv, banks_csv, logname))

                if len(procs) >= MAX_PARALLEL:
                    p0, f0 = procs.pop(0)
                    k0, sas0, mem0, log0 = meta.pop(0)
                    p0.wait(); f0.close()
                    line = last_nonempty_line(log0)
                    m = parse_metrics(line) if line else None
                    if m:
                        costs.setdefault(str(k0), {}).setdefault(sas0, {})[mem0] = m
                        save_now(costs)

    for (p, f), (k, subaccs_csv, banks_csv, logname) in zip(procs, meta):
        p.wait(); f.close()
        line = last_nonempty_line(logname)
        m = parse_metrics(line) if line else None
        if m:
            costs.setdefault(str(k), {}).setdefault(subaccs_csv, {})[banks_csv] = m
            save_now(costs)

    save_now(costs)
    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    main()
