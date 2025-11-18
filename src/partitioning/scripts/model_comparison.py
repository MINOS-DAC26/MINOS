"""
model_comparison.py

Runs a standardized set of experiments:
1. Base models with base configuration
2. Larger chip memory variants with more subaccelerators
3. High chiplet-count stress tests

For every scenario we:
- first run dreamer.py to get the lower bound / “dream” configuration
- then run partition.py to actually optimize the chiplet configuration

We run each scenario under:
    (a) powergating model
    (b) standard leakage behavior (-l 1)
and then we repeat both with the multi-input target:
    (c) --target EPT
    (d) --target EPT -l 1
"""

import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import sys
import csv
import re
from typing import List, Dict, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# CSV setup
# ---------------------------------------------------------------------------

CSV_FILE = "results/figures/model_comparison_results.csv"
os.makedirs("results/figures", exist_ok=True)

# results[scenario_name][target_key]["NoPG"|"PG"] = {
#     "E": float, "D": float or "T": float, "EDP_or_EPT": float
# }
results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}


def write_csv() -> None:
    """Rewrite the whole CSV from the current `results` dict."""
    fieldnames = [
        "Model",
        "EDP_NoPG_E", "EDP_NoPG_D", "EDP_NoPG_EDP",
        "EDP_PG_E", "EDP_PG_D", "EDP_PG_EDP",
        "EPT_NoPG_E", "EPT_NoPG_T", "EPT_NoPG_EPT",
        "EPT_PG_E", "EPT_PG_T", "EPT_PG_EPT",
    ]
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # write rows in deterministic order
        for scenario in sorted(results.keys()):
            row = {"Model": scenario}
            scenario_data = results[scenario]

            # EDP block
            edp_block = scenario_data.get("EDP", {})
            no_pg_edp = edp_block.get("NoPG", {})
            pg_edp = edp_block.get("PG", {})

            row["EDP_NoPG_E"] = format_pct(no_pg_edp.get("E"))
            row["EDP_NoPG_D"] = format_pct(no_pg_edp.get("D"))
            row["EDP_NoPG_EDP"] = format_pct(no_pg_edp.get("EDP"))

            row["EDP_PG_E"] = format_pct(pg_edp.get("E"))
            row["EDP_PG_D"] = format_pct(pg_edp.get("D"))
            row["EDP_PG_EDP"] = format_pct(pg_edp.get("EDP"))

            # EPT block
            ept_block = scenario_data.get("EPT", {})
            no_pg_ept = ept_block.get("NoPG", {})
            pg_ept = ept_block.get("PG", {})

            row["EPT_NoPG_E"] = format_pct(no_pg_ept.get("E"))
            row["EPT_NoPG_T"] = format_pct(no_pg_ept.get("T"))
            row["EPT_NoPG_EPT"] = format_pct(no_pg_ept.get("EPT"))

            row["EPT_PG_E"] = format_pct(pg_ept.get("E"))
            row["EPT_PG_T"] = format_pct(pg_ept.get("T"))
            row["EPT_PG_EPT"] = format_pct(pg_ept.get("EPT"))

            writer.writerow(row)


def format_pct(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val:.2f}%"


# ---------------------------------------------------------------------------
# Parsing helper (last line only)
# ---------------------------------------------------------------------------

def parse_metrics(output: str) -> Dict[str, float]:
    """
    Parse ONLY the last non-empty line of the output.

    Primary mode:
        - look for keywords (energy, delay, throughput) on that line.

    Fallback mode:
        - if no keywords are found, treat the line as a numeric triple:
          energy, delay, inv_throughput
    """
    lines = [l for l in output.splitlines() if l.strip()]
    if not lines:
        return {}

    last_line = lines[-1]
    metrics: Dict[str, float] = {}

    # --- keyword-based parsing on last line ---
    energy = _find_number_in_line(last_line, ["energy", "e=", "total energy"])
    if energy is not None:
        metrics["energy"] = energy

    delay = _find_number_in_line(last_line, ["delay", "latency", "d="])
    if delay is not None:
        metrics["delay"] = delay

    inv_throughput = _find_number_in_line(
        last_line,
        ["inverse throughput", "inv throughput", "et^-1", "et_inv", "t=", "throughput"],
    )
    if inv_throughput is not None:
        metrics["inv_throughput"] = inv_throughput

    # --- fallback: bare numeric triple like "E, D, invT" ---
    if not metrics:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", last_line)
        vals = [float(x) for x in nums]

        if len(vals) >= 1:
            metrics["energy"] = vals[0]
        if len(vals) >= 2:
            metrics["delay"] = vals[1]
        if len(vals) >= 3:
            metrics["inv_throughput"] = vals[2]

    return metrics


def _find_number_in_line(line: str, keywords: List[str]) -> Optional[float]:
    lower = line.lower()
    if any(k in lower for k in keywords):
        m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Helpers for running commands
# ---------------------------------------------------------------------------

def run_command(cmd: List[str]) -> str:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout)
    return proc.stdout


def base_args(model_name: str) -> List[str]:
    return [
        "--model", model_name,
        "--generation", "0",
        "--preloading", "0",
        "--solution", "0",
    ]


def add_optional_flags(args: List[str],
                       leakage: bool = False,
                       target: Optional[str] = None,
                       extra: Optional[List[str]] = None) -> List[str]:
    final = list(args)
    if leakage:
        final.extend(["-l", "1"])
    if target:
        final.extend(["--target", target])
    if extra:
        final.extend(extra)
    return final


# key helpers ---------------------------------------------------------------

def scenario_name(model_name: str, extra: Optional[List[str]]) -> str:
    if not extra:
        return model_name
    parts = []
    i = 0
    while i < len(extra):
        if extra[i].startswith("--") and i + 1 < len(extra):
            parts.append(f"{extra[i][2:]}={extra[i+1]}")
            i += 2
        else:
            parts.append(extra[i])
            i += 1
    return f"{model_name}[{','.join(parts)}]"


def leakage_key(leakage: bool) -> str:
    # leakage=False -> powergating model -> "PG"
    # leakage=True  -> -l 1 -> "NoPG"
    return "PG" if not leakage else "NoPG"


def target_key(target: Optional[str]) -> str:
    return "EPT" if target == "EPT" else "EDP"


# dream store: (scenario, target, "PG") -> metrics
# we deliberately keep only ONE baseline per scenario+target (the PG one)
dream_store: Dict[Tuple[str, str, str], Dict[str, float]] = {}


def run_dreamer(model_name: str,
                leakage: bool = False,
                target: Optional[str] = None,
                extra: Optional[List[str]] = None) -> None:
    cmd = ["python", "dreamer.py"]
    cmd.extend(add_optional_flags(base_args(model_name), leakage, target, extra))
    out = run_command(cmd)
    metrics = parse_metrics(out)

    scen = scenario_name(model_name, extra)
    tkey = target_key(target)

    # store ONLY the PG baseline so that -l 1 uses the same baseline
    if not leakage:
        dream_store[(scen, tkey, "PG")] = metrics
        print(f"[INFO] Stored dream metrics for {scen}, {tkey}, PG: {metrics}")
    else:
        print(f"[INFO] Ran dreamer for {scen}, {tkey}, NoPG (not stored as baseline)")


def run_partition(model_name: str,
                  leakage: bool = False,
                  target: Optional[str] = None,
                  extra: Optional[List[str]] = None) -> None:
    cmd = ["python", "partition.py"]
    cmd.extend(add_optional_flags(base_args(model_name), leakage, target, extra))
    out = run_command(cmd)
    part_metrics = parse_metrics(out)

    scen = scenario_name(model_name, extra)
    tkey = target_key(target)
    lkey = leakage_key(leakage)

    # always use the PG baseline
    dream = dream_store.get((scen, tkey, "PG"))
    if dream is None:
        # maybe reuse base scenario (no extra) PG baseline
        base_scen = model_name
        dream = dream_store.get((base_scen, tkey, "PG"))
        if dream is None:
            print(f"[WARN] No PG baseline found for {scen}, {tkey}, skipping overhead calc.")
            return

    # compute / store overheads
    scenario_results = results.setdefault(scen, {})
    target_results = scenario_results.setdefault(tkey, {})
    leak_results = target_results.setdefault(lkey, {})

    # energy overhead
    if "energy" in dream and "energy" in part_metrics and dream["energy"] != 0:
        e_over = (part_metrics["energy"] - dream["energy"]) / dream["energy"] * 100.0
        leak_results["E"] = e_over

    if tkey == "EDP":
        if "delay" in dream and "delay" in part_metrics and dream["delay"] != 0:
            d_over = (part_metrics["delay"] - dream["delay"]) / dream["delay"] * 100.0
            leak_results["D"] = d_over

        if ("energy" in dream and "delay" in dream and
            "energy" in part_metrics and "delay" in part_metrics and
            (dream["energy"] * dream["delay"]) != 0):
            dream_edp = dream["energy"] * dream["delay"]
            part_edp = part_metrics["energy"] * part_metrics["delay"]
            edp_over = (part_edp - dream_edp) / dream_edp * 100.0
            leak_results["EDP"] = edp_over
    else:  # EPT
        if ("inv_throughput" in dream and "inv_throughput" in part_metrics and
            dream["inv_throughput"] != 0):
            t_over = (
                (part_metrics["inv_throughput"] - dream["inv_throughput"])
                / dream["inv_throughput"] * 100.0
            )
            leak_results["T"] = t_over

        if ("energy" in dream and "inv_throughput" in dream and
            "energy" in part_metrics and "inv_throughput" in part_metrics and
            (dream["energy"] * dream["inv_throughput"]) != 0):
            dream_ept = dream["energy"] * dream["inv_throughput"]
            part_ept = part_metrics["energy"] * part_metrics["inv_throughput"]
            ept_over = (part_ept - dream_ept) / dream_ept * 100.0
            leak_results["EPT"] = ept_over

    # rewrite CSV
    write_csv()
    print(f"[INFO] Updated CSV after {scen}, {tkey}, {lkey}")


# ---------------------------------------------------------------------------
# Configuration (your original blocks)
# ---------------------------------------------------------------------------

BASE_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "bert",
    "bert-large",
    "llama1b",
    "llama8b",
    "llama70b",
]

LARGER_CHIP_VARIANTS: List[Dict[str, Any]] = [
    {
        "model": "bert",
        "extra": ["--chips", "4", "--subAccs", "2"],
    },
    {
        "model": "bert-large",
        "extra": ["--chips", "8", "--subAccs", "3"],
    },
]

HIGH_CHIPLET_TESTS: List[Dict[str, Any]] = [
    {
        "model": "llama1b",
        "extra": ["--chips", "128"],
    },
    {
        "model": "llama8b",
        "extra": ["--chips", "128"],
    },
]

LEAKAGE_VARIANTS = [False, True]    # False -> PG, True -> -l 1
TARGET_VARIANTS = [None, "EPT"]     # None -> EDP, "EPT" -> ...
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n=== Block 1: Base models with base configuration ===")
    for model in BASE_MODELS:
        model_hybrid = f"{model}_hybrid"
        for target in TARGET_VARIANTS:
            for leakage in LEAKAGE_VARIANTS:
                # dreamer first (for leakage=True we run but don't store)
                run_dreamer(model_hybrid, leakage=leakage, target=target)
                # then actual optimization
                run_partition(model_hybrid, leakage=leakage, target=target)

    print("\n=== Block 2: Larger chip memory / explicit chip layout ===")
    for variant in LARGER_CHIP_VARIANTS:
        model_hybrid = f"{variant['model']}_hybrid"
        extra = variant["extra"]
        for target in TARGET_VARIANTS:
            for leakage in LEAKAGE_VARIANTS:
                # only partition here, reuse PG baseline from block 1
                run_partition(model_hybrid,
                              leakage=leakage,
                              target=target,
                              extra=extra)

    print("\n=== Block 3: Testing against high chiplet counts ===")
    for test in HIGH_CHIPLET_TESTS:
        model_hybrid = f"{test['model']}_hybrid"
        extra = test["extra"]
        for target in TARGET_VARIANTS:
            for leakage in LEAKAGE_VARIANTS:
                # for high chiplet we actually run dreamer too
                run_dreamer(model_hybrid,
                            leakage=leakage,
                            target=target,
                            extra=extra)
                run_partition(model_hybrid,
                              leakage=leakage,
                              target=target,
                              extra=extra)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
