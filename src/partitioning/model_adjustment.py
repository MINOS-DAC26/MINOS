# model_adjustment.py

from typing import Dict, Tuple, Union, Optional, List
from itertools import product
from functools import lru_cache
import shutil
import json
import time
Number = Union[int, float]


@lru_cache(maxsize=1024)
def _fmt(val: Number) -> str:
    """
    Gurobi-style scientific formatting with:
    - 16 significant digits
    - lowercase 'e'
    - two-digit exponent (e.g. e+03, e-05)
    """
    # step 1: get uppercase form with 16 sigfigs
    s = f"{val:.16E}"               # e.g. "1.2345678901234567E+03"
    mantissa, exp = s.split("E")    # split mantissa / exponent
    
    sign = exp[0]                   # '+' or '-'
    exp_num = int(exp[1:])          # e.g. 3
    exp_str = f"{exp_num:02d}"      # "03"
    
    return f"{mantissa}e{sign}{exp_str}"


def _collect_bound_updates(
    latencies: List[Number],
    bounds: Dict[str, Number],
    *,
    num_chips: int,
    num_subaccs: int,
    max_slots: int,
    power: float,
    idle_factor: float,
    wakeup_cost: float,
) -> Tuple[Dict[str, Number], Dict[str, Number]]:
    """Return (lb_updates, ub_updates)."""

    lb, ub = {}, {}

    if 'latency_ub' in bounds:
        ub_val = bounds['latency_ub']
        ub['Total_Latency']         = ub_val
        ub['Total_Maxload']         = ub_val
        ub['Total_MaxloadPipeline'] = ub_val
        ub['IdleEnergy']            = (num_chips * num_subaccs - 1)*bounds['latency_ub']*power*idle_factor + num_subaccs*num_chips*wakeup_cost
        ub['IdleEnergyPipeline']    = (num_chips * num_subaccs - 1)*bounds['latency_ub']*power*idle_factor + num_subaccs*num_chips*wakeup_cost
        for i in range(len(latencies)):
            ub[f'Latency[{i}]']           = ub_val
            ub[f'TMinusLatencies[{i}]']   = ub_val
        for ci, pi in product(range(num_chips), range(num_subaccs)):
            for si in range(max_slots):
                tag = f'{ci},{pi},{si}'
                ub[f'SlotStartTime[{tag}]']     = ub_val
                ub[f'SlotStartTimeMin[{tag}]']  = ub_val
                ub[f'SlotEndTime[{tag}]']       = ub_val
                ub[f'SlotLoad[{tag}]']          = ub_val
                ub[f'SlotUsed[{tag}]']          = ub_val
                for ni in range(len(latencies)):
                    tag4 = f'{ci},{pi},{si},{ni}'
                    ub[f'SlotStartTimeHelper[{tag4}]'] = ub_val
                    ub[f'SlotEndTimeHelper[{tag4}]']   = ub_val
                    ub[f'HelperProd[{tag4}]']          = ub_val
            tag2 = f'{ci},{pi}'
            ub[f'SASlotIdling[{tag2}]'] = ub_val
            ub[f'sa_load[{tag2}]']      = ub_val

    if 'latency_lb' in bounds:
        lb_val = bounds['latency_lb']
        lb['Total_Latency']      = lb_val

    if 'maxload_lb' in bounds:
        lb['Total_Maxload']          = bounds['maxload_lb']
        lb['Total_MaxloadPipeline']  = bounds['maxload_lb']

    if 'energy_lb' in bounds:
        lb['Total_Energy']           = bounds['energy_lb']
        lb['Total_EnergyPipeline']   = bounds['energy_lb']
    if 'energy_ub' in bounds:
        ub_val = bounds['energy_ub']
        ub['Total_Energy']           = ub_val
        ub['Total_EnergyPipeline']   = ub_val
        ub['Computational_Energy']   = ub_val
        ub['Communication_Energy']   = ub_val
        ub['Wakeup_Energy']          = ub_val

    if 'messages_ub' in bounds:
        ub['Total_Messages'] = bounds['messages_ub']

    if 'energy_lb' in bounds and 'latency_lb' in bounds:
        lb['Total_EDP'] = bounds['energy_lb'] * bounds['latency_lb']
    if 'energy_ub' in bounds and 'latency_ub' in bounds:
        ub['Total_EDP'] = bounds['energy_ub'] * bounds['latency_ub']
    if 'energy_lb' in bounds and 'maxload_lb' in bounds:
        lb['Total_EPT'] = bounds['energy_lb'] * bounds['maxload_lb']
    if 'energy_ub' in bounds and 'latency_ub' in bounds:
        ub['Total_EPT'] = bounds['energy_ub'] * bounds['latency_ub']

    return lb, ub


def _update_json_bounds(
    input_json_path: str,
    output_json_path: str,
    lb_updates: Dict[str, Number],
    ub_updates: Dict[str, Number]
) -> None:
    with open(f"models/vars_{input_json_path}.json", "r") as f:
        data = json.load(f)
    
    for var_name, new_lb in lb_updates.items():
        if var_name in data and "lower_bound" in data[var_name]:
            data[var_name]["lower_bound"] = float(new_lb)
    
    for var_name, new_ub in ub_updates.items():
        if var_name in data and "upper_bound" in data[var_name]:
            data[var_name]["upper_bound"] = float(new_ub)
    
    with open(f"models/vars_{output_json_path}.json", "w") as f:
        json.dump(data, f, indent=2)


def _process_columns(tokens, coeff_updates):
    val = coeff_updates.get((tokens[1], tokens[0]))
    if val is None:
        return False
    tokens[2] = _fmt(val)
    return True


def _process_rhs(tokens, rhs_updates) -> bool:
    val = rhs_updates.get(tokens[1])
    if val is None:
        return False
    tokens[2] = _fmt(val)
    return True


def _process_bounds(tokens, lb_updates, ub_updates) -> bool:
    btype, _, var = tokens[:3]

    if btype == "LO":
        val = lb_updates.get(var)
        if val is not None:
            val_str = _fmt(val)
            if len(tokens) == 3:
                tokens.append(val_str)
            else:
                tokens[3] = val_str
            return True

    elif btype == "UP":
        val = ub_updates.get(var)
        if val is not None:
            tokens[3] = _fmt(val)
            return True

    return False


def _rebuild_with_original_spacing(orig, new_tokens):
    leading_ws = len(orig) - len(orig.lstrip())
    core = orig.lstrip().rstrip("\n")
    gaps = core.count(" ")
    parts = core.split()          
    for i, tok in enumerate(new_tokens):
        parts[i] = tok
    rebuilt = (" " * leading_ws) + " ".join(parts)
    return rebuilt + "\n"


def patch_mps_file(
    input_key: str,
    output_key: str,
    updates: Dict[Tuple[Optional[str], Optional[str]], Number],
    lb_updates: Dict[str, Number],
    ub_updates: Dict[str, Number],
) -> None:
    coeff_updates = {(r, v): val for (r, v), val in updates.items()
                     if r is not None and v is not None}
    rhs_updates   = {r: val for (r, v), val in updates.items() if v is None}

    section = None
    processing = False
    with open(f"models/{input_key}.mps", "r", buffering=1<<20) as fin, open(f"models/{output_key}.mps", "w", buffering=1<<20) as fout:
        write      = fout.write
        lstrip     = str.lstrip
        startswith = str.startswith
        for raw in fin:
            stripped = lstrip(raw)

            if not processing:
                write(raw)
                if startswith(stripped, "COLUMNS"):
                    processing = True        # enable parsing
                    section = "COLUMNS"
                continue

            if startswith(stripped, "QCMATRIX"):          # quadratic part begins
                write(raw)
                shutil.copyfileobj(fin, fout)           # dump the tail verbatim
                break

            first = stripped.split(maxsplit=1)[0]
            if first in {"COLUMNS", "RHS", "BOUNDS"}:
                section = first
                write(raw); continue

            tokens = raw.split()
            changed = False
            if section == "COLUMNS":
                changed = _process_columns(tokens, coeff_updates)
            elif section == "RHS":
                changed = _process_rhs(tokens, rhs_updates)
            elif section == "BOUNDS":
                changed = _process_bounds(tokens, lb_updates, ub_updates)

            if changed:                         # rebuild line with original indent
                raw = _rebuild_with_original_spacing(raw, tokens)
            write(raw)


# ──────────────────────────────────────────────────────────────────────
#  2.  Utility: build *updates* dict from simple arrays & bounds dict
# ──────────────────────────────────────────────────────────────────────
def _build_updates(
    latencies: List[Number],
    energies: List[Number],
    bounds: Dict[str, Number],
    *,
    num_chips: int = 1,
    num_subaccs: int = 1,
    max_slots: int = 1,
    edges: Optional[List[Tuple[int, int]]] = None,
    wakeup_time: float = 0.0,
    power: float = 1.0,                                         
    idle_factor: float = 1.0,
    incomparables: List[Tuple[int, int]],
) -> Dict[Tuple[Optional[str], Optional[str]], Number]:
    
    updates: Dict[Tuple[Optional[str], Optional[str]], Number] = {}

    # 1) ─ RHS rows depending on individual node latencies (and the t-minus variant)
    for i, lat in enumerate(latencies):
        updates[(f"LatencyLowerBound[{i}]", None)]       = lat
        updates[(f"TMinusLatenciesConstr[{i}]", None)]   = -lat

    # 2) ─ Computational energy single row
    updates[("Computational_Energy_Constraint", None)] = float(sum(energies))

    # 3) ─ Edge precedence (optional)
    for e, (u, v) in enumerate(edges):
        row = f"Edge_Latency_Constraints[{e},{u},{v}]"
        updates[(row, None)] = latencies[v]

    # 4) ─ Slot-load coefficients   Σ_n lat[n] · NSA[ci,pi,si,ni]
    idx = 0
    for ci in range(num_chips):
        for pi in range(num_subaccs):
            for si in range(max_slots):
                for ni, lat in enumerate(latencies):
                    row = f"SlotLoad[{ci},{pi},{si}]"
                    var = f"NodeSlotAssignment[{ci},{pi},{si},{ni}]"
                    updates[(row, var)] = -lat
                idx += 1

    # 5) ─ Big-M and helper constraints driven by latency_ub
    latency_ub = bounds.get("latency_ub", None)
    if latency_ub is not None:
        for ci in range(num_chips):
            for pi in range(num_subaccs):
                for si in range(max_slots):
                    # SlotStartTimeBigM
                    row_bm = f"SlotStartTimeBigM[{ci},{pi},{si}]"
                    var_bm = f"SlotUsed[{ci},{pi},{si}]"
                    updates[(row_bm, var_bm)] = -latency_ub
                    updates[(row_bm, None)]   = -latency_ub
                    row_sd = f"SlotDependencyTime[{ci},{pi},{si-1}]"
                    updates[(row_sd, var_bm)] = -(latency_ub+wakeup_time)
                    updates[(row_sd, None)]   = -latency_ub
                    # Slot helper rows
                    for ni in range(len(latencies)):
                        row_h = f"SSTHelper[{ci},{pi},{si},{ni}]"
                        var_h = f"NodeSlotAssignment[{ci},{pi},{si},{ni}]"
                        updates[(row_h, var_h)] = latency_ub
                        updates[(row_h, None)]  = latency_ub

    for ci in range(num_chips):
        for pi in range(num_subaccs):
            row_max = f"max_subaccLoad_Constraint[{ci},{pi}]"
            for ni, lat in enumerate(latencies):
                updates[(row_max, f"SA_Assignment[{ni},{ci},{pi}]")] = -float(lat)
                updates[("IdleEnergy_Constraint", f"SA_Assignment[{ni},{ci},{pi}]")] = power * idle_factor * lat
                updates[("IdleEnergyPipeline_Constraint", f"SA_Assignment[{ni},{ci},{pi}]")] = power * idle_factor * lat    

    for k, (i, j) in enumerate(incomparables):
        # DC1: coeff on z[k] is -L, RHS is latencies[j] - L
        row1 = f"DisjunctiveConstraints1[{k}]"
        updates[(row1, f"z[{k}]")] = -latency_ub
        updates[(row1, None)]         = latencies[j] - latency_ub

        # DC2: coeff on w[k] is +L, coeff on yz[k] is -L, RHS is latencies[i] - L
        row2 = f"DisjunctiveConstraints2[{k}]"
        updates[(row2, f"w[{k}]")]  = +latency_ub
        updates[(row2, f"yz[{k}]")] = -latency_ub
        updates[(row2, None)]         = latencies[i] - latency_ub           

    return updates


# ──────────────────────────────────────────────────────────────────────
#  3.  Single convenience wrapper the user calls
# ──────────────────────────────────────────────────────────────────────
def patch_model_mps(
    input_key: str,
    output_key: str,
    latencies,
    energies,
    bounds,
    *,
    num_chips: int = 1,
    num_subaccs: int = 1,
    max_slots: int = 1,
    edges: Optional[List[Tuple[int, int]]] = None,
    power: float = 1.0,
    idle_factor: float = 1.0,
    wakeup_time: float = 0.0,
    wakeup_cost: float = 0.0,
    incomparables: List[Tuple[int, int]],
) -> None:
    """
    Build update dict from simple arrays & dict, then stream-patch the file.
    """
    start_time = time.time()
    coeff_rhs_updates = _build_updates(
        latencies, energies, bounds,
        num_chips=num_chips, num_subaccs=num_subaccs,
        max_slots=max_slots,
        edges=edges,
        power=power,
        idle_factor=idle_factor,
        wakeup_time=wakeup_time,
        incomparables=incomparables)
    print(f"Built updates in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    lb_updates, ub_updates = _collect_bound_updates(
        latencies, bounds,
        num_chips=num_chips, num_subaccs=num_subaccs, max_slots=max_slots, idle_factor=idle_factor, power=power, wakeup_cost=wakeup_cost)
    print(f"Collected bound updates in {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    patch_mps_file(
        input_key, output_key,
        coeff_rhs_updates,
        lb_updates, ub_updates)
    print(f"Patched MPS file in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    _update_json_bounds(
        input_key, output_key,
        lb_updates, ub_updates)
    print(f"Updated JSON bounds in {time.time() - start_time:.2f} seconds")
