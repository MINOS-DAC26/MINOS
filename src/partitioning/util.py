# util.py

import numpy as np
import scipy.sparse as sp
import networkx as nx
import os
import time
import numpy as np
import networkx as nx
import gurobipy as gp
import wandb
import json
from gurobipy import Model
from heuristics import choose_heuristic


def get_incomparables(G):
    num_nodes = len(G.nodes)
    incomparable = np.ones((num_nodes, num_nodes), dtype=int) - np.identity(num_nodes, dtype=int)
    for node in G.nodes:
        for descendant in nx.descendants(G, node):
            incomparable[node, descendant] = 0
            incomparable[descendant, node] = 0
    I = sp.coo_matrix(incomparable)
    I_upper = sp.triu(I, k=1).tocoo()
    num_incomparable = I_upper.nnz
    return  I, I_upper, num_incomparable


# Precomputed model readout
def read_model(problem_key, wb):
    if os.path.exists(f"models/{problem_key}.mps.gz") or os.path.exists(f"models/{problem_key}.mps"):
        print(f"🔍 Model found at models/{problem_key}.mps(.gz)")
        start_time = time.time()
        model = gp.read(f"models/{problem_key}.mps")
        if wb: wandb.log({"Model Reading Time": time.time() - start_time})
        return model, True
    else:
        print(f"⚠️  Model {problem_key} not found. Setting model up from scratch again ... \n")
        return Model("minimize_edp"), False
    

# Save solution
def save_solution(model, problem_key):
    previous_data = {}
    os.path.exists('results') or os.makedirs('results', exist_ok = True)
    if os.path.exists("results/optimization_results.json"):
        with open("results/optimization_results.json", "r") as f: previous_data = json.load(f)
    else:
        with open("results/optimization_results.json", "w") as f:
            json.dump({}, f)
    previous_best = previous_data.get(problem_key, {}).get("best_objective", None)
    previous_bound = previous_data.get(problem_key, {}).get("objective_bound", None)

    if not previous_best or (model.ObjVal < previous_best and model.ObjBound >= previous_bound):
        previous_data[problem_key] = {"best_objective": model.ObjVal, "objective_bound": model.ObjBound}
        with open("results/optimization_results.json", "w") as f: json.dump(previous_data, f, indent=4)
        model.write(f"solutions/{problem_key}.sol.gz")
        print(f"✏️  Updated solution and objective bound for problem: {problem_key}")
    else:
        # model.write(f"solutions/{problem}.sol.gz")
        print(f"📉 New solution ({model.ObjVal:.3e}) is not better than existing ({previous_best:.3e}). Skipping save.")


# Warm starting
def warm_start(model, G, **kwargs):
    problem = kwargs.get("problem")
    lookups = kwargs.get("lookups")
    heuristic = kwargs.get("heuristic")
    dream = kwargs.get("dream")
    num_chips = kwargs.get("num_chips")
    chip_sizes = kwargs.get("chip_sizes")
    num_subaccs = kwargs.get("num_subaccs")
    max_number_of_slots = kwargs.get("max_number_of_slots")
    I_upper = kwargs.get("I_upper")
    powergating = kwargs.get("powergating")
    model_vars = kwargs.get("model_vars")
    H = kwargs.get("H")
    target = kwargs.get("target")
    wakeup_time = kwargs.get("wakeup_time")
    solution = kwargs.get("solution")
    dream_bound_value = kwargs.get("dream_bound_value")
    idle_leakage = kwargs.get("idle_leakage")
    w2s_etime = kwargs.get("w2s_etime")
    latency_ub = kwargs.get("latency_ub")
    overlap = kwargs.get("overlap")
    bank_size = kwargs.get("bank_size")
    
    # Read existing solution file
    best_bound = dream_bound_value
    value_instead_bound = len(lookups) > 1 
    for lookup in lookups:
        best_bound = read_lower_bound(model, lookup.get_solution_key(), target, best_bound, value_instead_bound=value_instead_bound)
        
    solution_loaded = False
    if solution: 
        solution_loaded = read_previous_solution(model, problem.get_solution_key())
    # Or run heuristics
    if not solution_loaded:
        print(f"🔍 Running heuristics {heuristic} for warm start ...")
        msts = choose_heuristic(heuristic, G, dream=dream,
                            num_chips=num_chips, 
                            chip_sizes=chip_sizes,
                            num_subaccs=num_subaccs,
                            max_number_of_slots=max_number_of_slots,
                            epsilon=wakeup_time,
                            I_upper=I_upper,
                            fheader=f"starts/{problem.get_solution_key()}",
                            model_vars=model_vars,
                            H=H,
                            powergating=powergating,
                            idle_leakage=idle_leakage,
                            w2s_etime=w2s_etime,
                            latency_ub=latency_ub,
                            overlap=overlap,
                            bank_size=bank_size, 
                            target=target,
                            subaccs_per_chip=kwargs.get("subaccs_per_chip", []),
                            )    
        set_start_values_from_mst_chunked(msts, model)


def set_start_values_from_mst_chunked(msts, model, chunk_size=100000):
    model.NumStart = len(msts)  
    for i, mst in enumerate(msts):
        model.params.StartNumber = i  
        print(f"Processing MST file {i + 1}: {mst}")
        
        with open(mst, 'r') as file:
            chunk = []
            line_count = 0
            
            for line in file:
                if line.startswith('#') or not line.strip(): continue
                chunk.append(line.strip())
                line_count += 1
                
                if line_count == chunk_size:
                    process_chunk(chunk, model)
                    chunk = []  
                    line_count = 0  
            
            if chunk: process_chunk(chunk, model)
        model.update()


def process_chunk(chunk, model):
    for line in chunk:
        parts = line.split()
        if len(parts) != 2: continue
        
        var_name = parts[0]
        var_value = float(parts[1])

        try:
            var = model.getVarByName(var_name)
            if var: var.setAttr('Start', var_value)
        except Exception as e: print(f"Error setting start value for {var_name}: {e}")


# Read lower bound
def read_lower_bound(model, lookup_key, target, best_value, value_instead_bound):
    if not os.path.exists("results/optimization_results.json"):
        print(f"⚠️  No previous results file found for {lookup_key}.")
        return best_value
    with open("results/optimization_results.json", "r") as f:
        previous_entry = json.load(f).get(lookup_key, None)
    if previous_entry:
        target_var = model.getVarByName(f'Total_{target}')
        if value_instead_bound: best_bound = previous_entry.get("best_objective", None)
        else: best_bound = previous_entry.get("objective_bound", None)
        if target_var and best_value < best_bound:
            target_var.LB = best_bound
            print(f"✅ Applied previous objective bound {target_var.LB:.3e} from {lookup_key}!")
            return best_bound
        else: 
            print(f"📉 Read bound value {best_bound:.3e} is not better than existing lower bound {best_value:.3}.")
            return best_value
    else: 
        print(f"⚠️  No previous bound entry found for {lookup_key}.")
        return best_value


# Read previous solution
def read_previous_solution(model, lookup_key):
    if os.path.exists(f"solutions/{lookup_key}.sol.gz"):
        print(f"📚 Preloading solution from earlier setup ...")
        model.NumStart = 1
        model.params.StartNumber = 0
        model.read(f"solutions/{lookup_key}.sol")
        model.update()
        return True
    else: 
        print(f"⚠️  No previous solution found for {lookup_key}.")
        return False


# Load model variables
def load_model_vars(model, num_nodes, num_chips, num_edges, num_subaccs, max_number_of_slots, I_upper):
    x = np.array([model.getVarByName(f"Assignment[{i},{j}]") for i in range(num_nodes) for j in range(num_chips)]).reshape(num_nodes, num_chips)
    if num_chips != 1:
        c = np.array([model.getVarByName(f"Communication_req[{i}]") for i in range(num_edges)])
    else:
        c = np.zeros
    y = np.array([model.getVarByName(f"SA_Assignment[{i},{j},{k}]") for i in range(num_nodes) for j in range(num_chips) for k in range(num_subaccs)]).reshape(num_nodes, num_chips, num_subaccs)
    incomparable_pairs = list(zip(I_upper.row, I_upper.col))
    yz, w = {}, {}
    for i, j in zip(I_upper.row, I_upper.col):
        yz[(i, j)] = model.getVarByName(f"yz_{i}_{j}")
        w[(i, j)] = model.getVarByName(f"w_{i}_{j}")

    t = np.array([model.getVarByName(f"Latency[{i}]") for i in range(num_nodes)])
        
    node_slot_assignment = np.array([model.getVarByName(f"NodeSlotAssignment[{j},{k},{s},{i}]") for i in range(num_nodes) for j in range(num_chips) for k in range(num_subaccs) for s in range(max_number_of_slots)]).reshape(num_chips, num_subaccs, max_number_of_slots,num_nodes)
    slot_used = np.array([model.getVarByName(f"SlotUsed[{j},{k},{s}]") for j in range(num_chips) for k in range(num_subaccs) for s in range(max_number_of_slots)]).reshape(num_chips, num_subaccs, max_number_of_slots)
    num_non_empty_slots = np.array([model.getVarByName(f"NumNonEmptySlots[{j},{k}]") for j in range(num_chips) for k in range(num_subaccs)]).reshape(num_chips, num_subaccs)
    sa_idle_time = np.array([model.getVarByName(f"SASlotIdling[{i},{j}]") for i in range(num_chips) for j in range(num_subaccs)]).reshape(num_chips, num_subaccs)
    return x, t, c, y, yz, w, node_slot_assignment, slot_used, num_non_empty_slots, sa_idle_time


# Slot time extraction
def extract_slot_times(model, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    max_number_of_slots = kwargs.get("max_number_of_slots")
    matrix_load_vars = kwargs.get("matrix_load_vars")
    if matrix_load_vars:
        return {
            (chip_id, sa_id): [
                (model.getVarByName(f"SlotStartTime[{chip_id*num_subaccs*max_number_of_slots+sa_id*max_number_of_slots+slot_id}]").X,
                model.getVarByName(f"SlotEndTime[{chip_id*num_subaccs*max_number_of_slots+sa_id*max_number_of_slots+slot_id}]").X)
                for slot_id in range(max_number_of_slots) if (slot_used_var := model.getVarByName(f"SlotUsed[{chip_id*num_subaccs*max_number_of_slots+sa_id*max_number_of_slots+slot_id}]")) and slot_used_var.X > 0.99
            ]
            for chip_id in range(num_chips) for sa_id in range(num_subaccs)
            if any(
                (slot_used_var := model.getVarByName(f"SlotUsed[{chip_id*num_subaccs*max_number_of_slots+sa_id*max_number_of_slots+slot_id}]")) and slot_used_var.X > 0.99
                for slot_id in range(max_number_of_slots)
            )
        }
    else:
        return {
                (chip_id, sa_id): [
                    (model.getVarByName(f"SlotStartTime[{chip_id},{sa_id},{slot_id}]").X,
                    model.getVarByName(f"SlotEndTime[{chip_id},{sa_id},{slot_id}]").X)
                    for slot_id in range(max_number_of_slots) if (slot_used_var := model.getVarByName(f"SlotUsed[{chip_id},{sa_id},{slot_id}]")) and slot_used_var.X > 0.99
                ]
                for chip_id in range(num_chips) for sa_id in range(num_subaccs)
                if any(
                    (slot_used_var := model.getVarByName(f"SlotUsed[{chip_id},{sa_id},{slot_id}]")) and slot_used_var.X > 0.99
                    for slot_id in range(max_number_of_slots)
                )
            }
    

# Extract assignments
def get_assignments(G, x, y, num_chips, num_subaccs):
    return [
        {
            "Machine": chip_id,
            "Subaccs": {
                f"SA_{sa_id}": [
                    node_id for node_id in G.nodes
                    if (x[node_id, chip_id].X if num_subaccs == 1 else y[node_id, chip_id, sa_id].X) > 0.99
                ]
                for sa_id in range(num_subaccs)
            }
        }
        for chip_id in range(num_chips)
    ]


# Extract capacity usage
def capacity_used_per_chip(G, x, num_chips, chip_sizes):
    capacity_used_per_chip = []
    for chip_id in range(num_chips):
        total_capacity = sum(G.nodes[node_id]['size'] for node_id in G.nodes if x[node_id, chip_id].X > 0.99)
        capacity_used_per_chip.append({'chip': chip_id, 'usage': round(total_capacity/chip_sizes[chip_id],3)})
    return capacity_used_per_chip


# Extract message traffic
def message_traffic(G, c, x, num_chips):
    message_traffic = []
    for e, (u, v) in enumerate(G.edges):
        source_chip = next(ch for ch in range(num_chips) if x[u, ch].X > 0.99)
        dest_chip = next(ch for ch in range(num_chips) if x[v, ch].X > 0.99)
        if source_chip != dest_chip:
            hops = c[e].X
            entry = {'source': u, 'destination': v, 'size': G.edges[u, v]['size'], 'energy': hops * G.edges[u, v]['energy'] if hops else None, 'latency': hops * G.edges[u, v]['latency'] if hops else None}
            message_traffic.append(entry)
    return message_traffic


# Detect conflicts in scheduling
def detect_conflicts(model, G, y, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    latencies = kwargs.get("latencies")

    times = {n: model.getVarByName(f"Latency[{n}]").X for n in G.nodes}
    conflicts = []
    for chip_id in range(num_chips):
        for sa_id in range(num_subaccs):
            timeframes = sorted(
                [(node_id, times[node_id] - latencies[node_id], times[node_id])
                 for node_id in G.nodes if y[node_id, chip_id, sa_id].X > 0.99],
                key=lambda x: x[1])
            for i in range(len(timeframes) - 1):
                (node1, start1, end1), (node2, start2, end2) = timeframes[i], timeframes[i + 1]
                if start2 < end1 and ((end1 - start2) > max(1e-5, 0.01 * min(latencies))):
                    conflicts.append((chip_id, sa_id, node1, node2, (start1, end1), (start2, end2)))
    if conflicts:
        print("⚠️  Conflicting nodes detected:")
        for chip_id, sa_id, node1, node2, timeframe1, timeframe2 in conflicts:
            if timeframe1[1] - timeframe1[0] == 0 or timeframe2[1] - timeframe2[0] == 0: continue
            print(f"    - Chip {chip_id}, SA {sa_id}: Node {node1} {timeframe1} overlaps with Node {node2} {timeframe2}")
    else: print("✅ No conflicts found.")
    return conflicts


def save_results_to_json(problem, value, target):
    os.path.exists('results') or os.makedirs('results', exist_ok = True)
    os.path.exists('results/dream_results.json') or open('results/dream_results.json', 'w').close()
    with open('results/dream_results.json', 'r') as f: 
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = {}
    key = problem.get_solution_key()
    if key not in results.keys():
        results[key] = {}
        results[key][target] = value
    else:
        if results[key][target] > value:
            results[key][target] = value
    with open('results/dream_results.json', 'w') as f: json.dump(results, f, indent=4)


def load_result_from_json(problem, target, num_chips, num_subaccs, dream = True):
    dream_problem = problem.update_values(com=1.0, coe=1.0, c=1, l=0, pg=1, ov=0, p=num_chips*num_subaccs, tp="2D_MESH")
    if not os.path.exists('results/dream_results.json'):
        return None
    with open('results/dream_results.json', 'r') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = {}
    if dream: key = dream_problem.get_solution_key()
    else: key = dream_problem.update_values(x="Dream").get_solution_key()
    
    if key not in results:
        print(f"⚠️  No results found for key {key}.")
        return None
    else:
        if target not in results[key]:
            print(f"⚠️  No target {target} found for key {key}.")
            return None
        else:
            return results[key][target]