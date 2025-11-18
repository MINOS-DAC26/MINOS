# constraints.py

import numpy as np
from tqdm import tqdm
from gurobipy import GRB, GRB, QuadExpr, hstack
from var_rep import Var
import scipy.sparse as sp
import pandas as pd


# Add chip assignment constraints (x)
def add_chip_assignment_constraints(G, model, **kwargs):
    num_nodes = kwargs.get("num_nodes")
    num_chips = kwargs.get("num_chips")
    chip_sizes = kwargs.get("chip_sizes", [1/num_chips] * num_chips)
    model_vars = kwargs.get("model_vars")
    dynamic = kwargs.get("dynamic")

    node_sizes = np.array([G.nodes[node_id]['size'] for node_id in G.nodes])
    kv_sizes = np.array([G.nodes[node_id].get('kv_size', 0) for node_id in G.nodes])

    # Chip assignment variables
    x = model.addMVar((num_nodes, num_chips), vtype=GRB.BINARY, name="Assignment") 
    model_vars.add_var(Var("Assignment", (num_nodes, num_chips), 0, 1, int))
    if num_chips != 1: model.addConstr(x.sum(axis=1) == 1, name="Node_Assignment")
    else: model.addConstrs((x[n, 0] == 1 for n in range(num_nodes)), name="Nodes_assigned_to_M0")

    # Add load balancing constraint and variable
    load = model.addMVar(num_chips, vtype=GRB.CONTINUOUS, lb=0, ub=1.1, name="Load")
    model_vars.add_var(Var("Load", (num_chips,), 0, 1, float))
    model.addConstr(load == node_sizes @ x, name="LoadConstraint")
    
    if not dynamic:
        model.addConstr(load <= np.array(chip_sizes), name="Load_Limit")
        model.addConstr(kv_sizes @ x <= np.array(chip_sizes), name="KV_Size_Constraint")
    return x, None, None, load


# Add chip cost constraints (exact method)
def add_chip_cost_constraints(model, y, load, **kwargs):
    num_chips = kwargs.get("num_chips")
    model_vars = kwargs.get("model_vars")
    max_subaccs = kwargs.get("max_subaccs")
    num_subaccs = kwargs.get("num_subaccs")
    num_nodes = kwargs.get("num_nodes")
    bank_size = kwargs.get("bank_size")

    # Chip costs
    if num_chips * num_subaccs == 96:
        df = pd.read_csv("scripts/96-SA_96-MEM_cost.csv")
    else:
        df = pd.read_csv("scripts/12-SA_20-MEM_cost.csv")
    sa_levels  = np.sort(df["SAs"].unique())
    mem_levels = np.sort(df["Memories"].unique())
    max_banks = int(mem_levels.max())
    grid = df.pivot(index="SAs", columns="Memories", values="Cost").reindex(index=sa_levels, columns=mem_levels).values.astype(float)
    S, M = len(sa_levels), len(mem_levels)
    SA_mat  = np.tile(sa_levels.reshape(S,1), (1,M)).astype(float)
    MEM_mat = np.tile(mem_levels.reshape(1,M), (S,1)).astype(float)

    # SA USAGE PER CHIP
    sa_active = model.addMVar((num_chips, num_subaccs), vtype=GRB.BINARY, name="SAActive")
    model_vars.add_var(Var("SAActive", (num_chips, num_subaccs), 0, 1, int))
    num_subaccs_used_per_chip = model.addMVar(num_chips, vtype=GRB.INTEGER, lb=0, ub=num_subaccs, name="subAccsUsedPerChip")
    model_vars.add_var(Var("subAccsUsedPerChip", (num_chips,), 0, num_subaccs, int))
    
    model.addConstrs((sa_active[c, p] >= y[n, c, p] for n in range(num_nodes) for c in range(num_chips) for p in range(num_subaccs)), name="SAActive_link_lb")   
    model.addConstr(sa_active.sum() <= max_subaccs, name="max_subaccs_Constraint")
    model.addConstrs((num_subaccs_used_per_chip[c] == sa_active[c, :].sum() for c in range(num_chips)), name="CountSubaccsPerChip")

    # Bucketing chip load into mem banks
    mem_banks = model.addMVar(num_chips, vtype=GRB.INTEGER, lb=0, ub=max_banks, name="MemBanks") 
    model_vars.add_var(Var("MemBanks", (num_chips,), 0, max_banks, int))
    model.addConstr(load <= mem_banks * bank_size, name="MemBanks_up")
    model.addConstr(load >= (mem_banks - 1) * bank_size, name="MemBanks_low")
    model.addConstrs((mem_banks[c] >= num_subaccs_used_per_chip[c] for c in range(num_chips)), name="MinBanksPerSA")

    # Pair selection variables
    cost_pair = model.addMVar((num_chips, S, M), vtype=GRB.BINARY, name="PairSelect"); 
    model_vars.add_var(Var("PairSelect", (num_chips, S, M), 0, 1, int))
    model.addConstrs((cost_pair[c, :, :].sum() == 1 for c in range(num_chips)), name="ChoosePair")
    model.addConstrs((num_subaccs_used_per_chip[c] == (cost_pair[c, :, :] * SA_mat).sum() for c in range(num_chips)), name="MatchSA")
    model.addConstrs((mem_banks[c]  == (cost_pair[c, :, :] * MEM_mat).sum() for c in range(num_chips)), name="MatchMem")

    cost = model.addMVar(num_chips, lb=0.0, ub=float(np.nanmax(grid)), name="Cost"); model_vars.add_var(Var("Cost", (num_chips,), 0, float(np.nanmax(grid)), float))
    model.addConstrs((cost[c] == (cost_pair[c, :, :] * grid).sum() for c in range(num_chips)), name="CostLookup")

    return cost


# Add SA assignment constraints (y)
def add_sa_assignment_constraints(model, x, **kwargs):
    num_nodes = kwargs.get("num_nodes")
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    dynamic = kwargs.get("dynamic", False)
    model_vars = kwargs.get("model_vars")

    # SA assignment variables
    y = model.addMVar((num_nodes, num_chips, num_subaccs), vtype=GRB.BINARY, name="SA_Assignment")
    model_vars.add_var(Var("SA_Assignment", (num_nodes, num_chips, num_subaccs), 0, 1, int))
    
    # Node assigned to exactly one SA on the given chip
    model.addConstr(y.sum(axis=2) == x, name="Chip_SA_Assignment")
    model.addConstr(y.sum(axis=2).sum(axis=1) == 1, name="Must_Schedule")

    # Dynamic resource allocation
    if dynamic:
        saUsed = model.addMVar((num_chips, num_subaccs), vtype=GRB.BINARY, name="SAUsed")
        model.addConstrs((saUsed >= y[k, :, :] for k in range(num_nodes)), name="SAUsed_link")
        model.addConstr(saUsed[:, 1:] <= saUsed[:, :-1], name="Sequential_SA_Use")
    else:
        saUsed = None

    return y, saUsed


# Add communication constraints (c)
def add_communication_constraints(model, G, x, **kwargs):
    num_edges = kwargs.get("num_edges")
    num_chips = kwargs.get("num_chips")
    H = kwargs.get("H")
    model_vars = kwargs.get("model_vars")

    # No communication in case of single chip
    if num_chips == 1: return np.zeros(num_edges, dtype=int)
    if num_chips != 1:
        # Communication variables
        c = model.addMVar(num_edges, vtype=GRB.INTEGER, lb=0, ub=np.max(H), name="Communication_req")
        model_vars.add_var(Var("Communication_req", (num_edges), 0, np.max(H), int))

        # Communication requirements
        for e, (u, v) in enumerate(G.edges):
            expr = QuadExpr()
            expr.addTerms(H.flatten(), [i for i in x[v, :].tolist() for j in range(num_chips)], [i for j in range(num_chips) for i in x[u, :].tolist() ])
            model.addConstr(c[e] == expr, name=f"N{u}_comm_to_N{v}")
        return c
    

# Add latency constraints (t)
def add_latency_constraints(model, G, c, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_nodes = kwargs.get("num_nodes")
    latencies = kwargs.get("latencies")
    latency_ub = kwargs.get("latency_ub")
    model_vars = kwargs.get("model_vars")

    edge_latencies = [G.edges[u, v]['latency'] for u, v in G.edges]
    # Latency variables
    t = model.addMVar(num_nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="Latency")
    model_vars.add_var(Var("Latency", (num_nodes), 0.0, latency_ub, float))

    # Limit finish times by their execution time
    model.addConstr(t >= latencies, name="LatencyLowerBound")
    # Scheduling constraints
    model.addConstrs((t[v] >= t[u] + G.nodes[v]['latency'] + (c[e] * edge_latencies[e] if num_chips != 1 else 0)for e, (u, v) in enumerate(G.edges)), name="Edge_Latency_Constraints")
    return t


# Add disjunctive scheduling constraints
def add_disjunctive_constraints(G, model, y, t, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    latency_ub = kwargs.get("latency_ub")
    I_upper = kwargs.get("I_upper")
    model_vars = kwargs.get("model_vars")
    num_nodes = kwargs.get("num_nodes")

    incomparable_pairs = list(zip(I_upper.row, I_upper.col)) 

    # Shared resource and ordering variables
    yz = model.addMVar(len(incomparable_pairs), vtype=GRB.BINARY, name="yz")
    model_vars.add_var(Var("yz", len(incomparable_pairs), 0, 1, int))
    w = model.addMVar(len(incomparable_pairs), vtype=GRB.BINARY, name="w")
    model_vars.add_var(Var("w", len(incomparable_pairs), 0, 1, int))
    z = model.addMVar(len(incomparable_pairs), vtype=GRB.BINARY, name="z")
    model_vars.add_var(Var("z", len(incomparable_pairs), 0, 1, int))
    yrast = y.reshape(num_nodes,num_chips*num_subaccs)
    y_lookup = {}
    for i in range(num_nodes):
        y_lookup[i] = yrast[i].tolist()
    
    row = []
    col = []
    data = []

    # Batched constraint addition for yz
    batch_size, num_constraints = 100000, len(I_upper.row)
    for batch_start in tqdm(range(0, num_constraints, batch_size), desc="⚙️  Batch Processing yz constraints"):
        batch_end = min(batch_start + batch_size, num_constraints)
        i_batch, j_batch = I_upper.row[batch_start:batch_end], I_upper.col[batch_start:batch_end]
        for k, (i, j) in enumerate(zip(i_batch, j_batch)):
            expr = QuadExpr()
            expr.addTerms([1]*num_chips*num_subaccs, y_lookup[i], y_lookup[j])
            model.addConstr(yz[batch_start+k] >= expr, name=f"yz_{i}_{j}_lb")

    model.addConstr(z <= yz, name="z_lb1")
    model.addConstr(z <= w, name="z_lb2")
    model.addConstr(z >= yz + w - 1, name="z_lb3")

    row = []
    col = []
    data = []
    row2 = []
    col2 = []
    data2 = []

    vars = hstack((yz,z,t))
    num_incomparables = len(incomparable_pairs)
    b = []
    b2 = []
    for k in range(num_incomparables):
        i,j = incomparable_pairs[k]
        b.append(G.nodes[j]['latency'] - latency_ub)
        b2.append(G.nodes[i]['latency'] - latency_ub)
        row.extend([k,k,k])
        col.extend([num_incomparables+k, num_incomparables*2+i, num_incomparables*2+j])
        data.extend([-latency_ub, -1, 1])
        row2.extend([k,k,k,k])
        col2.extend([k,num_incomparables+k,num_incomparables*2+i, num_incomparables*2+j])
        data2.extend([-latency_ub, latency_ub, 1, -1])

    A = sp.coo_matrix((data, (row, col)), shape=(num_incomparables, num_incomparables*2+num_nodes))
    b = np.array(b)
    model.addConstr(A @ vars >= b, name="DisjunctiveConstraints1")
    
    A2 = sp.coo_matrix((data2, (row2, col2)), shape=(num_incomparables, num_incomparables*2+num_nodes))
    b2 = np.array(b2)
    model.addConstr(A2 @ vars >= b2, name="DisjunctiveConstraints2")

    return yz, w


# Add slot assignment constraints
def add_slot_assignment_constraints(model, y, t, **kwargs):
    num_nodes = kwargs.get("num_nodes")
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    max_number_of_slots = kwargs.get("max_number_of_slots")
    latencies = kwargs.get("latencies")
    latency_ub = kwargs.get("latency_ub")
    wakeup_time = kwargs.get("wakeup_time")
    model_vars = kwargs.get("model_vars")

    # Slot assignment variables
    node_slot_assignment = model.addMVar((num_chips, num_subaccs, max_number_of_slots, num_nodes), vtype=GRB.BINARY, name="NodeSlotAssignment")
    model_vars.add_var(Var("NodeSlotAssignment", (num_chips, num_subaccs, max_number_of_slots, num_nodes), 0, 1, int))
    # Slot variables
    slot_start_time = model.addMVar((num_chips, num_subaccs, max_number_of_slots), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SlotStartTime")
    model_vars.add_var(Var("SlotStartTime", (num_chips, num_subaccs, max_number_of_slots), 0, latency_ub, float))
    slot_start_time_min = model.addMVar((num_chips, num_subaccs, max_number_of_slots), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SlotStartTimeMin")
    model_vars.add_var(Var("SlotStartTimeMin", (num_chips, num_subaccs, max_number_of_slots), 0, latency_ub, float))
    slot_start_time_helper = model.addMVar((num_chips, num_subaccs, max_number_of_slots, num_nodes), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SlotStartTimeHelper")
    model_vars.add_var(Var("SlotStartTimeHelper", (num_chips, num_subaccs, max_number_of_slots, num_nodes), 0, latency_ub, float))
    slot_end_time = model.addMVar((num_chips, num_subaccs, max_number_of_slots), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SlotEndTime")
    model_vars.add_var(Var("SlotEndTime", (num_chips, num_subaccs, max_number_of_slots), 0, latency_ub, float))
    slot_end_time_helper = model.addMVar((num_chips, num_subaccs, max_number_of_slots, num_nodes), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SlotEndTimeHelper")
    model_vars.add_var(Var("SlotEndTimeHelper", (num_chips, num_subaccs, max_number_of_slots, num_nodes), 0, latency_ub, float))
    slot_load = model.addMVar((num_chips, num_subaccs, max_number_of_slots), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SlotLoad")    
    model_vars.add_var(Var("SlotLoad", (num_chips, num_subaccs, max_number_of_slots), 0, latency_ub, float))
    slot_used = model.addMVar((num_chips, num_subaccs, max_number_of_slots), vtype=GRB.BINARY, name="SlotUsed")
    model_vars.add_var(Var("SlotUsed", (num_chips, num_subaccs, max_number_of_slots), 0, 1, int))
    # SA slot variables
    num_non_empty_slots = model.addMVar((num_chips, num_subaccs), vtype=GRB.INTEGER, lb=0, ub=max_number_of_slots, name="NumNonEmptySlots")
    model_vars.add_var(Var("NumNonEmptySlots", (num_chips, num_subaccs), 0, max_number_of_slots, int))
    sa_idle_time = model.addMVar((num_chips, num_subaccs), vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="SASlotIdling")
    model_vars.add_var(Var("SASlotIdling", (num_chips, num_subaccs), 0, latency_ub, float))

    # Assign each node a slot on its SA
    assign_sa = node_slot_assignment.sum(axis=2)
    for chip_id in range(num_chips):
        for sa_id in range(num_subaccs):
            model.addConstr(assign_sa[chip_id, sa_id] == y[:, chip_id, sa_id], name=f"NodeSlotAssignmentLink_chip{chip_id}_sa{sa_id}")
    
    temp2 = node_slot_assignment*latencies

    t_minus_latencies = model.addMVar(num_nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=latency_ub, name="TMinusLatencies")
    model_vars.add_var(Var("TMinusLatencies", (num_nodes,), 0.0, latency_ub, float))
    model.addConstr(t_minus_latencies == t - latencies, name="TMinusLatenciesConstr")

    model.addConstr(slot_start_time_helper == node_slot_assignment*t_minus_latencies + latency_ub*(1 - node_slot_assignment), name="SSTHelper")
    model.addConstr(slot_end_time_helper == node_slot_assignment*t, name="SlotEndTimeHelper")

    for i in range(num_chips):
        for s in range(num_subaccs):
            for q in range(max_number_of_slots):
                model.addGenConstrMin(slot_start_time_min[i, s, q], [slot_start_time_helper[i, s, q, z] for z in range(num_nodes)], name=f"MinSlotStartConstr_chip{i}_sa{s}_slot{q}")
                model.addGenConstrMax(slot_end_time[i, s, q], [slot_end_time_helper[i, s, q, z] for z in range(num_nodes)], name=f"MaxSlotEndConstr_chip{i}_sa{s}_slot{q}")
    model.addConstr(slot_start_time == slot_start_time_min - latency_ub*(1-slot_used), name="SlotStartTimeBigM")
    model.addConstr(slot_end_time >= slot_start_time, name="SlotEndTime")
    # Load for each slot
    model.addConstr(slot_load == temp2.sum(axis=3), name="SlotLoad")

    # Idle time for each SA
    model.addConstr(sa_idle_time == (slot_end_time - slot_start_time - slot_load).sum(axis=2), name="SASlotIdling")
    # Non-empty slots for each SA
    model.addConstr(slot_used[:, :, 1:] <= slot_used[:, :, :-1], name="SlotDependencyUsage")
    # A slot's start time must be epsilon larger than the preceding slot's finish time
    model.addConstr(slot_start_time[:, :, 1:] >= slot_end_time[:, :, :-1] + wakeup_time - (latency_ub+wakeup_time)*(1-slot_used[:, :, 1:]), name="SlotDependencyTime")

    model.addConstrs((slot_used >= node_slot_assignment[:,:,:,z] for z in range(num_nodes)), name="SlotUsedLink")

    model.addConstr(num_non_empty_slots == slot_used.sum(axis=2), name="NumNonEmptySlots")

    return node_slot_assignment, slot_used, num_non_empty_slots, sa_idle_time
