# heuristics.py

import networkx as nx
import math
import time
import random
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from cdlib import algorithms
import community as community_louvain
import os
import pandas as pd
from collections import defaultdict, deque


algos = {
    'rber_pots':    lambda H, es, ws, s: algorithms.rber_pots(H.to_undirected(), weights=es, node_sizes=ws, resolution_parameter=s).communities,
    'cpm':          lambda H, es, ws, s: algorithms.cpm(H.to_undirected(), weights=es, node_sizes=ws, resolution_parameter=s).communities,
    'supcomm':      lambda H, es, ws, s: algorithms.surprise_communities(H.to_undirected(), weights=es, node_sizes=ws).communities,
    'sigcomm':      lambda H, es, ws, s: algorithms.significance_communities(H.to_undirected(), node_sizes=ws).communities,
    'louvain':      lambda H, es, ws, s: algorithms.louvain(H.to_undirected(), weight='size', resolution=s).communities,
    'spectral':     lambda H, es, ws, s: algorithms.r_spectral_clustering(H.to_undirected(), n_clusters=s).communities,
    'greedy_mod':   lambda H, es, ws, s: nx.community.greedy_modularity_communities(H, resolution=s, weight='size'),
    'label':        lambda H, es, ws, s: nx.community.asyn_lpa_communities(H.to_undirected(), weight='size'),
    'fluidc':       lambda H, es, ws, s: nx.community.asyn_fluidc(H.to_undirected(), k=s),
    'walktrap':     lambda H, es, ws, s: algorithms.walktrap(H.to_undirected()).communities,
    'paris':        lambda H, es, ws, s: algorithms.paris(H.to_undirected()).communities,
    'em':           lambda H, es, ws, s: algorithms.em(H.to_undirected(), k=s).communities,
    'der':          lambda H, es, ws, s: algorithms.der(H.to_undirected(), walk_len=s).communities,
    'markov':       lambda H, es, ws, s: algorithms.markov_clustering(H.to_undirected(), iterations=100000, pruning_threshold=.1).communities,
    # 'current':      lambda H, es, ws, s: nx.community.edge_current_flow_betweenness_partition(H.to_undirected(), weight='size', number_of_sets=s),
    # 'betweeness':   lambda H, es, ws, s: nx.community.edge_betweenness_partition(H.to_undirected(), weight='size', number_of_sets=s),
    # 'siblinearity': lambda H, es, ws, s: algorithms.siblinarity_antichain(H, Lambda=s).communities,
    # 'belief':       lambda H, es, ws, s: algorithms.belief(H.to_undirected()).communities,
    # 'gdmp2':        lambda H, es, ws, s: algorithms.gdmp2(H, min_threshold=s).communities, # Bugged
    # 'girvman':      lambda H, es, ws, s: algorithms.girvan_newman(H.to_undirected(), level=s).communities,
}


def wrapper(G, name, seed, num_chips, chip_sizes, num_subaccs):
    weight_sizes = [G.nodes[n].get('size') for n in G.nodes()]
    edge_sizes = [G.edges[u, v].get('size') for u, v in G.edges()]

    groups = algos[name](G, edge_sizes, weight_sizes, seed) 
    return group_assignment(G, num_chips, chip_sizes, num_subaccs, groups)


def group_assignment(G, num_chips, chip_sizes, num_subaccs, groups):
    sorted_comms = sorted(groups, key=lambda group: -sum(G.nodes[n]['size'] for n in group))
    
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips
    
    for comm in sorted_comms:
        comm_size = sum(G.nodes[n]['size'] for n in comm)
        possible_chips = [i for i in range(num_chips) if chip_loads[i] + comm_size <= chip_sizes[i]]
        if possible_chips:
            target_chip = min(possible_chips, key=lambda i: chip_loads[i])
        else:
            for node in comm:
                for chip_id in range(num_chips):
                    if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                        partitions[chip_id].append(node)
                        chip_loads[chip_id] += G.nodes[node]['size']
                        break
                else:
                    total_chip_size = sum(chip_sizes)
                    return None
            continue
        
        partitions[target_chip].extend(comm)
        chip_loads[target_chip] += comm_size
    return partitions


# Heuristics for warm starts
def choose_heuristic(heuristics_list, G, dream=False, **kwargs):
    model_vars = kwargs.get("model_vars", None)
    num_chips = kwargs.get("num_chips", 1)
    chip_sizes = kwargs.get("chip_sizes", None)
    num_subaccs = kwargs.get("num_subaccs", 1)
    max_number_of_slots = kwargs.get("max_number_of_slots", 1)
    epsilon = kwargs.get("epsilon", 0.0)
    I_upper = kwargs.get("I_upper", 0.0)
    fheader = kwargs.get("fheader", None)
    H = kwargs.get("H", None)
    powergating = kwargs.get("powergating", True)
    idle_leakage = kwargs.get("idle_leakage", False)
    w2s_etime = kwargs.get("w2s_etime", 0.0)
    latency_ub = kwargs.get("latency_ub")
    overlap = kwargs.get("overlap")
    bank_size = kwargs.get("bank_size")
    target = kwargs.get("target")
    subaccs_per_chip = kwargs.get("subaccs_per_chip", [])

    heuristics = {
        "latency_focused": latency_asap_prefer_pred_chip,
        "energy_focused": energy_asap_topo_partition,
        "minmaxload": maxload_balanced_partition,
        "spectral": spectral_clustering_heuristic,          #0
        "louvain": louvain_partition,                       #1
        "order_based": order_based_partition,               #2
        "layered": layered_partition,                       #3
        "dfs": dfs_partition,                               #4
        "greedy_lookahead": greedy_with_lookahead,          #5
        "greedy": greedy_fill_first_partition,              #6
        "priority": priority_based_heuristic,               #7
        "edge_centric": edge_centric_partitioning,          #8
        "greedy_throughput": greedy_fill_first_throughput,  #9
        "balanced_partitions": chip_partition_balanced,     #10
        "balanced_partitions5": chip_partition_balanced5,   #11
        "waved_parallel": wave_parallel_partition,          #12
        "threshold": latency_threshold_asap,
        "ratio": edge_ratio_parallelizer,
        "threshold2": latency_threshold_asap2,
        "ratio2": edge_ratio_parallelizer2,
        "threshold3": latency_threshold_asap3,
        "ratio3": edge_ratio_parallelizer3,
        # Discarded:cost_aware_greedy, kmeans_partition, evolutionary_partitioning, greedy_merge_partitions, dynamic_priority_heuristic,
    }

    # Add all algos as heuristics using the wrapper
    for algo_name in algos:
        heuristics[algo_name] = lambda G, num_chips, chip_sizes, num_subaccs, name=algo_name: wrapper(
            G, name, num_chips, num_chips, chip_sizes, num_subaccs
        )

    if dream or num_chips == 1:
        mst = generate_dream_start(model_vars, G, num_subaccs, max_number_of_slots, epsilon, I_upper, fheader, "dream_asap", powergating, idle_leakage, w2s_etime, latency_ub)
        mst2 = generate_dream_start(model_vars, G, num_subaccs, max_number_of_slots, epsilon, I_upper, fheader, "dream_greedy_thoughput", powergating, idle_leakage, w2s_etime, latency_ub)
        mst3 = generate_dream_start(model_vars, G, 1, max_number_of_slots, epsilon, I_upper, fheader, "dream_asap-1", powergating, idle_leakage, w2s_etime, latency_ub)
        mst5 = generate_dream_start(model_vars, G, num_subaccs, max_number_of_slots, epsilon, I_upper, fheader, "test", powergating, idle_leakage, w2s_etime, latency_ub)
        mst6 = generate_dream_start(model_vars, G, num_subaccs, max_number_of_slots, epsilon, I_upper, fheader, "test-2", powergating, idle_leakage, w2s_etime, latency_ub)
        mst7 = generate_dream_start(model_vars, G, num_subaccs, max_number_of_slots, epsilon, I_upper, fheader, "all-one", powergating, idle_leakage, w2s_etime, latency_ub)
        return [mst,mst2, mst3,mst5, mst6, mst7]
    else:
        # Collect heuristics to run
        if isinstance(heuristics_list, str):
            if heuristics_list == "none": return []
            elif heuristics_list == "all": heuristics_list = list(heuristics.keys())
            else: heuristics_list = [heuristics_list]

        # Compute heuristics and collect results
        results = []
        for name in heuristics_list:
            start_time = time.time()
            file_name = fheader + f"_{name}.mst"
            if os.path.exists(file_name): 
                results.append(file_name)
                continue
            else:
                partition = heuristics[name](G, num_chips, chip_sizes, num_subaccs=num_subaccs)
            if not partition: 
                print(f"❌ Heuristic failed for {name}")
                continue
            if name == "greedy_throughput":
                schedule = multi_chip_asap_throughput(G, num_chips, num_subaccs, H, partition)
            elif target == "EDCP":
                schedule = multi_chip_asap_new_hetero(G, num_chips, num_subaccs, subaccs_per_chip, H, partition)
            else:
                schedule = multi_chip_asap_new(G, num_chips, num_subaccs, H, partition)
            node_to_chip = schedule[-1]
            comm_hops = [H[node_to_chip[u]][node_to_chip[v]] for u, v in G.edges]
            schedules, schedules_start, _, _, node_to_chip = schedule
            start_times = {
                node: schedules_start[chip][sa][idx]
                for chip in schedules
                for sa in schedules[chip]
                for idx, node in enumerate(schedules[chip][sa])
            }
            slots, num_non_empty_slots = assign_multi_chip_schedule(schedule, num_chips, num_subaccs, max_number_of_slots, epsilon, w2s_etime)
            add_multi_chip_assignment_start(model_vars, partition)
            add_multi_chip_schedule_start(model_vars, schedule)
            nsa = add_multi_chip_slot_start(model_vars, num_chips, num_subaccs, max_number_of_slots, slots, num_non_empty_slots, start_times, latency_ub)
            add_multi_chip_slot_start_times(model_vars, slots, start_times, num_chips, num_subaccs, max_number_of_slots, latency_ub)
            add_multi_chip_incomparable_start(model_vars, I_upper, schedule)
            latencies = np.array([G.nodes[n]['latency'] for n in G.nodes])
            t = add_finish_time(model_vars, start_times, G) 
            tminus = add_tminus_latencies(model_vars, t, latencies, G)
            add_slot_helper_vars(model_vars, slots, t, latencies, num_chips, num_subaccs, max_number_of_slots, latency_ub)
            add_slot_end_and_idle_vars(model_vars, slots, t, latencies, num_chips, num_subaccs, max_number_of_slots, latency_ub)
            add_target_starts(model_vars, schedule, t, latencies, comm_hops, G, 32, num_chips, num_subaccs, overlap)
            if num_chips > 1: add_communication_reqs(model_vars, comm_hops)  
            load = add_multi_chip_load_start(model_vars, G, partition)
            if target == "EDCP":
                _, num_subaccs_used_per_chip = add_multi_chip_sa_usage_starts(model_vars, schedule, num_chips, num_subaccs)
                mem_banks = add_multi_chip_mem_banks_start(model_vars, load, bank_size)
                add_multi_chip_cost_start(model_vars, num_chips, num_subaccs, num_subaccs_used_per_chip, mem_banks)
            fname = model_vars.write_var_starts(fheader,name)
            results.append(fname)
            print(f"✅  Heuristic '{name}' completed in {time.time() - start_time:.2f} seconds.")
        return results


###
def add_multi_chip_load_start(model_vars, G, partitions):
    num_chips = len(partitions)
    load = np.zeros((num_chips,), dtype=float)
    for c in range(num_chips): load[c] = sum(float(G.nodes[n].get("size", 0.0)) for n in partitions[c])
    model_vars.add_start("Load", load)
    return load


def add_multi_chip_sa_usage_starts(model_vars, schedule, num_chips, num_subaccs):
    node_to_resource = schedule[3]
    sa_active = np.zeros((num_chips, num_subaccs), dtype=int)
    for _, (chip, sa) in node_to_resource.items(): sa_active[chip, sa] = 1
    num_subaccs_used_per_chip = sa_active.sum(axis=1).astype(int)
    model_vars.add_start("SAActive", sa_active)
    model_vars.add_start("subAccsUsedPerChip", num_subaccs_used_per_chip)
    return sa_active, num_subaccs_used_per_chip


def add_multi_chip_mem_banks_start(model_vars, load, bank_size):
    mem_banks = np.ceil(load / float(bank_size)).astype(int)
    model_vars.add_start("MemBanks", mem_banks)
    return mem_banks


def add_multi_chip_cost_start(model_vars, num_chips, num_subaccs, num_subaccs_used_per_chip, mem_banks):
    if num_chips * num_subaccs == 96:
        df = pd.read_csv("scripts/96-SA_96-MEM_cost.csv")
    else:
        df = pd.read_csv("scripts/12-SA_20-MEM_cost.csv")
    sa_levels  = np.sort(df["SAs"].unique())
    mem_levels = np.sort(df["Memories"].unique())
    grid = (df.pivot(index="SAs", columns="Memories", values="Cost").reindex(index=sa_levels, columns=mem_levels).values.astype(float))
    cost = np.zeros((num_chips,), dtype=float)
    cost_pair = np.zeros((num_chips, len(sa_levels), len(mem_levels)), dtype=int)
    for c in range(num_chips):
        sa = int(num_subaccs_used_per_chip[c])
        mb = int(mem_banks[c])
        i = int(np.where(sa_levels == sa)[0][0])
        ge = np.where(mem_levels >= mb)[0]
        j = int(ge[0])
        cost[c] = float(grid[i, j])
        cost_pair[c, i, j] = 1
    model_vars.add_start("Cost", cost)
    model_vars.add_start("PairSelect", cost_pair)
    return cost, cost_pair


def add_dream_target_starts(model_vars, schedule, t, latencies, G, inputs, num_chips, num_subaccs):
    total_latency = np.max(t)
    model_vars.add_start("Total_Latency", total_latency)

    _, _, _, node_to_resource  = schedule
    sa_load = np.zeros((num_chips, num_subaccs), dtype=float)
    for node, sa in node_to_resource.items():
        sa_load[0, sa] += latencies[node]

    model_vars.add_start("sa_load", sa_load)

    maxload = np.max(sa_load)
    model_vars.add_start("Total_Maxload", maxload)

    maxload_pipeline = (total_latency + (inputs - 1) * maxload) / inputs
    model_vars.add_start("Total_MaxloadPipeline", maxload_pipeline)


def add_target_starts(model_vars, schedule, t, latencies, comm_hops, G, inputs, num_chips, num_subaccs, overlap):
    total_latency = np.max(t)
    model_vars.add_start("Total_Latency", total_latency)

    _, _, _, node_to_resource, _ = schedule
    sa_load = np.zeros((num_chips, num_subaccs), dtype=float)
    for node, (chip, sa) in node_to_resource.items():
        sa_load[chip, sa] += latencies[node]
    if not overlap:
        for e, (u, v) in enumerate(G.edges):
            chip_u, sa_u = node_to_resource[u]
            chip_v, sa_v = node_to_resource[v]
            comm_latency = G.edges[u, v]['latency'] * comm_hops[e]
            sa_load[chip_u, sa_u] += comm_latency
            sa_load[chip_v, sa_v] += comm_latency

    model_vars.add_start("sa_load", sa_load)

    maxload = np.max(sa_load)
    model_vars.add_start("Total_Maxload", maxload)

    maxload_pipeline = (total_latency + (inputs - 1) * maxload) / inputs
    model_vars.add_start("Total_MaxloadPipeline", maxload_pipeline)
    

def add_slot_helper_vars(model_vars, slots, t, latencies, num_chips, num_subaccs, max_slots, latency_ub):
    dim_start_helper = model_vars.get_dim("SlotStartTimeHelper")
    dim_end_helper = model_vars.get_dim("SlotEndTimeHelper")

    start_start_helper = np.full(dim_start_helper, latency_ub, dtype=float)
    start_end_helper = np.zeros(dim_end_helper, dtype=float)

    for chip in range(num_chips):
        for sa in range(num_subaccs):
            for slot in range(max_slots):
                nodes = slots[chip][sa].get(slot, [])
                for node in nodes:
                    start_start_helper[chip, sa,slot, node] = t[node] - latencies[node]
                    start_end_helper[chip, sa,slot, node] = t[node]

    model_vars.add_start("SlotStartTimeHelper", start_start_helper)
    model_vars.add_start("SlotEndTimeHelper", start_end_helper)


def add_slot_end_and_idle_vars(model_vars, slots, t, latencies, num_chips, num_subaccs, max_slots, latency_ub):
    dim_end = model_vars.get_dim("SlotEndTime")
    dim_load = model_vars.get_dim("SlotLoad")
    dim_idle = model_vars.get_dim("SASlotIdling")

    start_end = np.zeros(dim_end, dtype=float)
    start_load = np.zeros(dim_load, dtype=float)
    start_idle = np.zeros(dim_idle, dtype=float)

    slot_start = model_vars.starts["SlotStartTime"]

    for chip in range(num_chips):
        for sa in range(num_subaccs):
            idle_sum = 0.0
            for slot in range(max_slots):
                nodes = slots[chip][sa].get(slot, [])
                if not nodes:
                    continue

                # SlotEndTime
                end_times = [t[node] for node in nodes]
                start_end[chip, sa,slot] = max(end_times)

                # SlotLoad
                load = sum(latencies[node] for node in nodes)
                start_load[chip, sa,slot] = load

                # Idling
                idle_sum += max(0.0, start_end[chip, sa,slot] - slot_start[chip, sa,slot] - load)

            start_idle[chip, sa] = idle_sum

    model_vars.add_start("SlotEndTime", start_end)
    model_vars.add_start("SlotLoad", start_load)
    model_vars.add_start("SASlotIdling", start_idle)
    

def enforce_group_constraint_on_partition(partition, G, nodes, copies):
    grouped_nodes = {i: [i + nodes * n for n in range(copies)] for i in range(1, nodes+1)}
    
    node_to_chip = {}
    for chip_idx, chip in enumerate(partition):
        for node in chip:
            node_to_chip[node] = chip_idx
    
    for group in grouped_nodes.values():
        original_node = group[0]
        original_chip = node_to_chip[original_node]

        for copied_node in group:
            if copied_node not in partition[original_chip]:
                for chip in partition:
                    if copied_node in chip:
                        chip.remove(copied_node)
                partition[original_chip].append(copied_node)
    partition[0].append(0)
    partition[0].append(len(G.nodes) - 1)
    
    return partition


def generate_dream_start(model_vars, G, num_subaccs, num_slots, epsilon, I_upper, fheader, heuristic, powergating, idle_leakage, w2s_etime, latency_ub):
    file_name = fheader + f"_{heuristic}.mst"
    if os.path.exists(file_name): return file_name
    if heuristic == "dream_asap" or heuristic == "dream_asap-1":
        schedule = dream_asap_new(G, num_subaccs)
    elif heuristic =="dream_greedy_thoughput" or heuristic == "dream_greedy-1":
        schedule = dream_greedy_throughput(G, num_subaccs)
    elif heuristic == "test":
        schedule = dream_limited_jumps_balanced(G, num_subaccs, 1.01)
    elif heuristic == "test-2":
        schedule = dream_limited_jumps_balanced(G, num_subaccs, 1.05)
    elif heuristic == "all-one":
        schedule = dream_all_on_one_sa(G, num_subaccs)
    else:
        print("Invalid heuristic")
        return None
    
    slots, num_non_empty_slots = assign_dream_shedule(schedule, num_subaccs, num_slots, epsilon,w2s_etime)
    schedules, schedules_start, _, _ = schedule
    start_times = {
    node: schedules_start[sa][idx]
    for sa in range(len(schedules))
    for idx, node in enumerate(schedules[sa])
    }
    add_dream_assignment_start(model_vars)
    add_dream_schedule_start(model_vars, schedule)
    nsa = add_dream_slot_start(model_vars, 0, num_subaccs, num_slots, slots, num_non_empty_slots)
    add_dream_chip_slot_start_times(model_vars, slots, start_times, num_subaccs, num_slots, latency_ub)
    add_dream_incomparable_start(model_vars, I_upper, schedule)
    t = add_finish_time(model_vars, start_times, G) 
    latencies = np.array([G.nodes[n]['latency'] for n in G.nodes])
    tminus = add_tminus_latencies(model_vars, t, latencies, G)
    slots_dream = {0: slots}  # For dream_asap, we only have one chip in the schedule
    add_slot_helper_vars(model_vars, slots_dream, t, latencies, 1, num_subaccs, num_slots, latency_ub)
    add_slot_end_and_idle_vars(model_vars, slots_dream, t, latencies, 1, num_subaccs, num_slots, latency_ub)
    add_dream_target_starts(model_vars, schedule, t, latencies, G, 32, 1, num_subaccs)
    fname = model_vars.write_var_starts(fheader,heuristic)

    partitioning={}
    partitioning['assignment'] = {}
    return fname


def compute_approx_partition_cost(G, partitions, chip_sizes=None, comm_weight=1.0, size_weight=10.0):
    """
    Compute an approximate cost of a given partition solution.

    Args:
        G (nx.DiGraph): Graph with node 'size' attributes and edge 'size' or 'energy' attributes.
        partitions (list[list[int]]): A list of lists, where partitions[i] contains node IDs assigned to chip i.
        chip_sizes (list[float]): Capacity (maximum sum of node sizes) for each partition.
        comm_weight (float): Weight factor for communication cost in the approximate objective.
        size_weight (float): Penalty factor for partition-size violations (if any).
                             If partition exceeds capacity by X, cost is X * size_weight for that partition.

    Returns:
        float: An approximate objective value, lower is better.
    """
    num_chips = len(partitions)
    # Build a quick lookup: node -> chip
    node_to_chip = {}
    for chip_id, part_nodes in enumerate(partitions):
        for n in part_nodes:
            node_to_chip[n] = chip_id

    # 1) Sum of capacity violations
    capacity_violation_cost = 0.0
    if chip_sizes is not None:
        # compute size usage
        usage_per_chip = [0.0] * num_chips
        for chip_id, part_nodes in enumerate(partitions):
            usage_per_chip[chip_id] = sum(G.nodes[n]['size'] for n in part_nodes)
        for chip_id in range(num_chips):
            exceed = usage_per_chip[chip_id] - chip_sizes[chip_id]
            if exceed > 0:
                capacity_violation_cost += exceed * size_weight

    # 2) Communication cost: if an edge crosses partitions, add its cost
    #    (We assume 'size' or 'energy' attribute on edges as proxy for cost.)
    comm_cost = 0.0
    for u, v in G.edges():
        # Safeguard: skip edges where either node isn't in node_to_chip
        if u not in node_to_chip or v not in node_to_chip:
            continue

        if node_to_chip[u] != node_to_chip[v]:
            # A simple measure: edge size or energy
            edge_size = G.edges[u, v].get('size', 1.0)
            comm_cost += edge_size

    return capacity_violation_cost + comm_weight * comm_cost


def greedy_fill_first_partition(G, num_chips, chip_sizes, **kwargs):
    """
    Partitions nodes in the graph into a specified number of chips, attempting to balance 
    partition sizes by filling the smallest partition first. Nodes are sorted in descending 
    order of size, and each node is assigned to the partition with the smallest total size.
    
    Args:
        graph (networkx.Graph): The input graph with nodes having a 'size' attribute.
        num_chips (int): The number of partitions (or chips) to divide the nodes into.
        chip_size (float): The maximum allowable size for each partition.
    
    Returns:
        list: A list of lists, where each sublist represents a partition containing node IDs.
    """
    # Initialize empty partitions and their corresponding size trackers
    partitions = [[] for _ in range(num_chips)]
    partition_sizes = [0] * num_chips

    # Sort nodes by size in descending order
    nodes_sorted = sorted(G.nodes(data=True), key=lambda x: x[1]['size'], reverse=True)

    # Distribute nodes into partitions
    for node, data in nodes_sorted:
        if node == 0:
            # Place node 0 into the first partition as a starting point
            partitions[0].append(node)
            partition_sizes[0] += data['size']
            continue
        else:
            # Find the partition with the smallest current total size
            min_partition_idx = partition_sizes.index(min(partition_sizes))
            # Assign the node to this smallest partition
            partitions[min_partition_idx].append(node)
            # Update the size of this partition
            partition_sizes[min_partition_idx] += data['size']

            # Check if the partition size exceeds the chip size limit
            if partition_sizes[min_partition_idx] > chip_sizes[min_partition_idx]:
                print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / sum(chip_sizes)) * 100):.2f}% of the average chip size.")
                return None

    return partitions


def greedy_fill_first_throughput(G, num_chips, chip_sizes, **kwargs):
    """
    Partitions nodes in the graph into a specified number of chips, attempting to balance 
    partition sizes by filling the smallest partition first. Nodes are sorted in descending 
    order of size, and each node is assigned to the partition with the smallest total size.
    
    Args:
        graph (networkx.Graph): The input graph with nodes having a 'size' attribute.
        num_chips (int): The number of partitions (or chips) to divide the nodes into.
        chip_size (float): The maximum allowable size for each partition.
    
    Returns:
        list: A list of lists, where each sublist represents a partition containing node IDs.
    """
    # Initialize empty partitions and their corresponding size trackers
    partitions = [[] for _ in range(num_chips)]
    partition_sizes = [0] * num_chips
    partition_loads = [0] * num_chips

    # Sort nodes by size in descending order
    nodes_sorted_size = sorted(G.nodes(data=True), key=lambda x: x[1]['size'], reverse=True)
    nodes_sorted_work = sorted(G.nodes(data=True), key=lambda x: x[1]['latency'], reverse=True)
    # Distribute nodes into partitions
    for node, data in nodes_sorted_work:
        if node == 0:
            # Place node 0 into the first partition as a starting point
            partitions[0].append(node)
            partition_sizes[0] += data['size']
            partition_loads[0] += data['latency']
            continue
        else:
            # Find the partition with the smallest current total size
            min_partition_idx = partition_loads.index(min(partition_loads))
            # Assign the node to this smallest partition
            partitions[min_partition_idx].append(node)
            # Update the size of this partition
            partition_sizes[min_partition_idx] += data['size']
            partition_loads[min_partition_idx] += data['latency']

    # Now do a pair-wise swapping to balance the sizes
    # Check if the partition size exceeds the chip size limit
    def check_valid_partitions(partitions, partition_sizes, chip_sizes):
        for i in range(len(partitions)):
            if partition_sizes[i] > chip_sizes[i]:
                return False
        return True
    swaps = 0
    while not check_valid_partitions(partitions, partition_sizes, chip_sizes) and swaps < 100:
        for i in range(len(partitions)):
            for j in range(len(partitions)):
                if i != j and partition_sizes[i] > chip_sizes[i] and partition_sizes[j] < chip_sizes[j]:
                    # Swap nodes between partitions
                    node_i = partitions[i].pop()
                    node_j = partitions[j].pop()
                    partitions[i].append(node_j)
                    partitions[j].append(node_i)
                    partition_sizes[i] -= G.nodes[node_i]['size']
                    partition_sizes[j] += G.nodes[node_j]['size']
                    partition_loads[i] -= G.nodes[node_i]['latency']
                    partition_loads[j] += G.nodes[node_j]['latency']
                    swaps += 1
    if check_valid_partitions(partitions, partition_sizes, chip_sizes):
        print(f"Partitions balanced after {swaps} swaps.")
        print("Partition sizes:", partition_sizes)
        print("Partition loads:", partition_loads)
        return partitions
    else:
        print(f"Couldn't balance partitions after {swaps} swaps.")
        return None


def cost_aware_greedy(G, num_chips, chip_sizes, comm_weight=1.0, size_weight=10.0):
    """
    A more sophisticated greedy that considers both node size and potential communication cost.
    It sorts nodes by an importance metric (e.g., node size * out_degree), then places each
    node in the partition that *minimizes the incremental approximate cost*.

    Returns: A list of lists (partitions).
    """
    # Initialize empty partitions and usage
    partitions = [[] for _ in range(num_chips)]
    usage_per_chip = [0.0] * num_chips

    # Example scoring for node priority: size * out_degree
    nodes_sorted = sorted(G.nodes(), key=lambda n: G.nodes[n]['size'] * G.out_degree(n), reverse=True)

    for node in nodes_sorted:
        best_chip = None
        best_cost_change = math.inf

        # Try assigning 'node' to each chip, measure approximate cost delta
        for chip_id in range(num_chips):
            # place node temporarily
            partitions[chip_id].append(node)
            old_usage = usage_per_chip[chip_id]
            usage_per_chip[chip_id] += G.nodes[node]['size']

            # compute approximate cost
            cost_val = compute_approx_partition_cost(
                G, partitions, chip_sizes, comm_weight, size_weight
            )
            # revert
            partitions[chip_id].remove(node)
            usage_per_chip[chip_id] = old_usage

            if cost_val < best_cost_change:
                best_cost_change = cost_val
                best_chip = chip_id

        # Permanently place node in the best chip
        partitions[best_chip].append(node)
        usage_per_chip[best_chip] += G.nodes[node]['size']

    return partitions


def order_based_partition(G, num_chips, chip_sizes, order="topological", **kwargs):
    """
    Partitions nodes in an order-based manner. For example, follow the topological order
    and keep placing nodes in the current partition if they fit, else move to the next partition,
    or cycle through partitions in a round-robin.

    This can reduce communication for sequential chains.

    Returns: A list of lists (partitions).
    """
    partitions = [[] for _ in range(num_chips)]
    usage_per_chip = [0.0] * num_chips

    if order == "topological":
        node_order = list(nx.topological_sort(G))
    elif order == "bfs":
        # BFS from node 0, for instance
        node_order = list(nx.bfs_tree(G, source=0))
    else:
        # fallback: just numeric order or something
        node_order = sorted(G.nodes())

    current_chip = 0
    for node in node_order:
        node_size = G.nodes[node]['size']

        # If it doesn't fit in the current chip, move on
        if usage_per_chip[current_chip] + node_size <= chip_sizes[current_chip]:
            partitions[current_chip].append(node)
            usage_per_chip[current_chip] += node_size
        else:
            placed = False
            for _ in range(num_chips):
                current_chip = (current_chip + 1) % num_chips
                if usage_per_chip[current_chip] + node_size <= chip_sizes[current_chip]:
                    partitions[current_chip].append(node)
                    usage_per_chip[current_chip] += node_size
                    placed = True
                    break
            if not placed:
                total_chip_size = sum(chip_sizes)
                print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
                return None

        # Optionally just always increment chip index
        # current_chip = (current_chip + 1) % num_chips

    return partitions


def greedy_with_lookahead(G, num_chips, chip_sizes, **kwargs):
    # Step 1: Initialize partitions and chip loads
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0] * num_chips
    
    # Step 2: Compute communication cost for all nodes
    def communication_cost(node, chip_id, partitions, G):
        cost = 0
        for neighbor in G.neighbors(node):
            if neighbor in partitions[chip_id]:
                cost += G[node][neighbor].get('latency', 0)
        return cost

    # Step 3: Greedy assignment with lookahead
    for node in G.nodes:
        best_chip = None
        best_cost = float('inf')

        for chip_id in range(num_chips):
            if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                current_cost = communication_cost(node, chip_id, partitions, G)
                # Include a penalty for load imbalance
                current_cost += abs(sum(chip_loads) / num_chips - chip_loads[chip_id])
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_chip = chip_id

        if best_chip is None:
            total_chip_size = sum(chip_sizes)
            print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
            return None

        # Assign node to the best chip
        partitions[best_chip].append(node)
        chip_loads[best_chip] += G.nodes[node]['size']
    
    return partitions


def priority_based_heuristic(G, num_chips, chip_sizes, **kwargs):
    # Step 1: Compute priority scores for nodes
    def compute_priority(node):
        communication = sum(G[node][neighbor].get('latency', 0) for neighbor in G.neighbors(node))
        energy = G.nodes[node].get('energy', 1)
        latency = G.nodes[node].get('latency', 1)
        return communication + energy + latency

    priorities = [(node, compute_priority(node)) for node in G.nodes]
    priorities.sort(key=lambda x: x[1], reverse=True)

    # Step 2: Assign nodes based on priority
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0] * num_chips

    for node, _ in priorities:
        best_chip = None
        for chip_id in range(num_chips):
            if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                best_chip = chip_id
                break

        if best_chip is None:
            total_chip_size = sum(chip_sizes)
            print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
            return None

        partitions[best_chip].append(node)
        chip_loads[best_chip] += G.nodes[node]['size']
    
    return partitions


def energy_asap_topo_partition(G, num_chips, chip_sizes, threshold=0.9, **kwargs):
    """
    Memory-bound, energy-aware partitioning (no latency):
    - Maintains a Kahn-style ready set (topological).
    - Chips are seeded in increasing numeric order with the earliest topological node available.
    - While placing on the current chip:
        * If used/cap < threshold: choose ready node that maximizes energy to nodes already on this chip.
        * Else: choose ready node that minimizes outgoing energy to UNASSIGNED nodes (reduce boundary cut).
    - If no ready node fits, move to next chip.
    Returns: list[list[int]] partitions per chip, or None on failure.
    """

    def edge_energy(u, v):
        e = G.edges[u, v]
        return e.get('energy', e.get('size', 1.0))

    def energy_to_chip(n, chip_nodes):
        # Sum both directions between n and nodes already on this chip
        s = 0.0
        for u in chip_nodes:
            if G.has_edge(n, u):
                s += edge_energy(n, u)
            if G.has_edge(u, n):
                s += edge_energy(u, n)
        return s

    def outgoing_energy_to_unassigned(n, unassigned_set):
        # Only outgoing edges from n to still-unassigned nodes
        return sum(edge_energy(n, w) for w in G.successors(n) if w in unassigned_set)

    # Init per-chip bookkeeping
    partitions = [[] for _ in range(num_chips)]
    used = [0.0] * num_chips

    # Kahn-style topo init
    indeg = {v: G.in_degree(v) for v in G.nodes}
    ready = deque([v for v in G.nodes if indeg[v] == 0])
    unassigned = set(G.nodes)

    # Stable topological ranking for deterministic seeding & tie-breaks
    topo_order = list(nx.topological_sort(G))
    topo_rank = {v: i for i, v in enumerate(topo_order)}

    chip = 0

    while unassigned:
        if chip >= num_chips:
            print("ERROR: Out of chips before assigning all nodes.")
            return None

        cap = chip_sizes[chip]
        # Filter ready nodes that fit on current chip
        fit_ready = [v for v in ready if used[chip] + G.nodes[v].get('size', 0.0) <= cap]

        if not fit_ready:
            # If some ready nodes exist but none fit, open next chip
            if ready:
                chip += 1
                continue
            # Otherwise, graph has a cycle or data issue
            print("ERROR: No ready nodes available (cycle or metadata issue).")
            return None

        frac = used[chip] / cap if cap > 0 else 1.0
        chip_nodes = partitions[chip]

        if not chip_nodes:
            # Seed each chip with the earliest available topological node
            chosen = min(
                fit_ready,
                key=lambda v: (topo_rank[v], outgoing_energy_to_unassigned(v, unassigned - {v}))
            )
        else:
            if frac < threshold:
                # Cohesion phase: maximize linkage to current chip
                chosen = max(
                    fit_ready,
                    key=lambda v: (
                        energy_to_chip(v, chip_nodes),
                        -outgoing_energy_to_unassigned(v, unassigned - {v}),
                        -topo_rank[v]  # deterministic tie-breaker: earlier in topo is better
                    )
                )
            else:
                # Boundary phase: minimize outgoing to unassigned
                chosen = min(
                    fit_ready,
                    key=lambda v: (
                        outgoing_energy_to_unassigned(v, unassigned - {v}),
                        -energy_to_chip(v, chip_nodes),
                        topo_rank[v]   # deterministic tie-breaker
                    )
                )

        # Place chosen on current chip
        size_v = G.nodes[chosen].get('size', 0.0)
        partitions[chip].append(chosen)
        used[chip] += size_v
        unassigned.remove(chosen)

        # Remove chosen from ready and update successors
        try:
            ready.remove(chosen)
        except ValueError:
            pass
        for w in G.successors(chosen):
            indeg[w] -= 1
            if indeg[w] == 0 and w in unassigned:
                ready.append(w)

        # If chip just filled up, move to next
        if used[chip] >= cap:
            chip += 1

    return partitions


def latency_asap_prefer_pred_chip(G, num_chips, chip_sizes, **kwargs):
    """
    Latency-focused ASAP partitioning with 'stick-to-predecessor-chip' bias.

    Rules:
    - Process nodes in topological order.
    - For each node, identify the predecessor that determines data_ready
      (the one with the latest finish). Prefer that predecessor's chip.
    - If that chip is available at data_ready and has capacity, place there.
    - Otherwise, choose the chip that can start the node earliest
      (min max(data_ready, next_free_time[chip])) among chips with capacity.
      Tie-breaker: lower memory load, then lower chip id.

    Ignores communication latency entirely; only node 'latency' and memory ('size') matter.
    Returns: list[list[int]] partitions per chip, or None if placement fails.
    """
    # Per-chip bookkeeping
    partitions = [[] for _ in range(num_chips)]
    used = [0.0] * num_chips               # memory used per chip
    next_free = [0.0] * num_chips           # next available time per chip

    topo = list(nx.topological_sort(G))
    N = len(G.nodes)
    start = [0.0] * N
    finish = [0.0] * N

    # Map node -> assigned chip
    node_chip = {}

    total_cap = sum(chip_sizes) if chip_sizes else 0.0

    for v in topo:
        v_lat = G.nodes[v].get('latency', 0.0)
        v_sz  = G.nodes[v].get('size', 0.0)

        # Compute data-ready and anchor predecessor (latest finishing pred)
        if G.in_degree(v) == 0:
            data_ready = 0.0
            anchor_chip = None
        else:
            # Among predecessors, find the one with max finish time
            preds = list(G.predecessors(v))
            anchor_pred = max(preds, key=lambda u: finish[u])
            data_ready = finish[anchor_pred]
            anchor_chip = node_chip.get(anchor_pred, None)

        # Helper: check chips that have enough remaining memory
        def chips_that_fit():
            return [c for c in range(num_chips) if used[c] + v_sz <= chip_sizes[c]]

        candidates = chips_that_fit()
        if not candidates:
            print(
                f"Couldn't place node {v}/{len(G.nodes)}, all chips full. "
                f"Node would've taken {((v_sz / total_cap) * 100) if total_cap else 0:.2f}% "
                f"of the average chip size."
            )
            return None

        placed = False

        # 1) Try the anchor/predecessor chip first (strict preference).
        if anchor_chip is not None and anchor_chip in candidates:
            earliest_on_anchor = max(data_ready, next_free[anchor_chip])
            # Only choose another chip if anchor won't be available by data_ready
            if next_free[anchor_chip] <= data_ready:
                # Place on anchor chip
                s = data_ready
                e = s + v_lat
                partitions[anchor_chip].append(v)
                used[anchor_chip] += v_sz
                start[v], finish[v] = s, e
                next_free[anchor_chip] = e
                node_chip[v] = anchor_chip
                placed = True
            # else: fall through to choose best alternative

        if not placed:
            # 2) Choose the chip that can start the earliest; ties by lowest used mem, then chip id
            # (If anchor chip *is* a candidate but not free by data_ready, it’s included here — but
            # we only pick it if it still wins earliest-start.)
            best = min(
                candidates,
                key=lambda c: (max(data_ready, next_free[c]), used[c], c)
            )
            s = max(data_ready, next_free[best])
            e = s + v_lat
            partitions[best].append(v)
            used[best] += v_sz
            start[v], finish[v] = s, e
            next_free[best] = e
            node_chip[v] = best

    return partitions


def spectral_clustering_heuristic(G, num_chips, chip_sizes, verbose=False, **kwargs):
    num_nodes = len(G.nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for u, v in G.edges:
        adj_matrix[u, v] = G[u][v].get('latency', 1) * G[u][v].get('energy', 1)
        adj_matrix[v, u] = adj_matrix[u, v]  # Undirected

    clustering = SpectralClustering(
        n_clusters=num_chips, affinity='precomputed', assign_labels='kmeans', random_state=42
    )
    try:
        labels = clustering.fit_predict(adj_matrix)
    except Exception as e:
        print(f"[SpectralClustering] Failed: {e}")
        return None

    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips

    # Step 1: Initial assignment
    for node, label in enumerate(labels):
        node_size = G.nodes[node]['size']
        partitions[label].append(node)
        chip_loads[label] += node_size

    # Step 2: Rebalance overloaded chips
    for chip_id in range(num_chips):
        while chip_loads[chip_id] > chip_sizes[chip_id] and not chip_sizes[chip_id] == 0:
            # Pop the largest node
            largest_node = max(partitions[chip_id], key=lambda n: G.nodes[n]['size'])
            size = G.nodes[largest_node]['size']

            # Try to find a chip with room
            destination = None
            max_free = -1
            for other_id in range(num_chips):
                if other_id == chip_id:
                    continue
                free_space = chip_sizes[other_id] - chip_loads[other_id]
                if free_space >= size and free_space > max_free:
                    destination = other_id
                    max_free = free_space

            if destination is None:
                if verbose:
                    print(f"Node {largest_node} (size {size}) from chip {chip_id} could not be rebalanced.")
                return None

            partitions[chip_id].remove(largest_node)
            partitions[destination].append(largest_node)
            chip_loads[chip_id] -= size
            chip_loads[destination] += size

            if verbose:
                print(f"Moved node {largest_node} from chip {chip_id} to {destination}")

    for i in range(num_chips):
        if chip_loads[i] > chip_sizes[i]:
            if verbose:
                print(f"[Validation Failed] Chip {i} overloaded: {chip_loads[i]} > {chip_sizes[i]}")
            return None
        
    return partitions


def improved_greedy_with_communication(G, num_chips, chip_sizes):
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0] * num_chips

    # Helper to calculate inter-chip communication cost
    def inter_chip_cost(node, target_chip):
        cost = 0
        for neighbor in G.neighbors(node):
            for chip_id, partition in enumerate(partitions):
                if neighbor in partition and chip_id != target_chip:
                    cost += G[node][neighbor].get('latency', 1) * G[node][neighbor].get('size', 1)
        return cost

    # Assign nodes in order of decreasing degree (high connectivity first)
    nodes_sorted = sorted(G.nodes, key=lambda n: G.degree[n], reverse=True)

    for node in nodes_sorted:
        best_chip = None
        min_cost = float('inf')

        for chip_id in range(num_chips):
            if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                cost = inter_chip_cost(node, chip_id)
                if cost < min_cost:
                    min_cost = cost
                    best_chip = chip_id

        if best_chip is None:
            total_chip_size = sum(chip_sizes)
            print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
            return None

        # Assign node to the best chip
        partitions[best_chip].append(node)
        chip_loads[best_chip] += G.nodes[node]['size']
    
    return partitions


def dynamic_priority_heuristic(G, num_chips, chip_sizes):
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0] * num_chips

    # Priority calculation: communication + energy + latency
    def compute_priority(node):
        return sum(G[node][neighbor].get('latency', 1) for neighbor in G.neighbors(node)) + \
               G.nodes[node].get('energy', 1) + \
               G.nodes[node].get('latency', 1)

    # Assign nodes dynamically based on adjusted priorities
    nodes = list(G.nodes)
    while nodes:
        # Recalculate priorities
        priorities = [(node, compute_priority(node)) for node in nodes]
        priorities.sort(key=lambda x: x[1], reverse=True)
        node, _ = priorities[0]
        nodes.remove(node)

        # Assign node to the best chip
        best_chip = None
        min_cost = float('inf')

        for chip_id in range(num_chips):
            if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                cost = sum(
                    G[node][neighbor].get('latency', 1)
                    for neighbor in G.neighbors(node)
                    if neighbor in partitions[chip_id]
                )
                if cost < min_cost:
                    min_cost = cost
                    best_chip = chip_id

        if best_chip is None:
            total_chip_size = sum(chip_sizes)
            print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
            return None

        partitions[best_chip].append(node)
        chip_loads[best_chip] += G.nodes[node]['size']

    return partitions


def edge_centric_partitioning(G, num_chips, chip_sizes, **kwargs):
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0] * num_chips

    # Start with nodes with the highest degree
    nodes_sorted = sorted(G.nodes, key=lambda n: G.degree[n], reverse=True)
    for node in nodes_sorted:
        # Find the chip with the most neighbors already assigned
        best_chip = None
        max_shared_edges = -1

        for chip_id in range(num_chips):
            if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                shared_edges = sum(1 for neighbor in G.neighbors(node) if neighbor in partitions[chip_id])
                if shared_edges > max_shared_edges:
                    max_shared_edges = shared_edges
                    best_chip = chip_id

        if best_chip is None:
            total_chip_size = sum(chip_sizes)
            print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
            return None

        # Assign node to the best chip
        partitions[best_chip].append(node)
        chip_loads[best_chip] += G.nodes[node]['size']
    
    return partitions


def evolutionary_partitioning(G, num_chips, chip_sizes, generations=50, pop_size=100):
    # Define fitness function
    def evaluate(individual):
        partitions = [[] for _ in range(num_chips)]
        chip_loads = [0] * num_chips
        comm_cost = 0

        # Assign nodes to chips based on individual
        for node, chip in enumerate(individual):
            partitions[chip].append(node)
            chip_loads[chip] += G.nodes[node]['size']
        
        # Calculate load balance penalty
        load_penalty = sum(max(0, chip_loads[chip] - chip_sizes[chip]) for chip in range(num_chips))

        # Calculate inter-chip communication cost
        for u, v in G.edges:
            if individual[u] != individual[v]:
                comm_cost += G[u][v].get('latency', 1) + G[u][v].get('energy', 1)

        return (comm_cost + load_penalty,)

    # Genetic algorithm setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.randint, 0, num_chips - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(G.nodes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_chips - 1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run genetic algorithm
    population = toolbox.population(n=pop_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    # Return the best solution
    best_individual = tools.selBest(population, k=1)[0]
    partitions = [[] for _ in range(num_chips)]
    for node, chip in enumerate(best_individual):
        partitions[chip].append(node)
    return partitions


def louvain_partition(G, num_chips, chip_sizes, **kwargs):
    """
    Use Louvain community detection to partition the graph and then 
    map communities to chips, adjusting to respect chip sizes.
    """
    # Detect communities
    partition_dict = community_louvain.best_partition(G.to_undirected())
    
    # Group nodes by community
    communities = {}
    for node, com in partition_dict.items():
        communities.setdefault(com, []).append(node)
        
    # Sort communities by size (largest first)
    sorted_comms = sorted(communities.values(), key=lambda comm: -sum(G.nodes[n]['size'] for n in comm))
    
    # Initialize chips
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips
    
    for comm in sorted_comms:
        # assign community to chip with smallest load that can accommodate it
        comm_size = sum(G.nodes[n]['size'] for n in comm)
        possible_chips = [i for i in range(num_chips) if chip_loads[i] + comm_size <= chip_sizes[i]]
        if possible_chips:
            target_chip = min(possible_chips, key=lambda i: chip_loads[i])
        else:
            # If no chip can take the entire community, break community apart
            # and assign greedily
            for node in comm:
                for chip_id in range(num_chips):
                    if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                        partitions[chip_id].append(node)
                        chip_loads[chip_id] += G.nodes[node]['size']
                        break
                else:
                    total_chip_size = sum(chip_sizes)
                    print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
                    return None
            continue
        
        # assign entire community
        partitions[target_chip].extend(comm)
        chip_loads[target_chip] += comm_size

    return partitions


def kmeans_partition(G, num_chips, chip_sizes):
    """
    Use KMeans clustering on node features extracted from the graph to partition nodes.
    We'll use spectral embedding as features.
    """
    # Compute Laplacian matrix for spectral embedding
    laplacian = nx.normalized_laplacian_matrix(G)
    # Use a small number of eigenvectors as features (e.g., 10)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.todense())
    # Take first few eigenvectors (skip the first trivial one)
    features = eigenvectors[:, 1:11]  # shape: (num_nodes, 10)

    # KMeans on these features
    kmeans = KMeans(n_clusters=num_chips, random_state=42).fit(features)
    labels = kmeans.labels_

    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips

    for node, cluster_id in enumerate(labels):
        # Try to assign to the corresponding cluster/chip
        if chip_loads[cluster_id] + G.nodes[node]['size'] <= chip_sizes[cluster_id]:
            partitions[cluster_id].append(node)
            chip_loads[cluster_id] += G.nodes[node]['size']
        else:
            # If not possible, find another chip that fits
            for chip_id in range(num_chips):
                if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                    partitions[chip_id].append(node)
                    chip_loads[chip_id] += G.nodes[node]['size']
                    break
            else:
                total_chip_size = sum(chip_sizes)
                print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
                return None
    return partitions


def layered_partition(G, num_chips, chip_sizes, **kwargs):
    """
    Partition the graph by topological layers.
    """
    # Get topological levels: assign level = length of longest path from source
    levels = {}
    for node in nx.topological_sort(G):
        # level = 0 if no predecessors, else max(level of preds) + 1
        preds = list(G.predecessors(node))
        levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1

    # Group nodes by level
    level_groups = {}
    for node, lvl in levels.items():
        level_groups.setdefault(lvl, []).append(node)

    # Sort groups by level order
    sorted_levels = [level_groups[lvl] for lvl in sorted(level_groups)]

    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips
    current_chip = 0

    for group in sorted_levels:
        for node in group:
            node_size = G.nodes[node]['size']
            # Try to fit node in the current chip; if full, move to next
            if chip_loads[current_chip] + node_size <= chip_sizes[current_chip]:
                partitions[current_chip].append(node)
                chip_loads[current_chip] += node_size
            else:
                found = False
                # cycle through chips to find space
                for offset in range(num_chips):
                    chip_candidate = (current_chip + offset) % num_chips
                    if chip_loads[chip_candidate] + node_size <= chip_sizes[chip_candidate]:
                        partitions[chip_candidate].append(node)
                        chip_loads[chip_candidate] += node_size
                        current_chip = chip_candidate  # switch context
                        found = True
                        break
                if not found:
                    total_chip_size = sum(chip_sizes)
                    print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
                    return None
    return partitions


def chip_partition_balanced(G, num_chips, chip_sizes, **kwargs):
    """
    Greedy balanced partitioning of nodes onto chips with hard size constraints and soft latency balance.
    Nodes are assigned to chips with enough remaining size and lowest latency load.
    """
    topo_order = list(nx.topological_sort(G))
    latencies = [G.nodes[node]['latency'] for node in topo_order]
    sizes = [G.nodes[node]['size'] for node in topo_order]

    total_latency = sum(latencies)
    avg_latency_load = total_latency / num_chips if num_chips > 0 else 0.0

    partitions = {c: [] for c in range(num_chips)}
    chip_latency_load = [0.0] * num_chips
    chip_size_load = [0.0] * num_chips
    node_to_chip = {}
    unassigned_nodes = []

    current_chip = 0
    for i, node in enumerate(topo_order):
        node_latency = latencies[i]
        node_size = sizes[i]

        place_on_current = True
        if chip_latency_load[current_chip] + node_latency > avg_latency_load * 1.01:
            place_on_current = False
        if chip_size_load[current_chip] + node_size > chip_sizes[current_chip]:
            place_on_current = False

        if place_on_current:
            pass
        else:
            # Try to find a better chip
            found = False
            for chip_id in range(num_chips):
                if chip_size_load[chip_id] + node_size <= chip_sizes[chip_id]:
                    if chip_latency_load[chip_id] + node_latency <= avg_latency_load * 1.01:
                        current_chip = chip_id
                        found = True
                        break
            if not found:
                # Couldn't place now, will place later
                unassigned_nodes.append((node, node_latency, node_size))
                continue

        partitions[current_chip].append(node)
        chip_latency_load[current_chip] += node_latency
        chip_size_load[current_chip] += node_size
        node_to_chip[node] = current_chip

    # Handle unassigned nodes: put on chip with lowest latency load among those with enough space
    for node, node_latency, node_size in unassigned_nodes:
        best_chip = None
        lowest_latency = float('inf')
        for chip_id in range(num_chips):
            free_space = chip_sizes[chip_id] - chip_size_load[chip_id]
            if free_space >= node_size:
                if chip_latency_load[chip_id] < lowest_latency:
                    best_chip = chip_id
                    lowest_latency = chip_latency_load[chip_id]
        if best_chip is None:
            print(f"ERROR: Could not place node {node}. No chip has enough space.")
            return None
        partitions[best_chip].append(node)
        chip_latency_load[best_chip] += node_latency
        chip_size_load[best_chip] += node_size
        node_to_chip[node] = best_chip

    return [partitions[c] for c in range(num_chips)]


def maxload_balanced_partition(G, num_chips, chip_sizes, **kwargs):
    """
    Max-load-focused partitioning:
      - Objective: balance total *latency* per chip (minimize max load).
      - Constraint: per-chip size (sum of node 'size') may not exceed chip_sizes[chip].
      - Ignores edges/communication and topology.

    Strategy:
      1) Sort nodes by descending latency (tie-break by larger size, then node id for determinism).
      2) For each node, place it on the feasible chip (size fits) with the current lowest latency load.
         Tie-break by lower size load, then lower chip id.

    Returns:
        list[list[int]]: nodes per chip in order of assignment, or None if a node cannot fit anywhere.
    """
    # Per-chip bookkeeping
    partitions = [[] for _ in range(num_chips)]
    chip_latency_load = [0.0] * num_chips   # what we balance
    chip_size_load    = [0.0] * num_chips   # hard constraint

    # Sort nodes: high latency first, then larger size (reduce fragmentation), then node id
    nodes_sorted = sorted(
        G.nodes(),
        key=lambda n: (G.nodes[n].get('latency', 0.0),
                       G.nodes[n].get('size',    0.0),
                       -n),                     # stable tie-breaker (invert to keep descending)
        reverse=True
    )

    total_chip_size = sum(chip_sizes) if chip_sizes else 0.0

    for node in nodes_sorted:
        n_lat = float(G.nodes[node].get('latency', 0.0))
        n_sz  = float(G.nodes[node].get('size',    0.0))

        # Chips where the node fits wrt size
        feasible = [c for c in range(num_chips) if chip_size_load[c] + n_sz <= chip_sizes[c]]

        if not feasible:
            print(
                f"Couldn't place node {node}/{len(G.nodes)}, all chips full. "
                f"Node would've taken {((n_sz / total_chip_size) * 100) if total_chip_size else 0:.2f}% "
                f"of the average chip size."
            )
            return None

        # Choose the chip that keeps max-latency as low as possible:
        #   primary: lowest latency load
        #   secondary: lowest size load (less fragmentation)
        #   tertiary: lowest chip id (deterministic)
        best_chip = min(
            feasible,
            key=lambda c: (chip_latency_load[c], chip_size_load[c], c)
        )

        # Assign
        partitions[best_chip].append(node)
        chip_latency_load[best_chip] += n_lat
        chip_size_load[best_chip]    += n_sz

    return partitions


def chip_partition_balanced5(G, num_chips, chip_sizes, **kwargs):
    """
    Greedy balanced partitioning of nodes onto chips with hard size constraints and soft latency balance.
    Nodes are assigned to chips with enough remaining size and lowest latency load.
    """
    topo_order = list(nx.topological_sort(G))
    latencies = [G.nodes[node]['latency'] for node in topo_order]
    sizes = [G.nodes[node]['size'] for node in topo_order]

    total_latency = sum(latencies)
    avg_latency_load = total_latency / num_chips if num_chips > 0 else 0.0

    partitions = {c: [] for c in range(num_chips)}
    chip_latency_load = [0.0] * num_chips
    chip_size_load = [0.0] * num_chips
    node_to_chip = {}
    unassigned_nodes = []

    current_chip = 0
    for i, node in enumerate(topo_order):
        node_latency = latencies[i]
        node_size = sizes[i]

        place_on_current = True
        if chip_latency_load[current_chip] + node_latency > avg_latency_load * 1.05:
            place_on_current = False
        if chip_size_load[current_chip] + node_size > chip_sizes[current_chip]:
            place_on_current = False

        if place_on_current:
            pass
        else:
            # Try to find a better chip
            found = False
            for chip_id in range(num_chips):
                if chip_size_load[chip_id] + node_size <= chip_sizes[chip_id]:
                    if chip_latency_load[chip_id] + node_latency <= avg_latency_load * 1.05:
                        current_chip = chip_id
                        found = True
                        break
            if not found:
                # Couldn't place now, will place later
                unassigned_nodes.append((node, node_latency, node_size))
                continue

        partitions[current_chip].append(node)
        chip_latency_load[current_chip] += node_latency
        chip_size_load[current_chip] += node_size
        node_to_chip[node] = current_chip

    # Handle unassigned nodes: put on chip with lowest latency load among those with enough space
    for node, node_latency, node_size in unassigned_nodes:
        best_chip = None
        lowest_latency = float('inf')
        for chip_id in range(num_chips):
            free_space = chip_sizes[chip_id] - chip_size_load[chip_id]
            if free_space >= node_size:
                if chip_latency_load[chip_id] < lowest_latency:
                    best_chip = chip_id
                    lowest_latency = chip_latency_load[chip_id]
        if best_chip is None:
            print(f"ERROR: Could not place node {node}. No chip has enough space.")
            return None
        partitions[best_chip].append(node)
        chip_latency_load[best_chip] += node_latency
        chip_size_load[best_chip] += node_size
        node_to_chip[node] = best_chip

    return [partitions[c] for c in range(num_chips)]


def wave_parallel_partition(G, num_chips, chip_sizes, num_subaccs=5):
    """
    Partition G into chips with SA and memory limits, assigning nodes wave by wave.
    Assumes no memory constraints and identical chips.
    """
    max_parallel_per_chip = num_subaccs  # max parallel nodes per chip
    # Compute ASAP levels to get parallel waves
    node_levels = compute_asap_levels(G)
    
    # Group nodes by level
    level_nodes = defaultdict(list)
    for node, level in node_levels.items():
        level_nodes[level].append(node)
    
    partitions = [[] for _ in range(num_chips)]
    chip_wave_usage = [defaultdict(int) for _ in range(num_chips)]  # chip_wave_usage[chip][level] = count
    chip_pointer = 0  # Keep track of which chip to try next

    for level in sorted(level_nodes.keys()):
        wave = level_nodes[level]
        remaining = list(wave)
        
        while remaining:
            # Fit up to max_parallel_per_chip nodes on current chip
            count = 0
            chunk = []
            while remaining and count < max_parallel_per_chip:
                chunk.append(remaining.pop(0))
                count += 1
            
            # Assign chunk to current chip
            for node in chunk:
                partitions[chip_pointer].append(node)
                chip_wave_usage[chip_pointer][level] += 1
            
            # Move to next chip for overflow
            chip_pointer = (chip_pointer + 1) % num_chips

        # Reset chip pointer for next wave to encourage locality
        chip_pointer = 0

    return partitions


def compute_asap_levels(G):
    in_degrees = {n: G.in_degree(n) for n in G.nodes}
    levels = {}
    queue = deque([n for n in G.nodes if in_degrees[n] == 0])
    for n in queue:
        levels[n] = 0
    while queue:
        node = queue.popleft()
        for succ in G.successors(node):
            in_degrees[succ] -= 1
            levels[succ] = max(levels.get(succ, 0), levels[node] + 1)
            if in_degrees[succ] == 0:
                queue.append(succ)
    return levels


def dfs_partition(G, num_chips, chip_sizes, **kwargs):
    """
    Partition nodes based on DFS traversal order.
    """
    dfs_order = list(nx.dfs_preorder_nodes(G))
    partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips
    current_chip = 0

    for node in dfs_order:
        node_size = G.nodes[node]['size']
        # Try to assign to current_chip
        if chip_loads[current_chip] + node_size <= chip_sizes[current_chip]:
            partitions[current_chip].append(node)
            chip_loads[current_chip] += node_size
        else:
            # Move to next chip that fits
            found = False
            for chip_id in range(num_chips):
                if chip_loads[chip_id] + node_size <= chip_sizes[chip_id]:
                    current_chip = chip_id
                    partitions[current_chip].append(node)
                    chip_loads[current_chip] += node_size
                    found = True
                    break
            if not found:
                total_chip_size = sum(chip_sizes)
                print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
                return None
    return partitions


def greedy_merge_partitions(G, num_chips, chip_sizes):
    """
    Start with each node in its own partition, then greedily merge partitions 
    until reaching num_chips, guided by minimal increase in communication cost.
    """
    # Initialize each node as a separate partition
    partitions = [{node} for node in G.nodes()]
    partition_sizes = [G.nodes[node]['size'] for node in G.nodes()]

    # Precompute edge weights
    def inter_partition_cost(p1, p2):
        cost = 0
        for u in p1:
            for v in p2:
                if G.has_edge(u, v) or G.has_edge(v, u):
                    cost += G[u][v].get('size', 1) if G.has_edge(u, v) else 0
                    cost += G[v][u].get('size', 1) if G.has_edge(v, u) else 0
        return cost

    # Greedy merging until number of partitions equals num_chips
    while len(partitions) > num_chips:
        best_pair = None
        best_increase = float('inf')
        # Try all pairs of partitions for potential merge
        for i in range(len(partitions)):
            for j in range(i+1, len(partitions)):
                # Check capacity constraint for merge
                if partition_sizes[i] + partition_sizes[j] > max(chip_sizes):
                    continue
                cost_increase = inter_partition_cost(partitions[i], partitions[j])
                if cost_increase < best_increase:
                    best_increase = cost_increase
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        # Merge partitions i and j
        partitions[i] = partitions[i].union(partitions[j])
        partition_sizes[i] += partition_sizes[j]
        # Remove partition j
        partitions.pop(j)
        partition_sizes.pop(j)

    # Now assign merged partitions to chips greedily
    final_partitions = [[] for _ in range(num_chips)]
    chip_loads = [0.0] * num_chips
    for part in partitions:
        part_size = sum(G.nodes[n]['size'] for n in part)
        possible_chips = [i for i in range(num_chips) if chip_loads[i] + part_size <= chip_sizes[i]]
        if not possible_chips:
            # split partition if necessary (fallback)
            for node in part:
                for chip_id in range(num_chips):
                    if chip_loads[chip_id] + G.nodes[node]['size'] <= chip_sizes[chip_id]:
                        final_partitions[chip_id].append(node)
                        chip_loads[chip_id] += G.nodes[node]['size']
                        break
                else:
                    total_chip_size = sum(chip_sizes)
                    print(f"Couldn't place node {node}/{len(G.nodes)}, all chips full. Node would've taken {((G.nodes[node]['size'] / total_chip_size) * 100):.2f}% of the average chip size.")
                    return None
        else:
            target = min(possible_chips, key=lambda i: chip_loads[i])
            final_partitions[target].extend(list(part))
            chip_loads[target] += part_size

    return final_partitions


def dream_asap(G, num_subaccs):
    # Partitions for a single chip
    node_to_resource = {}  # Dictionary to map nodes to Subaccs
    schedules = {p:[] for p in range(num_subaccs)}  # Dictionary to map Subaccs to nodes
    schedules_start = {p:[] for p in range(num_subaccs)}  # Dictionary to map Subaccs to start times
    schedules_end = {p:[] for p in range(num_subaccs)}  # Dictionary to map Subaccs to end times
    # Topological ordering for ASAP scheduling
    topo_order = list(nx.topological_sort(G))
    next_free_time = [0.0] * num_subaccs  # Track next free time of each SA
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes  # Start time of each node
    end_time = [0.0] * num_nodes  # Start time of each node

    # ASAP scheduling across Subaccs
    for node in topo_order:
        # Determine the earliest time the node can start based on parent finishing times
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            earliest_data_ready = max(
                start_time[parent] + G.nodes[parent]['latency'] for parent in G.predecessors(node)
            )

        # Find the earliest available SA for this node
        best_start_time = float('inf')
        best_sa = 0
        for p in range(num_subaccs):
            candidate_start = max(earliest_data_ready, next_free_time[p])
            if candidate_start < best_start_time:
                best_start_time = candidate_start
                best_sa = p

        # Assign the node to the selected SA and update times
        schedules[best_sa].append(node)
        node_to_resource[node] = best_sa
        start_time[node] = best_start_time
        schedules_start[best_sa].append(best_start_time)
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_end[best_sa].append(end_time[node])
        # Update next free time for the SA
        next_free_time[best_sa] = best_start_time + G.nodes[node]['latency']
    schedule = (schedules, schedules_start, schedules_end, node_to_resource)
    return schedule


def latency_threshold_asap2(G, num_chips, chip_sizes, **kwargs):
    return latency_threshold_asap(G, num_chips, chip_sizes, thr=0.015, **kwargs)


def latency_threshold_asap3(G, num_chips, chip_sizes, **kwargs):
    return latency_threshold_asap(G, num_chips, chip_sizes, thr=0.0075, **kwargs)


def latency_threshold_asap(G, num_chips, chip_sizes, *, thr=0.01, **kwargs):
    """
    Greedy ASAP with a latency threshold:
      - Process nodes in topo order.
      - If node latency >= thr: try to start exactly at data_ready; pick any chip that can do so.
        Prefer current_chip if it can start at data_ready; otherwise pick earliest-start chip.
      - Else: keep locality by staying on current_chip if it fits; otherwise place on the chip with
        the earliest possible start; update current_chip to the chosen one.
    Returns: partitions (list[list[int]])
    """
    import math

    def nsize(n): return float(G.nodes[n].get("size", 0.0))
    def nlat(n):  return float(G.nodes[n].get("latency", 0.0))

    partitions = [[] for _ in range(num_chips)]
    used = [0.0] * num_chips
    next_free = [0.0] * num_chips
    start, finish = {}, {}
    total_cap = sum(chip_sizes) if chip_sizes else 0.0

    topo = list(nx.topological_sort(G))
    current_chip = 0

    for v in topo:
        v_lat, v_sz = nlat(v), nsize(v)
        preds = list(G.predecessors(v))
        data_ready = max((finish[u] for u in preds), default=0.0)

        feasible = [c for c in range(num_chips) if used[c] + v_sz <= chip_sizes[c]]
        if not feasible:
            print(f"Couldn't place node {v}/{len(G.nodes)}, all chips full. "
                  f"Node would've taken {((v_sz/total_cap)*100 if total_cap else 0):.2f}% of avg chip.")
            return None

        def start_time_on(c): return max(data_ready, next_free[c])

        chosen = None
        if v_lat >= thr:
            # high-latency: prioritize starting at DR; prefer current_chip if it can
            cand = [c for c in feasible if next_free[c] <= data_ready]
            if current_chip in cand:
                chosen = current_chip
            elif cand:
                # choose the one that was most recently busy (max next_free) to reduce idle bubbles
                chosen = max(cand, key=lambda c: next_free[c])
            else:
                chosen = min(feasible, key=start_time_on)
        else:
            # low-latency: stay on current chip if possible
            if current_chip in feasible:
                chosen = current_chip
            else:
                chosen = min(feasible, key=lambda c: (start_time_on(c), used[c], c))

        s = start_time_on(chosen)
        e = s + v_lat
        partitions[chosen].append(v)
        used[chosen] += v_sz
        start[v], finish[v] = s, e
        next_free[chosen] = e
        current_chip = chosen  # keep greedy locality

    return partitions


def edge_ratio_parallelizer2(G, num_chips, chip_sizes, **kwargs):
    return edge_ratio_parallelizer(G, num_chips, chip_sizes, scale=0.5, w_lat=1.0, w_energy=1.0, **kwargs)


def edge_ratio_parallelizer3(G, num_chips, chip_sizes, **kwargs):
    return edge_ratio_parallelizer(G, num_chips, chip_sizes, scale=0.05, w_lat=1.0, w_energy=1.0, **kwargs)


def edge_ratio_parallelizer(G, num_chips, chip_sizes, *, scale=0.1, w_lat=1.0, w_energy=1.0, **kwargs):
    """
    Decide parallelize vs. co-locate by comparing node latency to incident-edge cost:
      If latency(node) >= scale * sum_{(u,v) incident} [w_lat*edge.latency + w_energy*edge.energy],
      treat as 'parallelize': try to start it at data_ready on any chip; else co-locate with anchor pred.
    Returns: partitions (list[list[int]])
    """
    import math

    def nsize(n): return float(G.nodes[n].get("size", 0.0))
    def nlat(n):  return float(G.nodes[n].get("latency", 0.0))
    def e_lat(u,v): return float(G.edges[u, v].get("latency", 0.0))
    def e_en(u,v):  return float(G.edges[u, v].get("energy",  0.0))

    partitions = [[] for _ in range(num_chips)]
    used = [0.0] * num_chips
    next_free = [0.0] * num_chips
    start, finish = {}, {}
    total_cap = sum(chip_sizes) if chip_sizes else 0.0

    topo = list(nx.topological_sort(G))

    for v in topo:
        v_lat, v_sz = nlat(v), nsize(v)
        preds = list(G.predecessors(v))
        succs = list(G.successors(v))
        data_ready = max((finish[u] for u in preds), default=0.0)
        # anchor predecessor for co-location
        anchor = max(preds, key=lambda u: finish[u]) if preds else None
        anchor_chip = None
        if anchor is not None:
            for c in range(num_chips):
                if anchor in partitions[c]:
                    anchor_chip = c
                    break

        # compute incident edge cost
        edge_sum = 0.0
        for u in preds:
            edge_sum += w_lat * e_lat(u, v) + w_energy * e_en(u, v)
        for w in succs:
            edge_sum += w_lat * e_lat(v, w) + w_energy * e_en(v, w)

        parallel_prefer = (v_lat >= scale * edge_sum)

        feasible = [c for c in range(num_chips) if used[c] + v_sz <= chip_sizes[c]]
        if not feasible:
            print(f"Couldn't place node {v}/{len(G.nodes)}, all chips full. "
                  f"Node would've taken {((v_sz/total_cap)*100 if total_cap else 0):.2f}% of avg chip.")
            return None

        def start_time_on(c): return max(data_ready, next_free[c])

        if parallel_prefer:
            # try to start at data_ready; otherwise pick earliest-start chip
            cand = [c for c in feasible if next_free[c] <= data_ready]
            chosen = max(cand, key=lambda c: next_free[c]) if cand else min(feasible, key=start_time_on)
        else:
            # co-locate on anchor chip if feasible; else earliest-start chip
            if anchor_chip is not None and anchor_chip in feasible:
                chosen = anchor_chip
            else:
                chosen = min(feasible, key=lambda c: (start_time_on(c), used[c], c))

        s = start_time_on(chosen)
        e = s + v_lat
        partitions[chosen].append(v)
        used[chosen] += v_sz
        start[v], finish[v] = s, e
        next_free[chosen] = e

    return partitions


def dream_all_on_one_sa(G, num_subaccs, sa_id: int = 0):
    node_to_resource = {}
    schedules = {p: [] for p in range(num_subaccs)}
    schedules_start = {p: [] for p in range(num_subaccs)}
    schedules_end = {p: [] for p in range(num_subaccs)}

    topo_order = list(nx.topological_sort(G))
    next_free_time = [0.0] * num_subaccs
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes
    end_time = [0.0] * num_nodes

    for node in topo_order:
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            earliest_data_ready = max(
                start_time[parent] + G.nodes[parent]['latency'] for parent in G.predecessors(node)
            )

        best_start_time = max(next_free_time[sa_id], earliest_data_ready)
        schedules[sa_id].append(node)
        node_to_resource[node] = sa_id
        start_time[node] = best_start_time
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_start[sa_id].append(best_start_time)
        schedules_end[sa_id].append(end_time[node])
        next_free_time[sa_id] = end_time[node]

    return schedules, schedules_start, schedules_end, node_to_resource

def dream_asap_new(G, num_subaccs):
    node_to_resource = {}
    schedules = {p: [] for p in range(num_subaccs)}
    schedules_start = {p: [] for p in range(num_subaccs)}
    schedules_end = {p: [] for p in range(num_subaccs)}

    topo_order = list(nx.topological_sort(G))
    next_free_time = [0.0] * num_subaccs
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes
    end_time = [0.0] * num_nodes

    for node in topo_order:
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            earliest_data_ready = max(
                start_time[parent] + G.nodes[parent]['latency'] for parent in G.predecessors(node)
            )

        # Find all Subaccs that are free at earliest_data_ready
        candidate_subaccs = [p for p in range(num_subaccs) if next_free_time[p] <= earliest_data_ready]

        if candidate_subaccs:
            # Pick the one that was last used most recently (i.e., highest next_free_time)
            best_sa = max(candidate_subaccs, key=lambda p: next_free_time[p])
            best_start_time = earliest_data_ready
        else:
            # No free Subaccs, pick the one that becomes free the soonest
            best_sa = min(range(num_subaccs), key=lambda p: next_free_time[p])
            best_start_time = next_free_time[best_sa]

        # Assign node
        schedules[best_sa].append(node)
        node_to_resource[node] = best_sa
        start_time[node] = best_start_time
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_start[best_sa].append(best_start_time)
        schedules_end[best_sa].append(end_time[node])
        next_free_time[best_sa] = end_time[node]

    return schedules, schedules_start, schedules_end, node_to_resource


def dream_greedy_throughput(G, num_subaccs):
        # Partitions for a single chip
    node_to_resource = {}  # Dictionary to map nodes to Subaccs
    schedules = {p:[] for p in range(num_subaccs)}  # Dictionary to map Subaccs to nodes
    schedules_start = {p:[] for p in range(num_subaccs)}  # Dictionary to map Subaccs to start times
    schedules_end = {p:[] for p in range(num_subaccs)}  # Dictionary to map Subaccs to end times
    # Topological ordering for ASAP scheduling
    topo_order = list(nx.topological_sort(G))
    next_free_time = [0.0] * num_subaccs  # Track next free time of each SA
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes  # Start time of each node
    end_time = [0.0] * num_nodes  # Start time of each node
    sa_load = [0.0] * num_subaccs  # Track load on each SA

    # ASAP scheduling across Subaccs
    for node in topo_order:
        # Determine the earliest time the node can start based on parent finishing times
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            earliest_data_ready = max(
                start_time[parent] + G.nodes[parent]['latency'] for parent in G.predecessors(node)
            )

        # Find the earliest available SA for this node
        best_start_time = float('inf')
        best_sa = 0
        subaccs_sorted = sorted(range(num_subaccs), key=lambda p: sa_load[p])
        for p in subaccs_sorted:
            candidate_start = max(earliest_data_ready, next_free_time[p])
            if candidate_start < best_start_time:
                best_start_time = candidate_start
                best_sa = p

        # Assign the node to the selected SA and update times
        schedules[best_sa].append(node)
        node_to_resource[node] = best_sa
        start_time[node] = best_start_time
        schedules_start[best_sa].append(best_start_time)
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_end[best_sa].append(end_time[node])
        # Update next free time for the SA
        next_free_time[best_sa] = best_start_time + G.nodes[node]['latency']
        sa_load[best_sa] += G.nodes[node]['latency']
    schedule = (schedules, schedules_start, schedules_end, node_to_resource)
    return schedule


def dream_all_on_one_sa(G, num_subaccs, sa_id: int = 0):
    node_to_resource = {}
    schedules = {p: [] for p in range(num_subaccs)}
    schedules_start = {p: [] for p in range(num_subaccs)}
    schedules_end = {p: [] for p in range(num_subaccs)}

    topo_order = list(nx.topological_sort(G))
    next_free_time = [0.0] * num_subaccs
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes
    end_time = [0.0] * num_nodes

    for node in topo_order:
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            earliest_data_ready = max(
                start_time[parent] + G.nodes[parent]['latency'] for parent in G.predecessors(node)
            )

        best_start_time = max(next_free_time[sa_id], earliest_data_ready)
        schedules[sa_id].append(node)
        node_to_resource[node] = sa_id
        start_time[node] = best_start_time
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_start[sa_id].append(best_start_time)
        schedules_end[sa_id].append(end_time[node])
        next_free_time[sa_id] = end_time[node]

    return schedules, schedules_start, schedules_end, node_to_resource

def dream_limited_jumps_balanced(G, num_subaccs, wiggle, max_jumps=None):
    if max_jumps is None:
        max_jumps = len(G)

    topo_order = list(nx.topological_sort(G))
    latencies = [G.nodes[node]['latency'] for node in topo_order]
    total_latency = sum(latencies)
    avg_load = total_latency / num_subaccs if num_subaccs > 0 else 0.0

    schedules = {p: [] for p in range(num_subaccs)}
    schedules_start = {p: [] for p in range(num_subaccs)}
    schedules_end = {p: [] for p in range(num_subaccs)}
    node_to_resource = {}
    start_time = {}
    end_time = {}
    next_free_time = [0.0] * num_subaccs
    sa_load = [0.0] * num_subaccs 
    jumps_used = 0

    current_sa = 0
    for i, node in enumerate(topo_order):
        node_latency = latencies[i]
        
        if G.in_degree(node) == 0:
            data_ready_time = 0.0
        else:
            data_ready_time = max(end_time[pred] for pred in G.predecessors(node))

        place_on_current = True
        if sa_load[current_sa] + node_latency > avg_load * wiggle:
            place_on_current = False

        if place_on_current:
            pass
        else:
            if jumps_used < max_jumps:
                best_sa = min(range(num_subaccs), key=lambda p: sa_load[p])
                if sa_load[best_sa] + node_latency < sa_load[current_sa] + node_latency:
                    current_sa = best_sa
                    jumps_used += 1

        node_start_time = max(next_free_time[current_sa], data_ready_time)
        node_end_time = node_start_time + node_latency
        schedules[current_sa].append(node)
        schedules_start[current_sa].append(node_start_time)
        schedules_end[current_sa].append(node_end_time)
        node_to_resource[node] = current_sa
        start_time[node] = node_start_time
        end_time[node] = node_end_time
        next_free_time[current_sa] = node_end_time
        sa_load[current_sa] += node_latency

    schedule = (schedules, schedules_start, schedules_end, node_to_resource)
    return schedule


def multi_chip_asap(G, num_chips, num_subaccs, H, partitions):
    node_to_chip = {node:chip for chip,nodes in enumerate(partitions) for node in nodes}
    node_to_resource = {}
    schedules = {n:{p:[] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_start = {n:{p:[] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_end = {n:{p:[] for p in range(num_subaccs)} for n in range(num_chips)}
    # Topological ordering for ASAP scheduling
    topo_order = list(nx.topological_sort(G))
    next_free_time = {n:[0.0] * num_subaccs for n in range(num_chips)}  # Track next free time of each SA
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes  # Start time of each node
    end_time = [0.0] * num_nodes  # Start time of each node

    def get_hops(node1,node2):
        chip1 = node_to_chip[node1]
        chip2 = node_to_chip[node2]
        return H[chip1][chip2]
        
    #ASAP Scheduling across Chips & Subaccs
    for node in topo_order:
        # Determine the earliest time the node can start based on parent finishing times
        n = node_to_chip[node]
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            readys = []
            parents = G.predecessors(node)
            for parent in parents:
                m = node_to_chip[parent]
                readys.append(start_time[parent] + G.nodes[parent]['latency'] + G.edges[parent,node]['latency']*get_hops(parent,node))
            earliest_data_ready = max(readys)
        
        # Find the earliest available SA for this node
        best_start_time = float('inf')
        best_sa = 0
        for p in range(num_subaccs):
            candidate_start = max(earliest_data_ready, next_free_time[n][p])
            if candidate_start < best_start_time:
                best_start_time = candidate_start
                best_sa = p
        
        schedules[n][best_sa].append(node)
        node_to_resource[node] = (n,best_sa)
        start_time[node] = best_start_time
        schedules_start[n][best_sa].append(best_start_time)
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_end[n][best_sa].append(end_time[node])
        next_free_time[n][best_sa] = best_start_time + G.nodes[node]['latency']
    schedule = (schedules, schedules_start, schedules_end, node_to_resource, node_to_chip)
    return schedule


def multi_chip_asap_new(G, num_chips, num_subaccs, H, partitions):
    node_to_chip = {node: chip for chip, nodes in enumerate(partitions) for node in nodes}
    node_to_resource = {}
    
    schedules = {n: {p: [] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_start = {n: {p: [] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_end = {n: {p: [] for p in range(num_subaccs)} for n in range(num_chips)}

    topo_order = list(nx.topological_sort(G))
    next_free_time = {n: [0.0] * num_subaccs for n in range(num_chips)}
    
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes
    end_time = [0.0] * num_nodes

    def get_hops(node1, node2):
        chip1 = node_to_chip[node1]
        chip2 = node_to_chip[node2]
        return H[chip1][chip2]
    
    for node in topo_order:
        n = node_to_chip[node]

        # Determine earliest data-ready time
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            readys = []
            for parent in G.predecessors(node):
                parent_chip = node_to_chip[parent]
                parent_end = start_time[parent] + G.nodes[parent]['latency']
                comm_delay = G.edges[parent, node]['latency'] * get_hops(parent, node)
                readys.append(parent_end + comm_delay)
            earliest_data_ready = max(readys)

        # Choose best SA
        candidate_subaccs = []
        for p in range(num_subaccs):
            if next_free_time[n][p] <= earliest_data_ready:
                candidate_subaccs.append(p)

        if candidate_subaccs:
            # Prefer most recently used one (highest next_free_time)
            best_sa = max(candidate_subaccs, key=lambda p: next_free_time[n][p])
            best_start_time = earliest_data_ready
        else:
            # No Subaccs are free yet — pick the one that becomes free soonest
            best_sa = min(range(num_subaccs), key=lambda p: next_free_time[n][p])
            best_start_time = next_free_time[n][best_sa]

        # Schedule node
        schedules[n][best_sa].append(node)
        node_to_resource[node] = (n, best_sa)
        start_time[node] = best_start_time
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_start[n][best_sa].append(best_start_time)
        schedules_end[n][best_sa].append(end_time[node])
        next_free_time[n][best_sa] = end_time[node]

    return schedules, schedules_start, schedules_end, node_to_resource, node_to_chip


def multi_chip_asap_new_hetero(G, num_chips, num_subaccs, subaccs_per_chip, H, partitions):
    node_to_chip = {node: chip for chip, nodes in enumerate(partitions) for node in nodes}
    node_to_resource = {}
    
    schedules = {n: {p: [] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_start = {n: {p: [] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_end = {n: {p: [] for p in range(num_subaccs)} for n in range(num_chips)}

    topo_order = list(nx.topological_sort(G))
    next_free_time = {n: [0.0] * subaccs_per_chip[n] for n in range(num_chips)}
    
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes
    end_time = [0.0] * num_nodes

    def get_hops(node1, node2):
        chip1 = node_to_chip[node1]
        chip2 = node_to_chip[node2]
        return H[chip1][chip2]
    
    for node in topo_order:
        n = node_to_chip[node]

        # Determine earliest data-ready time
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            readys = []
            for parent in G.predecessors(node):
                parent_chip = node_to_chip[parent]
                parent_end = start_time[parent] + G.nodes[parent]['latency']
                comm_delay = G.edges[parent, node]['latency'] * get_hops(parent, node)
                readys.append(parent_end + comm_delay)
            earliest_data_ready = max(readys)

        # Choose best SA
        candidate_subaccs = []
        for p in range(subaccs_per_chip[n]):
            if next_free_time[n][p] <= earliest_data_ready:
                candidate_subaccs.append(p)

        if candidate_subaccs:
            # Prefer most recently used one (highest next_free_time)
            best_sa = max(candidate_subaccs, key=lambda p: next_free_time[n][p])
            best_start_time = earliest_data_ready
        else:
            # No Subaccs are free yet — pick the one that becomes free soonest
            best_sa = min(range(subaccs_per_chip[n]), key=lambda p: next_free_time[n][p])
            best_start_time = next_free_time[n][best_sa]

        # Schedule node
        schedules[n][best_sa].append(node)
        node_to_resource[node] = (n, best_sa)
        start_time[node] = best_start_time
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_start[n][best_sa].append(best_start_time)
        schedules_end[n][best_sa].append(end_time[node])
        next_free_time[n][best_sa] = end_time[node]

    return schedules, schedules_start, schedules_end, node_to_resource, node_to_chip


def assign_dream_shedule(schedule, num_subaccs, num_slots, epsilon, w2s_etime):
    slots = {p:{s:[] for s in range(num_slots)} for p in range(num_subaccs)} 
    schedules, schedules_start, schedules_end, _ = schedule
    num_non_empty_slots = {p:0 for p in range(num_subaccs)}
    for p in range(num_subaccs):
        if len(schedules[p]) == 0:
            continue
        slot = 0
        slots[p][slot].append(schedules[p][0])
        for i in range(1,len(schedules[p])):
            finish_time = schedules_end[p][i-1]
            if schedules_start[p][i] > finish_time + max(epsilon,w2s_etime) and slot < num_slots - 1:
                slot += 1
            slots[p][slot].append(schedules[p][i])
        num_non_empty_slots[p] = slot + 1
    return slots, num_non_empty_slots

def assign_multi_chip_schedule(schedule, num_chips, num_subaccs, num_slots, epsilon, w2s_etime):
    slots = {n:{p:{s:[] for s in range(num_slots)} for p in range(num_subaccs)} for n in range(num_chips)}
    schedules, schedules_start, schedules_end, node_to_resource, node_to_chip = schedule
    num_non_empty_slots = {n:{p:0 for p in range(num_subaccs)} for n in range(num_chips)}
    for n in range(num_chips):
        for p in range(num_subaccs):
            if len(schedules[n][p]) == 0:
                continue
            slot = 0
            slots[n][p][slot].append(schedules[n][p][0])
            for i in range(1,len(schedules[n][p])):
                finish_time = schedules_end[n][p][i-1]
                if schedules_start[n][p][i] > finish_time + max(epsilon,w2s_etime) and slot < num_slots - 1:
                    slot += 1
                slots[n][p][slot].append(schedules[n][p][i])
            num_non_empty_slots[n][p] = slot + 1
    return slots, num_non_empty_slots


def multi_chip_asap_throughput(G, num_chips, num_subaccs, H, partitions):
    node_to_chip = {node:chip for chip,nodes in enumerate(partitions) for node in nodes}
    node_to_resource = {}
    schedules = {n:{p:[] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_start = {n:{p:[] for p in range(num_subaccs)} for n in range(num_chips)}
    schedules_end = {n:{p:[] for p in range(num_subaccs)} for n in range(num_chips)}
    # Topological ordering for ASAP scheduling
    topo_order = list(nx.topological_sort(G))
    next_free_time = {n:[0.0] * num_subaccs for n in range(num_chips)}  # Track next free time of each SA
    num_nodes = len(G.nodes)
    start_time = [0.0] * num_nodes  # Start time of each node
    end_time = [0.0] * num_nodes  # Start time of each node
    sa_load = {n:[0.0] * num_subaccs for n in range(num_chips)}

    def get_hops(node1,node2):
        chip1 = node_to_chip[node1]
        chip2 = node_to_chip[node2]
        return H[chip1][chip2]
        
    #ASAP Scheduling across Chips & Subaccs
    for node in topo_order:
        # Determine the earliest time the node can start based on parent finishing times
        n = node_to_chip[node]
        if G.in_degree(node) == 0:
            earliest_data_ready = 0.0
        else:
            readys = []
            parents = G.predecessors(node)
            for parent in parents:
                m = node_to_chip[parent]
                readys.append(start_time[parent] + G.nodes[parent]['latency'] + G.edges[parent,node]['latency']*get_hops(parent,node))
            earliest_data_ready = max(readys)

        best_start_time = float('inf')
        best_sa = 0
        subaccs_sorted = sorted(range(num_subaccs), key=lambda p: sa_load[n][p])
        for p in subaccs_sorted:
            candidate_start = max(earliest_data_ready, next_free_time[n][p])
            if candidate_start < best_start_time:
                best_start_time = candidate_start
                best_sa = p
        
        sa_load[n][best_sa] += G.nodes[node]['latency']
        schedules[n][best_sa].append(node)
        node_to_resource[node] = (n,best_sa)
        start_time[node] = best_start_time
        schedules_start[n][best_sa].append(best_start_time)
        end_time[node] = best_start_time + G.nodes[node]['latency']
        schedules_end[n][best_sa].append(end_time[node])
        next_free_time[n][best_sa] = best_start_time + G.nodes[node]['latency']
    schedule = (schedules, schedules_start, schedules_end, node_to_resource, node_to_chip)
    return schedule


def add_dream_assignment_start(model_vars):
    dim = model_vars.get_dim("Assignment")
    start = np.ones(dim, dtype=int)
    model_vars.add_start("Assignment",start)


def add_multi_chip_assignment_start(model_vars,partition):
    dim = model_vars.get_dim("Assignment")
    start = np.zeros(dim, dtype=int)
    for chip in range(len(partition)):
        for node in partition[chip]:
            start[node,chip] = 1
    model_vars.add_start("Assignment", start)  


def add_dream_schedule_start(model_vars, schedule):
    dim = model_vars.get_dim("SA_Assignment")
    start = np.zeros(dim, dtype=int)
    schedules, _, _, _ = schedule
    for p in range(len(schedules)):
        for node in schedules[p]:
            start[node,0,p] = 1
    model_vars.add_start("SA_Assignment",start)

def add_finish_time(model_vars, start_time_dict, G):
    dim = model_vars.get_dim("Latency")  
    start = np.zeros(dim)
    for node in G.nodes:
        start[node] = start_time_dict[node] + G.nodes[node]['latency']
    model_vars.add_start("Latency", start)
    return start


def add_tminus_latencies(model_vars, t, latencies, G):
    dim = model_vars.get_dim("TMinusLatencies") 
    start = np.zeros(dim)
    for node in G.nodes:
        start[node] = t[node] - latencies[node]
    model_vars.add_start("TMinusLatencies", start)
    return start


def add_communication_reqs(model_vars, comm_hops):
    dim = model_vars.get_dim("Communication_req")
    start = np.zeros(dim, dtype=int)
    for e, hops in enumerate(comm_hops):
        start[e] = hops
    model_vars.add_start("Communication_req", start)


def add_multi_chip_schedule_start(model_vars, schedule):
    dim = model_vars.get_dim("SA_Assignment")
    start = np.zeros(dim, dtype=int)
    schedules, _, _, node_to_resource, _ = schedule
    for chip in range(len(schedules)):
        for p in range(len(schedules[chip])):
            for node in schedules[chip][p]:
                start[node,chip,p] = 1
    model_vars.add_start("SA_Assignment",start)


def add_dream_slot_start(model_vars, num_chips, num_subaccs, max_number_of_slots, slots, num_non_empty_slots):
    dim_nsa = model_vars.get_dim("NodeSlotAssignment")
    start_nsa = np.zeros(dim_nsa, dtype=int)
    dim_su = model_vars.get_dim("SlotUsed")
    start_su = np.zeros(dim_su, dtype=int)
    dim_nes = model_vars.get_dim("NumNonEmptySlots")
    start_nes = np.zeros(dim_nes, dtype=int)
    if model_vars.matrix_load_vars:
        N = num_chips
        S = num_subaccs
        Q = max_number_of_slots
        for s in range(S):
            start_nes[s] = num_non_empty_slots[s]
            for q in range(num_non_empty_slots[s]):
                start_su[s*Q+q] = 1
                for node in slots[s][q]:
                    start_nsa[node,s*Q+q] = 1
    else:
        for p in range(len(slots)):
            for s in range(num_non_empty_slots[p]):
                start_nes[0,p] = num_non_empty_slots[p]
                start_su[0,p,s] = 1
                for node in slots[p][s]:
                    start_nsa[0,p,s,node] = 1
    model_vars.add_start("NodeSlotAssignment",start_nsa)
    model_vars.add_start("SlotUsed",start_su)
    model_vars.add_start("NumNonEmptySlots",start_nes)
    return start_nsa


def add_multi_chip_slot_start(model_vars, num_chips, num_subaccs, max_number_of_slots, slots, num_non_empty_slots, start_times, latency_ub):
    dim_nsa = model_vars.get_dim("NodeSlotAssignment")
    start_nsa = np.zeros(dim_nsa, dtype=int)
    dim_su = model_vars.get_dim("SlotUsed")
    start_su = np.zeros(dim_su, dtype=int)
    dim_nes = model_vars.get_dim("NumNonEmptySlots")
    start_nes = np.zeros(dim_nes, dtype=int)
    if model_vars.matrix_load_vars:
        N = num_chips
        S = num_subaccs
        Q = max_number_of_slots
        for i in range(N):
            for s in range(S):
                start_nes[i*S+s] = num_non_empty_slots[i][s]
                for q in range(num_non_empty_slots[i][s]):
                    start_su[i*S*Q+s*Q+q] = 1
                    for node in slots[i][s][q]:
                        start_nsa[node,i*S*Q+s*Q+q] = 1
    else:
        for chip in range(len(slots)):
            for p in range(len(slots[chip])):
                for s in range(num_non_empty_slots[chip][p]):
                    start_nes[chip,p] = num_non_empty_slots[chip][p]
                    start_su[chip,p,s] = 1
                    for node in slots[chip][p][s]:
                        start_nsa[chip,p,s,node] = 1
    model_vars.add_start("NodeSlotAssignment",start_nsa)
    model_vars.add_start("SlotUsed",start_su)
    model_vars.add_start("NumNonEmptySlots",start_nes)
    return start_nsa


def add_dream_chip_slot_start_times(model_vars, slots, start_times, num_subaccs, max_number_of_slots, latency_ub):
    dim_sst = model_vars.get_dim("SlotStartTime")
    dim_sst_min = model_vars.get_dim("SlotStartTimeMin")
    
    start_sst = np.zeros(dim_sst, dtype=float)
    start_sst_min = np.full(dim_sst_min, latency_ub, dtype=float)  # ← default to latency_ub

    for sa in range(num_subaccs):
        for slot in range(len(slots[sa])):
            nodes = slots[sa][slot]
            if not nodes:
                continue
            # Compute t - latency (already in start_times)
            slot_start = min(start_times[node] for node in nodes)
            clamped_start = max(0.0, min(slot_start, latency_ub))
            start_sst[0, sa,slot] = clamped_start
            start_sst_min[0, sa,slot] = clamped_start  # valid only when slot is used!

    model_vars.add_start("SlotStartTime", start_sst)
    model_vars.add_start("SlotStartTimeMin", start_sst_min)


def add_multi_chip_slot_start_times(model_vars, slots, start_times, num_chips, num_subaccs, max_number_of_slots, latency_ub):
    dim_sst = model_vars.get_dim("SlotStartTime")
    dim_sst_min = model_vars.get_dim("SlotStartTimeMin")
    
    start_sst = np.zeros(dim_sst, dtype=float)
    start_sst_min = np.full(dim_sst_min, latency_ub, dtype=float)  # ← default to latency_ub

    for chip in range(num_chips):
        for sa in range(num_subaccs):
            for slot in range(len(slots[chip][sa])):
                nodes = slots[chip][sa][slot]
                if not nodes:
                    continue
                # Compute t - latency (already in start_times)
                slot_start = min(start_times[node] for node in nodes)
                clamped_start = max(0.0, min(slot_start, latency_ub))
                start_sst[chip, sa,slot] = clamped_start
                start_sst_min[chip, sa,slot] = clamped_start  # valid only when slot is used!

    model_vars.add_start("SlotStartTime", start_sst)
    model_vars.add_start("SlotStartTimeMin", start_sst_min)


def add_dream_incomparable_start(model_vars, I_upper, schedule):
    schedules, schedules_start, schedules_end, node_to_resource = schedule
    yz = model_vars.get_dim("yz")
    start_yz = np.zeros(yz, dtype=int)
    z = model_vars.get_dim("z")
    start_z = np.zeros(z, dtype=int)
    w = model_vars.get_dim("w")
    start_w = np.zeros(w, dtype=int)
    k = 0
    for i, j in zip(I_upper.row, I_upper.col):
        pos_i = schedules[node_to_resource[i]].index(i)
        pos_j = schedules[node_to_resource[j]].index(j)
        if node_to_resource[i] == node_to_resource[j]:
            start_yz[k] = 1

            if schedules_start[node_to_resource[i]][pos_i] <= schedules_start[node_to_resource[j]][pos_j]:
                start_w[k] = 1
            else: 
                start_w[k] = 0
        else:
            start_yz[k] = 0
            start_i = schedules_start[node_to_resource[i]][pos_i]
            start_j = schedules_start[node_to_resource[j]][pos_j]
            if start_i <= start_j:
                start_w[k] = 1
            else:
                start_w[k] = 0
        start_z[k] = start_w[k]*start_yz[k]  # z is 1 only if both yz and w are 1
        k += 1
    model_vars.add_start("yz",start_yz)
    model_vars.add_start("z",start_z)
    model_vars.add_start("w",start_w)


def add_multi_chip_incomparable_start(model_vars, I_upper, schedule):
    schedules, schedules_start, schedules_end, node_to_resource, node_to_chip = schedule
    yz = model_vars.get_dim("yz")
    start_yz = np.zeros(yz, dtype=int)
    w = model_vars.get_dim("w")
    start_w = np.zeros(w, dtype=int)
    z = model_vars.get_dim("z")
    start_z = np.zeros(z, dtype=int)
    k = 0
    for i, j in zip(I_upper.row, I_upper.col):
        chip_i = node_to_chip[i]
        chip_j = node_to_chip[j]
        sa_i = node_to_resource[i][1]
        sa_j = node_to_resource[j][1]
        pos_i = schedules[chip_i][sa_i].index(i)
        pos_j = schedules[chip_j][sa_j].index(j)
        start_i = schedules_start[chip_i][sa_i][pos_i]
        start_j = schedules_start[chip_j][sa_j][pos_j]
        if chip_i == chip_j:
            if sa_i == sa_j:
                start_yz[k] = 1
        else:
            start_yz[k] = 0
        if start_i <= start_j:
            start_w[k] = 1
        else:
            start_w[k] = 0
        start_z[k] = start_w[k]*start_yz[k]
        k += 1
    model_vars.add_start("yz",start_yz)
    model_vars.add_start("z",start_z)
    model_vars.add_start("w",start_w)


def write_hints_incomparables(G, I_upper, fname):
    # Label the graph by depth first search
    node_order = list(nx.bfs_tree(G, source=0))
  
    with open(fname, "w") as f:
        f.write(f"#Hints for incomparable nodes, those with deeper DFS difference are given higher priority\n")
        for k, (i, j) in enumerate(zip(I_upper.row, I_upper.col)):
            if node_order.index(i) < node_order.index(j):
                priority = node_order.index(j) - node_order.index(i)
                f.write(f"w[{k}] 1 {priority} \n")
    f.close()
    return