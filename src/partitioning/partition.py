import os
import argparse
import json
import networkx as nx
import time
import math
import wandb
import pathlib
import resource
from partitioner import optimize_parallel_graph_vector_bigM, normalize_graph
from network_topologies import NetworkTopology
from visualize import visualize_schedule, visualize_partitioning
from preprocessor import split_large_nodes
from problem_rep import Problem
from util import load_result_from_json


def calculate_total_model_size(graph):
    return sum(node[1]['size'] for node in graph.nodes(data=True))


# Function to calculate chip configuration based on the base model size
def calculate_chip_configuration(required_chips):
    grid_chips = next_power_of_two(required_chips)
    num_chips_x, num_chips_y = find_balanced_dimensions(grid_chips)
    return required_chips, num_chips_x, num_chips_y


# Supporting functions for chip configuration
def next_power_of_two(x):
    return 1 << (x - 1).bit_length()


def find_balanced_dimensions(total_chips):
    best_x, best_y = 1, total_chips
    for i in range(1, int(math.sqrt(total_chips)) + 1):
        if total_chips % i == 0:
            j = total_chips // i
            if abs(i - j) < abs(best_x - best_y):
                best_x, best_y = i, j
    return best_x, best_y


def find_balanced_dimensions_3d(total_chips):
    best_x, best_y, best_z = 1, 1, total_chips
    
    # Iterate over factors up to the cube root of total_chips
    for x in range(1, int(math.pow(total_chips, 1/3)) + 2):
        if total_chips % x == 0:
            yz = total_chips // x  # Remaining 2D area
            for y in range(1, int(math.sqrt(yz)) + 2):
                if yz % y == 0:
                    z = yz // y
                    # Find the most cubic-like shape (minimize dimension differences)
                    if abs(x - y) + abs(y - z) + abs(z - x) < abs(best_x - best_y) + abs(best_y - best_z) + abs(best_z - best_x):
                        best_x, best_y, best_z = x, y, z
                        
    return best_x, best_y, best_z


def calculate_chip_configuration_3d(required_chips):
    grid_chips = next_power_of_two(required_chips)
    num_chips_x, num_chips_y, num_chips_z = find_balanced_dimensions_3d(grid_chips)
    return required_chips, num_chips_x, num_chips_y, num_chips_z


def main(model_base_name, generation, num_subaccs, target, idle_leakage, heuristic, chips, estimates, powergating, preloading, solution, communication_time_scale, communication_energy_scale, idling_scale, power_scale, rt, dynamic, wakeup_time_scale, wakeup_cost_scale, early_termiantion, wb, lookup, topology, scale, overlap, max_subacc):
    pathlib.Path(f"visualizations/{target}").mkdir(parents=True, exist_ok=True)
    pathlib.Path("solutions").mkdir(parents=True, exist_ok=True)
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    pathlib.Path("starts").mkdir(parents=True, exist_ok=True)

    if target == "EnergyP": 
        target = "Energy"
        pipeline = True
    else: pipeline = False

    if max_subacc == 0:
        max_subacc = chips * num_subaccs

    if wb:
        wandb.init(project="ASPLOS", config={
            "Model": model_base_name,
            "Generation": generation,
            "Subaccelerators (SAs)": num_subaccs,
            "Optimization Target": target,
            "Leakage": idle_leakage,
            "Heuristic": heuristic,
            "Chips": chips,
            "Estimates": estimates,
            "Power Gating": powergating,
            "Target": target
        })

    wakeup_time = 25e-6       
    wakeup_cost = 50e-6
    if 'llama1b' in model_base_name.lower() or 'kv' in model_base_name.lower():
        base_chip_size = 120000000
    elif 'llama8b' in model_base_name:
        base_chip_size = 240000000
    elif 'llama70b' in model_base_name:
        base_chip_size = 1200000000
    else:
        base_chip_size = 12000000 
    single_chip_size = base_chip_size * (generation + 1)
    
    model_name = f"{model_base_name}_gen{generation}"
    file_name = model_name + f"_scale{scale}" if scale else model_name
    with open(f"../model_graphs/{estimates}/{file_name}.json", 'r') as f:
        data = json.load(f)
        print("ℹ️  Loaded model data for:", model_name)

    energy_factor = 5.7 if estimates == "rtl" else 1.0
    latency_factor = 13.3 if estimates == "rtl" else 1.0
    
    node_id_mapping = {node['id']: idx for idx, node in enumerate(data['nodes'])}
    G = nx.DiGraph()
    G.add_nodes_from((node_id_mapping[node['id']], {
        'name': node['name'],
        'latency': float(node['runtime']) or 0.0,
        'energy': node['energy'] or 0.0,
        'size': 0.0 if 'cache' in node['name'].lower() else (node['size'] or 0.0),
        'kv_size': (node['size'] or 0.0) if 'cache' in node['name'].lower() else 0.0,
    }) for node in data['nodes'])
    link_improvement = generation + 1 if not model_name.startswith("bert_deeper") else 1
    G.add_edges_from((node_id_mapping[edge['source']], node_id_mapping[edge['destination']], 
    {'size': edge['size'], 
     'energy': edge.get('energy', 0.0) / (energy_factor * (link_improvement)), 
     'latency': edge.get('latency', 0.0) / (latency_factor * (link_improvement))}) 
    for edge in data['edges'] if not str(edge['source']).startswith("input"))
    
    total_model_size = calculate_total_model_size(G) 
    graph, total_latency, total_energy, model_size, edge_size = normalize_graph(G, scale)
    wakeup_time = wakeup_time / total_latency if powergating else 0
    wakeup_cost = wakeup_cost / total_energy if powergating else 0
    
    config_chips = chips if chips != 0 else math.ceil(total_model_size / single_chip_size)

    if "3D" in topology:
        num_chips, num_chips_x, num_chips_y, num_chips_z = calculate_chip_configuration_3d(config_chips)
    else:
        num_chips, num_chips_x, num_chips_y = calculate_chip_configuration(config_chips)
   
    # Define network topology
    print("\n################## -- Architecture Setup -- ##################")
    if "1D" in topology or "ALL_TO_ALL" in topology:
        T = NetworkTopology(num_chips, type=topology)
    elif "2D" in topology:
        T = NetworkTopology(num_chips_x, num_chips_y, type=topology)
    elif "3D" in topology:
        T = NetworkTopology(num_chips_x, num_chips_y,  num_chips_z, type=topology)

    T.optimize_node_order()
        
    print(f"\nℹ️  Total Model Size: {total_model_size:.3e}, Single Chip Size: {single_chip_size:.3e} ==> Chip Count: {num_chips}")
    extra_nodes = num_chips_x*num_chips_y - num_chips
    if extra_nodes > 0:
        print(f"Removing {extra_nodes} excess nodes to match target chip count.")
        for i in range(extra_nodes):
            T.remove_node(f"M{num_chips_x*num_chips_y - i - 1}")
    hop_matrix = T.get_shortest_hops_matrix()

    # Set up chip sizes
    if chips != 0:
        needed_chips = math.ceil(total_model_size / single_chip_size)
        single_chip_size = single_chip_size / (num_chips / needed_chips) 
    single_chip_size *= 1.05
    graph = split_large_nodes(graph, single_chip_size / total_model_size, 1) 

    chip_sizes = [(single_chip_size / total_model_size)] * num_chips
    
    problem = Problem(m=model_name, c=num_chips, p=num_subaccs, e=estimates, l=idle_leakage, id=idling_scale, pg=powergating,
                    com=communication_time_scale, coe=communication_energy_scale, wts=wakeup_time_scale, wcs=wakeup_cost_scale,
                    po=power_scale,  t=target, tp=topology, sc=scale, ov=overlap, x='multi-chiplet')
    
    if lookup == 0: lookups = [problem]
    elif lookup == 1: 
        lookups = [
            problem.update_values(coe=communication_energy_scale/2, com=communication_time_scale/2, c=num_chips//2, p=num_subaccs*2),
            problem.update_values(coe=communication_energy_scale/2, com=communication_time_scale/2),
            problem.update_values(c=num_chips//2, p=num_subaccs*2),
            ]
    elif lookup == 2:
        lookups = [
            problem.update_values(id=idling_scale/2, wcs=wakeup_cost_scale/2),
            problem.update_values(id=idling_scale/2),
            problem.update_values(wcs=wakeup_cost_scale/2),
            ]
    elif lookup == 3:  
        lookups = [
            problem.update_values(coe=communication_energy_scale/2, com=communication_time_scale/2, p=num_subaccs*2),
            problem.update_values(coe=communication_energy_scale/2, com=communication_time_scale/2),
            problem.update_values(p=num_subaccs*2),
            ]

    dream_bound = load_result_from_json(problem,target, num_chips, num_subaccs, dream=False)
        
    # Run partitioning optimization
    partitioning = optimize_parallel_graph_vector_bigM(
        problem=problem, 
        lookups=lookups,
        G=graph, 
        num_chips=num_chips, 
        num_subaccs=num_subaccs, 
        heuristic=heuristic, 
        target=target, 
        T=T, 
        rt=rt,
        gap=0.025, 
        chip_sizes=chip_sizes, 
        idle_leakage=idle_leakage, 
        dynamic=dynamic, 
        wakeup_time=wakeup_time, 
        wakeup_cost=wakeup_cost, 
        dream_bound=dream_bound,
        dream=False,
        powergating=powergating,
        preloading=preloading,
        solution=solution,
        communication_time_scale=communication_time_scale,
        communication_energy_scale=communication_energy_scale,
        idling_scale=idling_scale,
        power_scale=power_scale,
        wakeup_cost_scale=wakeup_cost_scale,
        wakeup_time_scale=wakeup_time_scale,
        early_termination=early_termiantion,
        wb=wb,
        scale=scale,
        overlap=overlap,
        max_subaccs=max_subacc,
    )
    
    # Total message traffic calculation
    total_message_size = sum(entry['size'] * edge_size for entry in partitioning['message_traffic'])
    total_message_time = sum(entry['latency'] for entry in partitioning['message_traffic'])

    # Log results to WandB
    if wb:
        wandb.run.name = f"{model_name}_{num_chips}c_{num_subaccs}p"
        wandb.log({
            "Communication Energy": partitioning['communicationEnergy']*total_energy,
            "Computational Energy": partitioning['computationalEnergy']*total_energy,
            "Wakeup Energy": partitioning['wakeupEnergy']*total_energy,
            "Idle Energy": partitioning['totalIdleEnergy']*total_energy, 
            "Idle Energy (Pipeline)": partitioning['totalIdleEnergyPipeline']*total_energy,
            "Energy": partitioning['totalEnergy']*total_energy,
            "Energy (Pipeline)": partitioning['totalEnergyPipeline']*total_energy,

            "Maxload": partitioning['totalMaxload']*total_latency,
            "Maxload (Pipeline)": partitioning['totalMaxloadPipeline']*total_latency,
            "Latency": partitioning['totalLatency']*total_latency,

            "EDP": partitioning['totalEDP']*total_latency*total_energy,
            "EPT": partitioning['totalEPT']*total_energy*total_latency,
            
            "Total Message Traffic": total_message_size,
            "Total Message Time": total_message_time*total_latency,
            
            "Subaccelerators (SAs)": partitioning['num_subacc_used'] if partitioning['num_subacc_used'] else num_subaccs,
            "Chip Count": partitioning['num_chips_used'] if partitioning['num_chips_used'] else num_chips,
            "Chip Usage": [{'chip': entry['chip'], 'usage': entry['usage']} for entry in partitioning['capacity_used_per_chip']],
            "Number of Clusters": partitioning['numClusters'],

            "Model Variables": partitioning['num_vars'],
            "Model Constraints": partitioning['num_constraints'],
            "MIP Gap": partitioning['mip_gap'],
            "Dream Gap": (partitioning[f"total{target}"] - dream_bound) / dream_bound if dream_bound else None,
            "TTS": partitioning['time_to_solution'],
        })

        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        def to_human(num_bytes):
            units = ["B", "KB", "MB", "GB", "TB"]
            val = float(num_bytes)
            for u in units:
                if val < 1024 or u == units[-1]: return f"{val:.3f} {u}"
                val /= 1024
        human_peak = to_human(peak_bytes)
        wandb.summary["peak_ram"] = human_peak

    # Visualize and save partitioning results
    partitioning_vis_path = f"visualizations/{target}/{problem.get_solution_key()}_partitioning.png"
    schedule_vis_path = f"visualizations/{target}/{problem.get_solution_key()}_schedule.png"
    assigment_path = f"visualizations/{target}/{problem.get_solution_key()}_assignment.json"

    try:
        with open(assigment_path, 'w') as outfile:
            json.dump(partitioning['assignments'], outfile, indent=4)
        print("\n💾 Assignment saved to:", assigment_path)
        if len(G.nodes) < 10000:
            visualize_partitioning(graph, partitioning, model_name, num_subaccs, target, total_latency, problem.get_solution_key())
            visualize_schedule(partitioning, num_chips, num_subaccs, graph, model_name, hop_matrix, target, total_latency, wakeup_time*wakeup_time_scale*total_latency, problem.get_solution_key())
            if wb: wandb.log({"Partitioning": wandb.Image(partitioning_vis_path), "Schedule": wandb.Image(schedule_vis_path)})
    except Exception as e:
        print(f"⚠️  Warning: Could not generate visualizations. Error: {e}")

    # Print partitioning results
    print(f"\nPartitioning results for {model_name}:")
    print(f"    - Computational Energy: {partitioning['computationalEnergy']*total_energy} ({partitioning['computationalEnergy']/partitioning['totalEnergy']*100:.2f}%)")
    print(f"    - Communication Energy: {partitioning['communicationEnergy']*total_energy} ({partitioning['communicationEnergy']/partitioning['totalEnergy']*100:.2f}%)")
    print(f"    - Wakeup Energy: {partitioning['wakeupEnergy']*total_energy} ({partitioning['wakeupEnergy']/partitioning['totalEnergy']*100:.2f}%) {'--> inactive' if idle_leakage == 1 else ''}")
    print(f"    - Idle Energy: {partitioning['totalIdleEnergy']*total_energy} ({partitioning['totalIdleEnergy']/partitioning['totalEnergy']*100:.2f}%) {'--> inactive' if idle_leakage == 0 else ''}")

    print(f"\n    - Latency: {partitioning['totalLatency']*total_latency}")
    print(f"    - Energy: {partitioning['totalEnergy']*total_energy}")
    print(f"    - Load: {partitioning['totalMaxload']*total_latency}")
    print(f"    = EDP: {partitioning['totalEDP']*total_latency*total_energy}")
    
    print(f"\n    - Energy (Pipeline): {partitioning['totalEnergyPipeline']*total_energy}")
    print(f"    - Idle Energy (Pipeline): {partitioning['totalIdleEnergyPipeline']*total_energy}")
    print(f"    - Load (Pipeline): {partitioning['totalMaxloadPipeline']*total_latency}")
    print(f"    = EPT: {partitioning['totalEPT']*total_energy*total_latency}")
    
    print(f"\n    - Message Traffic: {total_message_size}")
    
    print(f"\n    - Chip Usage: {partitioning['capacity_used_per_chip']}")
    print(f"    - Number of Clusters: {partitioning['numClusters']}")
    
    if wb: wandb.finish()

    if os.path.exists("starts") and not ("70b" in model_name or "bert_hybrid_gen6" in model_name):
        print("🧹 Cleaning up start files...")
        for file in os.listdir("starts"): 
            if problem.get_solution_key() in file:  
                os.remove(os.path.join("starts", file))
                time.sleep(0.1)

    if target == "MaxloadPipeline" or target == "EPT" or pipeline: 
        print(f"{partitioning['totalEnergyPipeline']*total_energy}, {partitioning['totalLatency']*total_latency}, {partitioning['totalMaxloadPipeline']*total_latency}")
    else:
        print(f"{partitioning['totalEnergy']*total_energy}, {partitioning['totalLatency']*total_latency}, {partitioning['totalMaxload']*total_latency}")

# Parse command-line arguments for model, generation, and Subaccs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multi-chip partitioning with WandB logging.")
    parser.add_argument("-m", "--model", type=str, help="Base name of the model to partition.")
    parser.add_argument("-g", "--generation", type=int, default=0, help="Generation number to scale the model.")
    parser.add_argument("-p", "--subAccs", type=int, default=1, help="Number of Subaccelerators (SAs).")
    parser.add_argument("-t", "--target", type=str, default="EDP", help="Optimization target (EDP, Energy, Latency, Messages, Maxload, EPT).")
    parser.add_argument("-l", "--leakage", type=int, default=0, help="Conditional to account for idle leakage power.")
    parser.add_argument("-w", "--heuristic", type=str, default="all", help="Heuristic: louvain, spectral, dfs, order_based, layered, ...")
    parser.add_argument("-c", "--chips", type=int, default=0, help="Number of chips. 0 defaults to the minimum number of chips necessary")
    parser.add_argument("-e", "--estimates", type=str, default="rtl", help="Estimates: hw, rtl, ideal")
    parser.add_argument("-pg", "--powergating", type=int, default=1, help="Enable power gating")
    parser.add_argument("-pre", "--preloading", type=int, default=1, help="Enable preloading")
    parser.add_argument("-sol", "--solution", type=int, default=1, help="If loading a solution, set to 1")
    parser.add_argument("-com", "--communication_time_scale", type=float, default=1.0, help="Communication time scale")
    parser.add_argument("-coe", "--communication_energy_scale", type=float, default=1.0, help="Communication energy scale")
    parser.add_argument("-id", "--idling_scale", type=float, default=1.0, help="Idling scale")
    parser.add_argument("-po", "--power_scale", type=float, default=1.0, help="Power scale")
    parser.add_argument("-rt", "--runtime", type=int, default=1, help="Enable runtime")
    parser.add_argument("-wts", "--wakeup_time_scale", type=float, default=1.0, help="Wakeup time scale")
    parser.add_argument("-wcs", "--wakeup_cost_scale", type=float, default=1.0, help="Wakeup cost scale")
    parser.add_argument("-d", "--dynamic", type=int, default=0, help="Enable dynamic")
    parser.add_argument("-et", "--early_termination", type=int, default=1, help="Enable early termination")
    parser.add_argument("-wb", "--wandb", type=int, default=0, help="Enable WandB logging")
    parser.add_argument("-lo", "--lookup", type=int, default=0, help="Enable lookup")
    parser.add_argument("-tp", "--topology", type=str, default="2D_MESH", help="Network topology type")
    parser.add_argument("-sc", "--scale", type=float, default=0, help="Scale factor for node energy and latency")
    parser.add_argument("-ov", "--overlap", type=int, default=0, help="Enable overlap for maxload computation")
    parser.add_argument("-mp", "--max_subacc", type=int, default=0, help="Maximum number of Subaccs in total")
    args = parser.parse_args()

    # Execute main partitioning function with specified parameters
    main(args.model, args.generation, args.subAccs, args.target, args.leakage, args.heuristic, args.chips, args.estimates, args.powergating, args.preloading, args.solution, args.communication_time_scale, args.communication_energy_scale, args.idling_scale, args.power_scale, args.runtime, args.dynamic, args.wakeup_time_scale, args.wakeup_cost_scale, args.early_termination, args.wandb, args.lookup, args.topology, args.scale, args.overlap, args.max_subacc)
