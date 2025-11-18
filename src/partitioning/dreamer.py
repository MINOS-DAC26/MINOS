# dreamer.py

import os
import argparse
import json
import networkx as nx
import time
import math
import wandb
import pathlib
from partitioner import optimize_parallel_graph_vector_bigM, normalize_graph
from network_topologies import NetworkTopology
from visualize import visualize_schedule, visualize_partitioning
from preprocessor import split_large_nodes
from problem_rep import Problem
from util import save_results_to_json

def calculate_total_model_size(graph):
    return sum(node[1]['size'] for node in graph.nodes(data=True))

        
def main(model_base_name, generation, num_subaccs, target, heuristic, chips, estimates, powergating, preloading, solution, communication_time_scale, communication_energy_scale, idling_scale, power_scale, rt, dynamic, wakeup_time_scale, wakeup_cost_scale, wb, leakage, lookup):
    pathlib.Path(f"visualizations/{target}").mkdir(parents=True, exist_ok=True)
    pathlib.Path("solutions").mkdir(parents=True, exist_ok=True)
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    pathlib.Path("starts").mkdir(parents=True, exist_ok=True)
    if wb:
        wandb.init(project="ASPLOS", config={
            "Model": model_base_name,
            "Generation": generation,
            "Subaccelerators (SAs)": num_subaccs,
            "Optimization Target": target,
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
    with open(f"../model_graphs/{estimates}/{model_name}.json", 'r') as f:
        data = json.load(f)
    
    node_id_mapping = {node['id']: idx for idx, node in enumerate(data['nodes'])}
    G = nx.DiGraph()
    G.add_nodes_from((node_id_mapping[node['id']], {
        'name': node['name'],
        'latency': float(node['runtime']) or 0.0,
        'energy': node['energy'] or 0.0,
        'size': 0.0 if 'cache' in node['name'].lower() else (node['size'] or 0.0),
        'kv_size': (node['size'] or 0.0) if 'cache' in node['name'].lower() else 0.0,
    }) for node in data['nodes'])
    G.add_edges_from((node_id_mapping[edge['source']], node_id_mapping[edge['destination']], 
    {'size': edge['size'], 
     'energy': edge.get('energy', 0.0) / (5.7 * (generation + 1)), 
     'latency': edge.get('latency', 0.0) / (13.3 * (generation + 1))}) 
    for edge in data['edges'] if not str(edge['source']).startswith("input"))
    
    total_model_size = calculate_total_model_size(G)
    graph, total_latency, total_energy, model_size, edge_size = normalize_graph(G)
    wakeup_time = wakeup_time / total_latency if powergating else 0
    wakeup_cost = wakeup_cost / total_energy if powergating else 0

    num_chips = chips if chips != 0 else int(math.ceil(total_model_size / single_chip_size))
    if chips != 0:
        needed_chips = math.ceil(total_model_size / single_chip_size)
        single_chip_size = single_chip_size / (num_chips / needed_chips) 

    # Define network topology
    print("\n################## -- Architecture Setup -- ##################")
    T = NetworkTopology(1, 1, type="2D_MESH")
    hop_matrix = T.get_shortest_hops_matrix()

    graph = split_large_nodes(graph, single_chip_size*1.05 / total_model_size, 1)
    # Set up chip sizes
    chip_sizes = [1.0]
    
    problem = Problem(m=model_name, c=1, p=num_subaccs*num_chips, e=estimates, l=leakage, id=idling_scale, pg=powergating,
                    com=communication_time_scale, coe=communication_energy_scale, wts=wakeup_time_scale, wcs=wakeup_cost_scale,
                    po=power_scale,  t=target, tp="2D_MESH", sc=0, ov=0, x='Dream')
    
    if lookup == 0: lookups = [problem]
    elif lookup == 1: 
        lookups = [
            problem,
            ]
    elif lookup == 2:
        lookups = [
            problem.update_values(id=idling_scale/2, wcs=wakeup_cost_scale/2),
            problem.update_values(id=idling_scale/2),
            problem.update_values(wcs=wakeup_cost_scale/2),
            ]
    elif lookup == 3:  
        lookups = [
            problem,
            problem.update_values(p=num_subaccs*2),
            ]
    
    dream_bound = None
     
    # Run partitioning optimization
    partitioning = optimize_parallel_graph_vector_bigM(
        problem=problem, 
        lookups=lookups,
        G=graph, 
        num_chips=1, 
        num_subaccs=num_subaccs*num_chips, 
        heuristic=heuristic, 
        target=target, 
        T=T, 
        rt=rt,
        gap=0.1, 
        chip_sizes=chip_sizes, 
        idle_leakage=leakage, 
        dynamic=dynamic, 
        wakeup_time=wakeup_time, 
        wakeup_cost=wakeup_cost, 
        dream_bound=dream_bound,
        dream=True,
        powergating=powergating,
        preloading=preloading,
        solution=solution,
        communication_time_scale=communication_time_scale,
        communication_energy_scale=communication_energy_scale,
        idling_scale=idling_scale,
        power_scale=power_scale,
        wakeup_cost_scale=wakeup_cost_scale,
        wakeup_time_scale=wakeup_time_scale,
        wb=wb
    )

    save_results_to_json(problem, partitioning[f'total{target}'], target)

    # Log results to WandB
    if wb:
        wandb.run.name = f"{model_name}_{1}c_{num_chips*num_subaccs}p"
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
            
            "EDP": partitioning['totalEDP']*total_latency*total_energy,
            "EPT": partitioning['totalEPT']*total_energy*total_latency,

            "Total Message Traffic": 0,
            "Total Message Time": 0,
            "Latency": partitioning['totalLatency']*total_latency,
            
            "Subaccelerators (SAs)": partitioning['num_subacc_used'] if partitioning['num_subacc_used'] else num_subaccs*num_chips,
            "Chip Count": 1,
            "Chip Usage": [{'chip': entry['chip'], 'Bytes used': entry['usage'] * model_size} for entry in partitioning['capacity_used_per_chip']],
            "Number of Clusters": partitioning['numClusters'],

            "Model Variables": partitioning['num_vars'],
            "Model Constraints": partitioning['num_constraints'],
            "MIP Gap": partitioning['mip_gap'],
            "TTS": partitioning['time_to_solution'],
        })

    # Visualize and save partitioning results
    partitioning_vis_path = f"visualizations/{target}/{problem.get_solution_key()}_partitioning.png"
    schedule_vis_path = f"visualizations/{target}/{problem.get_solution_key()}_schedule.png"
    assigment_path = f"visualizations/{target}/{problem.get_solution_key()}_assignment.json"
    
    try:
        with open(assigment_path, 'w') as outfile:
            json.dump(partitioning['assignments'], outfile, indent=4)
        print("\n💾 Assignment saved to:", assigment_path)
        if len(G.nodes) < 10000:
            visualize_partitioning(graph, partitioning, model_name, num_subaccs*num_chips, target, total_latency, problem.get_solution_key())
            visualize_schedule(partitioning, 1, num_subaccs*num_chips, graph, model_name, hop_matrix, target, total_latency, wakeup_time*wakeup_time_scale*total_latency, problem.get_solution_key())
            if wb: wandb.log({"Partitioning": wandb.Image(partitioning_vis_path), "Schedule": wandb.Image(schedule_vis_path)})
    except Exception as e:
        print(f"⚠️  Warning: Could not generate visualizations. Error: {e}")

    # Print partitioning results
    print(f"\nPartitioning results for {model_name}:")
    print(f"    - Computational Energy: {partitioning['computationalEnergy']*total_energy} ({partitioning['computationalEnergy']/partitioning['totalEnergy']*100:.2f}%)")
    print(f"    - Communication Energy: {partitioning['communicationEnergy']*total_energy} ({partitioning['communicationEnergy']/partitioning['totalEnergy']*100:.2f}%)")
    print(f"    - Wakeup Energy: {partitioning['wakeupEnergy']*total_energy} ({partitioning['wakeupEnergy']/partitioning['totalEnergy']*100:.2f}%)")

    print(f"\n    - Latency: {partitioning['totalLatency']*total_latency}")
    print(f"    - Energy: {partitioning['totalEnergy']*total_energy}")
    print(f"    - Idle Energy: {partitioning['totalIdleEnergy']*total_energy}")
    print(f"    - Load: {partitioning['totalMaxload']*total_latency}")
    print(f"    = EDP: {partitioning['totalEDP']*total_latency*total_energy}")
    
    print(f"\n    - Energy (Pipeline): {partitioning['totalEnergyPipeline']*total_energy}")
    print(f"    - Idle Energy (Pipeline): {partitioning['totalIdleEnergyPipeline']*total_energy}")
    print(f"    - Load (Pipeline): {partitioning['totalMaxloadPipeline']*total_latency}")
    print(f"    = EPT: {partitioning['totalEPT']*total_energy*total_latency}")
    
    print(f"\n    - Message Traffic: {0}")
    
    print(f"\n    - Chip Usage: {partitioning['capacity_used_per_chip']}")
    print(f"    - Number of Clusters: {partitioning['numClusters']}")

    if wb: wandb.finish()

    if os.path.exists("starts") and not ("70b" in model_name or "bert_hybrid_gen6" in model_name):
        print("🧹 Cleaning up start files...")
        for file in os.listdir("starts"): 
            if problem.get_solution_key() in file:  
                os.remove(os.path.join("starts", file))
                time.sleep(0.1)

    if target == "MaxloadPipeline" or target == "EPT": 
        print(f"{partitioning['totalEnergyPipeline']*total_energy}, {partitioning['totalLatency']*total_latency}, {partitioning['totalMaxloadPipeline']*total_latency}")
    else:
        print(f"{partitioning['totalEnergy']*total_energy}, {partitioning['totalLatency']*total_latency}, {partitioning['totalMaxload']*total_latency}")


# Parse command-line arguments for model, generation, and Subaccs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multi-chip partitioning with WandB logging.")
    parser.add_argument("-m", "--model", type=str, help="Base name of the model to partition.")
    parser.add_argument("-g", "--generation", type=int, default=0, help="Generation number to scale the model.")
    parser.add_argument("-p", "--subAccs", type=int, default=1, help="Number of Subaccelerators (SAs).")
    parser.add_argument("-l", "--leakage", type=int, default=0, help="Conditional to account for idle leakage power.")
    parser.add_argument("-t", "--target", type=str, default="EDP", help="Optimization target (EDP, Energy, Latency, Messages, MaxloadPipeline, EPT).")
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
    parser.add_argument("-wb", "--wandb", type=int, default=0, help="Enable WandB logging")
    parser.add_argument("-lo", "--lookup", type=int, default=0, help="Enable lookup")
    args = parser.parse_args()

    # Execute main partitioning function with specified parameters
    main(args.model, args.generation, args.subAccs, args.target, args.heuristic, args.chips, args.estimates, args.powergating, args.preloading, args.solution, args.communication_time_scale, args.communication_energy_scale, args.idling_scale, args.power_scale, args.runtime, args.dynamic, args.wakeup_time_scale, args.wakeup_cost_scale, args.wandb, args.leakage, args.lookup)
