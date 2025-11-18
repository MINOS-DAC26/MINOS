import os
import time
import numpy as np
import networkx as nx
import wandb
from gurobipy import GRB, GRB
from heuristics import *
from constraints import *
from targets import * 
from graph import *
from util import *
from var_rep import ModelVars
from model_adjustment import patch_model_mps

# Infeasibility analysis
def infeasibility_analysis(model):
    print("\n################## -- Infeasibility Analysis -- ##################")
    print('🚨 The model is infeasible; computing IIS')
    model.computeIIS()
    if model.IISMinimal: print('IIS is minimal\n')
    else: print('IIS is not minimal\n')
    print('\n🔍 The following constraint(s) cannot be satisfied:')
    for cns in model.getConstrs():
        if cns.IISConstr: print('%s' % cns.constrName)
    return None


# Main optimization function
def optimize_parallel_graph_vector_bigM(
            problem,                        # Problem name for model file lookup
            lookups,                        # Lookup for previous problems
            G,                              # Graph to partition
            num_chips,                      # Number of chips
            num_subaccs,                        # Number of Subaccs per chip (SubAccelerators)
            heuristic,                      # Heuristic for warm start
            target="EDP",                   # Target objective: EDP, LATENCY, ENERGY, MESSAGES, EPT
            T=None,                         # Topology
            rt=False,                       # Runtime analysis flag
            log_file=None,                  # Log file name
            gap=0.01,                       # MIP gap
            chip_sizes=None,                # Chip sizes
            idle_leakage=0,                 # Idle leakage flag
            dynamic=False,                  # Dynamic resource allocation 
            wakeup_time=0,                  # Wake-up time
            wakeup_cost=0,                  # Wake-up cost
            wakeup_time_scale=1.0,          # Wake-up time scale
            wakeup_cost_scale=1.0,          # Wake-up cost scale
            dream_bound=None,               # Dream bound for early termination
            dream=False,                    # Dream flag
            preloading=1,                   # Model loading flag
            solution=1,                     # Solution loading flag
            communication_time_scale=1,     # Communication time scale
            communication_energy_scale=1,   # Communication energy scale
            idling_scale=1,                 # Idling scale
            powergating=1,                  # Power gating flag   
            power_scale=1,                  # Power scale
            early_termination=True,         # Early termination flag
            wb=True,                        # Wandb logging flag
            matrix_load_vars=False,         # Matrix load vars flag
            inputs=32,                      # Number of inputs
            scale=0.0,                       # Scale factor for nodes
            overlap=0,                   # Overlap flag for maxload computation
            max_subaccs=0,
            bank_size=0,
            subaccs_per_chip=[],  # Number of Subaccs per chip for EDCP
    ):
    
    print(f'ℹ️  Partitioning graph ({G.number_of_nodes()}x{G.number_of_edges()}) with {num_chips} chips of size {[f"{size:.3}" for size in chip_sizes]}, each with {num_subaccs} SA(s).')
    # Initialize variables for logging / setup                 
    has_printed_gap, preloaded = False, False

    # Define architecture properties
    H = T.get_shortest_hops_matrix()
    max_number_of_slots = min(len(G.nodes)//(num_chips*num_subaccs), 64)
    power = sum(G.nodes[u]['energy'] for u in G.nodes) / sum(G.nodes[u]['latency'] for u in G.nodes)

    # Extract graph properties
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    latencies = np.array([G.nodes[n]['latency'] for n in G.nodes])
    energies = np.array([G.nodes[n]['energy'] for n in G.nodes])
    I, I_upper, num_incomparable = get_incomparables(G)
    dream_bound_value = dream_bound if dream_bound else 0.0
    latency_ub = sum(G.nodes[u]['latency'] for u in G.nodes) + sum(G.edges[u, v]['latency'] for u, v in G.edges)
    critical_path = sum(G.nodes[node]['latency'] for node in nx.dag_longest_path(nx.DiGraph((u, v, {'weight': G.nodes[u]['latency']}) for u, v in G.edges))) 
    latency_lb = max(critical_path, sum(G.nodes[node]['latency'] for node in G.nodes)/(num_subaccs*num_chips))
    energy_ub = sum(G.nodes[u]['energy'] for u in G.nodes) + sum(G.edges[u, v]['energy'] for u, v in G.edges) + sum(G.edges[u, v]['latency']*power*0.5*2 for u, v in G.edges) + (latency_ub*num_chips*num_subaccs*power*0.5)
    energy_lb = sum(G.nodes[node_id]['energy'] for node_id in G.nodes) + wakeup_cost #* (num_chips * num_subaccs - 1) * 12
    messages_ub = sum(G.edges[u, v]['size'] * np.max(H) for u, v in G.edges)
    maxload_lb = sum(G.nodes[u]['latency'] for u in G.nodes) / (num_subaccs * num_chips)
    # Adjust certain model parameters
    wakeup_time = wakeup_time * wakeup_time_scale
    wakeup_cost = wakeup_cost * wakeup_cost_scale
    power = power * power_scale
    idle_factor = 0.5 * idling_scale
    for u, v in G.edges:
        G.edges[u, v]['latency'] *= communication_time_scale
        G.edges[u, v]['energy'] *= communication_energy_scale
    
    # Read existing model file
    print("\n################## -- Model Setup -- ##################")
    if preloading: 
        print(f"📚 Trying to preload model from earlier setup... \n")

        model, preloaded = read_model(problem.get_model_key(), wb)

        if scale and not preloaded: 
            input_key = problem.get_model_key(sc=0)
            output_key = problem.get_model_key()
            
            if os.path.exists(f"models/{input_key}.mps"):
                print(f"🔄 Using base model for scaling from {input_key}")
                
                if rt: start_time = time.time()
                patch_model_mps(
                    input_key=input_key,
                    output_key=output_key,
                    latencies=latencies,
                    energies=energies,
                    bounds={"latency_lb": latency_lb, "latency_ub": latency_ub, "energy_lb": energy_lb, "energy_ub": energy_ub, "maxload_lb": maxload_lb},
                    num_chips=num_chips,
                    num_subaccs=num_subaccs,
                    max_slots=max_number_of_slots,
                    edges=list(G.edges),
                    power=power,
                    idle_factor=idle_factor,
                    wakeup_time=wakeup_time,
                    wakeup_cost=wakeup_cost,
                    incomparables=list(zip(I_upper.row, I_upper.col)),
                )

                if rt: print(f"⏱️  Finished model patching in {time.time() - start_time:.2f}s")
                if wb: wandb.log({"Model Patching Time": time.time() - start_time})
                start_time = time.time()
                model, preloaded = gp.read(f"models/{problem.get_model_key()}.mps"), True
                if wb: wandb.log({"Model Reading Time": time.time() - start_time})
    else: 
        print("🔨 Setting up model from scratch... \n")
        model = Model("minimize_edp")

    # Set gurobi model parameter
    os.makedirs('logs', exist_ok=True)
    if log_file: model.setParam("LogFile", "logs/" + log_file)  # Custom log file
    else: model.setParam("LogFile", "logs/gurobi.log")          # Default log file
    model.setParam("LogToConsole", 1)                           # Console logging
    model.setParam("MIPGap", gap)                               # MIP gap
    model.setParam("TimeLimit", 3600*10)                        # Time limit
    model.setParam("MIPFocus", 3)                               # Optimization strategy
    model.setParam("NonConvex", 2)                              # Non-convex optimization
    model.setParam("IntFeasTol", 1e-6)                          # Integer feasibility tolerance     
    model.setParam("Threads", 64)                               # Number of threads
    model.setParam("Presolve", 0)                               # Presolve
    model.setParam("StartNodeLimit", 50000)

    # Retrieve variables from preloaded model for later use
    if preloaded:
        x, t, c, y, yz, w, node_slot_assignment, slot_used, num_non_empty_slots, sa_idle_time = load_model_vars(
            model, 
            num_nodes=num_nodes, 
            num_chips=num_chips, 
            num_edges=num_edges, 
            num_subaccs=num_subaccs, 
            max_number_of_slots=max_number_of_slots, 
            I_upper=I_upper
        )
        model_vars = ModelVars()
        model_vars.from_json(f"models/vars_{problem.get_model_key()}.json")
        matrix_load_vars = model_vars.matrix_load_vars

    # Setup model from scratch
    else:
        setup_time = time.time()
        model_vars = ModelVars(matrix_load_vars=matrix_load_vars)
        # -------------------------------------------------------------------------
        #  1) Define x[node, chip]: node -> assigned chip
        # -------------------------------------------------------------------------
        if rt: start_time = time.time()
        x, chipUsed, MaxCapacityUsed, load = add_chip_assignment_constraints(
            G, model, 
            num_nodes=num_nodes, 
            num_chips=num_chips, 
            chip_sizes=[size * 1.1 for size in chip_sizes], 
            dynamic=dynamic,
            model_vars=model_vars,
        )
        if rt: print(f"\n⏱️  Finished x setup in {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------------------
        #  2) Define y[node, chip, sa]: node -> assigned to (chip, sa)
        # -------------------------------------------------------------------------
        if rt: start_time = time.time()
        y, saUsed = add_sa_assignment_constraints(
            model, x, 
            num_nodes=num_nodes, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            dynamic=dynamic,
            model_vars=model_vars
        )
        if rt: print(f"⏱️  Finished y setup in {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------------------
        #  2.5) Define cost per chip 
        # -------------------------------------------------------------------------
        if target == "EDCP":
            if rt: start_time = time.time()
            cost = add_chip_cost_constraints(model, y, load, 
                num_chips=num_chips, 
                model_vars=model_vars,
                max_subaccs=max_subaccs,
                num_subaccs=num_subaccs,
                num_nodes=num_nodes,
                bank_size=bank_size
            )
            if rt: print(f"⏱️  Finished chip cost setup in {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------------------
        #  3) Communication variables c[e] for each edge, for inter-chip hops
        # -------------------------------------------------------------------------
        if rt: start_time = time.time()
        c = add_communication_constraints(
            model, G, x, 
            num_edges=num_edges, 
            num_chips=num_chips, 
            H=H,
            model_vars=model_vars,
        )
        if rt: print(f"⏱️  Finished c setup in {time.time() - start_time:.2f}s")

        # -------------------------------------------------------------------------
        #  4) t[node]: finishing time of each node, t[v] >= t[u] + lat(v) + comm.
        # -------------------------------------------------------------------------
        if rt: start_time = time.time()
        t = add_latency_constraints(
            model, G, c, 
            num_chips=num_chips, 
            num_nodes=num_nodes, 
            latencies=latencies, 
            latency_ub=latency_ub,
            model_vars=model_vars,
        )
        if rt: print(f"⏱️  Finished t setup in {time.time() - start_time:.2f}s")
        
        # -------------------------------------------------------------------------
        #  5) Big-M disjunctive scheduling for incomparable pairs
        # -------------------------------------------------------------------------
        if rt: start_time = time.time()
        yz, w = add_disjunctive_constraints(
            G, model, y, t, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            latency_ub=latency_ub, 
            I_upper=I_upper,
            model_vars=model_vars,
            num_nodes=num_nodes
        )
        if rt: print(f"⏱️  Finished yz and w setup in {time.time() - start_time:.2f}s")
        # -------------------------------------------------------------------------
        #  6) SA scheduling clusters for wake-up and shut-down cost
        # -------------------------------------------------------------------------
        if rt: start_time = time.time()
        node_slot_assignment, slot_used, num_non_empty_slots, sa_idle_time = add_slot_assignment_constraints(
            model, y, t, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            max_number_of_slots=max_number_of_slots, 
            latency_ub=latency_ub, 
            wakeup_time=wakeup_time, 
            latencies=latencies, 
            num_nodes=num_nodes,
            model_vars=model_vars,
            rt=rt
        )
        if rt: print(f"⏱️  Finished slot assignment setup in {time.time() - start_time:.2f}s")
        
        # -------------------------------------------------------------------------
        #  7) Build objectives
        # -------------------------------------------------------------------------
        TotalLatency = define_latency_target(
            model, G, t, 
            latency_ub=latency_ub, 
            latency_lb=latency_lb,
            sense="EQ",
            model_vars=model_vars,
        )
        MaxLoad = define_maxload_target(
            model, G,
            latency_ub=latency_ub, 
            maxload_lb=maxload_lb, 
            latencies=latencies, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            y=y,
            c=c,
            model_vars=model_vars,
            sense="EQ",
            overlap=overlap
        )
        MaxLoadPipeline = define_maxload_pipeline_target(
            model,
            latency_ub=latency_ub,
            maxload_lb=maxload_lb,
            num_subaccs=num_subaccs,
            num_chips=num_chips,
            inputs=inputs,
            MaxLoad=MaxLoad,
            TotalLatency = TotalLatency,
            sense="EQ",
            model_vars=model_vars,
            overlap=overlap
        )
        IdleEnergy = define_idle_energy_target(
            model, y, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            TotalLatency=TotalLatency, 
            latencies=latencies, 
            latency_ub=latency_ub, 
            power=power, 
            wakeup_cost=wakeup_cost,
            sense="EQ",
            idle_factor=idle_factor
        )

        IdleEnergyPipeline = define_idle_energy_pipeline_target(
            model, y, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            MaxLoad=MaxLoad, 
            latencies=latencies, 
            latency_ub=latency_ub, 
            power=power, 
            wakeup_cost=wakeup_cost,
            sense="EQ",
            idle_factor=idle_factor
        )
        ComputationalEnergy= computational_energy(
            model, G, 
            energy_ub=energy_ub,
            sense="EQ",
        )
        CommunicationEnergy = communication_energy(
            model, G, c, 
            energy_ub=energy_ub, 
            power=power,
            idle_factor=idle_factor,
            sense="EQ"
        )
        WakeupEnergy = wakeup_energy(
            model, 
            num_non_empty_slots=num_non_empty_slots, 
            sa_idle_time=sa_idle_time, 
            energy_ub=energy_ub, 
            power=power, 
            wakeup_cost=wakeup_cost,
            idle_factor=idle_factor,
            sense="EQ"
        )
        TotalEnergy = define_energy_target(
            model, G,
            energy_ub=energy_ub,
            energy_lb=energy_lb,
            idle_leakage=idle_leakage,
            IdleEnergy=IdleEnergy,
            powergating=powergating,
            ComputationalEnergy=ComputationalEnergy,
            CommunicationEnergy=CommunicationEnergy,
            WakeupEnergy=WakeupEnergy,
            sense="EQ",
            num_chips=num_chips,
            TotalLatency=TotalLatency,
        )
        TotalEnergyPipeline = define_energy_pipeline_target(
            model,
            energy_ub=energy_ub,
            energy_lb=energy_lb,
            idle_leakage=idle_leakage,
            IdleEnergyPipeline=IdleEnergyPipeline,
            powergating=powergating,
            inputs=inputs,
            TotalEnergy=TotalEnergy,
            ComputationalEnergy=ComputationalEnergy,
            CommunicationEnergy=CommunicationEnergy,
            WakeupEnergy=WakeupEnergy,
            sense="EQ",
        )
        TotalEDP = define_EDP_target(
            model, 
            TotalEnergy=TotalEnergy, 
            TotalLatency=TotalLatency, 
            energy_ub=energy_ub, 
            latency_ub=latency_ub, 
            energy_lb=energy_lb, 
            latency_lb=latency_lb,
            sense="EQ",
        )
        if target == "EDCP":
            TotalEDCP = define_EDCP_target(
                model, 
                TotalEnergy=TotalEnergy, 
                TotalLatency=TotalLatency, 
                energy_ub=energy_ub, 
                latency_ub=latency_ub, 
                energy_lb=energy_lb, 
                latency_lb=latency_lb,
                sense="EQ",
                cost=cost,
            )
        TotalMessages = define_messages_target(
            model, G, c, 
            messages_ub=messages_ub,
            sense="EQ",
        )
        TotalEPT = define_EPT_target(
            model, 
            energy_ub=energy_ub, 
            latency_ub=latency_ub, 
            energy_lb=energy_lb, 
            maxload_lb=maxload_lb, 
            TotalEnergyPipeline=TotalEnergyPipeline, 
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            MaxLoadPipeline=MaxLoadPipeline,
            sense="EQ",
        )

        print(f"⏱️  Model Setup Time: {time.time() - setup_time:.2f} seconds.")
        if wb: wandb.log({"Model Setup Time": time.time() - setup_time})

        write_time = time.time()
        model.write(f"models/{problem.get_model_key()}.mps")
        model_vars.to_json(f"models/vars_{problem.get_model_key()}.json")
        print(f"⏱️  Model written in {time.time() - write_time:.2f} seconds.")

    if target == "EDP": model.setObjective(model.getVarByName('Total_EDP'), GRB.MINIMIZE)
    elif target == "Latency": model.setObjective(model.getVarByName('Total_Latency'), GRB.MINIMIZE)
    elif target == "Energy": model.setObjective(model.getVarByName('Total_Energy'), GRB.MINIMIZE)
    elif target == "Messages": model.setObjective(model.getVarByName('Total_Messages'), GRB.MINIMIZE)
    elif target == "MaxloadPipeline": model.setObjective(model.getVarByName('Total_MaxloadPipeline'), GRB.MINIMIZE)
    elif target == "EPT": model.setObjective(model.getVarByName('Total_EPT'), GRB.MINIMIZE)
    elif target == "EDCP": model.setObjective(model.getVarByName('Total_EDCP'), GRB.MINIMIZE)
    else: raise ValueError("Invalid target objective: valid options are EDP, Latency, Energy, Messages, MaxloadPipeline, EPT")

    if dream_bound_value:
        target_var = model.getVarByName(f'Total_{target}')
        if target_var and target_var.LB < dream_bound_value: 
            target_var.LB = dream_bound_value
            print(f"\n✅ Applied dream bound value {dream_bound_value:.3}!")
        else: print(f"\n📉 Dream bound value {dream_bound_value:.3} is not better than existing lower bound {target_var.LB}.")
    else: print("\n❓ No dream bound value provided.")

    # -------------------------------------------------------------------------
    #  8) Warm Starting
    # -------------------------------------------------------------------------
    print("\n################## -- Starting Solve -- ##################")
    optim_time = time.time()
    if rt: start_time = time.time()

    if not heuristic == "none":
        warm_start(model, G, 
                problem=problem, 
                lookups=lookups,
                heuristic=heuristic, 
                dream=dream, 
                num_chips=num_chips, 
                chip_sizes=[size * 1.1 for size in chip_sizes], 
                num_subaccs=num_subaccs, 
                max_number_of_slots=max_number_of_slots, 
                I_upper=I_upper, 
                powergating=powergating, 
                model_vars=model_vars, 
                H=H, target=target, 
                wakeup_time=wakeup_time, 
                solution=solution,
                dream_bound_value=dream_bound_value,
                idle_leakage=idle_leakage,
                w2s_etime=(wakeup_cost)/(power*idle_factor),
                latency_ub=latency_ub,
                overlap=overlap, 
                bank_size=bank_size,
                subaccs_per_chip=subaccs_per_chip,
                )

    # write_hints_incomparables(G, I_upper, f"starts/{problem}_incomp.hnt")
    # model.read(f"starts/{problem}_incomp.hnt")
    if rt: print(f"⏱️  Finished heuristics in {time.time() - start_time:.2f} seconds \n")
        
    # -------------------------------------------------------------------------
    #  9) Solve
    # -------------------------------------------------------------------------
    # Callback for MIP logging & early termination
    def callback(model, where):
        nonlocal has_printed_gap
        # Catch MIP gap updates
        if where == GRB.Callback.MIP:
            obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)  
            obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND) 
            if obj_best < GRB.INFINITY:
                mip_gap = abs(obj_best - obj_bound) / abs(obj_best)
                if wb: wandb.log({"MIP Gap Over Time": mip_gap, "Incumbent Over Time": obj_best, "Objective Bound Over Time": obj_bound, "Dream Bound": dream_bound, "Time": time.time() - optim_time})
                # Catch initial MIP gap
                if not has_printed_gap:
                    print(f"Initial MIP Gap: {mip_gap:.4%} with objective: {obj_best:.4} and bound: {obj_bound:.4} (Time: {time.time() - optim_time:.2f}s)")
                    if wb: wandb.log({"IFS Gap": mip_gap, "IFS Time": time.time() - optim_time})
                    has_printed_gap = True
                    if early_termination: model.terminate()
                # Terminate early 
                if dream_bound_value and early_termination:
                    lower_bound_off = (obj_bound - dream_bound_value) / dream_bound_value > 0.10
                    upper_bound_close = (obj_best - dream_bound_value) / dream_bound_value <= 0.10
                    if lower_bound_off:
                        print(f"Early Termination: Infeasible (Lower bound {obj_bound:.4} is more than 10% off from dream bound {dream_bound_value:.4})")
                        if wb: wandb.log({"Early Termination": "Infeasible"})
                        model.terminate()
                    elif upper_bound_close:
                        print(f"Early Termination: Satisfied (Upper bound {obj_best:.4} is within 10% of dream bound {dream_bound_value:.4})")
                        if wb: wandb.log({"Early Termination": "Satisfied"})
                        model.terminate()

    model.optimize(callback)

    # -------------------------------------------------------------------------
    # 10) Collect results
    # -------------------------------------------------------------------------
    # Infeasibility analysis
    if model.Status == GRB.Status.INFEASIBLE: return infeasibility_analysis(model)

    # Collect results for logging
    result = {
        'status': ('optimal' if model.Status == GRB.Status.OPTIMAL else 'interrupted' if model.SolCount > 0 else 'unknown'),
        'num_vars': model.NumVars, 'num_constraints': model.NumConstrs, 'mip_gap': model.MIPGap, 'runtime': model.Runtime,
        'time_to_solution': time.time() - optim_time,
    }

    if model.SolCount > 0:
        print("\n################## -- Results -- ##################")
        # Save solution file to reload later
        save_solution(model, problem.get_solution_key())
        
        # Result check
        detect_conflicts(model, G, y, num_chips=num_chips, num_subaccs=num_subaccs, latencies=latencies)

        chips_used = np.array([model.getVarByName(f"Assignment[{i},{j}]").X for i in range(num_nodes) for j in range(num_chips)]).reshape(num_nodes, num_chips)
        num_chips_used = np.sum(np.any(chips_used > 0, axis=0))

        # Extract target values
        result['totalLatency'] = model.getVarByName("Total_Latency").X
        result['communicationEnergy'] = model.getVarByName("Communication_Energy").X
        result['computationalEnergy'] = model.getVarByName("Computational_Energy").X
        result['wakeupEnergy'] = model.getVarByName("Wakeup_Energy").X
        result['totalEnergy'] = model.getVarByName("Total_Energy").X
        result['totalEDP'] = model.getVarByName("Total_EDP").X
        result['message_traffic'] = message_traffic(G, c, x, num_chips)
        result['totalMaxload'] = model.getVarByName("Total_Maxload").X
        result['totalEPT'] = model.getVarByName("Total_EPT").X
        result['totalIdleEnergy'] = model.getVarByName("IdleEnergy").X
        result['totalIdleEnergyPipeline'] = model.getVarByName("IdleEnergyPipeline").X
        result['totalEnergyPipeline'] = model.getVarByName("Total_EnergyPipeline").X
        result['totalMaxloadPipeline'] = model.getVarByName("Total_MaxloadPipeline").X
        if target == "EDCP": 
            result['totalEDCP'] = model.getVarByName("Total_EDCP").X
            result['totalCost'] = sum(model.getVarByName(f"Cost[{chip_idx}]").X for chip_idx in range(num_chips))

        # Extract resource allocation values
        result['num_chips_used'] = num_chips_used
        if matrix_load_vars: result['num_subacc_used'] = num_subaccs if not dynamic else np.array([model.getVarByName(f"SAUsed[{i*num_subaccs+j}]").X for i in range(num_chips) for j in range(num_subaccs)]).reshape(num_chips, num_subaccs).sum()
        else: result['num_subacc_used'] = num_subaccs if not dynamic else np.array([model.getVarByName(f"SAUsed[{i},{j}]").X for i in range(num_chips) for j in range(num_subaccs)]).reshape(num_chips, num_subaccs).sum()
        if matrix_load_vars: result['numClusters'] = np.array([model.getVarByName(f"NumNonEmptySlots[{i*num_subaccs+j}]").X for i in range(num_chips) for j in range(num_subaccs)]).sum() 
        else: result['numClusters'] = np.array([model.getVarByName(f"NumNonEmptySlots[{i},{j}]").X for i in range(num_chips) for j in range(num_subaccs)]).sum() 
        result['capacity_used_per_chip'] = capacity_used_per_chip(G, x, num_chips, chip_sizes)

        if target == "EDCP":
            for chip_idx in range(num_chips):
                chip_load = model.getVarByName(f"Load[{chip_idx}]").X
                chip_mem = model.getVarByName(f"MemBanks[{chip_idx}]").X
                chip_cost = model.getVarByName(f"Cost[{chip_idx}]").X
                chip_num_subaccs = model.getVarByName(f"subAccsUsedPerChip[{chip_idx}]").X
                print(f"Chip {chip_idx}: Load={chip_load:.3f}, MemoryBlocks={int(chip_mem)}, Cost={chip_cost:.5f}, subAccsUsed={int(chip_num_subaccs)}")
                
        # Extract visualization metrics
        sorted_nodes = sorted(G.nodes, key=lambda x: (isinstance(x, str), x))
        result['node_finish_times'] = [model.getVarByName(f"Latency[{n}]").X for n in sorted_nodes]
        result["sa_slot_times"] = extract_slot_times(model,
            num_chips=num_chips, 
            num_subaccs=num_subaccs, 
            max_number_of_slots=max_number_of_slots,
            matrix_load_vars=matrix_load_vars
            )
        result['assignments'] = get_assignments(G, x, y, num_chips, num_subaccs)

    return result