# targets.py

from gurobipy import GRB, GRB, LinExpr, max_
from var_rep import Var


# Define idle energy target
def define_idle_energy_target(model, y, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    TotalLatency = kwargs.get("TotalLatency")
    latencies = kwargs.get("latencies")
    latency_ub = kwargs.get("latency_ub")
    power = kwargs.get("power")
    sense = kwargs.get("sense")
    wakeup_cost = kwargs.get("wakeup_cost")
    idle_factor = kwargs.get("idle_factor")

    IdleEnergy = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=((num_chips * num_subaccs - 1)*latency_ub*power*idle_factor + num_subaccs*num_chips*wakeup_cost)*8, name="IdleEnergy")
    sa_exec_time = (y * latencies[:, None, None]).sum(axis=0)
    energy_sum_idle = (TotalLatency - sa_exec_time) * power * idle_factor + num_chips * num_subaccs * wakeup_cost

    if sense == "EQ": model.addConstr(IdleEnergy == energy_sum_idle.sum(), name="IdleEnergy_Constraint")
    else: model.addConstr(IdleEnergy >= energy_sum_idle.sum(), name="IdleEnergy_Constraint")

    return IdleEnergy


def define_idle_energy_pipeline_target(model, y, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    MaxLoad = kwargs.get("MaxLoad")
    latencies = kwargs.get("latencies")
    latency_ub = kwargs.get("latency_ub")
    power = kwargs.get("power")
    sense = kwargs.get("sense")
    wakeup_cost = kwargs.get("wakeup_cost")
    idle_factor = kwargs.get("idle_factor")

    IdleEnergyPipeline = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=((num_chips * num_subaccs - 1)*latency_ub*power*idle_factor + num_subaccs*num_chips*wakeup_cost)*8, name="IdleEnergyPipeline")    
    sa_exec_time = (y * latencies[:, None, None]).sum(axis=0)
    energy_sum_idle_pipeline = (MaxLoad - sa_exec_time) * power * idle_factor + num_chips * num_subaccs * wakeup_cost

    if sense == "EQ": model.addConstr(IdleEnergyPipeline == energy_sum_idle_pipeline.sum(), name="IdleEnergyPipeline_Constraint")
    else: model.addConstr(IdleEnergyPipeline >= energy_sum_idle_pipeline.sum(), name="IdleEnergyPipeline_Constraint")
    
    return IdleEnergyPipeline


# Define latency target
def define_latency_target(model, G, t, **kwargs):
    latency_ub = kwargs.get("latency_ub")
    latency_lb = kwargs.get("latency_lb")
    sense = kwargs.get("sense")
    model_vars = kwargs.get("model_vars")

    TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=latency_lb, ub=latency_ub, name="Total_Latency")
    model_vars.add_var(Var("Total_Latency", (), 0, latency_ub, float))

    if sense == "EQ": model.addConstr(TotalLatency == max_([t[node_id] for node_id in G.nodes]), name=f"Makespan_Latency_Constraint")
    else: model.addConstrs((TotalLatency >= t[node_id] for node_id in G.nodes), name=f"Makespan_Latency_Constraint")

    return TotalLatency


def computational_energy(model, G, **kwargs):
    energy_ub = kwargs.get("energy_ub")
    sense = kwargs.get("sense")

    ComputationalEnergy = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=energy_ub, name="Computational_Energy")
    
    computational_energy = LinExpr()
    for node_id in G.nodes: computational_energy += G.nodes[node_id]['energy']
    if sense == "EQ": model.addConstr(ComputationalEnergy == computational_energy, name="Computational_Energy_Constraint")
    else: model.addConstr(ComputationalEnergy >= computational_energy, name="Computational_Energy_Constraint")

    return ComputationalEnergy


def communication_energy(model, G, c, **kwargs):
    energy_ub = kwargs.get("energy_ub")
    power = kwargs.get("power")
    idle_factor = kwargs.get("idle_factor")
    sense = kwargs.get("sense")

    CommunicationEnergy = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=energy_ub, name="Communication_Energy")
    
    communication_energy = LinExpr()
    for e, (u, v) in enumerate(G.edges): communication_energy += c[e] * (G.edges[u,v]['energy'] + 2 * idle_factor * G.edges[u,v]['latency'] * power)
    if sense == "EQ": model.addConstr(CommunicationEnergy == communication_energy, name="Communication_Energy_Constraint")
    else: model.addConstr(CommunicationEnergy >= communication_energy, name="Communication_Energy_Constraint")

    return CommunicationEnergy


def wakeup_energy(model, **kwargs):
    energy_ub = kwargs.get("energy_ub")
    power = kwargs.get("power")
    wakeup_cost = kwargs.get("wakeup_cost")
    sa_idle_time = kwargs.get("sa_idle_time")
    num_non_empty_slots = kwargs.get("num_non_empty_slots")
    idle_factor = kwargs.get("idle_factor")
    sense = kwargs.get("sense")

    WakeupEnergy = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=energy_ub, name="Wakeup_Energy")
    wakeup_energy = LinExpr()
    wakeup_energy += (num_non_empty_slots.sum() * wakeup_cost + sa_idle_time.sum() * power * idle_factor)
    if sense == "EQ": model.addConstr(WakeupEnergy == wakeup_energy, name="Wakeup_Energy_Constraint")
    else: model.addConstr(WakeupEnergy >= wakeup_energy, name="Wakeup_Energy_Constraint")

    return WakeupEnergy


# Define energy target
def define_energy_target(model, G, **kwargs):
    energy_ub = kwargs.get("energy_ub")
    energy_lb = kwargs.get("energy_lb")
    idle_leakage = kwargs.get("idle_leakage")
    IdleEnergy = kwargs.get("IdleEnergy")
    powergating = kwargs.get("powergating")
    ComputationalEnergy = kwargs.get("ComputationalEnergy")
    CommunicationEnergy = kwargs.get("CommunicationEnergy")
    WakeupEnergy = kwargs.get("WakeupEnergy")
    sense = kwargs.get("sense")

    TotalEnergy = model.addVar(vtype=GRB.CONTINUOUS, lb=energy_lb, ub=energy_ub, name="Total_Energy")
    
    energy_sum = LinExpr()
    energy_sum += ComputationalEnergy + CommunicationEnergy

    if idle_leakage: energy_sum += IdleEnergy
    elif powergating: energy_sum += WakeupEnergy

    # always_on = 560e-6 / sum(G.nodes[node_id]['energy'] for node_id in G.nodes)
    # energy_sum += always_on * TotalLatency * num_chips

    if sense == "EQ": model.addConstr(TotalEnergy == energy_sum, name="TotalEnergyConstraint")
    else: model.addConstr(TotalEnergy >= energy_sum, name="TotalEnergyConstraint")

    return TotalEnergy


def define_energy_pipeline_target(model, **kwargs):
    energy_ub = kwargs.get("energy_ub")
    energy_lb = kwargs.get("energy_lb")
    idle_leakage = kwargs.get("idle_leakage")
    IdleEnergyPipeline = kwargs.get("IdleEnergyPipeline")
    powergating = kwargs.get("powergating")
    inputs = kwargs.get("inputs")
    TotalEnergy = kwargs.get("TotalEnergy")
    sense = kwargs.get("sense")
    ComputationalEnergy = kwargs.get("ComputationalEnergy")
    CommunicationEnergy = kwargs.get("CommunicationEnergy")
    WakeupEnergy = kwargs.get("WakeupEnergy")

    TotalEnergyPipeline = model.addVar(vtype=GRB.CONTINUOUS, lb=energy_lb, ub=energy_ub, name="Total_EnergyPipeline")
    
    energy_sum_pipeline = LinExpr()
    energy_sum_pipeline += ComputationalEnergy + CommunicationEnergy
    
    if idle_leakage: energy_sum_pipeline += IdleEnergyPipeline
    elif powergating: energy_sum_pipeline += WakeupEnergy
    
    if sense == "EQ": model.addConstr(TotalEnergyPipeline == (energy_sum_pipeline*(inputs-1) + TotalEnergy)/(inputs), name="TotalEnergyPipelineConstraint")
    else: model.addConstr(TotalEnergyPipeline >= (energy_sum_pipeline*(inputs-1) + TotalEnergy)/(inputs), name="TotalEnergyPipelineConstraint")

    return TotalEnergyPipeline


# Define EDP target
def define_EDP_target(model, **kwargs):
    TotalEnergy = kwargs.get("TotalEnergy")
    TotalLatency = kwargs.get("TotalLatency")
    energy_ub = kwargs.get("energy_ub")
    latency_ub = kwargs.get("latency_ub")
    energy_lb = kwargs.get("energy_lb")
    latency_lb = kwargs.get("latency_lb")
    sense = kwargs.get("sense")

    TotalEDP = model.addVar(vtype=GRB.CONTINUOUS, lb=energy_lb*latency_lb, ub=latency_ub*energy_ub, name="Total_EDP")

    if sense == "EQ": model.addConstr(TotalEDP == TotalEnergy * TotalLatency, name="EDP_Constraint")
    else: model.addConstr(TotalEDP >= TotalEnergy * TotalLatency, name="EDP_Constraint")

    return TotalEDP


def define_EDCP_target(model, **kwargs):
    TotalEnergy = kwargs.get("TotalEnergy")
    TotalLatency = kwargs.get("TotalLatency")
    energy_ub = kwargs.get("energy_ub")
    latency_ub = kwargs.get("latency_ub")
    energy_lb = kwargs.get("energy_lb")
    latency_lb = kwargs.get("latency_lb")
    sense = kwargs.get("sense")
    cost = kwargs.get("cost")

    TotalEDCP = model.addVar(vtype=GRB.CONTINUOUS, lb=energy_lb*latency_lb*0.014451238, ub=latency_ub*energy_ub, name="Total_EDCP")

    if sense == "EQ": model.addConstr(TotalEDCP == TotalEnergy * TotalLatency * sum(cost), name="EDCP_Constraint")
    else: model.addConstr(TotalEDCP >= TotalEnergy * TotalLatency * sum(cost), name="EDCP_Constraint")

    return TotalEDCP


# Define messages target
def define_messages_target(model, G, c, **kwargs):
    messages_ub = kwargs.get("messages_ub")
    sense = kwargs.get("sense")

    TotalMessages = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=messages_ub, name="Total_Messages")
    msg_expr = LinExpr()
    for e, (u, v) in enumerate(G.edges): msg_expr += c[e] * G.edges[u,v]['size']

    if sense == "EQ": model.addConstr(TotalMessages == msg_expr, name="TotalMessages_Constraint")
    else: model.addConstr(TotalMessages >= msg_expr, name="TotalMessages_Constraint")
    
    return TotalMessages


# Define MaxLoad target
def define_maxload_target(model, G, y, c, **kwargs):
    num_chips = kwargs.get("num_chips")
    num_subaccs = kwargs.get("num_subaccs")
    latencies = kwargs.get("latencies")
    latency_ub = kwargs.get("latency_ub")
    maxload_lb = kwargs.get("maxload_lb")
    model_vars = kwargs.get("model_vars")
    sense = kwargs.get("sense")
    overlap = kwargs.get("overlap")

    MaxLoad = model.addVar(vtype=GRB.CONTINUOUS, lb=maxload_lb, ub=latency_ub, name="Total_Maxload")
    model_vars.add_var(Var("Total_Maxload", (), maxload_lb, latency_ub, float))
    
    sa_load = model.addMVar(shape=(num_chips, num_subaccs), vtype=GRB.CONTINUOUS, lb=0, ub=latency_ub, name="sa_load")
    model_vars.add_var(Var("sa_load", (num_chips, num_subaccs), 0.0, latency_ub, float))

    sa_load_expr = (y * latencies[:, None, None]).sum(axis=0)

    if not overlap:
        for e, (u, v) in enumerate(G.edges):
            latency = G.edges[u, v]['latency']
            sa_load_expr += c[e] * latency * (y[u, :, :] + y[v, :, :]) 
            
    model.addConstr(sa_load == sa_load_expr, name="max_subaccLoad_Constraint")

    if sense == "EQ": model.addConstr(MaxLoad == max_([sa_load[chip_id, sa_id] for chip_id in range(num_chips) for sa_id in range(num_subaccs)]), name="MaxLoad_Constraint")
    else: model.addConstrs((MaxLoad >= sa_load[chip_id, sa_id] for chip_id in range(num_chips) for sa_id in range(num_subaccs)), name="MaxLoad_Constraint")

    return MaxLoad


def define_maxload_pipeline_target(model, **kwargs):
    latency_ub = kwargs.get("latency_ub")
    maxload_lb = kwargs.get("maxload_lb")
    inputs = kwargs.get("inputs")
    MaxLoad = kwargs.get("MaxLoad")
    TotalLatency = kwargs.get("TotalLatency")
    sense = kwargs.get("sense")
    model_vars = kwargs.get("model_vars")

    MaxLoadPipeline = model.addVar(vtype=GRB.CONTINUOUS, lb=maxload_lb, ub=latency_ub, name="Total_MaxloadPipeline")
    model_vars.add_var(Var("Total_MaxloadPipeline", (), maxload_lb, latency_ub, float))    

    if sense == "EQ": model.addConstr(MaxLoadPipeline == (TotalLatency + (inputs-1) * MaxLoad)/(inputs), name="MaxLoadPipeline_Constraint")
    else: model.addConstr(MaxLoadPipeline >= (TotalLatency + (inputs-1) * MaxLoad)/(inputs), name="MaxLoadPipeline_Constraint")

    return MaxLoadPipeline


# Define EPT target
def define_EPT_target(model, **kwargs):
    energy_ub = kwargs.get("energy_ub")
    energy_lb = kwargs.get("energy_lb")
    latency_ub = kwargs.get("latency_ub")
    maxload_lb = kwargs.get("maxload_lb")
    TotalEnergyPipeline = kwargs.get("TotalEnergyPipeline")
    MaxLoadPipeline = kwargs.get("MaxLoadPipeline")
    sense = kwargs.get("sense")

    EPT = model.addVar(vtype=GRB.CONTINUOUS, lb=energy_lb*maxload_lb, ub=energy_ub*latency_ub, name="Total_EPT")

    if sense == "EQ": model.addConstr(EPT == TotalEnergyPipeline * MaxLoadPipeline, name="EPT_Constraint")
    else: model.addConstr(EPT >= TotalEnergyPipeline * MaxLoadPipeline, name="EPT_Constraint")
    
    return EPT