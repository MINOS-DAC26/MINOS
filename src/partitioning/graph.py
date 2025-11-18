# graph.py

import numpy as np
import scipy.sparse as sp
import networkx as nx

# Normalize graph for numerical stability
def normalize_graph(G, scale=1.0):
    # Print range before normalization
    print("\n################## -- Normalizing Graph -- ##################")
    print("Original Graph Specification:")
    print("  - Node Attributes:")
    print(f"    • Latency Range    : min = {min(G.nodes[u]['latency'] for u in G.nodes):.3e}, max = {max(G.nodes[u]['latency'] for u in G.nodes):.3e}")
    print(f"    • Energy Range     : min = {min(G.nodes[u]['energy'] for u in G.nodes):.3e}, max = {max(G.nodes[u]['energy'] for u in G.nodes):.3e}")
    print(f"    • Size Range       : min = {min(G.nodes[u]['size'] for u in G.nodes):.3e}, max = {max(G.nodes[u]['size'] for u in G.nodes):.3e}")
    print(f"    • Latency Sum      : {sum(G.nodes[u]['latency'] for u in G.nodes):.3e}")
    print(f"    • Energy Sum       : {sum(G.nodes[u]['energy'] for u in G.nodes):.3e}")
    print("  - Edge Attributes:")
    print(f"    • Latency Range    : min = {min(G.edges[u,v]['latency'] for u,v in G.edges):.3e}, max = {max(G.edges[u,v]['latency'] for u,v in G.edges):.3e}")
    print(f"    • Energy Range     : min = {min(G.edges[u,v]['energy'] for u,v in G.edges):.3e}, max = {max(G.edges[u,v]['energy'] for u,v in G.edges):.3e}\n")

    # Create copy to avoid modifying the original
    NG = G.copy()  

    # Upper bounds for normalization
    model_size = sum(NG.nodes[u]['size'] for u in NG.nodes)
    kv_size = sum(NG.nodes[u]['kv_size'] for u in NG.nodes)
    print(f"Model and KV Size: {model_size:.3e}, {kv_size:.3e}")
    total_latency = sum(NG.nodes[u]['latency'] for u in NG.nodes)
    if scale: total_latency /= scale
    total_energy = sum(NG.nodes[u]['energy'] for u in NG.nodes)
    if scale: total_energy /= scale
    edge_size = sum(NG.edges[u,v]['size'] for u,v in NG.edges)
    
    # Normalization
    for node_id in NG.nodes:
        NG.nodes[node_id]['size'] /= model_size
        NG.nodes[node_id]['kv_size'] /= kv_size if kv_size else 1
        NG.nodes[node_id]['latency'] /= total_latency
        NG.nodes[node_id]['energy'] /= total_energy
    for u, v in NG.edges:
        NG.edges[u, v]['latency'] /= total_latency
        NG.edges[u, v]['energy'] /= total_energy
        NG.edges[u, v]['size'] /= edge_size
    
    # Print range after normalization
    print("Normalized Graph Specification:")
    print(f"    • Latency Range    : min = {min(NG.nodes[u]['latency'] for u in NG.nodes):.3e}, max = {max(NG.nodes[u]['latency'] for u in NG.nodes):.3e}")
    print(f"    • Energy Range     : min = {min(NG.nodes[u]['energy'] for u in NG.nodes):.3e}, max = {max(NG.nodes[u]['energy'] for u in NG.nodes):.3e}")
    print(f"    • Size Range       : min = {min(NG.nodes[u]['size'] for u in NG.nodes):.3e}, max = {max(NG.nodes[u]['size'] for u in NG.nodes):.3e}")
    print(f"    • Latency Sum      : {sum(NG.nodes[u]['latency'] for u in NG.nodes):.3e}")
    print(f"    • Energy Sum       : {sum(NG.nodes[u]['energy'] for u in NG.nodes):.3e}")

    print("  - Normalized Edge Attributes:")
    print(f"    • Latency Range    : min = {min(NG.edges[u,v]['latency'] for u,v in NG.edges):.3e}, max = {max(NG.edges[u,v]['latency'] for u,v in NG.edges):.3e}")
    print(f"    • Energy Range     : min = {min(NG.edges[u,v]['energy'] for u,v in NG.edges):.3e}, max = {max(NG.edges[u,v]['energy'] for u,v in NG.edges):.3e}")

    return NG, total_latency, total_energy, model_size, edge_size


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
