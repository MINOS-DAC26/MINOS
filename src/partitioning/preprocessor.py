# preprocessor.py

def split_node(G, node, chip_size, margin, id_counter):
    """
    Splits a node in the graph `G` if its size exceeds a specified margin of the chip size.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the nodes to be split.
    - node (int): The ID of the node to potentially split.
    - chip_size (float): The target size of each chip. Nodes larger than `margin * chip_size` are split.
    - margin (float): A multiplier for determining if a node should be split.
    - id_counter (iterator): An iterator providing unique IDs for new nodes created during splitting.

    Returns:
    - None: The graph `G` is modified in place.
    
    This function splits the node either horizontally or vertically based on the comparison of input and output sizes.
    Each split creates two nodes with updated properties and reassigns edges based on the split type. Recursively calls 
    itself to further split large nodes if necessary.
    """
    node_data = G.nodes[node]
    node_size = node_data['size']

    # Check if the node size requires splitting
    if node_size <= margin * chip_size:
        return

    # Calculate total sizes of input and output edges
    input_size = sum(G[u][node]['size'] for u in G.predecessors(node))
    output_size = sum(G[node][v]['size'] for v in G.successors(node))

    # Determine split type based on output and input size
    split_type = 'horizontal' if output_size >= input_size else 'vertical'
    print(f"Splitting node {node} {split_type}ly into 2 parts")

    # Create identifiers for the split nodes
    new_node_1 = node  # Original ID for one of the split nodes
    new_node_2 = next(id_counter)  # New ID for the other split node

    # Define attributes for each split node
    new_node_data_1 = {
        'name': f"{node_data['name']}_1",  
        'size': node_size / 2,  
        'energy': node_data['energy'] / 2,  
        'latency': node_data['latency'] / 2  
    }
    new_node_data_2 = {
        'name': f"{node_data['name']}_2",  
        'size': node_size / 2,  
        'energy': node_data['energy'] / 2,  
        'latency': node_data['latency'] / 2 
    }

    # Update the existing node and add the new split node to the graph
    G.nodes[new_node_1].update(new_node_data_1)
    G.add_node(new_node_2, **new_node_data_2)

    # Reassign edges to the new nodes based on split type
    for u in G.predecessors(node):
        edge_data = G[u][node]
        if split_type == 'horizontal':
            G.add_edge(u, new_node_1, **edge_data)
            G.add_edge(u, new_node_2, **edge_data)
        else:
            # Divide edge properties for vertical splits
            split_edge_size = edge_data['size'] / 2
            new_edge_data_1 = {
                'size': split_edge_size,
                'energy': edge_data['energy'] / 2,  
                'latency': edge_data['latency'] / 2
            }
            new_edge_data_2 = new_edge_data_1.copy()
            G.add_edge(u, new_node_1, **new_edge_data_1)
            G.add_edge(u, new_node_2, **new_edge_data_2)

    for v in G.successors(node):
        edge_data = G[node][v]
        if split_type == 'horizontal':
            # Divide edge properties for horizontal splits
            split_edge_size = edge_data['size'] / 2
            new_edge_data_1 = {
                'size': split_edge_size,
                'energy': edge_data['energy'] / 2,  
                'latency': edge_data['latency'] / 2
            }
            new_edge_data_2 = new_edge_data_1.copy()
            G.add_edge(new_node_1, v, **new_edge_data_1)
            G.add_edge(new_node_2, v, **new_edge_data_2)
        else:
            G.add_edge(new_node_1, v, **edge_data)
            G.add_edge(new_node_2, v, **edge_data)

    # Recursively split each new node if it is still oversized
    split_node(G, new_node_1, chip_size, margin, id_counter)  
    split_node(G, new_node_2, chip_size, margin, id_counter)  


def split_large_nodes(G, chip_size, margin):
    """
    Splits all nodes in the graph `G` that exceed a given chip size threshold.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing nodes to be evaluated for splitting.
    - chip_size (float): The size threshold for each chip. Nodes larger than this size are candidates for splitting.
    - margin (float): A multiplier to adjust the threshold size for splitting nodes.

    Returns:
    - nx.DiGraph: A modified copy of the input graph with nodes split to meet the chip size constraints.
    
    This function identifies nodes exceeding the chip size, then calls `split_node` to split each of them 
    accordingly, creating a modified version of the graph where all nodes conform to the size constraint.
    """
    # Create a copy of the graph for splitting
    SG = G.copy()
    id_counter = iter(range(max(G.nodes) + 1, 10**6))  # Unique ID generator for new nodes

    # Identify nodes that exceed the chip size threshold
    nodes_to_split = [node for node, data in G.nodes(data=True) if data['size'] > chip_size * margin]
    for node in nodes_to_split:
        split_node(SG, node, chip_size, margin, id_counter)

    return SG


def split_linear_input(G, node, id_counter):
    """
    Splits a node in the graph `G` by dividing its size, energy, and latency properties in half, focusing on input edges.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the node to be split.
    - node (int): The ID of the node to split.
    - id_counter (iterator): An iterator that generates unique IDs for new nodes created during splitting.

    Returns:
    - tuple: (new_node_1, new_node_2) where new_node_1 is the updated original node, and new_node_2 is the newly created node.
    
    This function splits a node by creating two sub-nodes (`new_node_1` retains the original ID, while `new_node_2` 
    gets a new ID). It updates incoming edges by splitting their properties in half, while outgoing edges remain 
    connected to both split nodes.
    """
    node_data = G.nodes[node]
    node_size, node_energy, node_latency = node_data['size'] / 2, node_data['energy'] / 2, node_data['latency'] / 2

    # Define the IDs for the split nodes
    new_node_1, new_node_2 = node, next(id_counter)

    # Update original node and create a new split node
    G.nodes[new_node_1].update({
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_1'
    })
    G.add_node(new_node_2, **{
        **node_data,
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_2'
    })

    # Remove existing edges of the node to be split
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    G.remove_edges_from(in_edges + out_edges)

    # Reassign incoming edges with split properties
    for u, _, edge_data in in_edges:
        split_edge_data = {
            **edge_data,
            'size': edge_data['size'] / 2,
            'energy': edge_data['energy'] / 2,
            'latency': edge_data['latency'] / 2
        }
        G.add_edge(u, new_node_1, **split_edge_data)
        G.add_edge(u, new_node_2, **split_edge_data)

    # Reassign outgoing edges to both split nodes
    for _, v, edge_data in out_edges:
        G.add_edge(new_node_1, v, **edge_data)
        G.add_edge(new_node_2, v, **edge_data)

    return new_node_1, new_node_2


def split_linear_output(G, node, id_counter):
    """
    Splits a node in the graph `G` by dividing its size, energy, and latency properties in half, focusing on output edges.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the node to be split.
    - node (int): The ID of the node to split.
    - id_counter (iterator): An iterator that generates unique IDs for new nodes created during splitting.

    Returns:
    - tuple: (new_node_1, new_node_2) where new_node_1 is the updated original node, and new_node_2 is the newly created node.
    
    This function splits a node by creating two sub-nodes (`new_node_1` retains the original ID, while `new_node_2` 
    gets a new ID). Incoming edges remain connected to both split nodes, while outgoing edges are updated with 
    halved properties.
    """
    node_data = G.nodes[node]
    node_size, node_energy, node_latency = node_data['size'] / 2, node_data['energy'] / 2, node_data['latency'] / 2

    # Define the IDs for the split nodes
    new_node_1, new_node_2 = node, next(id_counter)

    # Update original node and create a new split node
    G.nodes[new_node_1].update({
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_1'
    })
    G.add_node(new_node_2, **{
        **node_data,
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_2'
    })

    # Remove existing edges of the node to be split
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    G.remove_edges_from(in_edges + out_edges)

    # Reassign incoming edges to both split nodes
    for u, _, edge_data in in_edges:
        G.add_edge(u, new_node_1, **edge_data)
        G.add_edge(u, new_node_2, **edge_data)

    # Reassign outgoing edges with split properties
    for _, v, edge_data in out_edges:
        split_edge_data = {
            **edge_data,
            'size': edge_data['size'] / 2,
            'energy': edge_data['energy'] / 2,
            'latency': edge_data['latency'] / 2
        }
        G.add_edge(new_node_1, v, **split_edge_data)
        G.add_edge(new_node_2, v, **split_edge_data)

    return new_node_1, new_node_2


def split_conv_input(G, node, id_counter):
    """
    Splits a convolutional node in the graph `G` by dividing its size, energy, and latency properties in half,
    with a focus on input edges.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the node to split.
    - node (int): The ID of the node to split.
    - id_counter (iterator): An iterator generating unique IDs for newly created nodes.

    Returns:
    - tuple: (new_node_1, new_node_2) where new_node_1 is the updated original node, and new_node_2 is the newly created node.
    
    This function splits the convolutional node by updating the original node and creating a new one. Incoming edges
    are split in half and assigned to each new node, while outgoing edges remain connected to both split nodes.
    """
    node_data = G.nodes[node]
    node_size, node_energy, node_latency = node_data['size'] / 2, node_data['energy'] / 2, node_data['latency'] / 2

    # Define IDs for the split nodes
    new_node_1, new_node_2 = node, next(id_counter)

    # Update the original node and create the new split node
    G.nodes[new_node_1].update({
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_1'
    })
    G.add_node(new_node_2, **{
        **node_data,
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_2'
    })

    # Remove existing edges for the node to be split
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    G.remove_edges_from(in_edges + out_edges)

    # Split and reassign incoming edges
    for u, _, edge_data in in_edges:
        split_edge_data = {
            **edge_data,
            'size': edge_data['size'] / 2,
            'energy': edge_data['energy'] / 2,
            'latency': edge_data['latency'] / 2
        }
        G.add_edge(u, new_node_1, **split_edge_data)
        G.add_edge(u, new_node_2, **split_edge_data)

    # Reassign outgoing edges to both split nodes
    for _, v, edge_data in out_edges:
        G.add_edge(new_node_1, v, **edge_data)
        G.add_edge(new_node_2, v, **edge_data)

    return new_node_1, new_node_2


def split_conv_output(G, node, id_counter):
    """
    Splits a convolutional node in the graph `G` by dividing its size, energy, and latency properties in half,
    with a focus on output edges.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the node to split.
    - node (int): The ID of the node to split.
    - id_counter (iterator): An iterator generating unique IDs for newly created nodes.

    Returns:
    - tuple: (new_node_1, new_node_2) where new_node_1 is the updated original node, and new_node_2 is the newly created node.
    
    This function splits the convolutional node by updating the original node and creating a new one. Incoming edges
    are kept as-is, while outgoing edges are split and assigned to each new node.
    """
    node_data = G.nodes[node]
    node_size, node_energy, node_latency = node_data['size'] / 2, node_data['energy'] / 2, node_data['latency'] / 2

    # Define IDs for the split nodes
    new_node_1, new_node_2 = node, next(id_counter)

    # Update the original node and create the new split node
    G.nodes[new_node_1].update({
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_1'
    })
    G.add_node(new_node_2, **{
        **node_data,
        'size': node_size,
        'energy': node_energy,
        'latency': node_latency,
        'name': node_data['name'] + '_2'
    })

    # Remove existing edges for the node to be split
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    G.remove_edges_from(in_edges + out_edges)

    # Reassign incoming edges to both split nodes
    for u, _, edge_data in in_edges:
        G.add_edge(u, new_node_1, **edge_data)
        G.add_edge(u, new_node_2, **edge_data)

    # Split and reassign outgoing edges
    for _, v, edge_data in out_edges:
        split_edge_data = {
            **edge_data,
            'size': edge_data['size'] / 2,
            'energy': edge_data['energy'] / 2,
            'latency': edge_data['latency'] / 2
        }
        G.add_edge(new_node_1, v, **split_edge_data)
        G.add_edge(new_node_2, v, **split_edge_data)

    return new_node_1, new_node_2

def split_batch(G, node, id_counter):
    """
    Splits a batch node in the graph `G` by dividing its energy and latency properties in half, focusing on both 
    input and output edges.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the node to split.
    - node (int): The ID of the node to split.
    - id_counter (iterator): An iterator generating unique IDs for newly created nodes.

    Returns:
    - tuple: (new_node_1, new_node_2) where new_node_1 is the updated original node, and new_node_2 is the newly created node.
    
    This function splits the batch node by updating the original node and creating a new one. Both input and output 
    edges are split and reassigned to the new nodes.
    """
    node_data = G.nodes[node]
    half_energy, half_latency = node_data['energy'] / 2, node_data['latency'] / 2

    # Define IDs for the split nodes
    new_node_1, new_node_2 = node, next(id_counter)

    # Update the original node and create the new split node
    G.nodes[new_node_1].update({
        'energy': half_energy,
        'latency': half_latency,
        'name': node_data['name'] + '_1'
    })
    G.add_node(new_node_2, **{
        **node_data,
        'energy': half_energy,
        'latency': half_latency,
        'name': node_data['name'] + '_2'
    })

    # Remove existing edges for the node to be split
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    G.remove_edges_from(in_edges + out_edges)

    # Split and reassign incoming edges
    for u, _, edge_data in in_edges:
        split_edge_data = {
            **edge_data,
            'size': edge_data['size'] / 2,
            'energy': edge_data['energy'] / 2,
            'latency': edge_data['latency'] / 2
        }
        G.add_edge(u, new_node_1, **split_edge_data)
        G.add_edge(u, new_node_2, **split_edge_data)

    # Split and reassign outgoing edges
    for _, v, edge_data in out_edges:
        split_edge_data = {
            **edge_data,
            'size': edge_data['size'] / 2,
            'energy': edge_data['energy'] / 2,
            'latency': edge_data['latency'] / 2
        }
        G.add_edge(new_node_1, v, **split_edge_data)
        G.add_edge(new_node_2, v, **split_edge_data)

    return new_node_1, new_node_2


def split_node_by_type(G, node, id_counter, split_strategy):
    """
    Splits a node in the graph `G` based on the specified splitting strategy.
    
    Parameters:
    - G (nx.DiGraph): The directed graph containing the node to split.
    - node (int): The ID of the node to split.
    - id_counter (iterator): An iterator generating unique IDs for new nodes.
    - split_strategy (str): The splitting strategy to use. Options include:
        'linear_input', 'linear_output', 'conv_input', 'conv_output', 'batch'.

    Returns:
    - tuple: (new_node_1, new_node_2) where new_node_1 is the updated original node, and new_node_2 is the newly created node.
    
    This function determines which split function to call based on the `split_strategy` parameter.
    Raises a ValueError if the specified strategy is unknown.
    """
    if split_strategy == 'linear_input':
        return split_linear_input(G, node, id_counter)
    elif split_strategy == 'linear_output':
        return split_linear_output(G, node, id_counter)
    elif split_strategy == 'conv_input':
        return split_conv_input(G, node, id_counter)
    elif split_strategy == 'conv_output':
        return split_conv_output(G, node, id_counter)
    elif split_strategy == 'batch':
        return split_batch(G, node, id_counter)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
