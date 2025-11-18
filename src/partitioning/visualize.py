# visualize.py

import pygraphviz as pgv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re

def format_scientific(value):
    """
    Formats a numeric value in scientific notation with 2 decimal places.
    
    Parameters:
    - value (float or str): The value to be formatted. If 'N/A', returns 'N/A'.
    
    Returns:
    - str: The formatted scientific notation string or 'N/A' if input is 'N/A'.
    """
    if value == 'N/A':
        return value
    return f"{float(value):.2e}"


def normalize_alpha(value, min_size, max_size, scale_min=0.2, scale_max=1.0):
    """
    Normalizes alpha (transparency) for a node based on its size between `scale_min` and `scale_max`.
    
    Parameters:
    - value (float): The size of the node.
    - min_size (float): The minimum node size in the graph.
    - max_size (float): The maximum node size in the graph.
    - scale_min (float): Minimum alpha value for transparency.
    - scale_max (float): Maximum alpha value for transparency.
    
    Returns:
    - float: The normalized alpha value for the node.
    """
    if max_size == min_size:
        return scale_min  # Avoid division by zero, return minimum alpha if no variation
    return scale_min + (scale_max - scale_min) * ((value - min_size) / (max_size - min_size))


def visualize_partitioning(G, partitioning, model_name, num_subaccs, target, total_latency, key):
    """
    Visualizes a graph `G` based on its partitioning across machines and Subaccelerators (SAs),
    saving the output as an image file.
    
    Parameters:
    - G (nx.DiGraph): The directed graph to be visualized.
    - partitioning (dict): A dictionary containing machine assignments for each node and SA.
    - model_name (str): The name of the model for labeling the output file.
    
    Returns:
    - None: Saves the visualization as a PNG file in the 'visualizations' folder.
    
    This function creates a visualization of `G` where nodes are color-coded by machine assignment,
    and edges are color-coded based on communication across machines and Subaccs. The alpha value of each 
    node is adjusted based on its size.
    """
    graph_plot = pgv.AGraph(strict=False, directed=True)

    # Extract unique machines and assign colors
    machines = {assignment['Machine'] for assignment in partitioning['assignments']}
    machine_list = sorted(machines)
    node_to_location = {}
    for assignment in partitioning['assignments']:
        machine_id = assignment['Machine']
        for sa_id, nodes in assignment['Subaccs'].items():
            for node_id in nodes:
                node_to_location[node_id] = (machine_id, sa_id)

    cmap = plt.get_cmap('Set3')  # Color map for machine colors
    machine_colors = {machine: cmap(i / len(machine_list)) for i, machine in enumerate(machine_list)}

    # Define border styles for Subaccs
    sa_border_styles = ['dotted', 'solid', 'dashed', 'bold']  # Cycle through these styles for each SA

    # Calculate min and max node sizes for alpha normalization
    sizes = [G.nodes[node_id].get('size', 0.0) for node_id in G.nodes]
    min_size, max_size = min(sizes), max(sizes)

    # Add nodes to the graph with visual attributes based on partitioning
    for node_id, node_data in G.nodes(data=True):
        node_name = node_data.get('name', str(node_id))
        machine_id, sa_id = node_to_location.get(node_id, ("Unassigned", "N/A"))
        
        # Normalize alpha based on node size
        node_size = G.nodes[node_id].get('size', 0.0)
        node_alpha = normalize_alpha(node_size, min_size, max_size)

        # Set node fill color with normalized alpha based on machine assignment
        node_color = "lightgray" if machine_id == "Unassigned" else "#{:02x}{:02x}{:02x}{:02x}".format(
            int(machine_colors[machine_id][0] * 255),
            int(machine_colors[machine_id][1] * 255),
            int(machine_colors[machine_id][2] * 255),
            int(node_alpha * 255)
        )

        # Select border style based on SA ID
        sa_number = int(re.search(r'\d+', sa_id).group()) if re.search(r'\d+', sa_id) else 0
        border_style = sa_border_styles[sa_number % len(sa_border_styles)]

        # Add the node with attributes for color, border style, and labels
        graph_plot.add_node(node_id, 
                label=f"{node_name} (N{node_id}, M{machine_id}, {sa_id}, l: {format_scientific(node_data.get('latency', 'N/A')*total_latency)}, e: {format_scientific(node_data.get('energy', 'N/A'))})", 
                style=f"filled,{border_style}",  # Apply border style
                shape="ellipse",
                fontname="Helvetica",
                fillcolor=node_color,  
                fontcolor="black")

    # Add edges to the graph with styles based on communication between machines and Subaccs
    for u, v, edge_data in G.edges(data=True):
        source_name = G.nodes[u].get('name', str(u))
        destination_name = G.nodes[v].get('name', str(v))
        latency = format_scientific(edge_data.get('latency', 'N/A')*total_latency)
        energy = format_scientific(edge_data.get('energy', 'N/A'))

        machine_id_u, sa_id_u = node_to_location.get(u, ("Unassigned", "N/A"))
        machine_id_v, sa_id_v = node_to_location.get(v, ("Unassigned", "N/A"))
        
        # Define edge color and width based on machine and SA assignments
        if machine_id_u != machine_id_v:
            edge_color = "red"   # Red for inter-machine edges
            penwidth = 3
            label_color = "red"
        elif sa_id_u != sa_id_v:
            edge_color = "blue"  # Blue for inter-SA edges within the same machine
            penwidth = 2
            label_color = "blue"
        else:
            edge_color = "black" # Black for edges within the same SA
            penwidth = 1
            label_color = "black"

        # Add edge with attributes for color and width
        graph_plot.add_edge(u, v, label=f"l: {latency}, e: {energy}", 
                   fontname="Helvetica", fontcolor=label_color, color=edge_color, penwidth=penwidth)

    # Save the graph visualization as a PNG file
    os.makedirs('visualizations', exist_ok=True)
    path = f'visualizations/{target}/{key}_partitioning.png'
    graph_plot.layout(prog='dot')
    graph_plot.draw(path)
    print("💾 Partitioning saved to:", path)


def visualize_schedule(partitioning, num_chips, num_subaccs, splitGraph, model_name, hop_matrix, target, total_latency, wakeup_time, key, labelling=True):
    """
    Visualizes the scheduling of tasks across chips and Subaccs, including inter-chip communication,
    and saves the output as an image file.

    Parameters:
    - partitioning (dict): Contains node finish times and assignments for each chip and SA.
    - num_chips (int): The total number of chips used in the partitioning.
    - num_subaccs (int): The number of SAs (subaccelerators) per chip.
    - splitGraph (nx.DiGraph): The directed graph containing nodes and edges with latency information.
    - model_name (str): The name of the model for labeling the output file.
    - hop_matrix (numpy.ndarray): Matrix of hop counts between chips.

    Returns:
    - None: Saves the visualization as a PNG file in the 'visualizations' folder.

    This function creates a bar chart where each bar represents a scheduled task on a specific chip and SA.
    Bars are color-coded by chip, with alpha variation for each SA. Communication latencies between chips 
    are represented by red bars labeled with the hop count on both the source and destination chips, 
    connected by a red line.
    """
    start_times = [(partitioning['node_finish_times'][node_id] - splitGraph.nodes[node_id]['latency']) * total_latency for node_id in sorted(splitGraph.nodes)]
    finish_times = [t * total_latency for t in partitioning['node_finish_times']]
    slot_times = {key: [(s * total_latency, e * total_latency) for s, e in times] for key, times in partitioning['sa_slot_times'].items()}
    assignments = partitioning['assignments']  
    
    base_width = 12  # Base width for the drawing pane
    base_height = 8  # Base height for the drawing pane
    adjusted_width = base_width + (len(splitGraph.nodes) // 100) * 2  # Add 2 units for every 100 nodes
    adjusted_height = base_height + (num_chips * num_subaccs // 12) * 2   # Add 2 units for every 12 Subaccs
    fig, ax = plt.subplots(figsize=(adjusted_width, adjusted_height))

    # Define colors for each chip and vary alpha levels for Subaccs
    colors = list(mcolors.XKCD_COLORS.values())[:num_chips]
    
    y_labels = []
    y_pos = []
    
    # Position each chip and SA along the y-axis
    for chip_id in range(num_chips):
        for sa_id in range(num_subaccs):
            y_labels.append(f"Chip {chip_id} - SA {sa_id}")
            y_pos.append(chip_id * num_subaccs + sa_id)
    
    # Plot each node as a bar with varying colors and alphas for different Subaccs
    for chip_id, machine in enumerate(assignments):
        color = colors[chip_id]
        for sa_id in range(num_subaccs):
            sa_nodes = machine['Subaccs'].get(f"SA_{sa_id}", [])

            alpha = 0.4 + 0.6 * (sa_id / (max(num_subaccs - 1, 1)))  # Alpha variation by SA
            
            for node_id in sa_nodes:
                start = start_times[node_id]
                finish = finish_times[node_id]
                duration = finish - start
                
                ax.barh(
                    chip_id * num_subaccs + sa_id, 
                    duration, 
                    left=start, 
                    color=color, 
                    alpha=alpha, 
                    edgecolor='black', 
                    linewidth=0.5,  # Thinner borders
                    height=0.85  # Stack bars tightly
                )
                
                # Annotate with node ID for visibility
                if labelling:
                    ax.text(start + duration / 2, chip_id * num_subaccs + sa_id, f"{node_id}", 
                            ha='center', va='center', color='white', fontsize=8)

    # Represent inter-chip communication latency on both source and destination chips
    for src_id, dst_id, edge_data in splitGraph.edges(data=True):
        latency = edge_data.get('latency')*total_latency

        # Find chips of source and destination nodes
        src_chip = next((chip_id for chip_id, machine in enumerate(assignments) 
                        if any(src_id in sa_nodes for sa_nodes in machine['Subaccs'].values())), None)
        dst_chip = next((chip_id for chip_id, machine in enumerate(assignments) 
                        if any(dst_id in sa_nodes for sa_nodes in machine['Subaccs'].values())), None)

        # Only process if chips are different
        if src_chip is not None and dst_chip is not None and src_chip != dst_chip:
            # Get SA of the source and destination nodes
            src_sa = int(next(sa_id.replace("SA_", "") for sa_id, sa_nodes in assignments[src_chip]['Subaccs'].items() if src_id in sa_nodes))
            dst_sa = int(next(sa_id.replace("SA_", "") for sa_id, sa_nodes in assignments[dst_chip]['Subaccs'].items() if dst_id in sa_nodes))

            # Position inter-chip communication latency on the source chip’s SA
            src_start = finish_times[src_id]
            dst_start = src_start + latency

            # Retrieve the hop count between source and destination chips
            hop_count = int(hop_matrix[src_chip, dst_chip])

            # Draw the red communication bars on the source and destination
            ax.barh(
                src_chip * num_subaccs + src_sa, 
                latency * hop_count, 
                left=src_start, 
                color='red', 
                alpha=0.6, 
                edgecolor='none',
                linewidth=1, 
                height=0.85
            )
            ax.barh(
                dst_chip * num_subaccs + dst_sa, 
                latency * hop_count, 
                left=src_start, 
                color='red', 
                alpha=0.6, 
                edgecolor='none', 
                linewidth=1, 
                height=0.85
            )

            # Define line start and end positions
            line_x = src_start + latency / 2  # Keep x constant for vertical line
            line_start_y = src_chip * num_subaccs + src_sa
            line_end_y = dst_chip * num_subaccs + dst_sa

            

            if labelling:
                midpoint_y = (line_start_y + line_end_y) / 2
                 # Add the hop count label at the midpoint, clearly interrupting the line
                ax.text(
                    line_x, midpoint_y, f"{hop_count}",
                    ha='center', va='center', color='red', fontsize=6,
                    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.1')  # Add a white background for clarity
                )
                # Draw the top segment of the vertical line (above the label)
                ax.plot(
                    [line_x, line_x], 
                    [line_start_y, midpoint_y - 0.2],  # Leave space for the label
                    color='red', linestyle='--', linewidth=0.5
                )
                # Draw the bottom segment of the vertical line (below the label)
                ax.plot(
                    [line_x, line_x], 
                    [midpoint_y + 0.2, line_end_y],  # Leave space for the label
                    color='red', linestyle='--', linewidth=0.5
                )
            else:
                ax.plot(
                    [line_x, line_x], 
                    [line_start_y, line_end_y], 
                    color='red', linestyle='--', linewidth=0.5
                )

                

    # Plot slot boundaries for each SA
    for (chip_id, sa_id), slot_intervals in slot_times.items():
        sa_y_position = chip_id * num_subaccs + sa_id  # Get the Y-axis position of this SA

        for start_time, end_time in slot_intervals:
            # Draw a vertical dashed line at the start of the slot
            ax.plot(
                [start_time, start_time],  # Vertical line at slot start time
                [sa_y_position - 0.4, sa_y_position + 0.4],  # Span within SA row
                linestyle="dashed", color="blue", linewidth=1, alpha=0.8  # Dashed green line for start
            )

            # Draw a vertical dashed line at the end of the slot
            ax.plot(
                [end_time, end_time],  # Vertical line at slot end time
                [sa_y_position - 0.4, sa_y_position + 0.4],  # Span within SA row
                linestyle="-", color="blue", linewidth=1, alpha=0.8  # Dashed orange-yellow line for end
            )

            # Add labels for start and end times
            if labelling:
                ax.text(
                    start_time, sa_y_position, f"{start_time:.2f}",
                    ha="right", va="center", fontsize=7, color="darkblue",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5)  # Background for readability
                )
                ax.text(
                    end_time, sa_y_position, f"{end_time:.2f}",
                    ha="right", va="center", fontsize=7, color="darkblue",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5)  # Background for readability
                )

            ax.barh(
                sa_y_position, 
                wakeup_time, 
                left=end_time, 
                alpha=0.6,
                facecolor='lightgray',
                edgecolor='gray',  # Gray outline
                linewidth=1,  # Visible outline
                hatch='//',  # Hatched pattern
                height=0.85
            )

            # Label with Greek letter epsilon (ε)
            if labelling:
                ax.text(
                    end_time + wakeup_time / 2, sa_y_position, "ε",
                    ha="center", va="center", fontsize=8, color="gray",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5)
                )


    # Set y-axis labels for combined Chip and SA identifiers
    # Set SA ticks restarting for each chip
    y_ticks = []
    y_labels = []
    chip_centers = []

    for chip_id in range(num_chips):
        for sa_id in range(num_subaccs):
            y_ticks.append(chip_id * num_subaccs + sa_id)  # Position each SA
            y_labels.append(f"{sa_id}")  # Restart SA numbering for each chip
        # Calculate the center of the current chip's Subaccs
        chip_centers.append(chip_id * num_subaccs + num_subaccs // 2)

    # Apply SA ticks
    # Set SA ticks restarting for each chip
    y_ticks = []
    y_labels = []
    chip_centers = []

    for chip_id in range(num_chips):
        for sa_id in range(num_subaccs):
            y_ticks.append(chip_id * num_subaccs + sa_id)
            y_labels.append(f"SA{sa_id}")
        chip_centers.append(chip_id * num_subaccs + (num_subaccs - 1) / 2)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    for chip_id, center in enumerate(chip_centers):
        ax.text(
            -0.04, center, f"Chip {chip_id}",
            ha='center', va='center', color=colors[chip_id], fontsize=8, fontweight='bold',
            transform=ax.get_yaxis_transform(), rotation=90
        )
    ax.set_xlabel("Time", fontsize=12, labelpad=10)
    title = f"{' '.join(word.capitalize() for word in model_name.split('_'))}: Schedule for {num_chips} Chips, {num_subaccs} Subaccs"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()  # Optimize layout
    os.makedirs('visualizations', exist_ok=True)
    path = f'visualizations/{target}/{key}_schedule.png'
    plt.savefig(path, dpi=300, format='png')  # Save the figure as a high-resolution PNG

    print("💾 Schedule saved at: ", path)


def visualize_graph(G, model_name):
    graph_plot = pgv.AGraph(strict=False, directed=True)

    # Add nodes and edges from G to graph_plot
    for node in G.nodes():
        graph_plot.add_node(node)
    for edge in G.edges():
        graph_plot.add_edge(edge[0], edge[1])
    
    # Set general graph attributes
    graph_plot.graph_attr['label'] = 'Graph Visualization'
    graph_plot.node_attr['shape'] = 'circle'
    graph_plot.node_attr['style'] = 'filled'
    graph_plot.node_attr['fillcolor'] = 'lightblue'
    graph_plot.edge_attr['color'] = 'black'
    
    # Render and save the graph
    os.makedirs('visualizations', exist_ok=True)
    path = f'visualizations/{model_name}.png'
    graph_plot.layout(prog='dot')
    graph_plot.draw(path)
    print("Saved model at", path)
