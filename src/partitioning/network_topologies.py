# network_topologies.py

import networkx as nx
import numpy as np


class NetworkTopology:
    """
    A class to create various network topologies and calculate hop distances between nodes.

    Args:
        n (int): Number of nodes or size in one dimension (e.g., for a 2D or 3D grid).
        type (str, optional): Type of network topology. If not provided, defaults to an all-to-all network.
        *args: Additional dimensions for 2D or 3D topologies.
    """
    def __init__(self, n, *args, type=None):
        self.G = None
        # Initialize the network topology based on the specified type
        if type is not None:
            self.create_network(n, *args, type=type)
        else:
            print("No network topology specified, defaulting to all-to-all network.")
            self.create_all_to_all(n)
        # Compute hops matrix to store shortest paths between nodes


    def create_1d_mesh(self, n):
        """
        Creates a 1D mesh (linear chain) topology.

        Args:
            n (int): Number of nodes.

        Returns:
            networkx.Graph: The 1D mesh graph.
        """
        self.G = nx.grid_graph([n])
        mapping = {i: f"M{i}" for i in range(n)}
        self.G = nx.relabel_nodes(self.G, mapping)
        return self.G


    def create_1d_ring(self, n):
        """
        Creates a 1D ring topology where endpoints are connected.

        Args:
            n (int): Number of nodes.

        Returns:
            networkx.Graph: The 1D ring graph.
        """
        self.G = nx.grid_graph([n], periodic=True)
        mapping = {i: f"M{i}" for i in range(n)}
        self.G = nx.relabel_nodes(self.G, mapping)
        return self.G


    def create_2d_mesh(self, n, m):
        """
        Creates a 2D mesh (grid) topology.

        Args:
            n (int): Number of rows.
            m (int): Number of columns.

        Returns:
            networkx.MultiDiGraph: The 2D mesh graph.
        """
        self.G = nx.grid_2d_graph(n, m, create_using=nx.MultiDiGraph)
        mapping = {node: f"M{rank}" for rank, node in enumerate(sorted(self.G.nodes()))}
        self.G = nx.relabel_nodes(self.G, mapping)
        return self.G


    def create_2d_torus(self, n, m):
        """
        Creates a 2D torus topology where opposite edges are connected.

        Args:
            n (int): Number of rows.
            m (int): Number of columns.

        Returns:
            networkx.MultiDiGraph: The 2D torus graph.
        """
        self.G = nx.grid_2d_graph(n, m, periodic=True, create_using=nx.MultiDiGraph)
        mapping = {node: f"M{rank}" for rank, node in enumerate(self.G.nodes())}
        self.G = nx.relabel_nodes(self.G, mapping)
        return self.G


    def create_3d_mesh(self, n, m, p):
        """
        Creates a 3D mesh (grid) topology.

        Args:
            n (int): Depth.
            m (int): Height.
            p (int): Width.

        Returns:
            networkx.MultiDiGraph: The 3D mesh graph.
        """
        self.G = nx.grid_graph([n, m, p])
        mapping = {node: f"M{rank}" for rank, node in enumerate(self.G.nodes())}
        self.G = nx.relabel_nodes(self.G, mapping)
        return self.G


    def create_3d_torus(self, n, m, p):
        """
        Creates a 3D torus topology with periodic boundaries on all dimensions.

        Args:
            n (int): Depth.
            m (int): Height.
            p (int): Width.

        Returns:
            networkx.MultiDiGraph: The 3D torus graph.
        """
        self.G = nx.grid_graph([n, m, p], periodic=True)
        mapping = {node: f"M{rank}" for rank, node in enumerate(self.G.nodes())}
        self.G = nx.relabel_nodes(self.G, mapping)
        return self.G
    

    def create_all_to_all(self, n):
        """
        Creates an all-to-all (fully connected) topology.

        Args:
            n (int): Number of nodes.

        Returns:
            None
        """
        self.G = nx.complete_graph(n, create_using=nx.MultiDiGraph)
        mapping = {node: f"M{rank}" for rank, node in enumerate(self.G.nodes())}
        self.G = nx.relabel_nodes(self.G, mapping)


    def create_hierarchical_cluster(self, n_total, cluster_type="1D_RING", interconnect_type="FULL", interconnect_weight=15, *args):
        n_cluster = n_total // 4
        cluster_graphs = []

        for cluster_id in range(4):
            cluster = NetworkTopology(n_cluster, *args, type=cluster_type)
            mapping = {node: f"C{cluster_id}_{node}" for node in cluster.G.nodes()}
            cluster.G = nx.relabel_nodes(cluster.G, mapping)
            cluster_graphs.append(cluster.G)

        self.G = nx.compose_all(cluster_graphs)

        cluster_centers = [f"C{i}_M0" for i in range(4)]  
        
        if interconnect_type == "FULL":
            for i in range(4):
                for j in range(i+1, 4):
                    self.G.add_edge(cluster_centers[i], cluster_centers[j], weight=interconnect_weight)
                    self.G.add_edge(cluster_centers[j], cluster_centers[i], weight=interconnect_weight)
        elif interconnect_type == "RING":
            for i in range(4):
                next_i = (i+1) % 4
                self.G.add_edge(cluster_centers[i], cluster_centers[next_i], weight=interconnect_weight)
                self.G.add_edge(cluster_centers[next_i], cluster_centers[i], weight=interconnect_weight)
        else:
            raise ValueError("Unsupported interconnect type. Use 'FULL' or 'RING'.")


    def create_network(self, n, *args, type="NONE"):
        """
        Creates the specified network topology based on the 'type' argument.

        Args:
            n (int): Number of nodes or dimension size.
            *args: Additional dimensions for 2D and 3D topologies.
            type (str): Type of topology to create.

        Returns:
            networkx.Graph or networkx.MultiDiGraph: The created network graph.
        """
        if type == "1D_MESH":
            print("Creating 1D mesh network.")
            return self.create_1d_mesh(n)
        elif type == "1D_RING":
            print("Creating 1D ring network.")
            return self.create_1d_ring(n)
        elif type == "2D_MESH":
            print("Creating 2D mesh network.")
            assert len(args) == 1, "2D mesh requires 2 total arguments."
            return self.create_2d_mesh(n, *args)
        elif type == "2D_TORUS":
            print("Creating 2D torus network.")
            assert len(args) == 1, "2D torus requires 2 total arguments."
            return self.create_2d_torus(n, *args)
        elif type == "3D_MESH":
            print("Creating 3D mesh network.")
            assert len(args) == 2, "3D mesh requires 3 total arguments."
            return self.create_3d_mesh(n, *args)
        elif type == "3D_TORUS":
            print("Creating 3D torus network.")
            assert len(args) == 2, "3D torus requires 3 total arguments."
            return self.create_3d_torus(n, *args)
        elif type == "ALL_TO_ALL":
            print("Creating all-to-all network.")
            return self.create_all_to_all(n)
        elif type == "HIERARCHICAL_CLUSTER":
            print("Creating hierarchical cluster network.")
            return self.create_hierarchical_cluster(n)

        else:
            raise ValueError("Invalid network topology type.")
    

    def get_shortest_hops_matrix(self):
        """
        Returns the shortest path matrix with rows/columns in node label order (M0, M1, ...).
        """
        matrix = nx.floyd_warshall_numpy(self.G)
        ordered_nodes = sorted(self.G.nodes(), key=lambda x: int(x[1:]))  # M0, M1, ...
        node_index = {node: i for i, node in enumerate(self.G.nodes())}
        indices = [node_index[node] for node in ordered_nodes]
        return matrix[np.ix_(indices, indices)]

    
    def get_shortest_hops(self, u, v):
        """
        Gets the shortest hop count between two specified nodes.

        Args:
            u (str): Starting node.
            v (str): Target node.

        Returns:
            int: Number of hops in the shortest path from u to v.
        """
        return nx.shortest_path_length(self.G, source=u, target=v)
    

    def get_simple_hops(self, u, v):
        """
        Gets the sequence of nodes in the shortest path between two specified nodes.

        Args:
            u (str): Starting node.
            v (str): Target node.

        Returns:
            list: Sequence of nodes in the shortest path from u to v.
        """
        return nx.shortest_path(self.G, source=u, target=v)
    

    def remove_node(self, node):
        """
        Removes a specified node from the network topology.

        Args:
            node (str): The node to remove.

        Returns:
            None
        """
        if node in self.G:
            self.G.remove_node(node)
            print(f"Node {node} removed from the network.")
            self.hops_matrix = nx.floyd_warshall_numpy(self.G)
        else:
            print(f"Node {node} does not exist in the network.")


    def optimize_node_order(self):
        print("🔄 Optimizing node order to minimize hop distance in sequence.")
        if self.G is None: return

        nodes = list(self.G.nodes())
        visited = set()
        new_order = []

        current = nodes[0]
        visited.add(current)
        new_order.append(current)

        while len(visited) < len(nodes):
            neighbors = [(node, nx.shortest_path_length(self.G, current, node)) for node in nodes if node not in visited]
            next_node = min(neighbors, key=lambda x: x[1])[0]
            visited.add(next_node)
            new_order.append(next_node)
            current = next_node

        mapping = {old: f"M{i}" for i, old in enumerate(new_order)}
        self.G = nx.relabel_nodes(self.G, mapping)
