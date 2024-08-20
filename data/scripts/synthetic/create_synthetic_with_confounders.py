'''
File to create synthetic datasets with confounder variables for ablation studies
1. Read in an initial PPI (processed data) and transforms it into a directed acyclic graph (DAG) by directing edges from higher to lower degree nodes.
2. Perform topological sort on graph so that it follows hierarchical organization (meaning, the nodes with lower indices are closer to the 'top'/'edges' of the graph)
    We need this because we need to know the hierarchical sorting to compute node values given structural equations
3. Define a simple structural equation for each node. Very simple, for now the gene expression for each node is computed as the summation of gene expression of its parents, multiplied by a weight, and passed through a logistic function to keep gene expression values between zero and one
    This is why we needed the topological sort: because we need to start from the nodes without parents in the graph (for these, we assign random gene expression values), and then we go down recursively in the graph to generate gene expression for the remaining genes
4. To generate observational data (equivalent to "diseased"), we just follow this process N times (this will generate different samples because we initialize the 'leaf' nodes with random gene expression)
5. To generate interventional data (equivalent to "treated"), we follow first simulate observational data, then we assign a specific value to the perturbed gene, and then we propagate the effects downstream (meaning that the gene expression value for nodes downstream is re-computed)
    This currently implements perturbations in only one gene, we need to extend to a set of genes
'''

import pickle
import torch
import networkx as nx


##############
#Function to transform PPI into DAG
import networkx as nx

def create_dag_from_ppi(ppi_network):
    """
    Convert an undirected PPI network to a DAG by directing edges from higher to lower degree nodes.
    Args:
    - ppi_network: An undirected NetworkX graph representing the PPI network.
    Returns:
    - dag: A directed NetworkX graph representing the approximated DAG.
    """
    # Create a deep copy of the network to keep the original network unchanged
    dag = ppi_network.copy(as_view=False)
    dag = dag.to_directed()

    # Iterate over all edges and redirect them
    for u, v in list(dag.edges()):
        if dag.degree[u] < dag.degree[v]:
            # Reverse the edge if the degree of u is less than v
            dag.remove_edge(u, v)
            dag.add_edge(v, u)
        elif dag.degree[u] == dag.degree[v] and u > v:
            # If degrees are equal, use node labels to decide direction, ensuring consistency
            dag.remove_edge(u, v)
            dag.add_edge(v, u)
        # If u's degree is higher than v's or they're equal and u < v, leave the edge as is

    # Remove cycles if any remain, using a simple heuristic
    try:
        cycle = nx.find_cycle(dag, orientation='original')
        while cycle:
            # Remove an edge from the cycle
            edge_to_remove = cycle[0]
            dag.remove_edge(*edge_to_remove[:2])
            cycle = nx.find_cycle(dag, orientation='original')
    except nx.NetworkXNoCycle:
        pass  # No cycle found, the DAG is ready

    return dag



##############
#Function to sort node indices in DAG so that it follows hierarchical organization
from collections import defaultdict, deque

def topological_sort(edges, num_nodes):
    """Perform topological sorting on a DAG represented as a list of edges."""
    # Create adjacency list and in-degree count
    adj_list = defaultdict(list)
    in_degree = defaultdict(int)
    for src, dest in edges:
        adj_list[src].append(dest)
        in_degree[dest] += 1
    
    # Initialize queue with nodes having in-degree of 0
    queue = deque([node for node in range(num_nodes) if node not in in_degree])
    sorted_nodes = []
    
    # Process nodes with in-degree of 0
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(sorted_nodes) != num_nodes:
        raise ValueError("Graph is not a DAG")
    
    return sorted_nodes

def reindex_dag_nodes(dag, sorted_nodes):
    """
    Reindex the nodes of a DAG based on the ordering provided in sorted_nodes.
    
    Args:
    - dag: A directed NetworkX graph representing the original DAG.
    - sorted_nodes: A list of nodes in the DAG sorted according to some criteria (e.g., topological sort).
    
    Returns:
    - new_dag: A new directed NetworkX graph with nodes reindexed according to sorted_nodes.
    """
    # Create a mapping from old indices to new indices based on sorted_nodes
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted_nodes)}
    
    # Apply the mapping to reindex nodes
    # NetworkX relabel_nodes function can be used to apply the mapping to the nodes
    new_dag = nx.relabel_nodes(dag, mapping=index_mapping, copy=True)
    
    return new_dag


def reindex_perturbed_nodes(perturbed_nodes, sorted_nodes):
    """
    Reindex a set of nodes perturbed based on the ordering provided in sorted_nodes.
    Args:
    - perturbed_nodes: A list with indices of perturbed nodes.
    - sorted_nodes: A list of nodes in the DAG sorted according to some criteria (e.g., topological sort).
    Returns:
    - new_perturbed_nodes: A list with reindexed perturbed nodes
    """
    # Create a mapping from old indices to new indices based on sorted_nodes
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted_nodes)}
    # Apply the mapping to reindex nodes
    new_perturbed_nodes = [index_mapping[e] for e in perturbed_nodes]
    return new_perturbed_nodes

################
#Define structural equations
import numpy as np

def logistic_function(x):
    """Logistic function to ensure values are between 0 and 1."""
    return 1 / (1 + np.exp(-x))

def generate_gene_expression(parent_expressions, weights, bias, confounder_bias):
    """Generate gene expression level based on expressions of parent genes."""
    # random_noise = np.random.normal(-0.04, 0.08)
    random_noise = np.random.normal(-0.04, 0.15)
    x = logistic_function(np.dot(weights, parent_expressions) + bias + confounder_bias)
    return np.clip(x + random_noise, 0, 1)

######################
#Simulate observational and interventional data 
#node indices follow hiarachical order in the DAG
def get_parents_for_each_node(dag):
    """
    For a given DAG, get the parents for each node.

    Args:
    - dag: A directed NetworkX graph representing the DAG.

    Returns:
    - parents_dict: A dictionary where keys are node indices and values are lists of parent nodes.
    """
    parents_dict = {node: list(dag.predecessors(node)) for node in dag.nodes()}
    return parents_dict


def generate_weights_and_biases(parents_dict):
    weights = {}
    biases = {}
    for node, parents in parents_dict.items():
        if parents:
            # Assign random weights and biases for each gene with parents
            weights[node] = np.random.normal(0.06, 0.20, size=len(parents))
            biases[node] = np.random.uniform(-0.05, 0.2)
        else:
            # For genes with no parents, weights are empty and bias is set
            weights[node] = np.array([])
            biases[node] = np.random.uniform(-0.05, 0.2)
    return weights, biases

def generate_confounder_biases(parents_dict, confounder_probability=0.2, mean=0.0, std=0.1):
    """
    Generate confounder biases for a subset of nodes in the parents_dict.
    Args:
    - parents_dict: A dictionary where keys are node indices and values are lists of parent nodes.
    - confounder_probability: The probability of assigning a confounder bias to a node.
    - mean: The mean of the normal distribution for generating biases.
    - std: The standard deviation of the normal distribution for generating biases.
    Returns:
    - confounder_biases: A dictionary where keys are node indices and values are confounder biases.
    """
    confounder_biases = {}
    for node in parents_dict.keys():
        if np.random.rand() < confounder_probability:
            confounder_biases[node] = np.random.normal(mean, std)
        else:
            confounder_biases[node] = 0.0
    return confounder_biases


def simulate_observational_data(parents_dict, num_samples, weights, biases, confounder_biases):
    """Simulate observational data for all genes based on parents_dict."""
    num_genes = len(parents_dict)
    gene_expressions = np.zeros((num_samples, num_genes))
    for i in range(num_genes):
        parents = parents_dict[i]
        for sample in range(num_samples):
            if parents:
                parent_expressions = gene_expressions[sample, parents]
                gene_expressions[sample, i] = generate_gene_expression(parent_expressions, weights[i], biases[i], confounder_biases[i])
            else:
                # If no parents, generate an initial expression level randomly
                gene_expressions[sample, i] = np.random.uniform(0.4, 0.8)  # Initial value
    return gene_expressions





def intervene_on_gene(observational_data, parents_dict, gene_indices, intervention_values, weights, biases, confounder_biases):
    """Simulate interventional data by setting the expression of a gene to an arbitrary value and observing downstream effects."""
    interventional_data = observational_data.copy()
    # Start with observational data as the base
    num_samples = interventional_data.shape[0]
    for gene_index, intervention_value in zip(gene_indices, intervention_values):
    # Intervene on the specified genes
        interventional_data[:, gene_index] = intervention_value
    # Propagate the intervention effects downstream 
    # This re-computes gene expression for all genes to add noise variability to nodes upstream of perturbed nodes, 
    # and to update gene expression accounting for new parents' values for those genes whose parents were perturbed
    for i in range(0, len(parents_dict)):   #for each gene
        if i in gene_indices:
            continue #do not recompute for other genes that have been intervened on
        parents = parents_dict[i]
        if len(parents)>0:
            for sample in range(num_samples):
                parent_expressions = interventional_data[sample, parents]
                interventional_data[sample, i] = generate_gene_expression(parent_expressions, weights[i], biases[i], confounder_biases[i])
    return interventional_data





####Functions to create synthetic manipulated data
#These are to be tested

#Remove random edges
def remove_random_edges(dag, fraction=0.1):
    """
    Remove a fraction of edges randomly from the DAG.
    
    Args:
    - dag: A directed NetworkX graph representing the DAG.
    - fraction: A float representing the fraction of edges to remove.
    
    Returns:
    - modified_dag: A new directed NetworkX graph with specified edges removed.
    """
    num_edges = dag.number_of_edges()
    num_remove = int(fraction * num_edges)
    edge_indices_to_remove = np.random.choice(range(dag.number_of_edges()), num_remove, replace=False)
    edges_to_remove = np.array(list(dag.edges()))[edge_indices_to_remove].tolist()

    modified_dag = dag.copy()
    modified_dag.remove_edges_from(edges_to_remove)
    
    return modified_dag

# Example usage:
# dag_with_random_edges_removed = remove_random_edges(dag, fraction=0.1)

#High betweeness centrality edge removal
def remove_high_betweenness_edges(dag, fraction=0.1):
    """
    Remove a fraction of edges with the highest betweenness centrality from the DAG.
    
    Args:
    - dag: A directed NetworkX graph representing the DAG.
    - fraction: A float representing the fraction of edges to remove.
    
    Returns:
    - modified_dag: A new directed NetworkX graph with high betweenness centrality edges removed.
    """
    edge_betweenness = nx.edge_betweenness_centrality(dag)
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
    num_remove = int(fraction * len(sorted_edges))
    edges_to_remove = [edge for edge, _ in sorted_edges[:num_remove]]

    modified_dag = dag.copy()
    modified_dag.remove_edges_from(edges_to_remove)
    
    return modified_dag

# Example usage:
# dag_with_high_betweenness_edges_removed = remove_high_betweenness_edges(dag, fraction=0.1)

#Bridge edge removal
def remove_bridge_edges(dag, fraction=0.1):
    """
    Remove a fraction of bridge edges (edges crucial for connectivity) from the DAG.
    Args:
    - dag: A directed NetworkX graph representing the DAG.
    - fraction: A float representing the fraction of bridge edges to remove.
    Returns:
    - modified_dag: A new directed NetworkX graph with specified bridge edges removed.
    """
    bridges = list(nx.bridges(dag.to_undirected()))
    num_remove = int(fraction * len(bridges))
    edges_to_remove = np.random.choice(range(len(bridges)), num_remove, replace=False)
    edges_to_remove = [bridges[e] for e in edges_to_remove]
    modified_dag = dag.copy()
    modified_dag.remove_edges_from(edges_to_remove)
    return modified_dag

# Example usage:
# dag_with_bridge_edges_removed = remove_bridge_edges(dag, fraction=0.1)
