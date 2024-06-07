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


################
#Define structural equations
import numpy as np

def logistic_function(x):
    """Logistic function to ensure values are between 0 and 1."""
    return 1 / (1 + np.exp(-x))

def generate_gene_expression(parent_expressions):
    """Generate gene expression level based on expressions of parent genes.
    Assuming parent_expressions is a list of expression levels of parent genes."""
    weight = 0.1  # This could be adjusted or made more complex
    x = sum(parent_expressions) * weight
    return logistic_function(x)



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


def simulate_observational_data(dag, num_samples):
    """Simulate observational data for all genes in the DAG.
    dag should be a structure that allows you to know the parents of each node."""
    # Initialize expressions for all genes in the DAG
    gene_expressions = np.zeros((num_samples, len(dag)))
    for i in range(len(dag)):
        parents = dag[i]  # Get parents of gene i
        for sample in range(num_samples):
            if parents:
                parent_expressions = gene_expressions[sample, parents]
                gene_expressions[sample, i] = generate_gene_expression(parent_expressions)
            else:
                # If no parents, generate an initial expression level randomly
                gene_expressions[sample, i] = np.random.rand()  #values in [0, 1)
    return gene_expressions


def intervene_on_gene(dag, gene_index, intervention_value, num_samples):
    """Simulate interventional data by setting the expression of a gene to an arbitrary value and observing downstream effects."""
    # Start with observational data as the base
    gene_expressions = simulate_observational_data(dag, num_samples)
    # Intervene on the specified gene
    gene_expressions[:, gene_index] = intervention_value
    # Propagate the intervention effects downstream
    for i in range(gene_index + 1, len(dag)):
        parents = dag[i]
        for sample in range(num_samples):
            parent_expressions = gene_expressions[sample, parents]
            gene_expressions[sample, i] = generate_gene_expression(parent_expressions)
    return gene_expressions