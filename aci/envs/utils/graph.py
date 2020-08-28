"""
Graph Coloring Environment
"""

# import torch as th
import numpy as np
import networkx as nx


# def get_colors(graph):
#     coloring = nx.algorithms.coloring.greedy_color(graph)
#     coloring = [coloring[i] for i in range(len(coloring))]
#     return th.tensor(coloring)


def determine_max_degree(constrains, n_nodes, graph_type, graph_args):
    max_degree = None
    if graph_type == 'watts_strogatz':
        if graph_args['p'] == 0:
            max_degree = graph_args['k']
    elif graph_type == 'random_regular':
       max_degree = graph_args['d']
    if max_degree is None:
        try: 
            max_degree = constrains['max_max_degree']
        except:
            raise ValueError("Max Degree of Graph is not limited.")
    return max_degree

def get_graph(graph_type, n_nodes, graph_args):
    if graph_type == 'watts_strogatz':
       return nx.watts_strogatz_graph(n=n_nodes, **graph_args)
    elif graph_type == 'random_regular':
       return nx.random_regular_graph(n=n_nodes, **graph_args)
    elif graph_type == 'erdos_renyi':
       return nx.erdos_renyi_graph(n=n_nodes, **graph_args)
    else:
        raise NotImplementedError(f'Graph type {graph_type} is not implemented.')

def check_constrains(
        graph, max_chromatic_number=None, min_chromatic_number=None, max_max_degree=None,
        min_min_degree=None, connected=None):
    chromatic_number = get_chromatic_number(graph)
    is_connected = nx.is_connected(graph)
    max_degree = get_max_degree(graph)
    min_degree = get_min_degree(graph)

    if (((max_chromatic_number is None) or (chromatic_number <= max_chromatic_number)) and
        ((min_chromatic_number is None) or (chromatic_number >= min_chromatic_number)) and
        ((max_max_degree is None) or (max_degree <= max_max_degree)) and
        ((min_min_degree is None) or (min_degree >= min_min_degree)) and
        (not connected or is_connected)):
        return {
            'max_degree': max_degree, 'is_connected': is_connected, 'chromatic_number': chromatic_number, 
            'fixed_degree': max_degree==min_degree,
        }
    else:
        return None

def search_graph(constrains, n_nodes, graph_type, graph_args):
    for i in range(10000):
        graph = get_graph(n_nodes=n_nodes, graph_type=graph_type, graph_args=graph_args)
        graph_info = check_constrains(graph, **constrains)
        if graph_info:
            return graph, graph_info
    raise Exception('Could not create graph with requested constrains number.')    

def get_chromatic_number(graph):
    return max(nx.algorithms.coloring.greedy_color(graph).values()) + 1

def get_max_degree(graph):
    return max([d for n, d in graph.degree()])

def get_min_degree(graph):
    return min([d for n, d in graph.degree()])

def get_neighbors(graph):
    neighbors = [
        list(np.array(list(graph[i].keys()))[np.random.permutation(len(graph[i]))])
        for i in range(len(graph))
    ]
    return neighbors

def get_adjacency_matrix(graph):
    return nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes()))).todense()

def create_graph(**kwargs):
    graph, graph_info = search_graph(**kwargs)
    neighbors = get_neighbors(graph)
    adjacency_matrix = get_adjacency_matrix(graph)
    graph = list(nx.to_edgelist(graph))
    return graph, neighbors, adjacency_matrix, graph_info


def pad_neighbors(neighbors, max_degree):
    padded_neighbors = [
        [i] + n + [-1]*(max_degree - len(n))
        for i, n in enumerate(neighbors)
    ]
    mask = [
        [False]*(len(n)+1) + [True]*(max_degree - len(n))
        for i, n in enumerate(neighbors)
    ]

    return np.array(padded_neighbors, dtype=np.int64), np.array(mask)