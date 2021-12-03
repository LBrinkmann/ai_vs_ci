"""
Graph Coloring Environment
"""

# import torch as th
import numpy as np
import networkx as nx
from numpy.lib.financial import ipmt
import pandas as pd
from aci.utils.df_utils import selector


def get_graph_properties(graph):
    betweenness = np.array(
        list(nx.algorithms.betweenness_centrality(graph).values()))
    clustering = np.array(list(nx.algorithms.clustering(graph).values()))
    closeness = np.array(list(nx.algorithms.clustering(graph).values()))
    constraint = np.array(
        list(nx.algorithms.structuralholes.constraint(graph).values()))
    chromatic_number = get_chromatic_number(graph)
    is_connected = nx.is_connected(graph)
    components = nx.number_connected_components(graph)
    max_degree = get_max_degree(graph)
    min_degree = get_min_degree(graph)

    return {
        'avg_betweenness': betweenness.mean(),
        'avg_clustering': clustering.mean(),
        'max_closeness': closeness.max(),
        'var_constraint': constraint.var(),
        'max_betweenness': betweenness.max(),
        'components': components,
        'max_degree': max_degree,
        'is_connected': is_connected,
        'chromatic_number': chromatic_number,
        'fixed_degree': max_degree == min_degree,
    }


# def get_colors(graph):
#     coloring = nx.algorithms.coloring.greedy_color(graph)
#     coloring = [coloring[i] for i in range(len(coloring))]
#     return th.tensor(coloring)


def get_graph(*, n_nodes, components=1, seed, **args):
    if components == 1:
        return _get_graph(seed=seed, nodes=n_nodes, **args)
    else:
        assert n_nodes % components == 0
        nodes_per_component = n_nodes//components
        graph = nx.empty_graph()
        for i in range(0, n_nodes, nodes_per_component):
            _seed = hash((seed, i)) % (2**32 - 1)
            nodes = range(i, i+nodes_per_component)
            node_map = {i: l for i, l in enumerate(nodes)}
            comp = _get_graph(seed=_seed, nodes=nodes_per_component, **args)
            comp = nx.relabel.relabel_nodes(comp, node_map)
            graph = nx.compose(graph, comp)
    return graph


def _get_graph(graph_type, nodes, seed, graph_args):
    if graph_type == 'watts_strogatz':
        return nx.watts_strogatz_graph(n=nodes, seed=seed, **graph_args)
    elif graph_type == 'random_regular':
        return nx.random_regular_graph(n=nodes, seed=seed, **graph_args)
    elif graph_type == 'erdos_renyi':
        return nx.erdos_renyi_graph(n=nodes, seed=seed, **graph_args)
    else:
        raise NotImplementedError(
            f'Graph type {graph_type} is not implemented.')


comp = {
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "nq": lambda a, b: a != b,
}


def check_constrains(properties, constrains):
    return all(comp[k](properties[p], v) for p, c in constrains.items() for k, v in c.items())


def search_graph(constrains, seed, idx, n_nodes, graph_type, components, graph_args):
    for i in range(10000):
        _seed = hash((seed, idx, i)) % (2**32 - 1)
        graph = get_graph(
            n_nodes=n_nodes, graph_type=graph_type, components=components, graph_args=graph_args, seed=_seed)
        g_props = get_graph_properties(graph)
        if check_constrains(g_props, constrains):
            return graph, g_props
    raise Exception('Could not create graph with requested constrains.')


def select_graph(props_df, quantiles, seed):
    # import ipdb
    # ipdb.set_trace()
    props_df_rank = props_df.rank(pct=True) * 100
    w = selector(props_df_rank, quantiles)
    return props_df[w].sample(1, random_state=seed).index[0]


def quantile_graph(quantiles, constrains, seed, n_nodes, graph_type, graph_args):
    graphs = []
    props = []
    for idx in range(1000):
        graph, prop = search_graph(
            constrains, idx, seed, n_nodes, graph_type, graph_args)
        graphs.append(graph)
        props.append(prop)
    props_df = pd.DataFrame(props)
    idx = select_graph(props_df, quantiles, seed)
    return graphs[idx], props[idx]


def get_chromatic_number(graph):
    return max(nx.algorithms.coloring.greedy_color(graph).values()) + 1


def get_max_degree(graph):
    return max([d for n, d in graph.degree()])


def get_min_degree(graph):
    return min([d for n, d in graph.degree()])


def get_neighbors(graph):
    neighbors = [
        list(np.array(list(graph[i].keys()))[
             np.random.permutation(len(graph[i]))])
        for i in range(len(graph))
    ]
    return neighbors

# def get_adjacency_matrix(graph):
#     return nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes()))).todense()


def create_graph(quantiles=None, **kwargs):
    if quantiles:
        graph, graph_info = quantile_graph(quantiles=quantiles, **kwargs)
    else:
        graph, graph_info = search_graph(idx=0, **kwargs)
    print(graph_info)
    neighbors = get_neighbors(graph)
    # adjacency_matrix = get_adjacency_matrix(graph)
    # graph = list(nx.to_edgelist(graph))
    return neighbors
