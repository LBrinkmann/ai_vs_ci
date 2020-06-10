"""
Graph Coloring Environment
"""

import random
import torch
import numpy as np
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def create_cycle_graph(n_nodes, degree):
    graph = nx.watts_strogatz_graph(n_nodes, degree, 0)
    n_actions = max(nx.algorithms.coloring.greedy_color(graph).values()) + 1
    n_neighbors = degree
    return graph, n_actions, n_neighbors

def random_regular_graph(n_nodes, degree, chromatic_number=None):
    for i in range(100):
        graph = nx.random_regular_graph(degree, n_nodes)
        n_actions = max(nx.algorithms.coloring.greedy_color(graph).values()) + 1
        n_neighbors = degree
        if (chromatic_number is None) or (n_actions == chromatic_number):
            return graph, n_actions, n_neighbors
    raise Exception('Could not create graph with requested chromatic number.')

def create_graph(graph_type, n_nodes, **kwargs):
    if graph_type == 'cycle':
        return create_cycle_graph(n_nodes, **kwargs)
    elif graph_type == 'random_regular':
        return random_regular_graph(n_nodes, **kwargs)
    else:
        raise NotImplementedError(f'Graph type {graph_type} is not implemented.')

class GraphColoring():
    def __init__(self, n_agents, graph_args, max_steps, global_reward_fraction, device):

        self.graph, self.n_actions, self.n_neighbors = create_graph(n_nodes=n_agents, **graph_args)
        self.graph_pos = nx.spring_layout(self.graph)
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)

        self.neighbors = torch.tensor([
            random.sample(self.graph[n].keys(), len(self.graph[n])) 
            for n in self.graph
        ], dtype=torch.int64, device=device) # n_agents, n_neighbors

        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.MultiDiscrete([n_colors] * (n_neighbors + 1))
        self.steps = 0
        self.max_steps = max_steps
        self.observation_shape = (self.n_neighbors + 1,)
        self.n_agents = n_agents
        self.global_reward_fraction = global_reward_fraction
        self.reset()

    def observations(self):
        state_ = self.state.reshape(1,-1).repeat(self.n_agents,1)
        neighbor_colors = torch.gather(state_, 1, self.neighbors)
        observations = torch.cat((self.state.reshape(-1,1), neighbor_colors), dim=1)
        return observations

    def step(self, actions):
        self.state = actions
        observations = self.observations()

        conflicts = (observations[:,[0]] == observations[:,1:])

        local_conflicts = conflicts.any(dim=1)
        global_conflicts = conflicts.any()

        local_reward = local_conflicts.type(torch.int) * -1
        global_reward = global_conflicts.type(torch.int) * -1

        reward = self.global_reward_fraction * global_reward + \
            (1-self.global_reward_fraction) * local_reward

        if not global_conflicts or (self.steps == self.max_steps):
            done = True
        elif self.steps > self.max_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
        self.steps += 1
        return observations, reward, done, {}

    def reset(self):
        self.steps = 0
        self.state = torch.tensor(np.random.randint(0, self.n_actions, self.n_agents), dtype=torch.int64)
        return self.observations()

    def render(self):
        fig = plt.figure()
        nx.draw(self.graph, self.graph_pos, node_color=self.state.cpu().numpy())
        plt.text(0, 0, str(self.steps), fontsize=30)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.moveaxis(data, 2, 0)
        plt.close()
        return torch.tensor(data[np.newaxis, np.newaxis])

    def close(self):
        pass

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GraphColoring(6, 'cycle', 5, 0.5, device)
    return env