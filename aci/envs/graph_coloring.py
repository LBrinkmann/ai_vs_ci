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

class GraphColoring():
    """
    Description:
        To be filled
    Source:
        To be filled
    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
    Actions:
        Type: Discrete(n_colors)
    Reward:
        Reward is -1 for every step taken and 0 for the termination step.
    Starting State:
        Each Agent has a random starting color.
    Episode Termination:
        If the coloring is correct, or more then 200 steps have been passed.
    """

    def __init__(self, n_agents, graph_type, max_steps, global_reward_fraction, device):

        if graph_type == 'cycle':
            self.graph = nx.cycle_graph(n_agents)
            n_colors = 2
            n_neighbors = 2
        else:
            raise NotImplementedError(f'Graph type {graph_type} is not implemented.')

        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)

        self.neighbors = torch.tensor([
            random.sample(self.graph[n].keys(), len(self.graph[n])) 
            for n in self.graph
        ], dtype=torch.int64, device=device) # n_agents, n_neighbors

        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.MultiDiscrete([n_colors] * (n_neighbors + 1))
        self.steps = 0
        self.max_steps = max_steps
        self.n_actions = n_colors
        self.n_neighbors = n_neighbors
        self.observation_shape = (self.n_neighbors + 1)
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
        nx.draw(self.graph)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def close(self):
        pass

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GraphColoring(6, 'cycle', 5, 0.5, device)
    return env