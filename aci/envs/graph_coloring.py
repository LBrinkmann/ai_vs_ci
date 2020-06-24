"""
Graph Coloring Environment
"""

import random
import torch as th
import numpy as np
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def get_graph_colors(graph):
    coloring = nx.algorithms.coloring.greedy_color(graph)
    coloring = [coloring[i] for i in range(len(coloring))]
    return th.tensor(coloring)

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
    def __init__(
            self, n_agents, graph_args, max_steps, fixed_length,
            global_reward_fraction, device):
        self.steps = 0
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.global_reward_fraction = global_reward_fraction
        self.graph_args = graph_args
        self.device = device
        self.fixed_length = fixed_length
        self.new_graph()
        self.reset()

    def new_graph(self):
        self.graph, self.n_actions, self.n_neighbors = create_graph(
            n_nodes=self.n_agents, **self.graph_args)
        self.graph_pos = nx.spring_layout(self.graph)
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)

        self.neighbors = th.tensor([
            random.sample(self.graph[n].keys(), len(self.graph[n])) 
            for n in range(len(self.graph))
        ], dtype=th.int64, device=self.device) # n_agents, n_neighbors
        self.observation_shape = (self.n_neighbors + 1,)

    def observations(self):
        state_ = self.state.reshape(1,-1).repeat(self.n_agents,1)
        neighbor_colors = th.gather(state_, 1, self.neighbors)
        observations = th.cat((self.state.reshape(-1,1), neighbor_colors), dim=1)
        return observations

    def get_rewards(self, observations):
        conflicts = (observations[:,[0]] == observations[:,1:])

        local_conflicts = conflicts.any(dim=1)
        global_conflicts = conflicts.any()

        local_reward = local_conflicts.type(th.int) * -1
        global_reward = global_conflicts.type(th.int) * -1

        reward = self.global_reward_fraction * global_reward + \
            (1-self.global_reward_fraction) * local_reward

        return reward, global_conflicts

    def step(self, actions):
        self.state = actions
        observations = self.observations()
        reward, conflicts = self.get_rewards(observations)
        if (not conflicts and not self.fixed_length) or (self.steps == self.max_steps):
            done = True
        elif self.steps > self.max_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
        self.steps += 1
        self.statetrace.append(self.state)
        self.rewardtrace.append(reward)
        self.conflictstrace.append(conflicts)
        return observations, reward, done, {}

    def reset(self):
        self.new_graph()
        self.steps = 0
        self.statetrace = []
        self.rewardtrace = []
        self.conflictstrace = []
        self.state = th.tensor(
            np.random.randint(0, self.n_actions, self.n_agents), dtype=th.int64)
        return self.observations()

    def render(self, show=True):
        fig = plt.figure()

        # colors = nx.algorithms.coloring.greedy_color(self.graph)
        # colors2 = [colors[i] for i in range(len(colors))]
        # state = get_graph_colors(self.graph)
        observations = self.observations()
        rewards, done = self.get_rewards(observations)
        avg_reward = rewards.mean()

        # bad fix to handle GraphColoringPartiallyFixed
        if hasattr(self, 'agent_pos'):
            labels = {i: f"{i}" for i in range(len(self.graph))}
            labels2 = {i: f"{i}:{r:.2f}" for i,r in zip(self.agent_pos,rewards) }
            labels = {**labels, **labels2}
        else:
            labels = {i: f"{i}:{rewards[i]:.2f}" for i in range(len(self.graph))}
        
        node_color = [self.state[i].item() for i in self.graph.nodes()]
        nx.draw(self.graph, self.graph_pos, node_color=node_color, labels=labels)
        plt.text(0, 0, f"{self.steps}:{avg_reward}", fontsize=30)
        if show:
            plt.savefig('temp.png')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.moveaxis(data, 2, 0)
        plt.close()
        return th.tensor(data[np.newaxis, np.newaxis])

    def close(self):
        pass

    def log(self):



        self.statetrace
        self.rewardtrace
        self.conflictstrace

        # writer.add_metrics(
        #     calc_metrics(rewards, episode_rewards),
        #     {'done': done},
        #     tf=['avg_reward', 'avg_episode_rewards'] if done else [],
        #     details_only=not done
        # )
        # writer.add_frame('{mode}.observations', lambda: env.render(), details_only=True)



def test():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = GraphColoring(6, 'cycle', 5, 0.5, device)
    return env


class GraphColoringPartiallyFixed(GraphColoring):
    def __init__(self, n_agents, fixed_agents, **kwargs):
        self.fixed_agents = fixed_agents
        self._n_agents = n_agents
        self.all_agents = self.fixed_agents + self._n_agents
        super().__init__(n_agents=self.all_agents, **kwargs)
        self.n_agents = n_agents


    def observations(self):
        state_ = self.state.reshape(1,-1).repeat(self.all_agents,1)
        neighbor_colors = th.gather(state_, 1, self.neighbors)
        observations = th.cat((self.state.reshape(-1,1), neighbor_colors), dim=1)
        return observations[self.agent_pos]

    def step(self, actions):
        _actions = self.state.clone()
        _actions[self.agent_pos] = actions
        return super().step(_actions)

    def reset(self):
        self.n_agents = self.all_agents
        self.new_graph()
        self.n_agents = self.all_agents - self.fixed_agents
        self.statetrace = []
        self.rewardtrace = []
        self.conflictstrace = []
        self.steps = 0
        self.agent_pos = np.random.choice(
            self.all_agents, self.n_agents, replace=False)
        self.state = get_graph_colors(self.graph)
        self.state[self.agent_pos] = th.tensor(
            np.random.randint(0, self.n_actions, self._n_agents), dtype=th.int64)
        return self.observations()
