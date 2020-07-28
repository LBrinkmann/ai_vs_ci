"""
Graph Coloring Environment
"""

import random
import torch as th
import numpy as np
import networkx as nx
import matplotlib as mpl
import string
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
        if not nx.is_connected(graph):
            continue
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

def int_to_alphabete(val):
    return string.ascii_uppercase[val]


class AdGraphColoring():
    def __init__(
            self, n_agents, all_agents, graph_args, max_steps, fixed_length, rewards_args, device,
            fixed_pos, fixed_network):
        self.steps = 0
        self.max_steps = max_steps
        self.n_agents = {'ci': n_agents, 'ai': n_agents}

        self.all_agents = all_agents
        self.graph_args = graph_args
        self.device = device
        self.fixed_length = fixed_length
        self.fixed_pos, self.fixed_network = fixed_pos, fixed_network
        self.rewards_args = rewards_args

        self.new_graph()
        self.reset(init=True)

    def new_graph(self):
        self.graph, n_actions, self.n_neighbors = create_graph(
            n_nodes=self.all_agents, **self.graph_args)
        self.graph_pos = nx.spring_layout(self.graph)
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)

        self.neighbors = th.tensor([
            random.sample(self.graph[n].keys(), len(self.graph[n])) 
            for n in range(len(self.graph))
        ], dtype=th.int64, device=self.device) # n_agents, n_neighbors
        self.observation_shape = {
            'ai': (self.all_agents, (self.all_agents, self.all_agents)),
            'ci': (self.n_neighbors + 1,)
        }
        self.n_actions = {
            'ai': n_actions,
            'ci': n_actions
        }

    def observations(self):
        state_ = self.state['ci'].reshape(1,-1).repeat(self.all_agents,1)
        neighbor_colors = th.gather(state_, 1, self.neighbors)
        observations = th.cat((self.state['ci'].reshape(-1,1), neighbor_colors), dim=1)
        return {
            'ai': [(self.state['ci'], self.adjacency_matrix)],
            'ci': observations[self.agent_pos]
        }

    def get_rewards(self, observations):
        conflicts = (observations['ci'][:,[0]] == observations['ci'][:,1:])

        ind_coordination = 1 - conflicts.any(dim=1).type(th.float)
        ind_catch = (
            self.state['ai'][self.agent_pos] == self.state['ci'][self.agent_pos]).type(th.float)

        metrics = {
            'ind_coordination': ind_coordination,
            'avg_coordination': ind_coordination.mean(0, keepdim=True).expand(self.n_agents['ci']),
            'ind_catch': ind_catch,
            'avg_catch': ind_catch.mean(0, keepdim=True).expand(self.n_agents['ci']),
        }

        # cat = [metrics[k] * v for k,v in self.rewards_args['ci'].items()]
        # cat = th.cat(cat)
        ci_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ci'].items()]).sum(0)

        ai_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ai'].items()]).sum(0).sum(0, keepdim=True)

        assert ai_reward.shape[0] == 1

        reward = {
            'ai': ai_reward,
            'ci': ci_reward
        }

        return reward, {**metrics, **reward}


    def step(self, actions, writer=None):
        if 'ai' in actions:
            self.state['ai'] = actions['ai']
        if 'ci' in actions:
            self.state['ci'][self.agent_pos] = actions['ci']

        observations = self.observations()
        rewards, info = self.get_rewards(observations)
        if (self.steps == self.max_steps):
            done = True
        elif self.steps > self.max_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
        self.steps += 1

        if writer:
            self._log(observations, info, done, writer) 

        return observations, rewards, done, {}

    def reset(self, init=False):
        if not self.fixed_network or init:
            self.new_graph()
        self.steps = 0

        self.agg_metrics = {}

        if not self.fixed_pos or init:
            self.agent_pos = np.random.choice(
                self.all_agents, self.n_agents['ci'], replace=False)
        _state = get_graph_colors(self.graph)
        self.state = {
            'ci': _state,
            'ai': _state
        }
        self.state['ci'][self.agent_pos] = th.tensor(
            np.random.randint(0, self.n_actions['ci'], self.n_agents['ci']), dtype=th.int64)
        self.state['ai'][self.agent_pos] = th.tensor(
            np.random.randint(0, self.n_actions['ai'], self.n_agents['ci']), dtype=th.int64)
        return self.observations()

    def _render(self, state, rewards):
        fig = plt.figure()

        avg_ai_reward = rewards['ai'].mean()
        avg_ci_reward = rewards['ci'].mean()

        labels = {i: f"{int_to_alphabete(i)}" for i in range(len(self.graph))}
        labels2 = {
            i: f"{int_to_alphabete(i)}:{r:.2f}" for i,r in zip(self.agent_pos, rewards['ci']) }
        labels = {**labels, **labels2}
        
        node_color_ci = [state['ci'][i].item() for i in self.graph.nodes()]        
        node_color_ai = [state['ai'][i].item() for i in self.graph.nodes()]
        nx.draw(self.graph, self.graph_pos, node_color=node_color_ai, node_size=500)
        nx.draw(
            self.graph, self.graph_pos, node_color=node_color_ci, 
            labels=labels, node_size=200, edgelist=[], font_size=20, font_color='magenta')

        plt.text(0, 0, f"{self.steps}:{avg_ai_reward}:{avg_ci_reward}", fontsize=30, color='magenta')
        # plt.text(-0.8, -0.5, f"s: {[i.item() for i in state['ci']]}", fontsize=15)
        # plt.text(-0.8, -0.7, f"nc ci: {node_color_ci}", fontsize=15)
        # plt.text(-0.8, -0.9, f"nc ai: {node_color_ai}", fontsize=15)
        fig.canvas.draw()
        # plt.savefig('temp.png')
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.moveaxis(data, 2, 0)
        plt.close()
        return th.tensor(data[np.newaxis, np.newaxis])

    def close(self):
        pass

    def _log(self, observations, metrics, done, writer):
        if len(self.agg_metrics) == 0:
            self.agg_metrics = metrics
        else:
            self.agg_metrics = {k: self.agg_metrics[k] + v for k, v in metrics.items()}

        writer.add_metrics(
            'trace',
            {
                'ai_reward': metrics['ai'],
                'sum_ci_reward':  metrics['ci'].sum(),
                'std_ci_reward':  metrics['ci'].std(),
                'avg_coordination': metrics['avg_coordination'][0],
                'avg_catch': metrics['avg_catch'][0],
            },
            {},
            tf=[],
            on='trace'
        )

        if done:
            writer.add_metrics(
                'final',
                {
                    'ai_reward': self.agg_metrics['ai'],
                    'sum_ci_reward':  self.agg_metrics['ci'].sum(),
                    'std_ci_reward':  self.agg_metrics['ci'].std(),
                    'avg_coordination': self.agg_metrics['avg_coordination'][0],
                    'avg_catch': self.agg_metrics['avg_catch'][0],
                },
                {},
                tf=[],
                on='final'
            )

            if writer.check_on(on='individual_trace'):
                for i in range(self.n_agents['ci']):
                    writer.add_metrics(
                        'individual_trace',
                        {
                            'ci_reward': self.agg_metrics['ci'][i],
                            'ind_coordination': self.agg_metrics['ind_coordination'][i],
                            'ind_catch': self.agg_metrics['ind_catch'][i],
                        },
                        {'done': done, 'agent': int_to_alphabete(i)},
                        tf=[],
                        on='individual_trace'
                    )

        writer.add_frame('{mode}.observations', lambda: self._render(self.state, rewards), on='video', flush=done)



