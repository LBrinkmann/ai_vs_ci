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
            self, all_agents, graph_args, max_steps, fixed_length, fixed_mapping, rewards_args, device,
            fixed_pos, fixed_network, ai_obs_mode='agents_matrix', fixed_agents=0):
        self.steps = 0
        self.max_steps = max_steps

        self.fixed_agents = fixed_agents
        self.all_agents = all_agents
        self.graph_args = graph_args
        self.device = device
        self.fixed_length = fixed_length
        self.fixed_pos, self.fixed_network, self.fixed_mapping = fixed_pos, fixed_network, fixed_mapping
        self.rewards_args = rewards_args
        self.ai_obs_mode = ai_obs_mode

        self.traces = {}

        self._new_graph()
        self._reset(init=True)

    def to_dict(self):
        # agent_pos: [c, a, b], agent c is positioned on node 0
        return {
            'graph': list(nx.to_edgelist(self.graph)),
            'node_of_agent': self.agent_pos.tolist(),
            'agent_at_node': self.node_agent.tolist()
        }


    def _calc_neighbors(self, random_pos=False):
        if random_pos:
            self.agent_pos = th.randperm(self.all_agents)
        else:
            self.agent_pos = th.arange(self.all_agents)
        
        self.node_agent = th.argsort(self.agent_pos)

        node_neighbors = [
            list(self.graph[i].keys())
            for i in range(len(self.graph))
        ]
        neighbors = [
            self.node_agent[node_neighbors[ap]][th.randperm(len(node_neighbors[ap]))]
            for ap in self.agent_pos
        ]
        self.neighbors = th.stack(neighbors).to(self.device) # n_agents, n_neighbors


    def _new_graph(self):
        self.graph, n_actions, self.n_neighbors = create_graph(
            n_nodes=self.all_agents, **self.graph_args)
        self.graph_pos = nx.spring_layout(self.graph)

        ci_obs_shape = (self.n_neighbors + 1,)
        ci_agents = self.all_agents - self.fixed_agents
        if self.ai_obs_mode == 'neighbors':
            ai_obs_shape = ci_obs_shape
            ai_agents = ci_agents
        # elif self.ai_obs_mode == 'agents_only':
        #     ai_obs_shape = (self.all_agents,)
        #     ai_agents = 1
        # elif self.ai_obs_mode == 'agents_matrix':
        #     ai_obs_shape = (self.all_agents,(self.all_agents, self.all_agents))
        #     ai_agents = None
        else:
            raise NotImplementedError(f'AI observation mode is not implemented yet.')

        self.n_agents = {'ci': ci_agents, 'ai': ai_agents}

        self.observation_shape = {
            'ai': ai_obs_shape,
            'ci': ci_obs_shape
        }
        self.n_actions = {
            'ai': n_actions,
            'ci': n_actions
        }

    def _map_incoming(self, values):
        return {'ai': values['ai'][self.ai_ci_map], 'ci': values['ci']}

    def _map_outgoing(self, values):
        return {'ai': values['ai'][self.ci_ai_map], 'ci': values['ci']}

    def _observations(self):
        state_ = self.state['ci'].reshape(1,-1).repeat(self.all_agents,1)
        neighbor_colors = th.gather(state_, 1, self.neighbors)
        observations = th.cat((self.state['ci'].reshape(-1,1), neighbor_colors), dim=1)

        ci_obs = observations[self.agent_pos]
        if self.ai_obs_mode == 'neighbors':
            ai_obs = ci_obs
        elif self.ai_obs_mode == 'agents_only':
            ai_obs = self.state['ci'].unsqueeze(0)
        elif self.ai_obs_mode == 'agents_matrix':
            raise NotImplementedError()
            # ai_obs = [(self.state['ci'], self.adjacency_matrix)]
        else:
            raise NotImplementedError(f'AI observation mode is not implemented yet.')

        return {
            'ai': ai_obs,
            'ci': ci_obs
        }

    def _get_rewards(self, observations):
        conflicts = (observations['ci'][:,[0]] == observations['ci'][:,1:])

        ind_coordination = 1 - conflicts.any(dim=1).type(th.float)
        ind_catch = (self.state['ai'] == self.state['ci']).type(th.float)

        metrics = {
            'ind_coordination': ind_coordination,
            'avg_coordination': ind_coordination.mean(0, keepdim=True).expand(self.n_agents['ci']),
            'ind_catch': ind_catch,
            'avg_catch': ind_catch.mean(0, keepdim=True).expand(self.n_agents['ci']),
        }

        ci_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ci'].items()]).sum(0)
        ai_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ai'].items()]).sum(0)
        
        if self.ai_obs_mode == 'agents_matrix':
            ai_reward = ai_reward.sum(0, keepdim=True)            

        reward = {
            'ai': ai_reward,
            'ci': ci_reward
        }

        return reward, metrics


    def _step(self, actions):
        self.state = actions

        observations = self._observations()
        rewards, info = self._get_rewards(observations)
        if (self.steps == self.max_steps):
            done = True
        elif self.steps > self.max_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
        self.steps += 1
        return observations, rewards, done, info

    def _reset(self, init=False):
        if not self.fixed_network or init:
            self._new_graph()
            self._calc_neighbors(random_pos=not self.fixed_pos)
        if not self.fixed_pos:
            self._calc_neighbors(random_pos=not self.fixed_pos)
        if not self.fixed_mapping:
            self.ai_ci_map = th.randperm(self.all_agents)
        elif init:
            self.ai_ci_map = th.arange(self.all_agents)
        self.ci_ai_map = th.argsort(self.agent_pos)

        self.steps = 0

        if self.fixed_agents:
            # TODO: current fixed agents are not supported
            raise NotImplementedError()
            # get a solution
            # _state = get_graph_colors(self.graph)
        else:
            # random starting state
            ci_start_state = th.tensor(
                np.random.randint(0, self.n_actions['ci'], self.n_agents['ci']), dtype=th.int64, device=self.device)
            ai_start_state = th.tensor(
                np.random.randint(0, self.n_actions['ai'], self.n_agents['ai']), dtype=th.int64, device=self.device)

        self.state = {
            'ci': ci_start_state,
            'ai': ai_start_state
        }
        return self._observations()

    def step(self, actions):
        observations, rewards, done, info = self._step(self._map_incoming(actions))
        return (
            self._map_outgoing(observations), 
            self._map_outgoing(rewards), 
            done, 
            info
        )

    def reset(self):
        observations = self._reset()
        return self._map_outgoing(observations)

    def close(self):
        pass
