"""
Graph Coloring Environment
"""

import random
import torch as th
import numpy as np
import networkx as nx
import matplotlib as mpl
import yaml
import string
from itertools import count
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
        # ci_ai_map: [2, 0, 1], ai agent 0 is ci agent 2
        return {
            'graph': list(nx.to_edgelist(self.graph)),
            'node_of_agent': self.agent_pos.tolist(),
            'agent_at_node': self.node_agent.tolist(),
            'ai_ci_map': self.ai_ci_map.tolist(),
            'ci_ai_map': self.ci_ai_map.tolist()
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
        ci_obs = th.cat((self.state['ci'].reshape(-1,1), neighbor_colors), dim=1)

        if self.ai_obs_mode == 'neighbors':
            ai_obs = ci_obs
        elif self.ai_obs_mode == 'agents_only':
            ai_obs = self.state['ci'].unsqueeze(0)
        elif self.ai_obs_mode == 'agents_matrix':
            raise NotImplementedError()
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
        self.steps += 1
        self.state = actions

        observations = self._observations()
        rewards, info = self._get_rewards(observations)
        if (self.steps == self.max_steps):
            done = True
        elif self.steps > self.max_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
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
        self.ci_ai_map = th.argsort(self.ai_ci_map)

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


# Tests


base_settings = """
ai_obs_mode: neighbors
all_agents: 20
fixed_agents: 0
fixed_length: true
fixed_network: true
fixed_pos: true
fixed_mapping: true
graph_args:
    chromatic_number: 3
    degree: 4
    graph_type: random_regular
max_steps: 50
rewards_args:
    ai:
        avg_catch: 0.5
        avg_coordination: 0
        ind_catch: 0.5
        ind_coordination: 0
    ci:
        avg_catch: 0
        avg_coordination: 0
        ind_catch: 0
        ind_coordination: 1
"""
base_grid = """
all_agents: [6, 20]
fixed_network: [true, false]
fixed_pos: [true, false]
fixed_mapping: [true, false]
"""

def expand(setting, grid):
    settings = [setting]
    labels = [{}]
    for key, values in grid.items():
        settings = [{**s, key: v} for s in settings for v in values]
        labels = [{**l, key: v} for l in labels for v in values]
    return settings, labels


def test():
    setting = yaml.safe_load(base_settings)
    grid = yaml.safe_load(base_grid)
    settings, labels = expand(setting, grid)
    for s, l in zip(settings, labels):
        print(l)
        test_setting(s)


def test_setting(setting):
    device = th.device('cpu')
    env = AdGraphColoring(**setting, device=device)

    # TEST:  correct self observation
    n_agents = setting['all_agents']
    n_colors = setting['graph_args']['chromatic_number']
    ai_color = th.randint(n_colors, size=(n_agents,))
    ci_color = th.randint(n_colors, size=(n_agents,))
    observation, rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )

    # AI does not sees its own color, this is correct, but might create problems for the model.
    assert (observation['ai'][env.ai_ci_map,0] == ci_color).all()
    assert (observation['ci'][:,0] == ci_color).all()

    # TEST:  correct ci neighbor observations

    test_agent = random.randint(0, n_agents-1)

    test_neighbor_colors_1 = th.sort(observation['ci'][test_agent,1:])[0]
    test_neighbors = env.neighbors[test_agent]
    test_neighbor_colors_2 = th.sort(observation['ci'][test_neighbors,0])[0]

    assert (test_neighbor_colors_1 == test_neighbor_colors_2).all(), 'wrong ci neighbor observations'

    # TEST: correct ai neighbor observations
    test_agent = random.randint(0, n_agents-1)

    test_neighbor_colors_1 = th.sort(observation['ai'][test_agent,1:])[0]

    # mapping agent from ai to ci idx
    test_agent_ci = env.ci_ai_map[test_agent] # inverse map to because we map indices
    # getting neighbors using ci idx
    test_neighbors_ci = env.neighbors[test_agent_ci]
    # mapping neighbors from ci to ai idx
    test_neighbors = env.ai_ci_map[test_neighbors_ci] # inverse map to because we map indices

    test_neighbor_colors_2 = th.sort(observation['ai'][test_neighbors,0])[0]

    assert (test_neighbor_colors_1 == test_neighbor_colors_2).all(), 'wrong ai neighbor observations'

    # TEST: ai reward
    if (setting['all_agents'] == 6):
        ai_color = th.tensor([0,1,0,1,2,2])[env.ci_ai_map]
        ci_color = th.tensor([2,2,2,1,2,2])
        expected_rewards = th.tensor([0.25,0.25,0.25, 0.75,0.75,0.75])[env.ci_ai_map]
        observation, rewards, done, info = env.step(
            {'ai': ai_color, 'ci': ci_color}
        )
        assert (rewards['ai'] ==  expected_rewards).all()

    # TEST: ci reward
    test_agent = random.randint(0, n_agents-1)
    test_neighbors = env.neighbors[test_agent]
    test_agents = test_neighbors.tolist() + [test_agent]
    ci_color = th.zeros(n_agents, dtype=th.int64)
    ci_color[test_agents] = th.randperm(len(test_agents))
    observation, rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )

    assert rewards['ci'][test_agent] == 1

    # TEST: step
    env.reset()
    done = False
    for i in count():
        ai_color = th.randint(n_colors, size=(n_agents,))
        ci_color = th.randint(n_colors, size=(n_agents,))
        observation, rewards, done, info = env.step(
            {'ai': ai_color, 'ci': ci_color}
        )
        if done:
            break
    if setting['fixed_length']:
        assert i == setting['max_steps'] - 1
    else:
        assert i <= setting['max_steps'] - 1


if __name__ == "__main__":
    test()    