from collections import namedtuple

import math
import random
import itertools
import torch as th
import numpy as np
import pandas as pd

from aci.components.simple_observation_cache import SimpleCache
from aci.utils.array_to_df import using_multiindex, map_columns, to_alphabete


def create_product_map(cache_size, observation_shape, n_actions):

    possible_single_node_traces = list(itertools.product(
        *itertools.repeat(list(range(n_actions)), cache_size)))

    possible_all_node_traces = list(itertools.product(
        *itertools.repeat(possible_single_node_traces, observation_shape[0])))

    lookup = {
        tuple(obs): idx for idx, obs in enumerate(possible_all_node_traces)
    }

    def func(ob):
        return lookup[ob]
    return func, lookup


def create_combinations_map(observation_shape, n_actions):
    n_neighbors = observation_shape[0] - 1
    neighbor_pos = itertools.combinations_with_replacement(list(range(n_actions)), n_neighbors)
    self_pos = list(range(n_actions))
    all_possible_obs = itertools.product(self_pos, neighbor_pos)
    lookup = {
        tuple(obs): idx for idx, obs in enumerate(all_possible_obs)
    }
    def func(ob):
        return lookup[(ob[0], tuple(sorted(ob[1:])))]
    return func, lookup


def get_idx(obs, omap):
    """
    obs input shape: trace, agents, neighbors + self
    """

    _obs = obs.permute(1, 2, 0)

    return [omap(tuple(tuple(ooo.item() for ooo in oo) for oo in o)) for o in _obs]

MAPS = {
    'product': create_product_map,
    'combinations': create_combinations_map
}

class TabularQ:
    def __init__(
            self, observation_shape, n_agents, n_actions, gamma, alpha,
            q_start, obs_map, cache_size, device):
        self.n_actions = n_actions
        self.device = device
        self.obs_idx_map, self.lookup = MAPS[obs_map](cache_size, observation_shape, n_actions)
        self.q_table = th.ones((n_agents, len(self.lookup), n_actions)) * q_start

        print(f'Created lookup table with {len(self.lookup)} entries for each of {n_agents} agents.')
        
        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma
        self.cache_size = cache_size

    def get_q(self, observations):
        historized_obs = self.evalcache.add_get(observations)
        observations_idx = get_idx(historized_obs, self.obs_idx_map)
        q_value = self.q_table[np.arange(self.n_agents), observations_idx]
        return q_value

    def init_episode(self, observations):
        self.traincache = SimpleCache(observations, self.cache_size + 1)
        self.evalcache = SimpleCache(observations, self.cache_size)

    def update(self, actions, observations, rewards, done, writer=None):
        historized_obs = self.traincache.add_get(observations)

        observations_idx = get_idx(historized_obs[:-1], self.obs_idx_map)
        prev_observations_idx = get_idx(historized_obs[1:], self.obs_idx_map)

        old_q_values = self.q_table[np.arange(self.n_agents), prev_observations_idx, actions]
        next_max_q_val = self.q_table[np.arange(self.n_agents), observations_idx].max(-1)[0]
        new_q_value = (1 - self.alpha) * old_q_values + self.alpha * (rewards + self.gamma * next_max_q_val)
        self.q_table[np.arange(self.n_agents), prev_observations_idx, actions] = new_q_value

        if writer and done:
            self.log(writer)

    def log(self, writer):
        if writer.check_on(on='table'):
            print(f'log {str(writer.step)}')
            df = using_multiindex(self.q_table.numpy(), ['agents', 'obs_idx', 'action'])
            df = map_columns(df, obs_idx=self.lookup.keys())
            df = to_alphabete(df, ['agents'])

            df = pd.pivot_table(df, index=['agents', 'obs_idx'], columns='action')
            writer.add_table(name='qtable', df=df, sheet=str(writer.step))
