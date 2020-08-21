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


def create_combinations_map(cache_size, observation_shape, n_actions):
    n_neighbors = observation_shape[0] - 1

    possible_single_node_traces = list(itertools.product(
        *itertools.repeat(list(range(n_actions)), cache_size)))

    neighbor_pos = itertools.combinations_with_replacement(possible_single_node_traces, n_neighbors)
    self_pos = possible_single_node_traces
    all_possible_obs = itertools.product(self_pos, neighbor_pos)
    lookup = {
        tuple(obs): idx for idx, obs in enumerate(all_possible_obs)
    }

    def func(ob):
        return lookup[(ob[0], tuple(sorted(ob[1:])))]
    return func, lookup


def discretisation(cache_size, observation_shape, n_actions, n_bins, minmax):
    assert len(minmax) == observation_shape[0]
    minmax = np.array(minmax)
    totrange = minmax[:,1] - minmax[:,0] + 0.0000001

    def func(ob):
        disc = tuple(tuple(int(((oo - minmax[i,0]) / totrange[i]) * n_bins) for j, oo in enumerate(o)) for i, o in enumerate(ob))
        return disc
    return func


def discrete_product(cache_size, observation_shape, n_actions, n_bins, minmax):
    dis = discretisation(cache_size, observation_shape, n_actions, n_bins, minmax)
    lfunc, lookup = create_product_map(cache_size, observation_shape, n_bins)
    def func(ob):
        ob = dis(ob)
        return lfunc(ob)
    return func, lookup


def get_idx(obs, omap):
    """
    obs input shape: trace, agents, neighbors + self
    obs input shape: agents, trace, neighbors + self
    """

    _obs = obs.permute(0, 2, 1)

    return [omap(tuple(tuple(ooo.item() for ooo in oo) for oo in o)) for o in _obs]


MAPS = {
    'product': create_product_map,
    'combinations': create_combinations_map,
    'discrete_product': discrete_product
}


class TabularQ:
    def __init__(
            self, observation_shape, n_agents, n_actions, gamma, alpha,
            q_start, obs_map, cache_size, device, share_table=False, map_args={}):
        self.n_actions = n_actions
        self.device = device
        self.obs_idx_map, self.lookup = MAPS[obs_map](cache_size, observation_shape, n_actions, **map_args)

        n_q_tables = 1 if share_table else n_agents
        self.q_table = th.ones((n_q_tables, len(self.lookup), n_actions), device=device) * q_start

        print(f'Created lookup table with {len(self.lookup)} entries for each of {n_agents} agents.')
        print(cache_size)

        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma
        self.cache_size = cache_size
        self.share_table = share_table

    def get_q(self, observations, training=False):
        """ Retrieves q values for all possible actions. 

        If observations are given, they are used. Otherwise last observations are used.

        """
        if training:
            historized_obs = self.cache.get()
        else:
            historized_obs = self.cache.add_get(observations)

        observations_idx = get_idx(historized_obs, self.obs_idx_map)
        q_table_idxs = np.zeros(self.n_agents, dtype=int) if self.share_table else np.arange(self.n_agents)
        q_value = self.q_table[q_table_idxs, observations_idx]
        return q_value

    def init_episode(self, observations, episode, training):
        self.cache = SimpleCache(observations, self.cache_size)
        # self.evalcache = SimpleCache(observations, self.cache_size)

    def update(self, actions, observations, rewards, done, writer=None):

        prev_observations_idx = get_idx(self.cache.get(), self.obs_idx_map)
        observations_idx = get_idx(self.cache.add_get(observations), self.obs_idx_map)

        q_table_idxs = np.zeros(self.n_agents, dtype=int) if self.share_table else np.arange(self.n_agents)
        old_q_values = self.q_table[q_table_idxs, prev_observations_idx, actions]
        next_max_q_val = self.q_table[q_table_idxs, observations_idx].max(-1)[0]
        new_q_value = (1 - self.alpha) * old_q_values + self.alpha * (rewards + self.gamma * next_max_q_val)

        if self.share_table:
            # currently this is ill behaving. There could be two subagents with the same prev_obs_id and actions, and it
            # is not clear of which agent the value is taken. 
            self.q_table[0, prev_observations_idx, actions] = new_q_value
        else:
            self.q_table[q_table_idxs, prev_observations_idx, actions] = new_q_value

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
