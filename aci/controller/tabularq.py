from collections import namedtuple

import math
import random
import itertools
import torch as th
import numpy as np

from aci.components.simple_observation_cache import SimpleCache


def create_product_map(cache_size, observation_shape, n_actions):
    all_possible_obs = itertools.product(
        *itertools.repeat(list(range(n_actions)), observation_shape[0]))
    lookup = {
        tuple(obs): idx for idx, obs in enumerate(all_possible_obs)
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
    return [omap(tuple(oo.item() for oo in o)) for o in obs]

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
        self.obs_idx_map, self.lookup = MAPS[obs_map](observation_shape, n_actions)
        self.q_table = th.ones((n_agents, len(self.lookup), n_actions)) * q_start
        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, observations):
        historized_obs = self.evalcache.add_get(observations)
        observations_idx = get_idx(historized_obs, self.obs_idx_map)
        q_value = self.q_table[np.arange(self.n_agents), observations_idx]
        return q_value

    def init_episode(self, observations):
        self.traincache = SimpleCache(observations, self.cache_size + 1)
        self.evalcache = SimpleCache(observations, self.cache_size)

    def update(self, actions, observations, rewards, done):
        historized_obs = self.traincache.add_get(observations)
        observations_idx = get_idx(historized_obs[:-1], self.obs_idx_map)
        prev_observations_idx = get_idx(observations[1:], self.obs_idx_map)

        old_q_values = self.q_table[np.arange(self.n_agents), prev_observations_idx, actions]
        next_max_q_val = self.q_table[np.arange(self.n_agents), observations_idx].max(-1)[0]
        new_q_value = (1 - self.alpha) * old_q_values + self.alpha * (rewards + self.gamma * next_max_q_val)
        self.q_table[np.arange(self.n_agents), prev_observations_idx, actions] = new_q_value

    def log(self, writer, details):
        if details:
            # tt = self.q_table.sum(-1).sum(0)
            # tt = tt.abs() > 0.00001
            for k, v in self.lookup.items():
                print(k, self.q_table[:, v].argmax(1))
