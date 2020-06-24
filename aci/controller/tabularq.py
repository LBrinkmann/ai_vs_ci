from collections import namedtuple

import math
import random
import itertools
import torch as th
import numpy as np


def create_product_map(observation_shape, n_actions):
    all_possible_obs = itertools.product(
        *itertools.repeat(list(range(n_actions)), observation_shape[0]))
    lookup = {
        tuple(obs): idx for idx, obs in enumerate(all_possible_obs)
    }
    def func(ob):
        return lookup[ob]
    return func, len(lookup)


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
    return func, len(lookup)

    
def get_idx(obs, omap):
    return [omap(tuple(oo.item() for oo in o)) for o in obs]

MAPS = {
    'product': create_product_map,
    'combinations': create_combinations_map
}

class TabularQ:
    def __init__(
            self, observation_shape, n_agents, n_actions, gamma, alpha, q_start, obs_map, device):
        self.n_actions = n_actions
        self.device = device
        self.obs_idx_map, n_idx = MAPS[obs_map](observation_shape, n_actions)
        self.q_table = th.ones((n_agents, n_idx, n_actions)) * q_start
        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, observations):
        assert observations.shape[0] == 1
        observations = observations.squeeze(0)
        observations_idx = get_idx(observations, self.obs_idx_map)
        return self.q_table[np.newaxis, np.arange(self.n_agents), observations_idx]

    def optimize(self, prev_observations, actions, observations, rewards, done):
        prev_observations_idx = get_idx(prev_observations, self.obs_idx_map)
        observations_idx = get_idx(observations, self.obs_idx_map)
        old_q_values = self.q_table[np.arange(self.n_agents), prev_observations_idx, actions]
        next_max_q_val = self.q_table[np.arange(self.n_agents), observations_idx].max(-1)[0]
        new_q_value = (1 - self.alpha) * old_q_values + self.alpha * (rewards + self.gamma * next_max_q_val)
        self.q_table[np.arange(self.n_agents), prev_observations_idx, actions] = new_q_value

    def init_episode(self, observations):
        self.prev_observations_idx = get_idx(observations, self.obs_idx_map)

    def update(self, actions, observations, rewards, done):
        observations_idx = get_idx(observations, self.obs_idx_map)
        old_q_values = self.q_table[np.arange(self.n_agents), self.prev_observations_idx, actions]
        next_max_q_val = self.q_table[np.arange(self.n_agents), observations_idx].max(-1)[0]
        new_q_value = (1 - self.alpha) * old_q_values + self.alpha * (rewards + self.gamma * next_max_q_val)
        self.q_table[np.arange(self.n_agents), self.prev_observations_idx, actions] = new_q_value        
        self.prev_observations_idx = observations_idx

    def log(self, writer, details):
        if details:
            tt = self.q_table.sum(-1).sum(0)
            tt = tt.abs() > 0.00001
            # print(self.q_table[:, tt])
            # print(tt.sum())