from collections import namedtuple
import numpy as np
import torch as th
import torch.optim as optim
import random


class FixedEpisodeMemory(object):
    def __init__(self, capacity, observation_shape, n_agents, n_actions, episode_length):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.episode_length = episode_length
        self.episode = -1
        self.memory = {
            'observations': th.zeros((self.n_agents, self.capacity, self.episode_length + 1, *self.observation_shape), dtype=th.int64),
            'actions': th.zeros((self.n_agents, self.capacity, self.episode_length), dtype=th.int64),
            'rewards': th.zeros((self.n_agents, self.capacity, self.episode_length), dtype=th.float32)
        }

    def init_episode(self, observations):
        self.episode += 1
        self.episode_step = 0
        self.current_idx = self.episode % self.capacity
        self.memory['observations'][:, self.current_idx, self.episode_step] = observations

    def push(self, observations, actions=None,  rewards=None):
        self.memory['observations'][:, self.current_idx, self.episode_step + 1] = observations
        self.memory['actions'][:, self.current_idx, self.episode_step] = actions
        self.memory['rewards'][:, self.current_idx, self.episode_step] = rewards
        self.episode_step += 1

    def sample(self, batch_size):
        assert len(self) >= batch_size, 'less elements in memory then batch size'
        idx = th.randperm(len(self))[:batch_size]
        observations = self.memory['observations'][:,idx,1:]
        prev_observations = self.memory['observations'][:,idx,:-1]
        actions = self.memory['actions'][:,idx]
        rewards = self.memory['rewards'][:,idx]

        return prev_observations, observations, actions, rewards

    def __len__(self):
            return min(self.episode, self.capacity)