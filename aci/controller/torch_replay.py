from collections import namedtuple
import numpy as np
import torch as th
import torch.optim as optim

class ReplayMemory(object):
    def __init__(self, capacity, observation_shape, action_shape, reward_shape):
        self.memory = {
            'observations': th.zeros((capacity, *observation_shape), dtype=th.float32),
            'done': th.zeros(capacity, dtype=th.bool),
            'actions': th.zeros((capacity, *action_shape), dtype=th.long),
            'rewards': th.zeros((capacity, *reward_shape), dtype=th.float32)
        }
        self.previous_map = th.tensor(
            [capacity - 1] + list(range(capacity - 1)), dtype=th.long)
        self.added = 0
        self.capacity = capacity

    def push(self, observations, rewards, done):
        position = (self.added) % self.capacity
        self.memory['observations'][position] = observations
        self.memory['rewards'][position] = rewards
        self.memory['done'][position] = done
        self.added += 1

    def sample(self, batch_size):
        assert len(self) >= batch_size, 'less elements in memory then batch size'
        sample_idx = np.random.choice(len(self), size=batch_size, replace=False)
        prev_idx = self.previous_map[sample_idx]
        observations = self.memory['observations'][sample_idx]
        prev_observations = self.memory['observations'][prev_idx]
        done = self.memory['done'][sample_idx]
        rewards = self.memory['rewards'][sample_idx]
        actions = self.memory['actions'][sample_idx]
        return prev_observations, actions, observations, rewards, done

    def __len__(self):
        return min(self.added, self.capacity)

