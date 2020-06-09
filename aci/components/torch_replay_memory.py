from collections import namedtuple
import numpy as np
import torch as th
import torch.optim as optim

class ReplayMemory(object):
    def __init__(self, capacity, observation_shape, n_actions, n_agents):
        self.memory = {
            'observations': th.zeros((capacity, n_agents, *observation_shape), dtype=th.float32),
            'done': th.zeros(capacity, dtype=th.bool),
            'actions': th.zeros((capacity, n_agents), dtype=th.long),
            'rewards': th.zeros((capacity, n_agents), dtype=th.float32)
        }
        self.valid = th.zeros(capacity, dtype=th.long)
        self.previous_map = th.tensor(
            [capacity - 1] + list(range(capacity - 1)), dtype=th.long)
        self.added = 0
        self.capacity = capacity

    def push(self, observations, actions=None,  rewards=None, done=False):
        if rewards is None:
            assert actions is None
            rewards = th.zeros_like(self.memory['rewards'][0])
            actions = th.zeros_like(self.memory['actions'][0])
            valid = 0
        else:
            valid = 1
        position = (self.added) % self.capacity
        self.memory['observations'][position] = observations
        self.memory['rewards'][position] = rewards
        self.memory['done'][position] = done
        self.memory['actions'][position] = actions
        self.valid[position] = valid
        self.added += 1

    def sample(self, batch_size):
        assert sum(self.valid) >= batch_size, 'less elements in memory then batch size'
        valid = self.valid.cpu().numpy()
        p = (valid/valid.sum())
        sample_idx = np.random.choice(
            self.capacity, size=batch_size, replace=False, p=p)
        prev_idx = self.previous_map[sample_idx]
        observations = self.memory['observations'][sample_idx]
        prev_observations = self.memory['observations'][prev_idx]
        done = self.memory['done'][sample_idx]
        rewards = self.memory['rewards'][sample_idx]
        actions = self.memory['actions'][sample_idx]
        return prev_observations, actions, observations, rewards, done

    def __len__(self):
        return min(sum(self.valid), self.capacity)

