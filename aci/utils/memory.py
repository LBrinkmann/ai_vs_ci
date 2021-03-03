"""
Memory
"""

import torch as th
import numpy as np
import collections


def create_tensor(example, max_history, episode_steps, device):
    shape = list(example.shape)
    shape[0] = max_history
    if episode_steps is not None:
        shape[1] = episode_steps
    return th.empty(shape, dtype=example.dtype, device=device)


class Memory():
    def __init__(self, max_history, episode_steps, device):
        self.max_history = max_history
        self.episode_steps = episode_steps

        self.history = None
        self.history_queue = None
        self.history_idx = None

        self.device = device
        self.history_queue = collections.deque([], maxlen=self.max_history)

    def init_history(self, data):
        self.history = {
            k: create_tensor(v, self.max_history, self.episode_steps, self.device)
            for k, v in data.items()
        }

    def start_episode(self, episode):
        self.history_idx = episode % self.max_history
        self.history_queue.appendleft(self.history_idx)

    def add(self, data, episode_step=None):
        if self.history is None:
            self.init_history(data)
        assert data.keys() == self.history.keys()
        for k, h in self.history.items():
            if episode_step is None:
                self.history[k][self.history_idx] = data[k].squeeze(0)
            else:
                self.history[k][self.history_idx, episode_step] = data[k].squeeze(0).squeeze(0)

    def last(self, batch_size, **kwargs):
        assert batch_size <= self.max_history
        relative_episodes = np.arange(batch_size)
        return self.get_relative(relative_episodes, **kwargs)

    def random(self, batch_size, horizon, **kwargs):
        eff_horizon = min(len(self), horizon)
        relative_episode = np.random.choice(eff_horizon, batch_size, replace=False)
        return self.get_relative(relative_episode, **kwargs)

    def stride(self, stride, keys=None):
        if keys is None:
            keys = self.history.keys()
        return {k: v[:self.history_idx + 1:stride] for k, v in self.history.items() if k in keys}

    def __len__(self):
        return len(self.history_queue)

    def finished(self):
        return self.history_idx != self.max_history - 1

    def get_relative(self, relative_episode, keys=None):
        if keys is None:
            keys = self.history.keys()
        hist_idx = th.tensor([self.history_queue[rp] for rp in relative_episode], dtype=th.int64)
        return {k: v[hist_idx] for k, v in self.history.items() if k in keys}

    def get(self, sample_type, **kwargs):
        if sample_type == 'last':
            return self.last(**kwargs)
        if sample_type == 'random':
            return self.random(**kwargs)
