from collections import namedtuple
import numpy as np
import torch as th
import torch.optim as optim

class ReplayMemory(object):
    def __init__(self, n_seq, seq_length, batch_size, n_agents, observation_shape):
        self.n_seq = n_seq
        self.seq_length = seq_length
        self.valid_seq = np.zeros(n_seq)
        self.batch_size = batch_size
        self.current_seq = None
        self.current_seq_pos = None

        self.memory = {
            'observations': th.zeros((seq_length + 1, n_seq, n_agents, *observation_shape), dtype=th.float32),
            'done': th.zeros(seq_length, n_seq, dtype=th.bool),
            'actions': th.zeros((seq_length, n_seq, n_agents), dtype=th.long),
            'rewards': th.zeros((seq_length, n_seq, n_agents), dtype=th.float32),
            'valid': th.zeros(seq_length, n_seq, dtype=th.bool)
        }

    def next_seq(self, prev_observations):
        if self.current_seq is not None:
            self.valid_seq[self.current_seq] = 1
            self.current_seq = self.current_seq + 1
        else:
            self.current_seq = 0
        
        self.current_seq_pos = 0
        for v in self.memory.values():
            v[self.current_seq].zero_()
        self.memory['observations'][self.current_seq,0] = prev_observations

    def push(self, observations, actions=None,  rewards=None, done=False):
        s_idx = self.current_seq
        p_idx = self.current_seq_pos
        self.memory['observations'][p_idx + 1, s_idx,] = observations
        self.memory['rewards'][p_idx, s_idx] = rewards
        self.memory['done'][p_idx, s_idx] = done
        self.memory['actions'][p_idx, s_idx] = actions
        self.memory['valid'][p_idx, s_idx] = True

        if self.current_seq_pos + 1 == self.seq_length:
            self.next_seq(observations)
        else:
            self.current_seq_pos = 0


    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert sum(self.valid_seq) >= batch_size, 'less elements in memory then batch size'
        p = (self.valid_seq/self.valid_seq.sum())
        sample_idx = np.random.choice(
            self.n_seq, size=batch_size, replace=False, p=p)
        observations = self.memory['observations'][1:,sample_idx]
        prev_observations = self.memory['observations'][:-1,sample_idx]
        done = self.memory['done'][:,sample_idx]
        rewards = self.memory['rewards'][:,sample_idx]
        actions = self.memory['actions'][:,sample_idx]
        valid = self.memory['valid'][:,sample_idx]

        return prev_observations, actions, observations, rewards, done, valid


    def can_sample(self):
        return self.valid_seq.sum() > self.batch_size

