import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce 

def prod(l):
    return reduce((lambda x, y: x * y), l)



class LinearFunction(nn.Module):
    def __init__(self, n_agents, observation_shape, n_actions):
        super(LinearFunction, self).__init__()
        self.in_channels = n_agents * prod(observation_shape)
        self.out_channels = n_agents * n_actions
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.linear = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, groups=self.n_agents)

    def forward(self, x):
        """
        x: batch x agents x *observation_shape
        returns: batch x agents x actions
        """
        _x = x.reshape(-1, self.in_channels, 1)
        _q = self.linear(_x)
        q = _q.reshape(-1, self.n_agents, self.n_actions)
        return q

    def log(self, writer, prefix='LinearFunction'):
        writer.write_module(self.linear, f'{prefix}.01.Linear', details_only=True)
        writer.add_image('linear.weights', self.linear.weight.unsqueeze(0), details_only=True)