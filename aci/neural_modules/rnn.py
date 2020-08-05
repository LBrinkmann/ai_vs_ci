import torch as th
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce 

def prod(l):
    return reduce((lambda x, y: x * y), l)



class RNN(nn.Module):
    def __init__(self, batch_size, n_agents, observation_shape, n_actions, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.ModuleList([
            nn.RNNCell(input_size=prod(observation_shape), hidden_size=hidden_size)
            for i in range(n_agents)
        ])
        self.linear = nn.Conv1d(n_agents*hidden_size, n_agents * n_actions, kernel_size=1, groups=n_agents)

        self.batch_size = batch_size
        self.n_agents = n_agents
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.hidden = th.zeros(self.batch_size, self.n_agents, self.hidden_size)

    def reset(self):
        self.hidden.zero_()

    def forward(self, x):
        """
        x: batch x agents x *observation_shape
        returns: batch x agents x actions
        """
        _x = x.reshape(self.batch_size, self.n_agents, prod(self.observation_shape))

        _new_hidden = [self.rnn[j](_x[:,j], self.hidden[:,j]) for j in range(self.n_agents)]

        _new_hidden = th.stack(_new_hidden, dim=1)

        self.hidden = _new_hidden.detach()

        # for j in range(self.n_agents):
        #     self.hidden[:,j] = self.rnn[j](_x[:,j], self.hidden[:,j].detach())
        _h = _new_hidden.reshape(self.batch_size, self.n_agents*self.hidden_size, 1)
        _q = self.linear(_h)
        q = _q.reshape(-1, self.n_agents, self.n_actions)
        return q

    def log(self, writer, prefix='LinearFunction'):
        pass
        # writer.write_module(self.linear, f'{prefix}.01.Linear', details_only=True)
        # writer.add_image('linear.weights', self.linear.weight.unsqueeze(0), details_only=True)