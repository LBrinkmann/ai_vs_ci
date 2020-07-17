import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearFunction(nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(LinearFunction, self).__init__()
        self.linear = nn.Linear(observation_shape, n_actions)

    def forward(self, x):
        q = self.linear(x)
        return q

    def log(self, writer, prefix='LinearFunction'):
        writer.write_module(self.linear, f'{prefix}.01.Linear', details_only=True)
        writer.add_image('linear.weights', self.linear.weight.unsqueeze(0), details_only=True)