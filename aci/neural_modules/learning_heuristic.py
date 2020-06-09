import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import write_module
import structlog

log = structlog.get_logger()

class LearningHeuristic(nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(LearningHeuristic, self).__init__()
        self.n_actions = n_actions
        self.linear = nn.Linear(n_actions, n_actions)

    def forward(self, x):
        one_hot = torch.zeros(x.shape + (self.n_actions,))
        one_hot = one_hot.scatter_(2, x.unsqueeze(2), 1)

        other = one_hot[:, 1:].sum(1)
        this = one_hot[:, 0]

        q = self.linear(other)
        return q

    def log(self, writer, step, prefix='LogisticRegression'):
        write_module(self.linear, f'{prefix}.01.Linear', writer, step)
        writer.add_image('linear.weights', self.linear.weight.unsqueeze(0), step)
