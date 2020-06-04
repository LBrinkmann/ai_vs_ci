import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedLogisticRegression(nn.Module):

    def __init__(self, observation_shape, n_agents, n_actions):
        super(SharedLogisticRegression, self).__init__()
        self.linear = nn.Linear(observation_shape, n_actions)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        q = self.linear(x)
        return q

    def log(self, writer, step, prefix='SharedLogisticRegression'):
        write_module(self.linear, f'{prefix}.01.Linear', writer, step)

def write_module(module, m_name, writer, step):
    for p_name, values in module.named_parameters():
        writer.add_histogram(f'{m_name}.{p_name}', values, step)