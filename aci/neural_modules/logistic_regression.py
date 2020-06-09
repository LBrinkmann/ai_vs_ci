import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(observation_shape, n_actions)

    def forward(self, x):
        q = self.linear(x)
        return q

    def log(self, writer, step, prefix='LogisticRegression'):
        write_module(self.linear, f'{prefix}.01.Linear', writer, step)

def write_module(module, m_name, writer, step):
    for p_name, values in module.named_parameters():
        writer.add_histogram(f'{m_name}.{p_name}', values, step)