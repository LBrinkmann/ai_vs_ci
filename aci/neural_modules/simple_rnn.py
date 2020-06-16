import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RecurrentModel(nn.Module):
    def __init__(
            self, observation_shape, n_actions, 
            rnn_input_size, rnn_hidden_size):
        super(RecurrentModel, self).__init__()
        self.linear1 = nn.Linear(observation_shape, rnn_input_size)
        self.gru = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_size)
        self.linear2 = nn.Linear(rnn_hidden_size, n_actions)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_input_size = rnn_input_size

    def forward(self, x, h):
        _x = F.relu(self.linear1(x))
        _x, _h = self.gru(_x, h)
        _x = self.linear2(_x)
        return _x, _h

    def init_hidden(self, batch_size):
        return th.zeros(1, batch_size, self.rnn_hidden_size, device=self.linear1.weight.device)

    def log(self, writer, prefix='RecurrentModel'):
        writer.write_module(self.linear1, f'{prefix}.linear1', details_only=True)
        writer.write_module(self.linear2, f'{prefix}.linear2', details_only=True)
        writer.write_module(self.gru, f'{prefix}.gru', details_only=True)
