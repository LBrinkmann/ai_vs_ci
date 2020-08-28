import torch.nn as nn
import torch.nn.functional as F
import torch as th


class GRUAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size, device):
        super(GRUAgent, self).__init__()
        self.linear1 = nn.Linear(in_features=input_shape*n_actions, out_features=hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=n_actions)

    def reset(self):
        self.hidden = None

    def forward(self, inputs, mask):
        inputs = inputs.reshape(*inputs.shape[:2],-1)
        x = F.relu(self.linear1(inputs))
        h, self.hidden = self.rnn(x, self.hidden)
        q = self.linear2(h)
        return q
