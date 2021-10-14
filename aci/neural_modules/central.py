from os import X_OK
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class CentralAgent(nn.Module):
    def __init__(self, view_size, n_actions, hidden_size, n_agents, rnn1, linear2, device, **kwargs):
        super(CentralAgent, self).__init__()
        self.n_actions = n_actions
        #  view_size, control_size, n_actions
        self.linear1 = nn.Linear(
            in_features=view_size*n_agents, out_features=hidden_size)
        if rnn1:
            self.rnn1 = nn.GRU(input_size=hidden_size,
                               hidden_size=hidden_size, batch_first=True)
        else:
            self.rnn1 = None
        if linear2:
            self.linear2 = nn.Linear(
                in_features=hidden_size, out_features=hidden_size)
        else:
            self.linear2 = None
        self.linear3 = nn.Linear(
            in_features=hidden_size, out_features=n_actions*n_agents)

    def reset(self):
        self.hidden = None

    def forward(self, view):
        b, s, p, i = view.shape
        inputs = view.reshape(b, s, p*i)
        x = F.relu(self.linear1(inputs))
        if self.rnn1:
            x, self.hidden = self.rnn1(x, self.hidden)
        if self.linear2:
            x = F.relu(self.linear2(x))
        q = self.linear3(x)
        q = q.reshape(b, s, p, self.n_actions)
        return q
