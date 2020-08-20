import torch.nn as nn
import torch.nn.functional as F
import torch as th


class GRUAgent(nn.Module):
    def __init__(self, input_shape, n_agents, n_actions, hidden_size, device):
        super(GRUAgent, self).__init__()
        self.linear1 = nn.ModuleList([
            nn.Linear(in_features=input_shape, out_features=hidden_size)
            for i in range(n_agents)
        ])

        self.rnn = nn.ModuleList([
            nn.RNNCell(input_size=hidden_size, hidden_size=hidden_size)
            for i in range(n_agents)
        ])

        self.linear2 = nn.ModuleList([
            nn.Linear(in_features=hidden_size, out_features=n_actions)
            for i in range(n_agents)
        ])
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.device = device


    def reset(self, batch_size):
        self.hidden = [
            th.zeros(batch_size, self.hidden_size, device=self.device)
            for i in range(self.n_agents)
        ]


    def forward(self, inputs):
        q = th.zeros(*inputs.shape[:3], self.n_actions, device=self.device)
        for agent_idx in range(self.n_agents):
            _q = []
            x = F.relu(self.linear1[agent_idx](inputs[agent_idx]))
            h = self.hidden[agent_idx]
            for idx in range(x.shape[1]):
                h = self.rnn[agent_idx](x[:,idx], h)
                q[agent_idx,:, idx] = self.linear2[agent_idx](h)
            self.hidden[agent_idx] = h
        return q
