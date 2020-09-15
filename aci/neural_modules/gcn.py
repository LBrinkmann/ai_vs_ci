from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class GCNModel(nn.Module):
    def __init__(self, n_actions, hidden_size, device):
        super(GCNModel, self).__init__()

        self.conv1 = GCNConv(n_actions, hidden_size, add_self_loops=False)
        self.conv2 = GCNConv(hidden_size, n_actions, add_self_loops=False)

    def reset(self):
        pass

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
